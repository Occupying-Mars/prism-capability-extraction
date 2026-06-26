#!/usr/bin/env python3
"""SmoothQuant W8A8 — isolate what activation-outlier migration buys on the
brittle substrate.

The earlier W8A8 result (563, -38 vs bf16) already ran SmoothQuant(0.8) inside
dynamic_quant_experiment.py, so "-38" was NOT naive W8A8. To make the SmoothQuant
claim honest we measure its *contribution*: W8A8 with NO SmoothQuant vs a
smoothing-strength sweep. SmoothQuant migrates per-channel activation outliers
into the weights (scale alpha), so int8 activation quant has fewer outliers to
clip — the standard fix for W8A8 accuracy loss.

GPTQ W8A8 weights (calibrated on the leak-gated b007 BFCL mix) + per-token
dynamic int8 activations. Output compressed-tensors -> vLLM int8. Eval the held
out 1007 via vllm_eval.py.

--smoothing -1  -> no SmoothQuant (naive W8A8 control)
--smoothing a   -> SmoothQuantModifier(smoothing_strength=a)

Usage (pod, .venv-llmc with llmcompressor; transformers<5):
  python smoothquant_w8a8.py --model out/qwen3-8b-b007-mace90-dense \
      --train train_data/train_mixed.jsonl --smoothing 0.8 --out out/sq-w8a8-a08
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gptq_quantize", HERE / "gptq_quantize.py")
gq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gq)


def build_calib(model_id, train_jsonl, nsamples, maxlen):
    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    texts = gq.build_calibration(Path(train_jsonl), tok, nsamples, maxlen, seed=42)
    ds = Dataset.from_dict({"text": texts})
    return ds.map(
        lambda s: tok(s["text"], max_length=maxlen, truncation=True, add_special_tokens=False),
        remove_columns=["text"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dense baked substrate dir")
    ap.add_argument("--train", required=True, help="leak-gated calib (NOT eval)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoothing", type=float, default=0.8, help="-1 disables SmoothQuant")
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--maxlen", type=int, default=512)
    args = ap.parse_args()

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[sq] loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = build_calib(args.model, args.train, args.nsamples, args.maxlen)

    recipe = []
    if args.smoothing >= 0:
        recipe.append(SmoothQuantModifier(smoothing_strength=args.smoothing))
        print(f"[sq] SmoothQuant alpha={args.smoothing}", flush=True)
    else:
        print("[sq] NO SmoothQuant (naive W8A8 control)", flush=True)
    recipe.append(GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]))

    print(f"[sq] W8A8 GPTQ on {len(ds)} leak-gated calib rows", flush=True)
    oneshot(model=model, dataset=ds, recipe=recipe,
            max_seq_length=args.maxlen, num_calibration_samples=args.nsamples)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)
    print(f"[sq] saved W8A8 (smoothing={args.smoothing}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
