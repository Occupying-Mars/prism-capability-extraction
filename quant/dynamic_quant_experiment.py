#!/usr/bin/env python3
"""Dynamic (activation) quantization of the dense substrate via llm-compressor.

Weight-only quant (NF4/AutoRound W4A16) never touched activations. Dynamic quant
quantizes activations at runtime per-token, which lets the matmul run on INT8/FP8
tensor cores -> a COMPUTE/throughput win (prefill, batch) that weight-only can't
give. Near-lossless at 8-bit.

Schemes (run on A100 sm_80 = native INT8, NO native FP8):
  W8A8        int8 weights (GPTQ) + per-token DYNAMIC int8 activations + SmoothQuant
              -> the headline: native INT8 math on A100, near-lossless + throughput
  FP8_DYNAMIC fp8 weights + per-token dynamic fp8 activations, DATA-FREE
              -> accuracy ceiling; emulated on A100 (no speedup), real win on Hopper
  W4A8        int4 weights (GPTQ g128) + dynamic int8 activations + SmoothQuant
              -> cake-and-eat-it (4-bit mem + int8 compute); may degrade to W4A16
                 on stock vLLM (issue #38064)

Calibration (weights only; dynamic activations need NONE) uses the leak-gated BFCL
train mix, formatted exactly like eval — never the held-out 1007.

Usage (pod, .venv):
  python dynamic_quant_experiment.py --scheme W8A8 \
      --model models/dense-bf16 --train data/train_mixed.jsonl --out out/dyn-w8a8
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
    """Leak-gated BFCL calib formatted as chat+tools+gold (matches eval activation ranges)."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    texts = gq.build_calibration(Path(train_jsonl), tok, nsamples, maxlen, seed=42)
    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(
        lambda s: tok(s["text"], max_length=maxlen, truncation=True, add_special_tokens=False),
        remove_columns=["text"],
    )
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["W4A16", "W8A8", "FP8_DYNAMIC", "W4A8"])
    ap.add_argument("--model", required=True, help="dense bf16 substrate dir")
    ap.add_argument("--train", default="data/train_mixed.jsonl", help="leak-gated calib (NOT eval)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--smoothing", type=float, default=0.8)
    args = ap.parse_args()

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[dyn] loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.scheme == "FP8_DYNAMIC":
        # data-free RTN; per-token dynamic fp8 activations, no calibration, no SmoothQuant
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
        print("[dyn] FP8_DYNAMIC (data-free)", flush=True)
        oneshot(model=model, recipe=recipe)
    elif args.scheme == "W4A16":
        # weight-only 4-bit GPTQ (activations stay fp16); no SmoothQuant
        ds = build_calib(args.model, args.train, args.nsamples, args.maxlen)
        print(f"[dyn] W4A16 GPTQ: {len(ds)} leak-gated calib rows", flush=True)
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
        oneshot(model=model, dataset=ds, recipe=recipe,
                max_seq_length=args.maxlen, num_calibration_samples=args.nsamples)
    else:
        # W8A8 / W4A8: GPTQ weights (need calib) + per-token dynamic int8 activations
        ds = build_calib(args.model, args.train, args.nsamples, args.maxlen)
        print(f"[dyn] {args.scheme}: {len(ds)} leak-gated calib rows + SmoothQuant {args.smoothing}", flush=True)
        recipe = [
            SmoothQuantModifier(smoothing_strength=args.smoothing),
            GPTQModifier(targets="Linear", scheme=args.scheme, ignore=["lm_head"]),
        ]
        oneshot(
            model=model, dataset=ds, recipe=recipe,
            max_seq_length=args.maxlen, num_calibration_samples=args.nsamples,
        )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)
    print(f"[dyn] saved {args.scheme} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
