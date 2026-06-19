#!/usr/bin/env python3
"""AutoRound (SignRound) W4A16 quantization of the dense baked substrate,
calibrated on the leak-gated BFCL mix. Pure-torch SignSGD rounding -> runs on
Blackwell sm_120 (no nvcc). Export format='auto_round' (sym INT4) -> vLLM
gptq:marlin prebuilt cubin for serving.

Eval-aware + drift-safe: AutoRound learns only weight rounding/clipping to
minimize per-layer output error on the BFCL calibration — it never touches the
(already-merged) adapter, so it cannot reproduce the LoRA-recovery collapse.

Usage (pod, .venv with `uv pip install auto-round`):
  python autoround_quantize.py --model out/qwen3-8b-b007-mace90-dense \
      --train train_data/train_mixed.jsonl --out out/qwen3-8b-ar-w4
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gptq_quantize", HERE / "gptq_quantize.py")
gq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dense baked substrate dir")
    ap.add_argument("--train", type=Path, required=True, help="leak-gated train mix (NOT eval)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--format", default="auto_round")
    args = ap.parse_args()

    from auto_round import AutoRound
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # leak-gated BFCL calibration: exact eval chat/tool template + gold tool_call
    calib_texts = gq.build_calibration(args.train, tokenizer, args.nsamples, args.seqlen, seed=42)
    print(f"[autoround] {len(calib_texts)} leak-gated BFCL calib texts", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="cuda")
    ar = AutoRound(
        model,
        tokenizer,
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        dataset=calib_texts,        # list of formatted task strings
        nsamples=args.nsamples,
        iters=args.iters,
        seqlen=args.seqlen,
    )
    ar.quantize_and_save(output_dir=args.out, format=args.format)
    print(f"[autoround] saved W{args.bits} ({args.format}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
