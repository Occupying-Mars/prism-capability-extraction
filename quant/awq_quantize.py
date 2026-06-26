#!/usr/bin/env python3
"""AWQ (Activation-aware Weight Quantization) W4 of the dense baked substrate,
calibrated on the leak-gated b007 BFCL mix. Closes the completeness gap next to
GPTQ (568) and AutoRound (588): AWQ protects the salient weight channels picked
by activation magnitude, the standard 3rd PTQ baseline.

Same pipeline as autoround_quantize.py: dense baked substrate in, W4 AWQ
checkpoint out, served + scored via vllm_eval.py (vLLM awq_marlin). Calibration
is the exact eval chat/tool template + gold tool_call from the leak-gated mix
(reuses gptq_quantize.build_calibration), so it is apples-to-apples with the
AutoRound/GPTQ numbers.

Runs in an isolated venv (AutoAWQ pins older transformers); quantize here, eval
in the vLLM venv.

Usage (pod, .venv-awq with `uv pip install autoawq`):
  python awq_quantize.py --model out/qwen3-8b-b007-mace90-dense \
      --train train_data/train_mixed.jsonl --out out/qwen3-8b-awq-w4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
    ap.add_argument("--zero-point", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--version", default="GEMM", help="AWQ kernel version: GEMM (Marlin-servable) or GEMV")
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    args = ap.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    calib_texts = gq.build_calibration(args.train, tokenizer, args.nsamples, args.seqlen, seed=42)
    print(f"[awq] {len(calib_texts)} leak-gated BFCL calib texts", flush=True)

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": args.version,
    }
    model = AutoAWQForCausalLM.from_pretrained(args.model, device_map="cuda", safetensors=True)
    print(f"[awq] quantizing w{args.bits} g{args.group_size} version={args.version} ...", flush=True)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "calibration_provenance.json").write_text(json.dumps({
        "method": "awq", "source": str(args.train), "n_calib": len(calib_texts),
        "bits": args.bits, "group_size": args.group_size, "version": args.version,
        "zero_point": args.zero_point, "leak_gated": True,
        "note": "AWQ on dense baked substrate, calib=b007 train mix (leak-audited vs held-out 1007)",
    }, indent=2))
    print(f"[awq] saved W{args.bits} AWQ -> {out}", flush=True)


if __name__ == "__main__":
    main()
