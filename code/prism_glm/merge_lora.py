#!/usr/bin/env python3
"""Merge a GLM LoRA adapter into a local attribution/eval target."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

if hasattr(torch, "distributed") and not hasattr(torch.distributed, "tensor"):
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        pass

from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    args.output.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    if not hasattr(config, "max_length"):
        config.max_length = getattr(config, "seq_length", 8192)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map=args.device_map,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    print(f"[merge] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
