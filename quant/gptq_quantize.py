#!/usr/bin/env python3
"""Eval-aware GPTQ 4-bit quantization of Qwen3-8B, calibrated on the LEAK-GATED
b007 BFCL training mix (never the held-out 1007 eval).

GPTQ is 2nd-order PTQ: it minimizes layer output error on calibration activations,
so calibrating on the *task distribution* (function-calling prompts + gold tool
calls) is the "eval-aware" part — far better than WikiText RTN. The eval set
stays held out; see leak_audit.py / mixed_overlap_audit.json for the gate.

Output: a 4-bit GPTQ checkpoint loadable by transformers (with gptqmodel
installed). Quantize the base only; the b007 LoRA + MACE mask + eval are applied
downstream by quantize_substrate.py --method gptq --gptq-path <out>.

Usage (pod, .venv):
  python gptq_quantize.py --train train_data/train_mixed.jsonl \
      --out out/qwen3-8b-gptq4 --n-calib 256
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_calibration(train_path: Path, tokenizer, n_calib: int, max_len: int, seed: int):
    """Format leak-gated train rows as full task strings: chat prompt + gold call."""
    rows = [r for r in read_jsonl(train_path) if (r.get("target_text") or "").strip()]
    random.Random(seed).shuffle(rows)
    calib = []
    for r in rows:
        prompt = tokenizer.apply_chat_template(
            r["messages"],
            tools=r.get("tools") or None,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        text = prompt + r["target_text"].strip()
        ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
        if len(ids) < 8:
            continue
        calib.append(text)
        if len(calib) >= n_calib:
            break
    print(f"[calib] built {len(calib)} examples from {train_path} (leak-gated)", flush=True)
    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train", type=Path, required=True, help="leak-gated train mix (NOT eval)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    calib = build_calibration(args.train, tokenizer, args.n_calib, args.max_len, args.seed)

    qcfg = QuantizeConfig(bits=args.bits, group_size=args.group_size)
    print(f"[gptq] loading {args.model} bits={args.bits} group_size={args.group_size}", flush=True)
    model = GPTQModel.load(args.model, qcfg)
    print(f"[gptq] quantizing on {len(calib)} calibration rows ...", flush=True)
    model.quantize(calib, batch_size=1)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    (args.out / "calibration_provenance.json").write_text(
        json.dumps(
            {
                "source": str(args.train),
                "n_calib": len(calib),
                "bits": args.bits,
                "group_size": args.group_size,
                "leak_gated": True,
                "note": "calibrated on b007 train mix, leak-audited vs held-out 1007 eval",
            },
            indent=2,
        )
    )
    print(f"[gptq] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
