#!/usr/bin/env python3
"""Filter BFCL-style training rows by encoded chat-template length."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_bfcl_masked_lora import encode_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-length", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    dropped = []
    lengths = []
    for idx, row in enumerate(rows):
        enc = encode_row(row, tokenizer, 1_000_000)
        if enc is None:
            dropped.append({"idx": idx, "length": None, "row": row})
            continue
        length = int(enc["input_ids"].shape[0])
        lengths.append(length)
        if length <= args.max_length:
            kept.append(row)
        else:
            dropped.append({"idx": idx, "length": length, "row": row})

    with args.output.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "max_length": args.max_length,
        "rows": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "max_seen_length": max(lengths) if lengths else None,
        "dropped_rows": [
            {
                "idx": item["idx"],
                "length": item["length"],
                "source": item["row"].get("source"),
                "prompt_preview": ((item["row"].get("messages") or [{}])[0].get("content") or "")[:200],
            }
            for item in dropped[:20]
        ],
    }
    args.output.with_name("manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
