#!/usr/bin/env python3
"""Audit overlap between BFCL-style train rows and held-out eval rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--near-threshold", type=float, default=0.85)
    p.add_argument("--shingle-size", type=int, default=5)
    p.add_argument("--max-reports", type=int, default=100)
    p.add_argument("--fail-on-overlap", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = stable_json(value)
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any], idx: int) -> str:
    return str(row.get("id") or row.get("mix_id") or row.get("source_id") or idx)


def row_parts(row: dict[str, Any]) -> dict[str, str]:
    prompt_obj = {
        "messages": row.get("messages"),
        "tools": row.get("tools"),
        "question": row.get("question"),
        "prompt": row.get("prompt"),
    }
    target_obj = {
        "target_text": row.get("target_text"),
        "target": row.get("target"),
        "reference_calls": row.get("reference_calls"),
        "answer": row.get("answer"),
    }
    prompt = normalize_text(prompt_obj)
    target = normalize_text(target_obj)
    return {
        "prompt": prompt,
        "target": target,
        "combined": f"{prompt}\n{target}",
    }


def token_shingles(text: str, size: int) -> set[str]:
    tokens = re.findall(r"[a-z0-9_./:-]+|[{}()[\\],:=<>]", text)
    if not tokens:
        return set()
    if len(tokens) <= size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def summarize_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    parts = row_parts(row)
    return {
        "id": row_id(row, idx),
        "source": row.get("source"),
        "prompt_preview": parts["prompt"][:240],
        "target_preview": parts["target"][:240],
    }


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_jsonl)
    eval_rows = read_jsonl(args.eval_jsonl)

    train_records = []
    eval_records = []
    for idx, row in enumerate(train_rows):
        parts = row_parts(row)
        train_records.append(
            {
                "idx": idx,
                "row": row,
                "id": row_id(row, idx),
                "parts": parts,
                "row_hash": sha(parts["combined"]),
                "prompt_hash": sha(parts["prompt"]),
                "target_hash": sha(parts["target"]),
                "shingles": token_shingles(parts["combined"], args.shingle_size),
            }
        )
    for idx, row in enumerate(eval_rows):
        parts = row_parts(row)
        eval_records.append(
            {
                "idx": idx,
                "row": row,
                "id": row_id(row, idx),
                "parts": parts,
                "row_hash": sha(parts["combined"]),
                "prompt_hash": sha(parts["prompt"]),
                "target_hash": sha(parts["target"]),
                "shingles": token_shingles(parts["combined"], args.shingle_size),
            }
        )

    eval_by_row_hash = defaultdict(list)
    eval_by_prompt_hash = defaultdict(list)
    eval_by_target_hash = defaultdict(list)
    inverted: dict[str, list[int]] = defaultdict(list)
    for rec in eval_records:
        eval_by_row_hash[rec["row_hash"]].append(rec)
        eval_by_prompt_hash[rec["prompt_hash"]].append(rec)
        eval_by_target_hash[rec["target_hash"]].append(rec)
        for shingle in rec["shingles"]:
            inverted[shingle].append(rec["idx"])

    exact_rows = []
    exact_prompts = []
    target_only = []
    near = []
    max_near_similarity = 0.0

    for train in train_records:
        for ev in eval_by_row_hash.get(train["row_hash"], []):
            exact_rows.append({"train": summarize_row(train["row"], train["idx"]), "eval": summarize_row(ev["row"], ev["idx"])})
        for ev in eval_by_prompt_hash.get(train["prompt_hash"], []):
            exact_prompts.append({"train": summarize_row(train["row"], train["idx"]), "eval": summarize_row(ev["row"], ev["idx"])})
        for ev in eval_by_target_hash.get(train["target_hash"], []):
            target_only.append({"train": summarize_row(train["row"], train["idx"]), "eval": summarize_row(ev["row"], ev["idx"])})

        if not train["shingles"]:
            continue
        candidates = Counter()
        for shingle in train["shingles"]:
            for eval_idx in inverted.get(shingle, []):
                candidates[eval_idx] += 1
        for eval_idx, shared in candidates.most_common():
            ev = eval_records[eval_idx]
            union = len(train["shingles"] | ev["shingles"])
            if union == 0:
                continue
            similarity = shared / union
            max_near_similarity = max(max_near_similarity, similarity)
            if similarity >= args.near_threshold:
                near.append(
                    {
                        "similarity": similarity,
                        "train": summarize_row(train["row"], train["idx"]),
                        "eval": summarize_row(ev["row"], ev["idx"]),
                    }
                )
                if len(near) >= args.max_reports:
                    break
        if len(near) >= args.max_reports:
            break

    report = {
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "near_threshold": args.near_threshold,
        "shingle_size": args.shingle_size,
        "exact_row_overlaps": exact_rows[: args.max_reports],
        "exact_prompt_overlaps": exact_prompts[: args.max_reports],
        "exact_target_only_overlaps": target_only[: args.max_reports],
        "near_overlaps": near[: args.max_reports],
        "counts": {
            "exact_row_overlaps": len(exact_rows),
            "exact_prompt_overlaps": len(exact_prompts),
            "exact_target_only_overlaps": len(target_only),
            "near_overlaps_reported": len(near),
        },
        "max_near_similarity": max_near_similarity,
        "passed": len(exact_rows) == 0 and len(exact_prompts) == 0 and len(near) == 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_overlap and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
