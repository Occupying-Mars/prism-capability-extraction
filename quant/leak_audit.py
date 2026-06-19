#!/usr/bin/env python3
"""Independent leak gate: assert a training/calibration corpus does NOT overlap
the held-out BFCL eval pairs.

Eval-aware quant calibrates/recovers on *task-distribution training data* that is
leak-gated against the held-out eval — it must never touch the 1007 eval rows.
This re-checks that, on top of tokenbender's own mixed_overlap_audit.json, before
we GPTQ-calibrate or train any recovery LoRA. Exits non-zero on any overlap so it
can hard-gate a pipeline.

Checks (per row, on the user-prompt text and the gold tool-call target):
  - exact prompt overlap
  - exact target overlap
  - near-duplicate prompt (Jaccard over 5-grams >= --near-threshold)

Usage:
  python leak_audit.py --train train_mixed.jsonl --eval pairs.jsonl --out audit.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prompt_text(row) -> str:
    """Concatenate user/system message text, robust to schema variants."""
    msgs = row.get("messages") or row.get("question") or row.get("prompt") or ""
    if isinstance(msgs, str):
        return msgs.strip()
    parts = []
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            elif isinstance(m, list):
                for mm in m:
                    if isinstance(mm, dict):
                        parts.append(str(mm.get("content", "")))
            else:
                parts.append(str(m))
    return " ".join(parts).strip()


def target_text(row) -> str:
    for k in ("target", "reference_calls", "answer", "ground_truth"):
        if row.get(k) not in (None, "", []):
            return json.dumps(row[k], sort_keys=True, ensure_ascii=False)
    return ""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def shingles(s: str, n: int = 5):
    toks = norm(s).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--near-threshold", type=float, default=0.85)
    ap.add_argument("--shingle-size", type=int, default=5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    eval_rows = list(read_jsonl(args.eval))
    eval_prompts = {norm(prompt_text(r)) for r in eval_rows}
    eval_targets = {norm(target_text(r)) for r in eval_rows}
    eval_shingles = [shingles(prompt_text(r), args.shingle_size) for r in eval_rows]

    train_rows = list(read_jsonl(args.train))
    exact_prompt = exact_target = 0
    max_near = 0.0
    near_hits = 0
    for r in train_rows:
        p = norm(prompt_text(r))
        t = norm(target_text(r))
        if p and p in eval_prompts:
            exact_prompt += 1
        if t and t in eval_targets:
            exact_target += 1
        sh = shingles(prompt_text(r), args.shingle_size)
        best = max((jaccard(sh, es) for es in eval_shingles), default=0.0)
        max_near = max(max_near, best)
        if best >= args.near_threshold:
            near_hits += 1

    passed = exact_prompt == 0 and exact_target == 0 and near_hits == 0
    audit = {
        "train_jsonl": str(args.train),
        "eval_jsonl": str(args.eval),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "near_threshold": args.near_threshold,
        "shingle_size": args.shingle_size,
        "exact_prompt_overlaps": exact_prompt,
        "exact_target_overlaps": exact_target,
        "near_overlaps": near_hits,
        "max_near_similarity": max_near,
        "passed": passed,
    }
    print(json.dumps(audit, indent=2))
    if args.out:
        args.out.write_text(json.dumps(audit, indent=2))
    if not passed:
        raise SystemExit("LEAK DETECTED — refusing to use this corpus for calibration/training")


if __name__ == "__main__":
    main()
