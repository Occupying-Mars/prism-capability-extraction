#!/usr/bin/env python3
"""Summarize BFCL issue #12 masked-eval outputs into a frontier table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_candidates(path: Path) -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in read_jsonl(path)}


def summarize_generation(
    path: Path,
    *,
    meta_by_id: dict[str, dict[str, Any]],
    full_anchor: int,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    total = len(rows)
    correct = sum(int(row.get("normalized_correct", row.get("correct", False))) for row in rows)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_repair: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        meta = meta_by_id.get(row["id"], {})
        ok = bool(row.get("normalized_correct", row.get("correct", False)))
        value = "correct" if ok else "total"
        category = meta.get("category", "unknown")
        split = meta.get("split_role", "unknown")
        by_category[category]["total"] += 1
        by_category[category]["correct"] += int(ok)
        by_split[split]["total"] += 1
        by_split[split]["correct"] += int(ok)
        repair_buckets = meta.get("repair_buckets") or ["none"]
        for bucket in repair_buckets:
            by_repair[bucket]["total"] += 1
            by_repair[bucket]["correct"] += int(ok)
    return {
        "generations": str(path),
        "examples": total,
        "normalized_exact_correct": correct,
        "normalized_exact_accuracy": correct / total if total else None,
        "recovery_vs_full_anchor": correct / full_anchor if full_anchor else None,
        "category_scores": {
            key: {
                "correct": val["correct"],
                "total": val["total"],
                "accuracy": val["correct"] / val["total"] if val["total"] else None,
            }
            for key, val in sorted(by_category.items())
        },
        "split_scores": {
            key: {
                "correct": val["correct"],
                "total": val["total"],
                "accuracy": val["correct"] / val["total"] if val["total"] else None,
            }
            for key, val in sorted(by_split.items())
        },
        "repair_bucket_scores": {
            key: {
                "correct": val["correct"],
                "total": val["total"],
                "accuracy": val["correct"] / val["total"] if val["total"] else None,
            }
            for key, val in sorted(by_repair.items())
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query-manifest", type=Path, required=True)
    p.add_argument("--candidate-jsonl", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--full-anchor", type=int, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_by_id = {row["eval_id"]: row for row in read_jsonl(args.query_manifest)}
    candidates = read_candidates(args.candidate_jsonl)
    frontier: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []

    for gen_path in sorted(args.eval_dir.glob("*.jsonl")):
        candidate_id = gen_path.stem
        if candidate_id.endswith(".summary"):
            continue
        candidate = candidates.get(candidate_id, {"candidate_id": candidate_id})
        summary = summarize_generation(gen_path, meta_by_id=meta_by_id, full_anchor=args.full_anchor)
        merged = dict(candidate)
        merged.update(
            {
                "candidate_id": candidate_id,
                "score": summary["normalized_exact_correct"],
                "examples": summary["examples"],
                "accuracy": summary["normalized_exact_accuracy"],
                "recovery_vs_full_anchor": summary["recovery_vs_full_anchor"],
            }
        )
        frontier.append(merged)
        detailed.append({"candidate": candidate, "summary": summary})

    frontier.sort(
        key=lambda row: (
            row.get("recovery_vs_full_anchor") or 0.0,
            -(row.get("selected_mlp_channels") or 10**12),
        ),
        reverse=True,
    )

    (args.out_dir / "frontier.json").write_text(json.dumps(frontier, indent=2))
    (args.out_dir / "detailed_scores.json").write_text(json.dumps(detailed, indent=2))

    csv_path = args.out_dir / "frontier.csv"
    fieldnames = [
        "candidate_id",
        "kind",
        "selected_mlp_channels",
        "mlp_fraction",
        "score",
        "examples",
        "accuracy",
        "recovery_vs_full_anchor",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in frontier:
            writer.writerow(row)

    thresholds = {
        "80": int((0.80 * args.full_anchor) + 0.999999),
        "85": int((0.85 * args.full_anchor) + 0.999999),
        "90": int((0.90 * args.full_anchor) + 0.999999),
        "95": int((0.95 * args.full_anchor) + 0.999999),
    }
    threshold_hits = {}
    for name, threshold in thresholds.items():
        hits = [row for row in frontier if int(row.get("score") or 0) >= threshold]
        threshold_hits[name] = {
            "threshold_count": threshold,
            "hit_count": len(hits),
            "smallest_hit": min(hits, key=lambda row: row.get("selected_mlp_channels") or 10**12) if hits else None,
        }
    (args.out_dir / "threshold_hits.json").write_text(json.dumps(threshold_hits, indent=2))

    print(json.dumps({"frontier_count": len(frontier), "threshold_hits": threshold_hits}, indent=2))


if __name__ == "__main__":
    main()
