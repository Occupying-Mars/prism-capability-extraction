#!/usr/bin/env python3
"""Build issue #12 category-floor repair masks around a MACE incumbent.

The builder compares a base masked eval against the full-anchor eval, finds
selection-split examples that the full model gets right and the base mask gets
wrong, and ranks non-selected MLP channels that activate on those failures.
It then swaps a small tail of the base mask for category- or bucket-focused
repair donors while keeping the selected-channel budget fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_SELECT = {"train", "calibration", "validation"}

SEGMENT_WEIGHT = {
    "prompt": 0.85,
    "target": 1.20,
    "full": 1.00,
}

STAT_WEIGHT = {
    "mean_abs": 1.00,
    "rms": 1.05,
    "max_abs": 0.75,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scores(path: Path) -> np.ndarray:
    scores = np.load(path)["mlp_scores"].astype(np.float32, copy=False)
    if scores.ndim != 2:
        raise ValueError(f"expected 2D mlp_scores in {path}, got {scores.shape}")
    return scores


def ranking(scores: np.ndarray) -> list[int]:
    flat = scores.reshape(-1)
    idx = np.arange(flat.size)
    ordered = idx[np.lexsort((idx, -flat))]
    return [int(gid) for gid in ordered]


def layer_channel(gid: int, d_ffn: int) -> tuple[int, int]:
    return int(gid // d_ffn), int(gid % d_ffn)


def channel_id(layer: int, channel: int, d_ffn: int) -> int:
    return int(layer) * d_ffn + int(channel)


def eval_correct(path: Path) -> dict[str, bool]:
    return {
        str(row["id"]): bool(row.get("normalized_correct", row.get("correct", False)))
        for row in read_jsonl(path)
    }


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def top_channel_weight(row: dict[str, Any], item: dict[str, Any]) -> float:
    rank = max(int(item.get("rank", 1)), 1)
    local_score = float(item.get("local_score", 1))
    global_score = float(item.get("global_score", 1))
    value = max(float(item.get("value", 0.0)), 0.0)
    return (
        SEGMENT_WEIGHT.get(str(row.get("segment")), 1.0)
        * STAT_WEIGHT.get(str(row.get("stat")), 1.0)
        * (1.0 / math.sqrt(rank))
        * (0.6 * local_score + 0.4 * global_score)
        * math.log1p(value)
    )


def write_mask(path: Path, selected: list[int], *, n_layers: int, d_ffn: int) -> None:
    scores = np.zeros((n_layers, d_ffn), dtype=np.float32)
    for rank, gid in enumerate(selected):
        layer, channel = layer_channel(gid, d_ffn)
        scores[layer, channel] = float(len(selected) - rank)
    np.savez_compressed(path, mlp_scores=scores)


def fill_selected(
    keep: list[int],
    donor_rank: list[int],
    fallback_rank: list[int],
    *,
    budget: int,
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for source in (keep, donor_rank, fallback_rank):
        for gid in source:
            if gid in seen:
                continue
            selected.append(gid)
            seen.add(gid)
            if len(selected) >= budget:
                return selected
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-mask", type=Path, required=True)
    p.add_argument("--base-topk", type=int, required=True)
    p.add_argument("--base-eval", type=Path, required=True)
    p.add_argument("--full-eval", type=Path, required=True)
    p.add_argument("--query-manifest", type=Path, required=True)
    p.add_argument("--top-channels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--replace-counts", default="500,1000,2000,5000,10000")
    p.add_argument("--top-per-plane", type=int, default=64)
    p.add_argument("--floor-categories", default="java,javascript,live_simple")
    p.add_argument("--repair-buckets", default="live_slot_values,time_normalization")
    p.add_argument("--category-focus-weight", type=float, default=2.0)
    p.add_argument("--bucket-focus-weight", type=float, default=1.7)
    p.add_argument("--stable-success-weight", type=float, default=-0.04)
    p.add_argument("--base-only-success-weight", type=float, default=-0.30)
    p.add_argument("--full-and-base-fail-weight", type=float, default=0.20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "candidate_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    base_scores = load_scores(args.base_mask)
    n_layers, d_ffn = base_scores.shape
    total_channels = n_layers * d_ffn
    base_rank = ranking(base_scores)
    base_selected = base_rank[: args.base_topk]
    base_set = set(base_selected)
    fallback_rank = [gid for gid in base_rank if gid not in base_set]
    replace_counts = parse_ints(args.replace_counts)
    floor_categories = set(parse_names(args.floor_categories))
    repair_buckets = set(parse_names(args.repair_buckets))

    meta_by_id = {str(row["eval_id"]): row for row in read_jsonl(args.query_manifest)}
    base_correct = eval_correct(args.base_eval)
    full_correct = eval_correct(args.full_eval)
    eligible = {
        eval_id
        for eval_id, meta in meta_by_id.items()
        if meta.get("split_role") in SPLIT_SELECT and eval_id in base_correct and eval_id in full_correct
    }

    global_scores: Counter[int] = Counter()
    category_scores: dict[str, Counter[int]] = defaultdict(Counter)
    bucket_scores: dict[str, Counter[int]] = defaultdict(Counter)
    outcome_counts: Counter[str] = Counter()
    outcome_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    outcome_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    rows_seen = rows_scored = entries_scored = 0

    query_weights: dict[str, float] = {}
    for eval_id in eligible:
        meta = meta_by_id[eval_id]
        category = str(meta.get("category", "unknown"))
        buckets = set(meta.get("repair_buckets") or [])
        focus = 1.0
        if category in floor_categories:
            focus *= args.category_focus_weight
        if buckets & repair_buckets:
            focus *= args.bucket_focus_weight
        if full_correct[eval_id] and not base_correct[eval_id]:
            outcome = "full_correct_base_wrong"
            weight = focus
        elif full_correct[eval_id] and base_correct[eval_id]:
            outcome = "both_correct"
            weight = args.stable_success_weight
        elif not full_correct[eval_id] and base_correct[eval_id]:
            outcome = "base_only_correct"
            weight = args.base_only_success_weight
        else:
            outcome = "both_wrong"
            weight = args.full_and_base_fail_weight * focus
        query_weights[eval_id] = weight
        outcome_counts[outcome] += 1
        outcome_by_category[category][outcome] += 1
        for bucket in buckets or {"none"}:
            outcome_by_bucket[bucket][outcome] += 1

    for row in read_jsonl(args.top_channels):
        rows_seen += 1
        eval_id = str(row["eval_id"])
        q_weight = query_weights.get(eval_id)
        if q_weight is None:
            continue
        rows_scored += 1
        meta = meta_by_id.get(eval_id, {})
        category = str(meta.get("category", "unknown"))
        buckets = set(meta.get("repair_buckets") or [])
        for item in row.get("top", [])[: args.top_per_plane]:
            gid = channel_id(item["layer"], item["channel"], d_ffn)
            if gid in base_set:
                continue
            value = q_weight * top_channel_weight(row, item)
            global_scores[gid] += value
            category_scores[category][gid] += value
            for bucket in buckets:
                bucket_scores[bucket][gid] += value
            entries_scored += 1

    candidates: list[dict[str, Any]] = []
    seen_masks: set[str] = set()

    def donor_rank(scores: Counter[int]) -> list[int]:
        return sorted(scores, key=lambda gid: (scores[gid], -gid), reverse=True)

    def add_candidate(candidate_id: str, kind: str, donor: list[int], replace: int, lineage: dict[str, Any]) -> None:
        if replace <= 0 or replace >= args.base_topk:
            return
        keep = base_selected[: args.base_topk - replace]
        selected = fill_selected(keep, donor, fallback_rank, budget=args.base_topk)
        key = hashlib.sha1(np.asarray(selected, dtype=np.int32).tobytes()).hexdigest()
        if key in seen_masks:
            return
        seen_masks.add(key)
        path = mask_dir / f"{candidate_id}.npz"
        write_mask(path, selected, n_layers=n_layers, d_ffn=d_ffn)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "kind": kind,
                "mask_path": str(path.relative_to(args.out_dir)),
                "selected_mlp_channels": len(selected),
                "mlp_fraction": len(selected) / total_channels,
                "topk_for_eval": len(selected),
                "selected_sha1": key,
                "lineage": lineage | {"replace": replace, "base_topk": args.base_topk},
            }
        )

    global_rank = donor_rank(global_scores)
    for replace in replace_counts:
        add_candidate(
            f"category_repair_global_r{replace}",
            "category_floor_global_repair",
            global_rank,
            replace,
            {"donor": "global_full_correct_base_wrong_pressure"},
        )
        for category in sorted(floor_categories):
            add_candidate(
                f"category_repair_{category}_r{replace}",
                "category_floor_specific_repair",
                donor_rank(category_scores[category]),
                replace,
                {"donor": "category_specific_pressure", "category": category},
            )
        for bucket in sorted(repair_buckets):
            add_candidate(
                f"bucket_repair_{bucket}_r{replace}",
                "repair_bucket_specific_repair",
                donor_rank(bucket_scores[bucket]),
                replace,
                {"donor": "repair_bucket_pressure", "repair_bucket": bucket},
            )

    manifest = {
        "issue": 12,
        "artifact": "bfcl_issue12_category_repair_candidates",
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_channels": total_channels,
        "selection_splits": sorted(SPLIT_SELECT),
        "heldout_policy": "heldout rows are excluded from category-repair channel scoring and branch construction",
        "inputs": {
            "base_mask": str(args.base_mask),
            "base_mask_sha256": sha256_file(args.base_mask),
            "base_eval": str(args.base_eval),
            "base_eval_sha256": sha256_file(args.base_eval),
            "full_eval": str(args.full_eval),
            "full_eval_sha256": sha256_file(args.full_eval),
            "query_manifest": str(args.query_manifest),
            "query_manifest_sha256": sha256_file(args.query_manifest),
            "top_channels": str(args.top_channels),
            "top_channels_sha256": sha256_file(args.top_channels),
        },
        "build_params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {
            "replace_counts": replace_counts,
            "floor_categories": sorted(floor_categories),
            "repair_buckets": sorted(repair_buckets),
        },
        "outcome_counts_selection_only": dict(outcome_counts),
        "outcome_by_category": {key: dict(val) for key, val in sorted(outcome_by_category.items())},
        "outcome_by_bucket": {key: dict(val) for key, val in sorted(outcome_by_bucket.items())},
        "top_channel_rows_seen": rows_seen,
        "top_channel_rows_scored": rows_scored,
        "top_channel_entries_scored": entries_scored,
        "candidate_count": len(candidates),
    }
    (args.out_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.out_dir / "candidate_masks.jsonl").open("w") as f:
        for row in candidates:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
