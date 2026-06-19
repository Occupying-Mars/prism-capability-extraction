#!/usr/bin/env python3
"""Build issue #12 rescue-swap masks from two evaluated parent frontiers.

This operator is meant to pressure a known MACE-90 incumbent downward. It
compares a smaller parent mask against a larger incumbent, finds examples that
the incumbent rescues, ranks outside-prefix channels that activate on those
rescued examples, and swaps those channels into smaller fixed-budget masks.

Selection uses train/calibration/validation rows only. Heldout rows may appear
in the reference eval outputs, but they do not contribute channel scores.
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


def layer_channel(gid: int, d_ffn: int) -> tuple[int, int]:
    return int(gid // d_ffn), int(gid % d_ffn)


def channel_id(layer: int, channel: int, d_ffn: int) -> int:
    return int(layer) * d_ffn + int(channel)


def ranking(scores: np.ndarray) -> list[int]:
    flat = scores.reshape(-1)
    idx = np.arange(flat.size)
    ordered = idx[np.lexsort((idx, -flat))]
    return [int(gid) for gid in ordered]


def eval_correct(path: Path) -> dict[str, bool]:
    rows = read_jsonl(path)
    return {str(row["id"]): bool(row.get("normalized_correct", row.get("correct", False))) for row in rows}


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def top_channel_weight(row: dict[str, Any], item: dict[str, Any]) -> float:
    segment = str(row.get("segment"))
    stat = str(row.get("stat"))
    rank = max(int(item.get("rank", 1)), 1)
    local_score = float(item.get("local_score", 1))
    global_score = float(item.get("global_score", 1))
    value = max(float(item.get("value", 0.0)), 0.0)
    rank_weight = 1.0 / math.sqrt(rank)
    return (
        SEGMENT_WEIGHT.get(segment, 1.0)
        * STAT_WEIGHT.get(stat, 1.0)
        * rank_weight
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
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--low-eval", type=Path, required=True)
    p.add_argument("--high-eval", type=Path, required=True)
    p.add_argument("--query-manifest", type=Path, required=True)
    p.add_argument("--top-channels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--budgets", default="140000,150000,160000,170000,175000")
    p.add_argument("--replace-counts", default="2000,5000,10000,15100,20000,30000")
    p.add_argument("--donor-uppers", default="175100,200000,240000")
    p.add_argument("--top-per-plane", type=int, default=64)
    p.add_argument("--still-fail-weight", type=float, default=0.35)
    p.add_argument("--lost-weight", type=float, default=-0.60)
    p.add_argument("--stable-success-weight", type=float, default=-0.05)
    p.add_argument("--emit-band-shift", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--emit-rescue-swap", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "candidate_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    parent_scores = load_scores(args.parent)
    n_layers, d_ffn = parent_scores.shape
    total_channels = n_layers * d_ffn
    parent_rank = ranking(parent_scores)
    rank_pos = {gid: pos for pos, gid in enumerate(parent_rank)}

    budgets = parse_ints(args.budgets)
    replace_counts = parse_ints(args.replace_counts)
    donor_uppers = parse_ints(args.donor_uppers)

    low_correct = eval_correct(args.low_eval)
    high_correct = eval_correct(args.high_eval)
    meta_by_id = {str(row["eval_id"]): row for row in read_jsonl(args.query_manifest)}

    eligible_ids = {
        eval_id
        for eval_id, meta in meta_by_id.items()
        if meta.get("split_role") in SPLIT_SELECT and eval_id in low_correct and eval_id in high_correct
    }
    rescued = {
        eval_id
        for eval_id in eligible_ids
        if high_correct[eval_id] and not low_correct[eval_id]
    }
    lost = {
        eval_id
        for eval_id in eligible_ids
        if low_correct[eval_id] and not high_correct[eval_id]
    }
    stable_success = {
        eval_id
        for eval_id in eligible_ids
        if low_correct[eval_id] and high_correct[eval_id]
    }
    stable_fail = {
        eval_id
        for eval_id in eligible_ids
        if not low_correct[eval_id] and not high_correct[eval_id]
    }

    query_weights: dict[str, float] = {}
    for eval_id in rescued:
        query_weights[eval_id] = 1.0
    for eval_id in stable_fail:
        query_weights[eval_id] = args.still_fail_weight
    for eval_id in lost:
        query_weights[eval_id] = args.lost_weight
    for eval_id in stable_success:
        query_weights[eval_id] = args.stable_success_weight

    rescue_scores: Counter[int] = Counter()
    category_scores: dict[str, Counter[int]] = defaultdict(Counter)
    rows_seen = 0
    rows_scored = 0
    entries_scored = 0
    for row in read_jsonl(args.top_channels):
        rows_seen += 1
        eval_id = str(row["eval_id"])
        q_weight = query_weights.get(eval_id)
        if q_weight is None:
            continue
        rows_scored += 1
        category = str(meta_by_id.get(eval_id, {}).get("category", "unknown"))
        for item in row.get("top", [])[: args.top_per_plane]:
            gid = channel_id(item["layer"], item["channel"], d_ffn)
            value = q_weight * top_channel_weight(row, item)
            rescue_scores[gid] += value
            category_scores[category][gid] += value
            entries_scored += 1

    candidates: list[dict[str, Any]] = []
    seen_mask_keys: set[str] = set()

    def add_candidate(candidate_id: str, kind: str, selected: list[int], lineage: dict[str, Any]) -> None:
        selected = list(dict.fromkeys(selected))
        if len(selected) != lineage["budget"]:
            raise ValueError(f"{candidate_id} selected {len(selected)} != budget {lineage['budget']}")
        mask_key = hashlib.sha1(np.asarray(selected, dtype=np.int32).tobytes()).hexdigest()
        if mask_key in seen_mask_keys:
            return
        seen_mask_keys.add(mask_key)
        mask_path = mask_dir / f"{candidate_id}.npz"
        write_mask(mask_path, selected, n_layers=n_layers, d_ffn=d_ffn)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "kind": kind,
                "mask_path": str(mask_path.relative_to(args.out_dir)),
                "selected_mlp_channels": len(selected),
                "mlp_fraction": len(selected) / total_channels,
                "topk_for_eval": len(selected),
                "selected_sha1": mask_key,
                "lineage": lineage,
            }
        )

    def rescue_rank_for_budget(budget: int, donor_upper: int, scores: Counter[int]) -> list[int]:
        donor_upper = min(donor_upper, total_channels)
        pool = parent_rank[budget:donor_upper]
        return sorted(
            pool,
            key=lambda gid: (scores.get(gid, 0.0), -rank_pos[gid]),
            reverse=True,
        )

    for budget in budgets:
        if budget <= 0 or budget > total_channels:
            raise ValueError(f"invalid budget {budget}")
        for replace in replace_counts:
            if replace <= 0 or replace >= budget:
                continue
            keep = parent_rank[: budget - replace]
            for donor_upper in donor_uppers:
                if donor_upper <= budget:
                    continue
                donor_upper = min(donor_upper, total_channels)
                band = parent_rank[budget:donor_upper]
                if len(band) < replace:
                    continue
                if args.emit_band_shift:
                    add_candidate(
                        f"band_shift_b{budget}_u{donor_upper}_r{replace}",
                        "parent_tail_replaced_by_parent_rescue_band",
                        fill_selected(keep, band, parent_rank, budget=budget),
                        {
                            "budget": budget,
                            "replace": replace,
                            "donor_upper": donor_upper,
                            "parent_prefix_kept": budget - replace,
                            "donor": "parent_rank_band",
                            "selection_splits": sorted(SPLIT_SELECT),
                        },
                    )
                if args.emit_rescue_swap:
                    add_candidate(
                        f"rescue_swap_b{budget}_u{donor_upper}_r{replace}",
                        "parent_tail_replaced_by_rescued_query_channels",
                        fill_selected(
                            keep,
                            rescue_rank_for_budget(budget, donor_upper, rescue_scores),
                            parent_rank,
                            budget=budget,
                        ),
                        {
                            "budget": budget,
                            "replace": replace,
                            "donor_upper": donor_upper,
                            "parent_prefix_kept": budget - replace,
                            "donor": "rescued_query_top_channels",
                            "selection_splits": sorted(SPLIT_SELECT),
                            "rescued_query_count": len(rescued),
                            "still_fail_weight": args.still_fail_weight,
                            "lost_weight": args.lost_weight,
                            "stable_success_weight": args.stable_success_weight,
                        },
                    )

    by_category = {
        category: {
            "rescued": sum(1 for eval_id in rescued if meta_by_id[eval_id].get("category") == category),
            "lost": sum(1 for eval_id in lost if meta_by_id[eval_id].get("category") == category),
            "stable_success": sum(1 for eval_id in stable_success if meta_by_id[eval_id].get("category") == category),
            "stable_fail": sum(1 for eval_id in stable_fail if meta_by_id[eval_id].get("category") == category),
        }
        for category in sorted({str(meta.get("category", "unknown")) for meta in meta_by_id.values()})
    }
    top_rescue_channels = [
        {
            "global_channel": int(gid),
            "layer": layer_channel(gid, d_ffn)[0],
            "channel": layer_channel(gid, d_ffn)[1],
            "score": float(score),
            "parent_rank": int(rank_pos.get(gid, -1) + 1),
        }
        for gid, score in sorted(rescue_scores.items(), key=lambda kv: (kv[1], -rank_pos.get(kv[0], total_channels)), reverse=True)[:200]
    ]

    manifest = {
        "issue": 12,
        "artifact": "bfcl_issue12_rescue_swap_candidates",
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_channels": total_channels,
        "selection_splits": sorted(SPLIT_SELECT),
        "heldout_policy": "heldout rows are excluded from rescue channel scoring and branch construction",
        "inputs": {
            "parent": str(args.parent),
            "parent_sha256": sha256_file(args.parent),
            "low_eval": str(args.low_eval),
            "low_eval_sha256": sha256_file(args.low_eval),
            "high_eval": str(args.high_eval),
            "high_eval_sha256": sha256_file(args.high_eval),
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
            "budgets": budgets,
            "replace_counts": replace_counts,
            "donor_uppers": donor_uppers,
        },
        "reference_outcome_counts_selection_only": {
            "eligible": len(eligible_ids),
            "rescued": len(rescued),
            "lost": len(lost),
            "stable_success": len(stable_success),
            "stable_fail": len(stable_fail),
        },
        "reference_outcome_counts_by_category": by_category,
        "top_channel_rows_seen": rows_seen,
        "top_channel_rows_scored": rows_scored,
        "top_channel_entries_scored": entries_scored,
        "top_rescue_channels": top_rescue_channels,
        "candidate_count": len(candidates),
    }

    (args.out_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.out_dir / "candidate_masks.jsonl").open("w") as f:
        for row in candidates:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
