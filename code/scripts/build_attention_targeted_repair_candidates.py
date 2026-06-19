#!/usr/bin/env python3
"""Build targeted attention repair candidates from an incumbent OV mask.

This is the attention analogue of the token issue12 targeted repair pass:
keep most of an already-good mask, replace a suffix with channels ranked by
examples/categories/buckets where a larger reference mask succeeds and the
incumbent fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SEGMENTS = ("prompt", "target", "full")
STATS = ("mean_abs", "rms", "max_abs")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def axis_indices(names: tuple[str, ...], selected: tuple[str, ...]) -> list[int]:
    index = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in selected if name not in index]
    if missing:
        raise ValueError(f"unknown axis values {missing}; allowed={names}")
    return [index[name] for name in selected]


def top_order(scores: np.ndarray) -> list[int]:
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    idx = np.arange(flat.size, dtype=np.int64)
    ordered = idx[np.lexsort((idx, -flat))]
    return [int(item) for item in ordered]


def positive_order(scores: np.ndarray) -> list[int]:
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    positive = np.nonzero(flat > 0)[0].astype(np.int64)
    if positive.size == 0:
        return []
    ordered = positive[np.lexsort((positive, -flat[positive]))]
    return [int(item) for item in ordered]


def unique_fill(prefixes: list[list[int]], fallback: list[int], budget: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for source in [*prefixes, fallback]:
        for item in source:
            gid = int(item)
            if gid in seen:
                continue
            selected.append(gid)
            seen.add(gid)
            if len(selected) >= budget:
                return selected
    return selected


def selected_scores(selected: list[int], shape: tuple[int, int]) -> np.ndarray:
    scores = np.zeros(shape[0] * shape[1], dtype=np.float32)
    n = len(selected)
    for rank, gid in enumerate(selected):
        scores[int(gid)] = float(n - rank)
    return scores.reshape(shape)


def sha1_selected(selected: list[int]) -> str:
    arr = np.asarray(selected, dtype=np.int32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def layer_stats(selected: list[int], hidden: int, n_layers: int) -> dict[str, Any]:
    counts = np.zeros(n_layers, dtype=np.int64)
    touched_heads: set[tuple[int, int]] = set()
    head_dim = hidden // 32
    for gid in selected:
        layer = int(gid // hidden)
        channel = int(gid % hidden)
        counts[layer] += 1
        touched_heads.add((layer, channel // head_dim))
    probs = counts[counts > 0].astype(np.float64) / max(len(selected), 1)
    entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
    return {
        "active_layers": int((counts > 0).sum()),
        "max_layer_channels": int(counts.max()) if counts.size else 0,
        "layer_entropy": entropy,
        "heads_touched": len(touched_heads),
    }


def eval_ok(row: dict[str, Any]) -> bool:
    return bool(row.get("normalized_correct", row.get("correct", False)))


def target_function_names(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for call in row.get("target") or row.get("reference_calls") or []:
        if isinstance(call, dict):
            out.update(str(key) for key in call.keys())
    return out


def prediction_function_names(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for call in row.get("prediction_calls") or []:
        if isinstance(call, dict):
            name = call.get("name")
            if name is not None:
                out.add(str(name))
    return out


def target_arg_names(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for call in row.get("target") or row.get("reference_calls") or []:
        if not isinstance(call, dict):
            continue
        for spec in call.values():
            if isinstance(spec, dict):
                out.update(str(key) for key in spec.keys())
    return out


def prediction_arg_names(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for call in row.get("prediction_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
            out.update(str(key) for key in call["arguments"].keys())
    return out


def failure_bucket(row: dict[str, Any]) -> str:
    prediction_calls = row.get("prediction_calls") or []
    if not prediction_calls:
        text = str(row.get("prediction_text") or "")
        return "no_call" if text.strip() else "empty_generation"
    pred_names = prediction_function_names(row)
    target_names = target_function_names(row)
    if target_names and pred_names.isdisjoint(target_names):
        return "wrong_function"
    pred_args = prediction_arg_names(row)
    target_args = target_arg_names(row)
    if target_args and pred_args != target_args:
        if pred_args.issubset(target_args):
            return "missing_arg"
        if target_args.issubset(pred_args):
            return "extra_arg"
        return "arg_schema"
    return "arg_value_or_format"


def query_indices(atlas_dir: Path, eval_ids: list[str]) -> list[int]:
    by_id: dict[str, int] = {}
    for row in read_jsonl(atlas_dir / "query_manifest.jsonl"):
        by_id[str(row["eval_id"])] = int(row["global_index"])
    return [by_id[eval_id] for eval_id in eval_ids if eval_id in by_id]


def order_from_indices(
    atlas_dir: Path,
    indices: list[int],
    *,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
    contrast_indices: list[int] | None = None,
    contrast_weight: float = 0.0,
) -> list[int]:
    if not indices:
        return []
    path = atlas_dir / "ov_channel_scores_global_uint8.npy"
    scores = np.load(path, mmap_mode="r")
    seg_idx = axis_indices(SEGMENTS, segments)
    stat_idx = axis_indices(STATS, stats)
    primary = np.asarray(scores[np.asarray(indices, dtype=np.int64)][:, seg_idx][:, :, stat_idx], dtype=np.float32).mean(axis=(0, 1, 2))
    if contrast_indices and contrast_weight:
        contrast = np.asarray(
            scores[np.asarray(contrast_indices, dtype=np.int64)][:, seg_idx][:, :, stat_idx],
            dtype=np.float32,
        ).mean(axis=(0, 1, 2))
        primary = primary - float(contrast_weight) * contrast
    return top_order(primary)


def category_key(category: str) -> str:
    return f"category_{sanitize(category)}_ov_global_score_mean_float16"


def order_from_category_rollup(
    atlas_dir: Path,
    category: str,
    *,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> list[int]:
    path = atlas_dir / "category_attention_channel_heatmaps.npz"
    if not path.exists():
        return []
    data = np.load(path)
    key = category_key(category)
    if key not in data.files:
        return []
    seg_idx = axis_indices(SEGMENTS, segments)
    stat_idx = axis_indices(STATS, stats)
    score = np.asarray(data[key], dtype=np.float32)[np.ix_(seg_idx, stat_idx)].mean(axis=(0, 1))
    return top_order(score)


def write_score_file(path: Path, *, head_scores: np.ndarray, selected: list[int], ov_shape: tuple[int, int], metadata: dict[str, Any]) -> None:
    ov_scores = selected_scores(selected, ov_shape)
    np.savez_compressed(
        path,
        head_scores=head_scores.astype(np.float32),
        ov_scores=ov_scores.astype(np.float32),
        metadata_json=np.array(json.dumps(jsonable(metadata), sort_keys=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-attribution", type=Path, required=True)
    parser.add_argument("--parent-attribution", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--small-eval", type=Path, required=True)
    parser.add_argument("--large-eval", type=Path, required=True)
    parser.add_argument("--budgets", default="106496")
    parser.add_argument("--replace-counts", default="512,1024,2048,4096,8192,12288,16384")
    parser.add_argument("--segments", default="target,full")
    parser.add_argument("--stats", default="mean_abs,rms")
    parser.add_argument("--contrast-weight", type=float, default=0.35)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.out_dir / "score_files"
    score_dir.mkdir(parents=True, exist_ok=True)

    base = np.load(args.base_attribution)
    parent = np.load(args.parent_attribution)
    parent_head = parent["head_scores"].astype(np.float32)
    parent_ov = parent["ov_scores"].astype(np.float32)
    base_ov = base["ov_scores"].astype(np.float32)
    n_layers, hidden = base_ov.shape
    total_ov = int(base_ov.size)
    base_order = positive_order(base_ov)
    fallback_order = top_order(parent_ov)
    if not base_order:
        raise ValueError(f"{args.base_attribution} has no positive OV scores")

    budgets = [b for b in parse_ints(args.budgets) if 0 < b <= len(base_order)]
    replace_counts = parse_ints(args.replace_counts)
    segments = parse_csv(args.segments)
    stats = parse_csv(args.stats)

    small_rows = {str(row["id"]): row for row in read_jsonl(args.small_eval)}
    large_rows = {str(row["id"]): row for row in read_jsonl(args.large_eval)}
    eval_ids = sorted(set(small_rows) & set(large_rows))
    rescued = [eval_id for eval_id in eval_ids if eval_ok(large_rows[eval_id]) and not eval_ok(small_rows[eval_id])]
    lost = [eval_id for eval_id in eval_ids if eval_ok(small_rows[eval_id]) and not eval_ok(large_rows[eval_id])]
    stable_success = [eval_id for eval_id in eval_ids if eval_ok(small_rows[eval_id]) and eval_ok(large_rows[eval_id])]
    stable_fail = [eval_id for eval_id in eval_ids if (not eval_ok(small_rows[eval_id])) and (not eval_ok(large_rows[eval_id]))]

    donor_orders: dict[str, list[int]] = {}
    donor_meta: dict[str, dict[str, Any]] = {}
    rescued_idx = query_indices(args.atlas_dir, rescued)
    lost_idx = query_indices(args.atlas_dir, lost)
    if rescued_idx:
        donor_orders["rescued_all"] = order_from_indices(args.atlas_dir, rescued_idx, segments=segments, stats=stats)
        donor_orders["rescued_minus_lost"] = order_from_indices(
            args.atlas_dir,
            rescued_idx,
            segments=segments,
            stats=stats,
            contrast_indices=lost_idx,
            contrast_weight=args.contrast_weight,
        )
        donor_meta["rescued_all"] = {"rescued": len(rescued), "lost": len(lost)}
        donor_meta["rescued_minus_lost"] = {"rescued": len(rescued), "lost": len(lost), "contrast_weight": args.contrast_weight}

    rescued_by_category: dict[str, list[str]] = defaultdict(list)
    for eval_id in rescued:
        rescued_by_category[str(small_rows[eval_id].get("category", "unknown"))].append(eval_id)
    for category, ids in sorted(rescued_by_category.items(), key=lambda item: (-len(item[1]), item[0])):
        idx = query_indices(args.atlas_dir, ids)
        if idx:
            key = f"rescued_category_{sanitize(category)}"
            donor_orders[key] = order_from_indices(args.atlas_dir, idx, segments=segments, stats=stats)
            donor_meta[key] = {"category": category, "rescued": len(ids)}
        rollup = order_from_category_rollup(args.atlas_dir, category, segments=segments, stats=stats)
        if rollup:
            key = f"category_rollup_{sanitize(category)}"
            donor_orders[key] = rollup
            donor_meta[key] = {"category": category, "source": "category_rollup"}

    failed_small = [eval_id for eval_id in eval_ids if not eval_ok(small_rows[eval_id])]
    failed_by_bucket: dict[str, list[str]] = defaultdict(list)
    for eval_id in failed_small:
        failed_by_bucket[failure_bucket(small_rows[eval_id])].append(eval_id)
    for bucket, ids in sorted(failed_by_bucket.items(), key=lambda item: (-len(item[1]), item[0])):
        idx = query_indices(args.atlas_dir, ids)
        if idx:
            key = f"small_failure_bucket_{sanitize(bucket)}"
            donor_orders[key] = order_from_indices(args.atlas_dir, idx, segments=segments, stats=stats)
            donor_meta[key] = {"bucket": bucket, "small_failures": len(ids)}

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate_id: str, family: str, selected: list[int], lineage: dict[str, Any], priority: int) -> None:
        budget = int(lineage["budget"])
        selected = unique_fill([selected], fallback_order, budget)
        if len(selected) != budget:
            raise ValueError(f"{candidate_id} selected {len(selected)} != budget {budget}")
        digest = sha1_selected(selected)
        if digest in seen:
            return
        seen.add(digest)
        score_path = score_dir / f"{candidate_id}.npz"
        write_score_file(
            score_path,
            head_scores=parent_head,
            selected=selected,
            ov_shape=base_ov.shape,
            metadata={
                "candidate_id": candidate_id,
                "family": family,
                "lineage": lineage,
                "base_attribution": str(args.base_attribution),
                "parent_attribution": str(args.parent_attribution),
                "atlas_dir": str(args.atlas_dir),
                "small_eval": str(args.small_eval),
                "large_eval": str(args.large_eval),
            },
        )
        row = {
            "candidate_id": candidate_id,
            "family": family,
            "score_file": str(score_path.relative_to(args.out_dir)),
            "topks": str(budget),
            "unit": "ov",
            "projection_sites": "ov",
            "mask_strategy": "global",
            "ablation": "zero",
            "selected_attention_ov_channels": budget,
            "attention_ov_fraction": budget / total_ov,
            "selected_sha1": digest,
            "priority": priority,
            "lineage": lineage,
        }
        row.update(layer_stats(selected, hidden, n_layers))
        candidates.append(row)

    for budget in budgets:
        add(
            f"base_prefix_b{budget}",
            "base_prefix",
            base_order[:budget],
            {"budget": budget},
            priority=9000,
        )
        for replace in replace_counts:
            if replace <= 0 or replace >= budget:
                continue
            keep = base_order[: budget - replace]
            for donor_name, donor_order in donor_orders.items():
                if not donor_order:
                    continue
                meta = donor_meta.get(donor_name, {})
                family = f"targeted_repair_{donor_name}"
                add(
                    f"{family}_b{budget}_r{replace}",
                    family,
                    unique_fill([keep, donor_order], fallback_order, budget),
                    {
                        "budget": budget,
                        "replace": replace,
                        "donor": donor_name,
                        "kept_base_prefix": budget - replace,
                        "segments": segments,
                        "stats": stats,
                        **meta,
                    },
                    priority=1000
                    + abs(replace - 4096) // 512
                    + (0 if donor_name.startswith("rescued") else 50 if donor_name.startswith("rescued_category") else 100),
                )

    candidates.sort(
        key=lambda row: (
            int(row["priority"]),
            -float(row["layer_entropy"]),
            str(row["candidate_id"]),
        )
    )
    for rank, row in enumerate(candidates, start=1):
        row["proposal_rank"] = rank
        row["selected_for_first_eval"] = rank == 1

    manifest = {
        "artifact": "issue3_attention_targeted_repair_candidates",
        "base_attribution": str(args.base_attribution),
        "parent_attribution": str(args.parent_attribution),
        "atlas_dir": str(args.atlas_dir),
        "small_eval": str(args.small_eval),
        "large_eval": str(args.large_eval),
        "total_attention_ov_channels": total_ov,
        "base_selected_ov_channels": len(base_order),
        "budgets": budgets,
        "replace_counts": replace_counts,
        "segments": segments,
        "stats": stats,
        "candidate_count": len(candidates),
        "comparison": {
            "rescued_large_over_small": len(rescued),
            "lost_large_vs_small": len(lost),
            "stable_success": len(stable_success),
            "stable_fail": len(stable_fail),
            "rescued_categories": dict(Counter(str(small_rows[eval_id].get("category", "unknown")) for eval_id in rescued)),
            "lost_categories": dict(Counter(str(small_rows[eval_id].get("category", "unknown")) for eval_id in lost)),
            "small_failure_buckets": dict(Counter(failure_bucket(small_rows[eval_id]) for eval_id in failed_small)),
        },
        "notes": [
            "proposal candidates only; behavior claims require full autoregressive BFCL eval",
            "fixed-budget repairs preserve the incumbent base prefix and swap only a suffix",
            "failure buckets are heuristic from generated tool calls, not scorer-provided BFCL labels",
        ],
    }
    write_jsonl(args.out_dir / "candidate_scores.jsonl", candidates)
    json_dump(args.out_dir / "candidate_manifest.json", manifest)
    print(json.dumps(jsonable({"candidate_count": len(candidates), "first": candidates[:12], "comparison": manifest["comparison"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
