#!/usr/bin/env python3
"""Build second-stage scaffold repair candidates from a base mask to a target mask.

This is for issue 3 after a fixed-budget scaffold repair has a known remaining
gap to a larger scaffold frontier.  It keeps candidate masks inside the union of
the exact base mask and target scaffold mask, then proposes fixed-budget swaps
using atlas donors from examples the target solves and the base misses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bfcl_attention_qwen3 import load_attention_scores, make_keep_mask


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


def eval_ok(row: dict[str, Any]) -> bool:
    return bool(row.get("normalized_correct", row.get("correct", False)))


def selected_scores(selected: list[int], shape: tuple[int, int]) -> np.ndarray:
    scores = np.zeros(shape[0] * shape[1], dtype=np.float32)
    n = len(selected)
    for rank, gid in enumerate(selected):
        scores[int(gid)] = float(n - rank)
    return scores.reshape(shape)


def sha1_selected(selected: list[int]) -> str:
    return hashlib.sha1(np.asarray(selected, dtype=np.int32).tobytes()).hexdigest()


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


def ordered_positive(scores: np.ndarray) -> list[int]:
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    idx = np.nonzero(flat > 0)[0].astype(np.int64)
    ordered = idx[np.lexsort((idx, -flat[idx]))]
    return [int(item) for item in ordered]


def ordered_mask(mask: np.ndarray, scores: np.ndarray) -> list[int]:
    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    idx = np.nonzero(flat_mask)[0].astype(np.int64)
    ordered = idx[np.lexsort((idx, -flat_scores[idx]))]
    return [int(item) for item in ordered]


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


def query_index_maps(atlas_dir: Path) -> tuple[dict[str, int], dict[str, str]]:
    idx_by_id: dict[str, int] = {}
    category_by_id: dict[str, str] = {}
    for row in read_jsonl(atlas_dir / "query_manifest.jsonl"):
        eval_id = str(row["eval_id"])
        idx_by_id[eval_id] = int(row["global_index"])
        category_by_id[eval_id] = str(row.get("category", "unknown"))
    return idx_by_id, category_by_id


def ids_to_indices(ids: list[str], idx_by_id: dict[str, int]) -> list[int]:
    return [idx_by_id[eval_id] for eval_id in ids if eval_id in idx_by_id]


def ids_from_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [str(row["id"]) for row in read_jsonl(path)]


def atlas_order(
    atlas_dir: Path,
    indices: list[int],
    *,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
    allowed: set[int],
    contrast_indices: list[int] | None = None,
    contrast_weight: float = 0.0,
) -> list[int]:
    if not indices:
        return []
    scores = np.load(atlas_dir / "ov_channel_scores_global_uint8.npy", mmap_mode="r")
    seg_idx = axis_indices(SEGMENTS, segments)
    stat_idx = axis_indices(STATS, stats)
    primary = np.asarray(scores[np.asarray(indices, dtype=np.int64)][:, seg_idx][:, :, stat_idx], dtype=np.float32)
    primary = primary.mean(axis=(0, 1, 2))
    if contrast_indices and contrast_weight:
        contrast = np.asarray(
            scores[np.asarray(contrast_indices, dtype=np.int64)][:, seg_idx][:, :, stat_idx],
            dtype=np.float32,
        ).mean(axis=(0, 1, 2))
        primary = primary - float(contrast_weight) * contrast
    flat = primary.reshape(-1)
    allowed_idx = np.asarray(sorted(allowed), dtype=np.int64)
    ordered = allowed_idx[np.lexsort((allowed_idx, -flat[allowed_idx]))]
    return [int(item) for item in ordered]


def write_score_file(path: Path, *, head_scores: np.ndarray, selected: list[int], ov_shape: tuple[int, int], metadata: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        head_scores=head_scores.astype(np.float32),
        ov_scores=selected_scores(selected, ov_shape).astype(np.float32),
        metadata_json=np.array(json.dumps(jsonable(metadata), sort_keys=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-score-file", type=Path, required=True)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--target-eval", type=Path, required=True)
    parser.add_argument("--target-attribution", type=Path, required=True)
    parser.add_argument("--target-budget", type=int, required=True)
    parser.add_argument("--head-scaffold-topk", type=int, required=True)
    parser.add_argument("--head-scaffold-layer-floor", type=int, default=0)
    parser.add_argument("--head-scaffold-multiplier", type=float, default=2.0)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--base-failure-buckets", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replace-counts", default="512,1024,2048,4096")
    parser.add_argument("--segments", default="target,full")
    parser.add_argument("--stats", default="mean_abs,rms")
    parser.add_argument("--contrast-weight", type=float, default=0.35)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.out_dir / "score_files"
    score_dir.mkdir(parents=True, exist_ok=True)

    base = np.load(args.base_score_file)
    base_head = base["head_scores"].astype(np.float32)
    base_ov = base["ov_scores"].astype(np.float32)
    head_scores_t, ov_scores_t = load_attention_scores(args.target_attribution)
    ov_scores = ov_scores_t.numpy()
    n_layers, hidden = base_ov.shape
    if tuple(ov_scores.shape) != tuple(base_ov.shape):
        raise ValueError(f"target attribution shape {tuple(ov_scores.shape)} != base {tuple(base_ov.shape)}")

    target_keep_t, target_info = make_keep_mask(
        head_scores=head_scores_t,
        ov_scores=ov_scores_t,
        unit="ov",
        topk=args.target_budget,
        random_seed=None,
        mask_strategy="head-scaffold-ov",
        head_scaffold_topk=args.head_scaffold_topk,
        head_scaffold_layer_floor=args.head_scaffold_layer_floor,
        head_scaffold_multiplier=args.head_scaffold_multiplier,
    )
    target_keep = target_keep_t.numpy()
    target_order = ordered_mask(target_keep, ov_scores)
    base_order = ordered_positive(base_ov)
    budget = len(base_order)
    total_ov = int(base_ov.size)
    base_set = set(base_order)
    target_set = set(target_order)
    allowed = base_set | target_set
    fallback_order = unique_fill([base_order, target_order], ordered_mask(np.ones_like(base_ov, dtype=bool), ov_scores), budget)

    base_rows = {str(row["id"]): row for row in read_jsonl(args.base_eval)}
    target_rows = {str(row["id"]): row for row in read_jsonl(args.target_eval)}
    eval_ids = sorted(set(base_rows) & set(target_rows))
    target_over_base = [eval_id for eval_id in eval_ids if eval_ok(target_rows[eval_id]) and not eval_ok(base_rows[eval_id])]
    base_over_target = [eval_id for eval_id in eval_ids if eval_ok(base_rows[eval_id]) and not eval_ok(target_rows[eval_id])]
    stable_success = [eval_id for eval_id in eval_ids if eval_ok(base_rows[eval_id]) and eval_ok(target_rows[eval_id])]
    stable_fail = [eval_id for eval_id in eval_ids if (not eval_ok(base_rows[eval_id])) and (not eval_ok(target_rows[eval_id]))]

    idx_by_id, category_by_id = query_index_maps(args.atlas_dir)
    gain_idx = ids_to_indices(target_over_base, idx_by_id)
    lost_idx = ids_to_indices(base_over_target, idx_by_id)

    donor_orders: dict[str, list[int]] = {
        "target_only_boundary": [gid for gid in target_order if gid not in base_set],
    }
    donor_meta: dict[str, dict[str, Any]] = {
        "target_only_boundary": {"target_only_channels": len(donor_orders["target_only_boundary"])},
    }
    if gain_idx:
        donor_orders["target_over_base_all"] = atlas_order(
            args.atlas_dir,
            gain_idx,
            segments=parse_csv(args.segments),
            stats=parse_csv(args.stats),
            allowed=allowed,
        )
        donor_orders["target_over_base_minus_lost"] = atlas_order(
            args.atlas_dir,
            gain_idx,
            segments=parse_csv(args.segments),
            stats=parse_csv(args.stats),
            allowed=allowed,
            contrast_indices=lost_idx,
            contrast_weight=args.contrast_weight,
        )
        donor_meta["target_over_base_all"] = {"target_over_base": len(target_over_base), "base_over_target": len(base_over_target)}
        donor_meta["target_over_base_minus_lost"] = {
            "target_over_base": len(target_over_base),
            "base_over_target": len(base_over_target),
            "contrast_weight": args.contrast_weight,
        }

    by_category: dict[str, list[str]] = defaultdict(list)
    for eval_id in target_over_base:
        by_category[category_by_id.get(eval_id, str(base_rows[eval_id].get("category", "unknown")))].append(eval_id)
    for category, ids in sorted(by_category.items(), key=lambda item: (-len(item[1]), item[0])):
        idx = ids_to_indices(ids, idx_by_id)
        if idx:
            key = f"target_over_base_category_{sanitize(category)}"
            donor_orders[key] = atlas_order(args.atlas_dir, idx, segments=parse_csv(args.segments), stats=parse_csv(args.stats), allowed=allowed)
            donor_meta[key] = {"category": category, "target_over_base": len(ids)}

    if args.base_failure_buckets:
        for bucket_path in sorted((args.base_failure_buckets / "repair_buckets").glob("*.jsonl")):
            bucket = bucket_path.stem
            bucket_ids = ids_from_jsonl(bucket_path)
            gain_bucket = sorted(set(bucket_ids) & set(target_over_base))
            all_bucket_idx = ids_to_indices(bucket_ids, idx_by_id)
            gain_bucket_idx = ids_to_indices(gain_bucket, idx_by_id)
            if gain_bucket_idx:
                key = f"gap_bucket_{sanitize(bucket)}"
                donor_orders[key] = atlas_order(
                    args.atlas_dir,
                    gain_bucket_idx,
                    segments=parse_csv(args.segments),
                    stats=parse_csv(args.stats),
                    allowed=allowed,
                    contrast_indices=lost_idx,
                    contrast_weight=args.contrast_weight,
                )
                donor_meta[key] = {"bucket": bucket, "target_over_base": len(gain_bucket), "base_bucket_failures": len(bucket_ids)}
            if all_bucket_idx and bucket in {"arg_value_exactness", "function_name_disambiguation", "live_slot_values"}:
                key = f"base_failure_bucket_{sanitize(bucket)}"
                donor_orders[key] = atlas_order(args.atlas_dir, all_bucket_idx, segments=parse_csv(args.segments), stats=parse_csv(args.stats), allowed=allowed)
                donor_meta[key] = {"bucket": bucket, "base_bucket_failures": len(bucket_ids)}

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    segments = parse_csv(args.segments)
    stats = parse_csv(args.stats)

    def add(candidate_id: str, family: str, selected: list[int], lineage: dict[str, Any], priority: int) -> None:
        selected = unique_fill([selected], fallback_order, budget)
        if len(selected) != budget:
            raise ValueError(f"{candidate_id} selected {len(selected)} != {budget}")
        digest = sha1_selected(selected)
        if digest in seen:
            return
        seen.add(digest)
        score_path = score_dir / f"{candidate_id}.npz"
        write_score_file(
            score_path,
            head_scores=base_head,
            selected=selected,
            ov_shape=base_ov.shape,
            metadata={
                "candidate_id": candidate_id,
                "family": family,
                "lineage": lineage,
                "base_score_file": str(args.base_score_file),
                "target_attribution": str(args.target_attribution),
                "atlas_dir": str(args.atlas_dir),
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

    add("base_mask", "base_mask", base_order, {"budget": budget}, priority=9000)
    replace_counts = parse_ints(args.replace_counts)
    for replace in replace_counts:
        if replace <= 0 or replace >= budget:
            continue
        keep = base_order[: budget - replace]
        for donor_name, donor_order in donor_orders.items():
            if not donor_order:
                continue
            family = f"scaffold_gap_repair_{donor_name}"
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
                    **donor_meta.get(donor_name, {}),
                },
                priority=1000
                + abs(replace - 2048) // 512
                + (0 if donor_name == "target_only_boundary" else 20 if donor_name.startswith("target_over_base") else 40),
            )

    candidates.sort(key=lambda row: (int(row["priority"]), -float(row["layer_entropy"]), str(row["candidate_id"])))
    for rank, row in enumerate(candidates, start=1):
        row["proposal_rank"] = rank
        row["selected_for_first_eval"] = rank == 1

    manifest = {
        "artifact": "issue3_attention_scaffold_gap_repair_candidates",
        "base_score_file": str(args.base_score_file),
        "base_eval": str(args.base_eval),
        "target_eval": str(args.target_eval),
        "target_attribution": str(args.target_attribution),
        "target_budget": args.target_budget,
        "target_mask": target_info,
        "atlas_dir": str(args.atlas_dir),
        "base_failure_buckets": str(args.base_failure_buckets) if args.base_failure_buckets else None,
        "total_attention_ov_channels": total_ov,
        "base_budget": budget,
        "replace_counts": replace_counts,
        "segments": segments,
        "stats": stats,
        "candidate_count": len(candidates),
        "comparison": {
            "target_over_base": len(target_over_base),
            "base_over_target": len(base_over_target),
            "stable_success": len(stable_success),
            "stable_fail": len(stable_fail),
            "target_over_base_categories": dict(Counter(category_by_id.get(eval_id, "unknown") for eval_id in target_over_base)),
            "base_over_target_categories": dict(Counter(category_by_id.get(eval_id, "unknown") for eval_id in base_over_target)),
        },
        "notes": [
            "candidate score files are exact global masks; evaluate them with --mask-strategy global",
            "all proposed swaps stay inside base mask union target scaffold mask",
            "behavior claims still require full autoregressive BFCL eval",
        ],
    }
    write_jsonl(args.out_dir / "candidate_scores.jsonl", candidates)
    json_dump(args.out_dir / "candidate_manifest.json", manifest)
    print(json.dumps(jsonable({"candidate_count": len(candidates), "first": candidates[:12], "comparison": manifest["comparison"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
