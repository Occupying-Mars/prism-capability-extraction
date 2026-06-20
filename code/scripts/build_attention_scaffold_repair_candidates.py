#!/usr/bin/env python3
"""Build scaffold-constrained attention repair candidates.

This keeps the head scaffold fixed, reconstructs the exact small/large OV
masks produced by ``bfcl_attention_qwen3.py --mask-strategy head-scaffold-ov``,
then proposes fixed-budget swaps inside that same scaffold.
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


def axis_indices(names: tuple[str, ...], selected: tuple[str, ...]) -> list[int]:
    index = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in selected if name not in index]
    if missing:
        raise ValueError(f"unknown axis values {missing}; allowed={names}")
    return [index[name] for name in selected]


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def eval_ok(row: dict[str, Any]) -> bool:
    return bool(row.get("normalized_correct", row.get("correct", False)))


def selected_scores(selected: list[int], shape: tuple[int, int]) -> np.ndarray:
    scores = np.zeros(shape[0] * shape[1], dtype=np.float32)
    n = len(selected)
    for rank, gid in enumerate(selected):
        scores[int(gid)] = float(n - rank)
    return scores.reshape(shape)


def sha1_selected(selected: list[int]) -> str:
    arr = np.asarray(selected, dtype=np.int32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


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


def ordered_selected(mask: np.ndarray, scores: np.ndarray) -> list[int]:
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


def query_indices(atlas_dir: Path, eval_ids: list[str]) -> list[int]:
    by_id = {str(row["eval_id"]): int(row["global_index"]) for row in read_jsonl(atlas_dir / "query_manifest.jsonl")}
    return [by_id[eval_id] for eval_id in eval_ids if eval_id in by_id]


def order_from_indices(
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
    primary = np.asarray(scores[np.asarray(indices, dtype=np.int64)][:, seg_idx][:, :, stat_idx], dtype=np.float32).mean(axis=(0, 1, 2))
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
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--small-eval", type=Path, required=True)
    parser.add_argument("--large-eval", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--small-budget", type=int, required=True)
    parser.add_argument("--large-budget", type=int, required=True)
    parser.add_argument("--head-scaffold-topk", type=int, required=True)
    parser.add_argument("--head-scaffold-layer-floor", type=int, default=0)
    parser.add_argument("--head-scaffold-multiplier", type=float, default=2.0)
    parser.add_argument("--replace-counts", default="512,1024,2048,4096")
    parser.add_argument("--segments", default="target,full")
    parser.add_argument("--stats", default="mean_abs,rms")
    parser.add_argument("--contrast-weight", type=float, default=0.35)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.out_dir / "score_files"
    score_dir.mkdir(parents=True, exist_ok=True)

    head_scores_t, ov_scores_t = load_attention_scores(args.attribution)
    head_scores = head_scores_t.numpy()
    ov_scores = ov_scores_t.numpy()
    n_layers, hidden = ov_scores.shape
    total_ov = int(ov_scores.size)
    replace_counts = parse_ints(args.replace_counts)
    segments = parse_csv(args.segments)
    stats = parse_csv(args.stats)

    small_keep_t, small_info = make_keep_mask(
        head_scores=head_scores_t,
        ov_scores=ov_scores_t,
        unit="ov",
        topk=args.small_budget,
        random_seed=None,
        mask_strategy="head-scaffold-ov",
        head_scaffold_topk=args.head_scaffold_topk,
        head_scaffold_layer_floor=args.head_scaffold_layer_floor,
        head_scaffold_multiplier=args.head_scaffold_multiplier,
    )
    large_keep_t, large_info = make_keep_mask(
        head_scores=head_scores_t,
        ov_scores=ov_scores_t,
        unit="ov",
        topk=args.large_budget,
        random_seed=None,
        mask_strategy="head-scaffold-ov",
        head_scaffold_topk=args.head_scaffold_topk,
        head_scaffold_layer_floor=args.head_scaffold_layer_floor,
        head_scaffold_multiplier=args.head_scaffold_multiplier,
    )
    small_keep = small_keep_t.numpy()
    large_keep = large_keep_t.numpy()
    scaffold_allowed = set(np.nonzero(large_keep.reshape(-1) | small_keep.reshape(-1))[0].astype(int).tolist())
    small_order = ordered_selected(small_keep, ov_scores)
    large_order = ordered_selected(large_keep, ov_scores)
    large_only_order = [gid for gid in large_order if not small_keep.reshape(-1)[gid]]
    fallback_order = unique_fill([small_order, large_order], ordered_selected(np.ones_like(small_keep, dtype=bool), ov_scores), args.small_budget)

    small_rows = {str(row["id"]): row for row in read_jsonl(args.small_eval)}
    large_rows = {str(row["id"]): row for row in read_jsonl(args.large_eval)}
    eval_ids = sorted(set(small_rows) & set(large_rows))
    rescued = [eval_id for eval_id in eval_ids if eval_ok(large_rows[eval_id]) and not eval_ok(small_rows[eval_id])]
    lost = [eval_id for eval_id in eval_ids if eval_ok(small_rows[eval_id]) and not eval_ok(large_rows[eval_id])]
    stable_success = [eval_id for eval_id in eval_ids if eval_ok(small_rows[eval_id]) and eval_ok(large_rows[eval_id])]
    stable_fail = [eval_id for eval_id in eval_ids if (not eval_ok(small_rows[eval_id])) and (not eval_ok(large_rows[eval_id]))]

    rescued_idx = query_indices(args.atlas_dir, rescued)
    lost_idx = query_indices(args.atlas_dir, lost)
    donor_orders: dict[str, list[int]] = {
        "large_only_boundary": large_only_order,
    }
    donor_meta: dict[str, dict[str, Any]] = {
        "large_only_boundary": {"large_only_channels": len(large_only_order)},
    }
    if rescued_idx:
        donor_orders["rescued_all"] = order_from_indices(
            args.atlas_dir,
            rescued_idx,
            segments=segments,
            stats=stats,
            allowed=scaffold_allowed,
        )
        donor_orders["rescued_minus_lost"] = order_from_indices(
            args.atlas_dir,
            rescued_idx,
            segments=segments,
            stats=stats,
            allowed=scaffold_allowed,
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
            donor_orders[key] = order_from_indices(args.atlas_dir, idx, segments=segments, stats=stats, allowed=scaffold_allowed)
            donor_meta[key] = {"category": category, "rescued": len(ids)}

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate_id: str, family: str, selected: list[int], lineage: dict[str, Any], priority: int) -> None:
        selected = unique_fill([selected], fallback_order, args.small_budget)
        if len(selected) != args.small_budget:
            raise ValueError(f"{candidate_id} selected {len(selected)} != {args.small_budget}")
        digest = sha1_selected(selected)
        if digest in seen:
            return
        seen.add(digest)
        score_path = score_dir / f"{candidate_id}.npz"
        write_score_file(
            score_path,
            head_scores=head_scores,
            selected=selected,
            ov_shape=ov_scores.shape,
            metadata={
                "candidate_id": candidate_id,
                "family": family,
                "lineage": lineage,
                "attribution": str(args.attribution),
                "atlas_dir": str(args.atlas_dir),
                "small_eval": str(args.small_eval),
                "large_eval": str(args.large_eval),
            },
        )
        row = {
            "candidate_id": candidate_id,
            "family": family,
            "score_file": str(score_path.relative_to(args.out_dir)),
            "topks": str(args.small_budget),
            "unit": "ov",
            "projection_sites": "ov",
            "mask_strategy": "global",
            "ablation": "zero",
            "selected_attention_ov_channels": args.small_budget,
            "attention_ov_fraction": args.small_budget / total_ov,
            "selected_sha1": digest,
            "priority": priority,
            "lineage": lineage,
        }
        row.update(layer_stats(selected, hidden, n_layers))
        candidates.append(row)

    add("scaffold_small_prefix", "scaffold_small_prefix", small_order, {"budget": args.small_budget}, priority=9000)
    for replace in replace_counts:
        if replace <= 0 or replace >= args.small_budget:
            continue
        keep = small_order[: args.small_budget - replace]
        for donor_name, donor_order in donor_orders.items():
            if not donor_order:
                continue
            family = f"scaffold_repair_{donor_name}"
            add(
                f"{family}_b{args.small_budget}_r{replace}",
                family,
                unique_fill([keep, donor_order], fallback_order, args.small_budget),
                {
                    "budget": args.small_budget,
                    "replace": replace,
                    "donor": donor_name,
                    "kept_small_prefix": args.small_budget - replace,
                    "segments": segments,
                    "stats": stats,
                    **donor_meta.get(donor_name, {}),
                },
                priority=1000
                + abs(replace - 2048) // 512
                + (0 if donor_name == "large_only_boundary" else 20 if donor_name.startswith("rescued") else 50),
            )

    candidates.sort(key=lambda row: (int(row["priority"]), -float(row["layer_entropy"]), str(row["candidate_id"])))
    for rank, row in enumerate(candidates, start=1):
        row["proposal_rank"] = rank
        row["selected_for_first_eval"] = rank == 1

    manifest = {
        "artifact": "issue3_attention_scaffold_repair_candidates",
        "attribution": str(args.attribution),
        "atlas_dir": str(args.atlas_dir),
        "small_eval": str(args.small_eval),
        "large_eval": str(args.large_eval),
        "total_attention_ov_channels": total_ov,
        "small_budget": args.small_budget,
        "large_budget": args.large_budget,
        "head_scaffold_topk": args.head_scaffold_topk,
        "small_mask": small_info,
        "large_mask": large_info,
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
        },
        "notes": [
            "candidate score files are exact global masks; evaluate them with --mask-strategy global",
            "all proposed swaps stay within the fixed head scaffold used by the small/large comparison",
            "behavior claims still require full autoregressive BFCL eval",
        ],
    }
    write_jsonl(args.out_dir / "candidate_scores.jsonl", candidates)
    json_dump(args.out_dir / "candidate_manifest.json", manifest)
    print(json.dumps(jsonable({"candidate_count": len(candidates), "first": candidates[:12], "comparison": manifest["comparison"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
