#!/usr/bin/env python3
"""Build anchored atlas repair candidates for attention OV masks.

This keeps a working exact OV mask as the anchor, preserves an optional tail,
and swaps a small middle suffix using category-level attention-atlas donors.
The output score files stay compatible with ``bfcl_attention_qwen3.py
eval-ladder`` as exact global OV masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SEGMENTS = ("prompt", "target", "full")
STATS = ("mean_abs", "rms", "max_abs")
DEFAULT_CATEGORIES = ("simple", "live_simple", "exec_simple", "java", "javascript", "sql")
CODE_LIVE_CATEGORIES = ("live_simple", "exec_simple", "java", "javascript", "sql")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


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


def category_key(category: str, site: str) -> str:
    return f"category_{sanitize(category)}_{site}_global_score_mean_float16"


def selected_mean(array: np.ndarray, *, segments: tuple[str, ...], stats: tuple[str, ...]) -> np.ndarray:
    seg_idx = axis_indices(SEGMENTS, segments)
    stat_idx = axis_indices(STATS, stats)
    sub = np.asarray(array, dtype=np.float32)[np.ix_(seg_idx, stat_idx)]
    return sub.mean(axis=(0, 1)).astype(np.float32)


def target_delta(array: np.ndarray, *, stats: tuple[str, ...]) -> np.ndarray:
    stat_idx = axis_indices(STATS, stats)
    arr = np.asarray(array, dtype=np.float32)
    target_full = arr[np.ix_([1, 2], stat_idx)].mean(axis=(0, 1))
    prompt = arr[np.ix_([0], stat_idx)].mean(axis=(0, 1))
    return np.maximum(target_full - prompt, 0.0).astype(np.float32)


def category_score(
    heatmaps: np.lib.npyio.NpzFile,
    *,
    categories: tuple[str, ...],
    weights: dict[str, float] | None,
    mode: str,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> np.ndarray:
    pieces = []
    used_weights = []
    for category in categories:
        key = category_key(category, "ov")
        if key not in heatmaps.files:
            continue
        array = heatmaps[key]
        if mode == "hotness":
            score = selected_mean(array, segments=segments, stats=stats)
        elif mode == "target_delta":
            score = target_delta(array, stats=stats)
        else:
            raise ValueError(f"unknown mode {mode}")
        pieces.append(score)
        used_weights.append(float((weights or {}).get(category, 1.0)))
    if not pieces:
        raise ValueError(f"no category scores found for {categories}")
    weights_arr = np.asarray(used_weights, dtype=np.float32)
    weights_arr = weights_arr / max(float(weights_arr.sum()), 1e-8)
    return np.tensordot(weights_arr, np.stack(pieces, axis=0), axes=(0, 0)).astype(np.float32)


def read_failure_category_weights(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    counts = Counter(data.get("category_failure_counts") or {})
    return {str(k): float(v) for k, v in counts.items() if float(v) > 0}


def write_score_file(
    path: Path,
    *,
    head_scores: np.ndarray,
    selected: list[int],
    ov_shape: tuple[int, int],
    metadata: dict[str, Any],
) -> None:
    np.savez_compressed(
        path,
        head_scores=head_scores.astype(np.float32),
        ov_scores=selected_scores(selected, ov_shape).astype(np.float32),
        metadata_json=np.array(json.dumps(jsonable(metadata), sort_keys=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-attribution", type=Path, required=True)
    parser.add_argument("--parent-attribution", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--replace-counts", default="1024,2048,4096,8192,12288")
    parser.add_argument("--protected-tail-counts", default="0,4096,8192,12288")
    parser.add_argument("--segments", default="target,full")
    parser.add_argument("--stats", default="mean_abs,rms")
    parser.add_argument("--failure-summary", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.out_dir / "score_files"
    score_dir.mkdir(parents=True, exist_ok=True)

    base = np.load(args.base_attribution)
    parent = np.load(args.parent_attribution)
    heatmaps = np.load(args.atlas_dir / "category_attention_channel_heatmaps.npz")
    parent_head = parent["head_scores"].astype(np.float32)
    parent_ov = parent["ov_scores"].astype(np.float32)
    base_ov = base["ov_scores"].astype(np.float32)
    n_layers, hidden = base_ov.shape
    total_ov = int(base_ov.size)
    base_order = positive_order(base_ov)
    fallback_order = top_order(parent_ov)
    if args.budget > len(base_order):
        raise ValueError(f"budget {args.budget} exceeds base mask size {len(base_order)}")

    segments = parse_csv(args.segments)
    stats = parse_csv(args.stats)
    replace_counts = parse_ints(args.replace_counts)
    protected_tail_counts = parse_ints(args.protected_tail_counts)
    available_categories = tuple(str(x) for x in heatmaps["categories"].tolist())
    failure_weights = read_failure_category_weights(args.failure_summary)

    groups: dict[str, tuple[tuple[str, ...], dict[str, float] | None]] = {
        "balanced": (tuple(category for category in DEFAULT_CATEGORIES if category in available_categories), None),
        "code_live": (tuple(category for category in CODE_LIVE_CATEGORIES if category in available_categories), None),
        "failure_weighted": (available_categories, failure_weights or None),
    }
    for category in available_categories:
        groups[f"category_{sanitize(category)}"] = ((category,), None)

    donor_orders: dict[str, list[int]] = {}
    donor_meta: dict[str, Any] = {}
    for group_name, (categories, weights) in groups.items():
        if not categories:
            continue
        for mode in ("target_delta", "hotness"):
            donor_name = f"{group_name}_{mode}"
            scores = category_score(
                heatmaps,
                categories=categories,
                weights=weights,
                mode=mode,
                segments=segments,
                stats=stats,
            )
            donor_orders[donor_name] = top_order(scores)
            donor_meta[donor_name] = {
                "categories": categories,
                "weights": weights,
                "mode": mode,
                "segments": segments,
                "stats": stats,
            }

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate_id: str, family: str, selected_parts: list[list[int]], lineage: dict[str, Any], priority: int) -> None:
        selected = unique_fill(selected_parts, fallback_order, args.budget)
        if len(selected) != args.budget:
            raise ValueError(f"{candidate_id} selected {len(selected)} != {args.budget}")
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
            },
        )
        row = {
            "candidate_id": candidate_id,
            "family": family,
            "score_file": str(score_path.relative_to(args.out_dir)),
            "topks": str(args.budget),
            "unit": "ov",
            "projection_sites": "ov",
            "mask_strategy": "global",
            "ablation": "zero",
            "selected_attention_ov_channels": args.budget,
            "attention_ov_fraction": args.budget / total_ov,
            "selected_sha1": digest,
            "priority": priority,
            "lineage": lineage,
        }
        row.update(layer_stats(selected, hidden, n_layers))
        candidates.append(row)

    add("base_mask", "base_mask", [base_order[: args.budget]], {"budget": args.budget}, priority=9000)
    for replace in replace_counts:
        if replace <= 0 or replace >= args.budget:
            continue
        for protected_tail in protected_tail_counts:
            if protected_tail < 0 or protected_tail >= args.budget:
                continue
            prefix_count = args.budget - protected_tail - replace
            if prefix_count < 0:
                continue
            tail = base_order[-protected_tail:] if protected_tail else []
            for donor_name, donor_order in donor_orders.items():
                donor_slice = donor_order[:replace]
                mode_penalty = 0 if donor_name.endswith("target_delta") else 40
                group_penalty = 0 if donor_name.startswith("failure_weighted") else 10 if donor_name.startswith("balanced") else 20
                tail_penalty = abs(protected_tail - 12288) // 1024
                replace_penalty = abs(replace - 4096) // 512
                family = f"anchored_atlas_{donor_name}"
                add(
                    f"{family}_b{args.budget}_r{replace}_p{protected_tail}",
                    family,
                    [base_order[:prefix_count], donor_slice, tail],
                    {
                        "budget": args.budget,
                        "replace": replace,
                        "protected_tail": protected_tail,
                        "kept_base_prefix": prefix_count,
                        "donor": donor_name,
                        **donor_meta[donor_name],
                    },
                    priority=1000 + mode_penalty + group_penalty + tail_penalty + replace_penalty,
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
        "artifact": "issue3_attention_anchored_atlas_repair_candidates",
        "base_attribution": str(args.base_attribution),
        "parent_attribution": str(args.parent_attribution),
        "atlas_dir": str(args.atlas_dir),
        "failure_summary": str(args.failure_summary) if args.failure_summary else None,
        "total_attention_ov_channels": total_ov,
        "budget": args.budget,
        "replace_counts": replace_counts,
        "protected_tail_counts": protected_tail_counts,
        "segments": segments,
        "stats": stats,
        "candidate_count": len(candidates),
        "available_categories": available_categories,
        "notes": [
            "proposal candidates only; behavior claims require full autoregressive BFCL eval",
            "keeps a working base mask anchored and uses atlas category heatmaps only for suffix repair",
            "protected_tail preserves the useful tail from the incumbent exact mask",
        ],
    }
    write_jsonl(args.out_dir / "candidate_scores.jsonl", candidates)
    json_dump(args.out_dir / "candidate_manifest.json", manifest)
    print(json.dumps(jsonable({"candidate_count": len(candidates), "first": candidates[:12]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
