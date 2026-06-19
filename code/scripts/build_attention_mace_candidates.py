#!/usr/bin/env python3
"""Build token-style attention MACE candidate score files for issue 3.

The output stays compatible with ``bfcl_attention_qwen3.py eval-ladder``:
each candidate is an NPZ containing ``head_scores`` and ``ov_scores``.  For
OV-only candidates the score values encode an exact selected-channel order, so
evaluating with ``--unit ov --topks <budget>`` tests that specific mask.
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


def category_donor_orders(
    atlas_dir: Path,
    *,
    categories: tuple[str, ...],
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> dict[str, list[int]]:
    path = atlas_dir / "category_attention_channel_heatmaps.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    out: dict[str, list[int]] = {}
    for category in categories:
        key = category_key(category, "ov")
        if key not in data.files:
            continue
        score = selected_mean(data[key], segments=segments, stats=stats)
        out[f"category_{sanitize(category)}"] = top_order(score)
    if out:
        score = np.stack(
            [selected_mean(data[category_key(category, "ov")], segments=segments, stats=stats) for category in categories if category_key(category, "ov") in data.files],
            axis=0,
        ).mean(axis=0)
        out["category_balanced"] = top_order(score)
    return out


def load_eval_correct(path: Path) -> dict[str, bool]:
    rows = read_jsonl(path)
    return {str(row["id"]): bool(row.get("normalized_correct", row.get("correct", False))) for row in rows}


def rescue_query_indices(
    *,
    manifest_path: Path,
    small_eval: Path,
    large_eval: Path,
) -> tuple[list[int], dict[str, Any]]:
    small = load_eval_correct(small_eval)
    large = load_eval_correct(large_eval)
    rescued = [eval_id for eval_id, ok in large.items() if ok and not small.get(eval_id, False)]
    lost = [eval_id for eval_id, ok in small.items() if ok and not large.get(eval_id, False)]
    stable_success = [eval_id for eval_id, ok in large.items() if ok and small.get(eval_id, False)]
    stable_fail = [eval_id for eval_id, ok in large.items() if (not ok) and (not small.get(eval_id, False))]
    idx_by_id = {}
    category_by_id = {}
    for row in read_jsonl(manifest_path):
        idx_by_id[str(row["eval_id"])] = int(row["global_index"])
        category_by_id[str(row["eval_id"])] = row.get("category")
    indices = [idx_by_id[eval_id] for eval_id in rescued if eval_id in idx_by_id]
    info = {
        "rescued": len(rescued),
        "lost": len(lost),
        "stable_success": len(stable_success),
        "stable_fail": len(stable_fail),
        "rescued_with_atlas": len(indices),
        "rescued_categories": dict(Counter(category_by_id.get(eval_id, "unknown") for eval_id in rescued)),
        "lost_categories": dict(Counter(category_by_id.get(eval_id, "unknown") for eval_id in lost)),
        "small_eval": str(small_eval),
        "large_eval": str(large_eval),
    }
    return indices, info


def rescue_donor_order(
    atlas_dir: Path,
    indices: list[int],
    *,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> list[int]:
    if not indices:
        return []
    path = atlas_dir / "ov_channel_scores_global_uint8.npy"
    if not path.exists():
        return []
    scores = np.load(path, mmap_mode="r")
    seg_idx = axis_indices(SEGMENTS, segments)
    stat_idx = axis_indices(STATS, stats)
    sub = np.asarray(scores[np.asarray(indices, dtype=np.int64)][:, seg_idx][:, :, stat_idx], dtype=np.float32)
    return top_order(sub.mean(axis=(0, 1, 2)))


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
    parser.add_argument("--parent-attribution", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="73728,81920,90112,98304,106496,114688")
    parser.add_argument("--replace-counts", default="1024,2048,4096,8192,16384")
    parser.add_argument("--donor-uppers", default="114688,131072,147456")
    parser.add_argument("--band-offsets", default="0,1,2,3")
    parser.add_argument("--categories", default="simple,live_simple,exec_simple,java,javascript,sql")
    parser.add_argument("--segments", default="target,full")
    parser.add_argument("--stats", default="mean_abs,rms")
    parser.add_argument("--rescue-small-eval", type=Path)
    parser.add_argument("--rescue-large-eval", type=Path)
    parser.add_argument("--max-candidates-per-family", type=int, default=80)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.out_dir / "score_files"
    score_dir.mkdir(parents=True, exist_ok=True)

    parent = np.load(args.parent_attribution)
    parent_head = parent["head_scores"].astype(np.float32)
    parent_ov = parent["ov_scores"].astype(np.float32)
    n_layers, hidden = parent_ov.shape
    total_ov = int(parent_ov.size)
    parent_order = top_order(parent_ov)
    budgets = [b for b in parse_ints(args.budgets) if 0 < b <= total_ov]
    replace_counts = parse_ints(args.replace_counts)
    donor_uppers = [min(u, total_ov) for u in parse_ints(args.donor_uppers)]
    band_offsets = parse_ints(args.band_offsets)
    categories = parse_csv(args.categories)
    segments = parse_csv(args.segments)
    stats = parse_csv(args.stats)

    donor_orders = category_donor_orders(args.atlas_dir, categories=categories, segments=segments, stats=stats)
    rescue_info: dict[str, Any] | None = None
    if args.rescue_small_eval and args.rescue_large_eval:
        rescue_indices, rescue_info = rescue_query_indices(
            manifest_path=args.atlas_dir / "query_manifest.jsonl",
            small_eval=args.rescue_small_eval,
            large_eval=args.rescue_large_eval,
        )
        rescue_order = rescue_donor_order(args.atlas_dir, rescue_indices, segments=segments, stats=stats)
        if rescue_order:
            donor_orders["rescue_large_over_small"] = rescue_order

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate_id: str, family: str, selected: list[int], lineage: dict[str, Any], priority: int) -> None:
        budget = int(lineage["budget"])
        selected = unique_fill([selected], parent_order, budget)
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
            ov_shape=parent_ov.shape,
            metadata={
                "candidate_id": candidate_id,
                "family": family,
                "lineage": lineage,
                "parent_attribution": str(args.parent_attribution),
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

    for budget in budgets:
        add(
            f"parent_prefix_b{budget}",
            "parent_prefix",
            parent_order[:budget],
            {"budget": budget},
            priority=9000,
        )

    family_counts: Counter[tuple[str, int]] = Counter()
    for budget in budgets:
        for replace in replace_counts:
            if replace <= 0 or replace >= budget:
                continue
            keep = parent_order[: budget - replace]
            for upper in donor_uppers:
                if upper <= budget:
                    continue
                for offset in band_offsets:
                    donor_start = budget + offset * replace
                    donor_end = min(donor_start + replace, upper)
                    if donor_start >= upper or donor_end - donor_start < replace:
                        continue
                    donor = parent_order[donor_start:donor_end]
                    family = "band_shift"
                    family_key = (family, budget)
                    if family_counts[family_key] < args.max_candidates_per_family:
                        add(
                            f"band_shift_b{budget}_u{upper}_o{offset}_r{replace}",
                            family,
                            unique_fill([keep, donor], parent_order, budget),
                            {
                                "budget": budget,
                                "replace": replace,
                                "donor_upper": upper,
                                "donor_offset": offset,
                                "donor_start_rank": donor_start,
                                "donor_end_rank": donor_end,
                                "kept_parent_prefix": budget - replace,
                            },
                            priority=1000 + (abs(budget - 98304) // 1024) * 10 + abs(offset - 3) * 2 + abs(replace - 4096) // 1024,
                        )
                        family_counts[family_key] += 1
            for donor_name, donor_order in donor_orders.items():
                if len(donor_order) < replace:
                    continue
                family = f"atlas_repair_{donor_name}"
                family_key = (family, budget)
                if family_counts[family_key] >= args.max_candidates_per_family:
                    continue
                add(
                    f"{family}_b{budget}_r{replace}",
                    family,
                    unique_fill([keep, donor_order], parent_order, budget),
                    {
                        "budget": budget,
                        "replace": replace,
                        "donor": donor_name,
                        "kept_parent_prefix": budget - replace,
                        "segments": segments,
                        "stats": stats,
                    },
                    priority=2000 + (abs(budget - 98304) // 1024) * 10 + abs(replace - 4096) // 1024,
                )
                family_counts[family_key] += 1

    candidates.sort(
        key=lambda row: (
            int(row["priority"]),
            abs(int(row["selected_attention_ov_channels"]) - 98304),
            -float(row["layer_entropy"]),
            str(row["candidate_id"]),
        )
    )
    for rank, row in enumerate(candidates, start=1):
        row["proposal_rank"] = rank
        row["selected_for_first_eval"] = rank == 1

    manifest = {
        "artifact": "issue3_attention_mace_candidates",
        "parent_attribution": str(args.parent_attribution),
        "atlas_dir": str(args.atlas_dir),
        "total_attention_ov_channels": total_ov,
        "budgets": budgets,
        "replace_counts": replace_counts,
        "donor_uppers": donor_uppers,
        "categories": categories,
        "segments": segments,
        "stats": stats,
        "candidate_count": len(candidates),
        "rescue_info": rescue_info,
        "notes": [
            "proposal candidates only; behavior claims require full autoregressive BFCL eval",
            "pairs have category metadata but no train/calibration/heldout split, so this is not a sealed-heldout selection surface",
            "families mirror token issue12: parent prefix, band shift, category/bucket-style donor repair, and optional rescue donor repair",
        ],
    }
    write_jsonl(args.out_dir / "candidate_scores.jsonl", candidates)
    json_dump(args.out_dir / "candidate_manifest.json", manifest)
    print(json.dumps(jsonable({"candidate_count": len(candidates), "first": candidates[:10], "rescue_info": rescue_info}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
