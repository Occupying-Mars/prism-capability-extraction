#!/usr/bin/env python3
"""Build atlas-derived attention score files for issue 3 candidate evals.

This is a cheap proposal stage. It reads the attention atlas and emits NPZ
files with ``head_scores`` and ``ov_scores`` so the existing BFCL attention
evaluator can run full autoregressive evals without a new masking path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SEGMENTS = ("prompt", "target", "full")
STATS = ("mean_abs", "rms", "max_abs")
CHANNEL_SITES = ("q", "k", "v", "ov")
UNIT_SITES = ("q_head", "k_head", "v_head", "ov_head", "qk_group", "qkv_group")
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


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def sanitize_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def parse_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def axis_indices(names: tuple[str, ...], selected: tuple[str, ...]) -> list[int]:
    index = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in selected if name not in index]
    if missing:
        raise ValueError(f"unknown axis values: {missing}; allowed={names}")
    return [index[name] for name in selected]


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


def normalize01(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def rank_score_summary(scores: np.ndarray, fractions: tuple[float, ...]) -> dict[str, Any]:
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    total = float(flat.sum())
    out: dict[str, Any] = {
        "shape": list(scores.shape),
        "total_items": int(flat.size),
        "score_sum": total,
        "score_min": float(flat.min()) if flat.size else None,
        "score_max": float(flat.max()) if flat.size else None,
        "score_mean": float(flat.mean()) if flat.size else None,
    }
    for frac in fractions:
        k = int(round(float(frac) * flat.size))
        if k <= 0:
            out[f"top_{frac:.3f}_mass_frac"] = 0.0
            continue
        k = min(k, flat.size)
        top = np.partition(flat, flat.size - k)[flat.size - k :]
        out[f"top_{frac:.3f}_items"] = int(k)
        out[f"top_{frac:.3f}_mass_frac"] = float(top.sum() / total) if total else None
    return out


def sha1_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha1()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def category_key(category: str, site: str, suffix: str) -> str:
    return f"category_{sanitize_key(category)}_{site}_{suffix}"


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


class AtlasSources:
    def __init__(self, atlas_dir: Path):
        self.atlas_dir = atlas_dir
        self.channel_category = load_npz(atlas_dir / "category_attention_channel_heatmaps.npz")
        self.unit_category = load_npz(atlas_dir / "category_attention_unit_heatmaps.npz")
        self.query_channel_scores: dict[str, np.memmap] = {}
        self.query_unit_scores: np.lib.npyio.NpzFile | None = None

        for site in CHANNEL_SITES:
            path = atlas_dir / f"{site}_channel_scores_global_uint8.npy"
            if path.exists():
                self.query_channel_scores[site] = np.load(path, mmap_mode="r")

        unit_path = atlas_dir / "attention_unit_scores_global_uint8.npz"
        if unit_path.exists():
            self.query_unit_scores = np.load(unit_path)

    @property
    def categories(self) -> tuple[str, ...]:
        values = self.channel_category["categories"]
        return tuple(str(x) for x in values.tolist())

    @property
    def has_query_scores(self) -> bool:
        return bool(self.query_channel_scores) and self.query_unit_scores is not None


def category_channel_score(
    sources: AtlasSources,
    *,
    site: str,
    categories: tuple[str, ...],
    mode: str,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> np.ndarray:
    pieces = []
    for category in categories:
        key = category_key(category, site, "global_score_mean_float16")
        if key not in sources.channel_category.files:
            raise KeyError(f"missing category channel key {key}")
        array = sources.channel_category[key]
        if mode == "hotness":
            pieces.append(selected_mean(array, segments=segments, stats=stats))
        elif mode == "target_delta":
            pieces.append(target_delta(array, stats=stats))
        else:
            raise ValueError(f"unknown mode: {mode}")
    return np.stack(pieces, axis=0).mean(axis=0).astype(np.float32)


def category_unit_score(
    sources: AtlasSources,
    *,
    site: str,
    categories: tuple[str, ...],
    mode: str,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> np.ndarray:
    pieces = []
    for category in categories:
        key = category_key(category, site, "global_score_mean_float16")
        if key not in sources.unit_category.files:
            raise KeyError(f"missing category unit key {key}")
        array = sources.unit_category[key]
        if mode == "hotness":
            pieces.append(selected_mean(array, segments=segments, stats=stats))
        elif mode == "target_delta":
            pieces.append(target_delta(array, stats=stats))
        else:
            raise ValueError(f"unknown mode: {mode}")
    return np.stack(pieces, axis=0).mean(axis=0).astype(np.float32)


def query_channel_score(
    sources: AtlasSources,
    *,
    site: str,
    mode: str,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> np.ndarray:
    if site not in sources.query_channel_scores:
        raise FileNotFoundError(sources.atlas_dir / f"{site}_channel_scores_global_uint8.npy")
    arr = np.asarray(sources.query_channel_scores[site], dtype=np.float32)
    if mode == "hotness":
        seg_idx = axis_indices(SEGMENTS, segments)
        stat_idx = axis_indices(STATS, stats)
        return arr[:, seg_idx][:, :, stat_idx].mean(axis=(0, 1, 2)).astype(np.float32)
    if mode == "target_delta":
        stat_idx = axis_indices(STATS, stats)
        target_full = arr[:, [1, 2]][:, :, stat_idx].mean(axis=(0, 1, 2))
        prompt = arr[:, [0]][:, :, stat_idx].mean(axis=(0, 1, 2))
        return np.maximum(target_full - prompt, 0.0).astype(np.float32)
    raise ValueError(f"unknown mode: {mode}")


def query_unit_score(
    sources: AtlasSources,
    *,
    site: str,
    mode: str,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> np.ndarray:
    if sources.query_unit_scores is None or site not in sources.query_unit_scores.files:
        raise FileNotFoundError(sources.atlas_dir / "attention_unit_scores_global_uint8.npz")
    arr = np.asarray(sources.query_unit_scores[site], dtype=np.float32)
    if mode == "hotness":
        seg_idx = axis_indices(SEGMENTS, segments)
        stat_idx = axis_indices(STATS, stats)
        return arr[:, seg_idx][:, :, stat_idx].mean(axis=(0, 1, 2)).astype(np.float32)
    if mode == "target_delta":
        stat_idx = axis_indices(STATS, stats)
        target_full = arr[:, [1, 2]][:, :, stat_idx].mean(axis=(0, 1, 2))
        prompt = arr[:, [0]][:, :, stat_idx].mean(axis=(0, 1, 2))
        return np.maximum(target_full - prompt, 0.0).astype(np.float32)
    raise ValueError(f"unknown mode: {mode}")


def expand_kv_to_heads(group_scores: np.ndarray, n_heads: int) -> np.ndarray:
    n_layers, n_kv_heads = group_scores.shape
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"cannot expand {n_kv_heads} kv groups to {n_heads} heads")
    return np.repeat(group_scores, n_heads // n_kv_heads, axis=1).astype(np.float32)


def make_head_scores(unit_scores: dict[str, np.ndarray]) -> np.ndarray:
    q_head = unit_scores["q_head"]
    ov_head = unit_scores["ov_head"]
    n_layers, n_heads = q_head.shape
    k_head = expand_kv_to_heads(unit_scores["k_head"], n_heads)
    v_head = expand_kv_to_heads(unit_scores["v_head"], n_heads)
    qk_group = expand_kv_to_heads(unit_scores["qk_group"], n_heads)
    qkv_group = expand_kv_to_heads(unit_scores["qkv_group"], n_heads)

    parts = [
        normalize01(q_head),
        normalize01(k_head),
        normalize01(v_head),
        normalize01(ov_head),
        normalize01(qk_group),
        normalize01(qkv_group),
    ]
    return np.stack(parts, axis=0).mean(axis=0).astype(np.float32)


def ov_head_prior(ov_channel_scores: np.ndarray, head_scores: np.ndarray) -> np.ndarray:
    n_layers, hidden = ov_channel_scores.shape
    n_heads = head_scores.shape[1]
    if hidden % n_heads != 0:
        return ov_channel_scores.astype(np.float32)
    head_dim = hidden // n_heads
    expanded = np.repeat(normalize01(head_scores), head_dim, axis=1)
    return (normalize01(ov_channel_scores) + expanded).astype(np.float32)


def build_family(
    sources: AtlasSources,
    *,
    source_kind: str,
    categories: tuple[str, ...],
    mode: str,
    segments: tuple[str, ...],
    stats: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    unit_scores: dict[str, np.ndarray] = {}
    if source_kind == "category":
        ov_scores = category_channel_score(
            sources,
            site="ov",
            categories=categories,
            mode=mode,
            segments=segments,
            stats=stats,
        )
        for site in UNIT_SITES:
            unit_scores[site] = category_unit_score(
                sources,
                site=site,
                categories=categories,
                mode=mode,
                segments=segments,
                stats=stats,
            )
    elif source_kind == "query":
        ov_scores = query_channel_score(sources, site="ov", mode=mode, segments=segments, stats=stats)
        for site in UNIT_SITES:
            unit_scores[site] = query_unit_score(sources, site=site, mode=mode, segments=segments, stats=stats)
    else:
        raise ValueError(f"unknown source_kind: {source_kind}")

    head_scores = make_head_scores(unit_scores)
    ov_scores = ov_head_prior(ov_scores, head_scores)
    meta = {
        "source_kind": source_kind,
        "categories": categories,
        "mode": mode,
        "segments": segments,
        "stats": stats,
        "head_formula": "mean normalize01(q_head,k_head,v_head,ov_head,expanded qk_group,expanded qkv_group)",
        "ov_formula": "normalize01(ov_channel_score) + expanded normalize01(head_scores)",
    }
    return head_scores.astype(np.float32), ov_scores.astype(np.float32), meta


def write_score_file(
    path: Path,
    *,
    head_scores: np.ndarray,
    ov_scores: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        head_scores=head_scores.astype(np.float32),
        ov_scores=ov_scores.astype(np.float32),
        metadata_json=np.array(json.dumps(jsonable(metadata), sort_keys=True)),
    )
    return {
        "score_file": str(path),
        "sha1": sha1_arrays(head_scores.astype(np.float32), ov_scores.astype(np.float32)),
        "head": rank_score_summary(head_scores, (0.5, 2 / 3, 7 / 9)),
        "ov": rank_score_summary(ov_scores, (0.5, 2 / 3, 7 / 9)),
    }


def add_eval_suggestions(
    *,
    candidate_id: str,
    score_file: str,
    attention_fraction: float,
    n_layers: int,
    n_heads: int,
    hidden: int,
    base_priority: int,
    selected_for_first_eval: bool,
) -> list[dict[str, Any]]:
    total_heads = n_layers * n_heads
    total_ov = n_layers * hidden
    head_topk = int(round(attention_fraction * total_heads))
    ov_topk = int(round(attention_fraction * total_ov))
    return [
        {
            "candidate_id": f"{candidate_id}__ov_global_{ov_topk}",
            "score_file": score_file,
            "unit": "ov",
            "projection_sites": "ov",
            "mask_strategy": "global",
            "topks": str(ov_topk),
            "ablation": "zero",
            "attention_fraction": attention_fraction,
            "priority": base_priority * 10,
            "selected_for_first_eval": selected_for_first_eval,
            "reason": "first full BFCL candidate: 50pct OV-channel atlas MACE with fixed MLP substrate",
        },
        {
            "candidate_id": f"{candidate_id}__ov_head_scaffold_{ov_topk}",
            "score_file": score_file,
            "unit": "ov",
            "projection_sites": "ov",
            "mask_strategy": "head-scaffold-ov",
            "topks": str(ov_topk),
            "head_scaffold_topk": str(head_topk),
            "ablation": "zero",
            "attention_fraction": attention_fraction,
            "priority": base_priority * 10 + 1,
            "selected_for_first_eval": False,
            "reason": "structured OV candidate; keeps channels inside atlas-hot heads",
        },
        {
            "candidate_id": f"{candidate_id}__qkv_ov_head_{head_topk}",
            "score_file": score_file,
            "unit": "head",
            "projection_sites": "qkv-ov",
            "mask_strategy": "global",
            "topks": str(head_topk),
            "ablation": "zero",
            "attention_fraction": attention_fraction,
            "priority": 900 + base_priority,
            "selected_for_first_eval": False,
            "reason": "diagnostic only unless OV candidates give signal; previous 50pct qkv/ov collapsed",
        },
    ]


def candidate_priority(candidate_id: str) -> int:
    order = {
        "category_balanced_hotness": 1,
        "category_balanced_target_delta": 2,
        "category_code_live_hotness": 3,
        "category_code_live_target_delta": 4,
        "query_weighted_hotness": 5,
        "query_weighted_target_delta": 6,
    }
    return order.get(candidate_id, 99)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--segments", default="target,full")
    parser.add_argument("--stats", default="mean_abs,rms")
    parser.add_argument("--attention-fraction", type=float, default=0.5)
    parser.add_argument("--category-groups", default="balanced,code_live")
    parser.add_argument("--include-query-mean", action="store_true")
    args = parser.parse_args()

    sources = AtlasSources(args.atlas_dir)
    segments = parse_csv(args.segments, ("target", "full"))
    stats = parse_csv(args.stats, ("mean_abs", "rms"))
    category_groups = parse_csv(args.category_groups, ("balanced", "code_live"))
    fractions = parse_float_csv(f"{args.attention_fraction},0.6666667,0.7777778")

    available_categories = set(sources.categories)
    group_map = {
        "balanced": tuple(category for category in DEFAULT_CATEGORIES if category in available_categories),
        "code_live": tuple(category for category in CODE_LIVE_CATEGORIES if category in available_categories),
    }
    missing_groups = [group for group in category_groups if group not in group_map]
    if missing_groups:
        raise ValueError(f"unknown category groups: {missing_groups}; allowed={sorted(group_map)}")

    score_dir = args.out_dir / "score_files"
    rows: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []

    def add_candidate(candidate_id: str, source_kind: str, categories: tuple[str, ...], mode: str) -> None:
        head_scores, ov_scores, metadata = build_family(
            sources,
            source_kind=source_kind,
            categories=categories,
            mode=mode,
            segments=segments,
            stats=stats,
        )
        score_path = score_dir / f"{candidate_id}.npz"
        summary = write_score_file(
            score_path,
            head_scores=head_scores,
            ov_scores=ov_scores,
            metadata={
                "candidate_id": candidate_id,
                "atlas_dir": str(args.atlas_dir),
                **metadata,
            },
        )
        n_layers, n_heads = head_scores.shape
        hidden = ov_scores.shape[1]
        rel_score_path = str(score_path.relative_to(args.out_dir))
        selected_for_first_eval = candidate_id == "category_balanced_hotness"
        row = {
            "candidate_id": candidate_id,
            "score_file": rel_score_path,
            "selected_for_first_eval": selected_for_first_eval,
            "metadata": metadata,
            "summary": summary,
            "budget_probe_fractions": fractions,
        }
        rows.append(row)
        suggestions.extend(
            add_eval_suggestions(
                candidate_id=candidate_id,
                score_file=rel_score_path,
                attention_fraction=args.attention_fraction,
                n_layers=n_layers,
                n_heads=n_heads,
                hidden=hidden,
                base_priority=candidate_priority(candidate_id),
                selected_for_first_eval=selected_for_first_eval,
            )
        )

    for group in category_groups:
        categories = group_map[group]
        if not categories:
            continue
        for mode in ("hotness", "target_delta"):
            add_candidate(f"category_{group}_{mode}", "category", categories, mode)

    if args.include_query_mean:
        if not sources.has_query_scores:
            raise FileNotFoundError("query mean requested, but full query score arrays are missing")
        for mode in ("hotness", "target_delta"):
            add_candidate(f"query_weighted_{mode}", "query", tuple(), mode)

    suggestions.sort(key=lambda row: (int(row["priority"]), str(row["candidate_id"])))
    manifest = {
        "artifact": "issue3_attention_atlas_candidate_scores",
        "atlas_dir": str(args.atlas_dir),
        "out_dir": str(args.out_dir),
        "segments": segments,
        "stats": stats,
        "available_categories": sources.categories,
        "category_groups": {key: value for key, value in group_map.items() if key in category_groups},
        "include_query_mean": bool(args.include_query_mean),
        "candidate_count": len(rows),
        "attention_fraction_goal": args.attention_fraction,
        "files": {
            "candidate_scores_jsonl": "candidate_scores.jsonl",
            "eval_suggestions_jsonl": "eval_suggestions.jsonl",
            "score_files": "score_files/*.npz",
        },
        "notes": [
            "proposal scores only; behavior claims require full autoregressive BFCL eval",
            "category-balanced candidates avoid simple-category dominance",
            "qkv-ov suggestions are diagnostic because earlier 50pct and 66pct qkv/ov collapsed",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "candidate_scores.jsonl", rows)
    write_jsonl(args.out_dir / "eval_suggestions.jsonl", suggestions)
    json_dump(args.out_dir / "candidate_manifest.json", manifest)
    print(json.dumps(jsonable({"candidate_count": len(rows), "first_eval": suggestions[:4]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
