#!/usr/bin/env python3
"""Build GLM issue-12 style aggregate mask candidates.

This is the cheap first stage before full BFCL generation eval. It emits masks
in the existing ``mlp_scores`` NPZ format and ranks them with attribution-only
coverage/proxy metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def layer_channel(gid: int, d_ffn: int) -> tuple[int, int]:
    return int(gid // d_ffn), int(gid % d_ffn)


def write_mask(path: Path, selected: list[int], *, n_layers: int, d_ffn: int) -> None:
    scores = np.zeros((n_layers, d_ffn), dtype=np.float32)
    for rank, gid in enumerate(selected):
        layer, channel = layer_channel(gid, d_ffn)
        scores[layer, channel] = float(len(selected) - rank)
    np.savez_compressed(path, mlp_scores=scores)


def fill_to_budget(prefixes: list[list[int]], fallback: list[int], budget: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for source in [*prefixes, fallback]:
        for gid in source:
            if gid in seen:
                continue
            selected.append(gid)
            seen.add(gid)
            if len(selected) >= budget:
                return selected
    return selected


def top_by_scores(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(scores, -k)[-k:]
    ordered = idx[np.lexsort((idx, -scores[idx]))]
    return [int(gid) for gid in ordered]


def layer_quota_mask(
    scores_2d: np.ndarray,
    budget: int,
    *,
    mode: str,
    minimum_per_layer: int,
) -> list[int]:
    n_layers, d_ffn = scores_2d.shape
    layer_mass = scores_2d.sum(axis=1).astype(np.float64)
    if mode == "sqrt_mass":
        weights = np.sqrt(np.maximum(layer_mass, 0.0))
    elif mode == "uniform_mass":
        weights = np.ones(n_layers, dtype=np.float64) + layer_mass / max(float(layer_mass.sum()), 1e-12)
    elif mode == "late_boost":
        ramp = np.linspace(0.75, 1.25, n_layers, dtype=np.float64)
        weights = np.sqrt(np.maximum(layer_mass, 0.0)) * ramp
    else:
        raise ValueError(f"unknown quota mode: {mode}")
    weights = weights / max(float(weights.sum()), 1e-12)
    quotas = np.floor(weights * budget).astype(int)
    quotas = np.maximum(quotas, min(minimum_per_layer, d_ffn))
    while int(quotas.sum()) > budget:
        idx = int(np.argmax(quotas))
        quotas[idx] -= 1
    while int(quotas.sum()) < budget:
        idx = int(np.argmax(weights * budget - quotas))
        quotas[idx] += 1

    selected: list[int] = []
    for layer, q in enumerate(quotas):
        if q <= 0:
            continue
        local = top_by_scores(scores_2d[layer], int(q))
        selected.extend(layer * d_ffn + ch for ch in local)
    return selected[:budget]


def metrics(selected: list[int], scores: np.ndarray, *, total_score: float) -> dict[str, Any]:
    arr = np.asarray(selected, dtype=np.int64)
    kept = float(scores[arr].sum()) if arr.size else 0.0
    layers = arr // scores.reshape(-1).shape[0] if False else None
    return {
        "coverage_score": kept,
        "coverage_frac": kept / total_score if total_score else None,
    }


def layer_stats(selected: list[int], n_layers: int, d_ffn: int) -> dict[str, Any]:
    counts = np.zeros(n_layers, dtype=np.int64)
    for gid in selected:
        counts[gid // d_ffn] += 1
    probs = counts[counts > 0].astype(np.float64) / max(len(selected), 1)
    entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
    return {
        "active_layers": int((counts > 0).sum()),
        "max_layer_count": int(counts.max()) if counts.size else 0,
        "layer_entropy": entropy,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--budgets", default="240000,260000,280000,297222,320000")
    p.add_argument("--replace-counts", default="5000,10000,20000,30000")
    p.add_argument("--donor-uppers", default="320000,360000,420000,500000")
    p.add_argument("--minimum-per-layer", type=int, default=128)
    p.add_argument("--top-eval", type=int, default=12)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "candidate_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    parent = np.load(args.parent)
    scores_2d = parent["mlp_scores"].astype(np.float32, copy=False)
    if scores_2d.ndim != 2:
        raise ValueError(f"expected 2d mlp_scores, got {scores_2d.shape}")
    n_layers, d_ffn = scores_2d.shape
    flat = scores_2d.reshape(-1)
    total_channels = int(flat.size)
    total_score = float(flat.sum())
    parent_rank = top_by_scores(flat, total_channels)

    budgets = parse_ints(args.budgets)
    replace_counts = parse_ints(args.replace_counts)
    donor_uppers = parse_ints(args.donor_uppers)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate_id: str, kind: str, selected: list[int], lineage: dict[str, Any]) -> None:
        selected = list(dict.fromkeys(int(gid) for gid in selected))
        budget = int(lineage["budget"])
        if len(selected) != budget:
            selected = fill_to_budget([selected], parent_rank, budget)
        key = hashlib.sha1(np.asarray(selected, dtype=np.int32).tobytes()).hexdigest()
        if key in seen:
            return
        seen.add(key)
        mask_path = mask_dir / f"{candidate_id}.npz"
        write_mask(mask_path, selected, n_layers=n_layers, d_ffn=d_ffn)
        m = metrics(selected, flat, total_score=total_score)
        row = {
            "candidate_id": candidate_id,
            "kind": kind,
            "mask_path": str(mask_path.relative_to(args.out_dir)),
            "selected_mlp_channels": len(selected),
            "mlp_fraction": len(selected) / total_channels,
            "topk_for_eval": len(selected),
            "selected_sha1": key,
            "mask_only_score": m["coverage_frac"],
            "coverage_score": m["coverage_score"],
            "coverage_frac": m["coverage_frac"],
            "lineage": lineage,
        }
        row.update(layer_stats(selected, n_layers, d_ffn))
        candidates.append(row)

    for budget in budgets:
        if budget <= 0 or budget > total_channels:
            continue
        add(f"parent_topk_b{budget}", "parent_topk", parent_rank[:budget], {"budget": budget})
        for mode in ("sqrt_mass", "uniform_mass", "late_boost"):
            add(
                f"layer_quota_{mode}_b{budget}",
                f"layer_quota_{mode}",
                layer_quota_mask(scores_2d, budget, mode=mode, minimum_per_layer=args.minimum_per_layer),
                {"budget": budget, "minimum_per_layer": args.minimum_per_layer},
            )
        for replace in replace_counts:
            if replace <= 0 or replace >= budget:
                continue
            keep = parent_rank[: budget - replace]
            for donor_upper in donor_uppers:
                donor_upper = min(donor_upper, total_channels)
                if donor_upper <= budget:
                    continue
                donor_band = parent_rank[budget:donor_upper]
                if len(donor_band) < replace:
                    continue
                add(
                    f"band_shift_b{budget}_u{donor_upper}_r{replace}",
                    "issue12_parent_tail_replaced_by_donor_band",
                    fill_to_budget([keep, donor_band[:replace]], parent_rank, budget),
                    {
                        "budget": budget,
                        "replace": replace,
                        "donor_upper": donor_upper,
                        "parent_prefix_kept": budget - replace,
                    },
                )

    candidates.sort(
        key=lambda row: (
            float(row["mask_only_score"] or 0.0),
            float(row["layer_entropy"]),
            -int(row["selected_mlp_channels"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(candidates, start=1):
        row["mask_only_rank"] = rank
        row["selected_for_eval"] = rank <= args.top_eval

    manifest = {
        "issue": 12,
        "artifact": "glm_issue12_aggregate_mask_candidates",
        "parent": str(args.parent),
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_channels": total_channels,
        "total_parent_score": total_score,
        "budgets": budgets,
        "replace_counts": replace_counts,
        "donor_uppers": donor_uppers,
        "candidate_count": len(candidates),
        "top_eval": args.top_eval,
        "mask_only_metric": "sum(parent aggregate ReLP score kept) / sum(all aggregate ReLP scores)",
    }
    (args.out_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.out_dir / "candidate_masks.jsonl").open("w") as f:
        for row in candidates:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"candidate_count": len(candidates), "top": candidates[: args.top_eval]}, indent=2))


if __name__ == "__main__":
    main()
