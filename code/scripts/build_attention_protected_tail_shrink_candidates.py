#!/usr/bin/env python3
"""Build protected-tail attention shrink candidates.

This is the attention analogue of token issue12 protected-tail pressure:
start from a repaired incumbent mask, shrink the total OV-channel budget, and
optionally preserve a suffix of the incumbent ordering while removing channels
from the middle/prefix boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def write_score_file(
    path: Path,
    *,
    head_scores: np.ndarray,
    selected: list[int],
    ov_shape: tuple[int, int],
    metadata: dict[str, Any],
) -> None:
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--protected-tail-counts", default="0,512,1024,2048,4096,8192,12288")
    parser.add_argument("--label", default="protected_tail")
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

    budgets = [budget for budget in parse_ints(args.budgets) if 0 < budget <= len(base_order)]
    protected_tail_counts = parse_ints(args.protected_tail_counts)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate_id: str, family: str, selected_parts: list[list[int]], lineage: dict[str, Any], priority: int) -> None:
        budget = int(lineage["budget"])
        selected = unique_fill(selected_parts, fallback_order, budget)
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
            f"{args.label}_prefix_shrink_b{budget}",
            f"{args.label}_prefix_shrink",
            [base_order[:budget]],
            {"budget": budget, "protected_tail": 0},
            priority=1000,
        )
        for protected_tail in protected_tail_counts:
            if protected_tail <= 0 or protected_tail >= budget:
                continue
            if protected_tail >= len(base_order):
                continue
            prefix_count = budget - protected_tail
            candidate_id = f"{args.label}_b{budget}_p{protected_tail}"
            add(
                candidate_id,
                f"{args.label}_protected_tail",
                [base_order[:prefix_count], base_order[-protected_tail:]],
                {
                    "budget": budget,
                    "protected_tail": protected_tail,
                    "kept_base_prefix": prefix_count,
                    "removed_middle_count": max(len(base_order) - prefix_count - protected_tail, 0),
                },
                priority=1000 + abs(protected_tail - min(8192, max(1, budget // 12))) // 512,
            )

    candidates.sort(
        key=lambda row: (
            int(row["priority"]),
            int(row["selected_attention_ov_channels"]),
            -float(row["layer_entropy"]),
            str(row["candidate_id"]),
        )
    )
    for rank, row in enumerate(candidates, start=1):
        row["proposal_rank"] = rank
        row["selected_for_first_eval"] = rank == 1

    manifest = {
        "artifact": "issue3_attention_protected_tail_shrink_candidates",
        "base_attribution": str(args.base_attribution),
        "parent_attribution": str(args.parent_attribution),
        "total_attention_ov_channels": total_ov,
        "base_selected_ov_channels": len(base_order),
        "budgets": budgets,
        "protected_tail_counts": protected_tail_counts,
        "candidate_count": len(candidates),
        "notes": [
            "proposal candidates only; behavior claims require full autoregressive BFCL eval",
            "protected-tail candidates preserve the incumbent suffix while shrinking the total OV budget",
        ],
    }
    write_jsonl(args.out_dir / "candidate_scores.jsonl", candidates)
    json_dump(args.out_dir / "candidate_manifest.json", manifest)
    print(json.dumps(jsonable({"candidate_count": len(candidates), "first": candidates[:12]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
