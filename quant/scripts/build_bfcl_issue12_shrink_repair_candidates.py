#!/usr/bin/env python3
"""Build issue #12 shrink candidates around a repaired MACE incumbent.

The category-repair candidates can put useful repair channels near the end of
the ranked mask. A plain prefix shrink would immediately discard those repair
channels. This builder emits both prefix shrinks and protected-tail shrinks so
the next eval round can test whether the repair tail is worth carrying into a
smaller substrate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


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


def write_mask(path: Path, selected: list[int], *, n_layers: int, d_ffn: int) -> None:
    scores = np.zeros((n_layers, d_ffn), dtype=np.float32)
    for rank, gid in enumerate(selected):
        layer, channel = layer_channel(gid, d_ffn)
        scores[layer, channel] = float(len(selected) - rank)
    np.savez_compressed(path, mlp_scores=scores)


def unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_base_lineage(path: Path | None, candidate_id: str | None) -> dict[str, Any]:
    if path is None or candidate_id is None:
        return {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("candidate_id") == candidate_id:
            return row
    raise ValueError(f"candidate_id {candidate_id!r} not found in {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-mask", type=Path, required=True)
    p.add_argument("--base-topk", type=int, required=True)
    p.add_argument("--base-candidate-id", required=True)
    p.add_argument("--base-candidate-jsonl", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--budgets", default="169500,169000,168000,165000,160000")
    p.add_argument("--protect-tail-counts", default="0,500,1000,2000,5000,10000")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "candidate_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(args.base_mask)
    n_layers, d_ffn = scores.shape
    total_channels = n_layers * d_ffn
    if args.base_topk <= 0 or args.base_topk > total_channels:
        raise ValueError(f"invalid base_topk {args.base_topk}")

    base_rank = ranking(scores)[: args.base_topk]
    budgets = sorted(set(parse_ints(args.budgets)), reverse=True)
    protect_tail_counts = sorted(set(parse_ints(args.protect_tail_counts)))
    base_lineage = load_base_lineage(args.base_candidate_jsonl, args.base_candidate_id)

    candidates: list[dict[str, Any]] = []
    seen_masks: set[str] = set()

    def add_candidate(candidate_id: str, kind: str, selected: list[int], lineage: dict[str, Any]) -> None:
        selected = unique_in_order(selected)
        budget = int(lineage["budget"])
        if len(selected) != budget:
            raise ValueError(f"{candidate_id} selected {len(selected)} != budget {budget}")
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
                "lineage": lineage,
            }
        )

    for budget in budgets:
        if budget <= 0 or budget >= args.base_topk:
            continue
        add_candidate(
            f"{args.base_candidate_id}_prefix_shrink_b{budget}",
            "repaired_mask_prefix_shrink",
            base_rank[:budget],
            {
                "budget": budget,
                "base_candidate_id": args.base_candidate_id,
                "base_topk": args.base_topk,
                "removed_channels": args.base_topk - budget,
                "protected_tail": 0,
            },
        )
        for protect_tail in protect_tail_counts:
            if protect_tail <= 0 or protect_tail >= budget or protect_tail > args.base_topk:
                continue
            prefix_count = budget - protect_tail
            selected = base_rank[:prefix_count] + base_rank[args.base_topk - protect_tail : args.base_topk]
            add_candidate(
                f"{args.base_candidate_id}_protect_tail_b{budget}_p{protect_tail}",
                "repaired_mask_protected_tail_shrink",
                selected,
                {
                    "budget": budget,
                    "base_candidate_id": args.base_candidate_id,
                    "base_topk": args.base_topk,
                    "removed_channels": args.base_topk - budget,
                    "protected_tail": protect_tail,
                    "prefix_kept": prefix_count,
                    "middle_drop_start": prefix_count,
                    "middle_drop_end": args.base_topk - protect_tail,
                },
            )

    manifest = {
        "issue": 12,
        "artifact": "bfcl_issue12_shrink_repair_candidates",
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_channels": total_channels,
        "inputs": {
            "base_mask": str(args.base_mask),
            "base_mask_sha256": sha256_file(args.base_mask),
            "base_candidate_id": args.base_candidate_id,
            "base_candidate_jsonl": str(args.base_candidate_jsonl) if args.base_candidate_jsonl else None,
            "base_lineage": base_lineage,
        },
        "build_params": {
            "base_topk": args.base_topk,
            "budgets": budgets,
            "protect_tail_counts": protect_tail_counts,
        },
        "candidate_count": len(candidates),
    }
    (args.out_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.out_dir / "candidate_masks.jsonl").open("w") as f:
        for row in candidates:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
