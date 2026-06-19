#!/usr/bin/env python3
"""Build issue #12 hybrid repair masks from trained-tree parents plus donors.

The purpose is to test whether co-activation or failure-conditioned donors can
repair a strong trained parent at the same MLP-channel budget. A candidate keeps
the top parent channels, replaces a small suffix with donor-ranked channels,
and is then evaluated by the usual BFCL masked harness.
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


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("empty NAME")
    return name, Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def layer_channel(gid: int, d_ffn: int) -> tuple[int, int]:
    return int(gid // d_ffn), int(gid % d_ffn)


def write_mask(path: Path, selected: list[int], *, n_layers: int, d_ffn: int) -> None:
    scores = np.zeros((n_layers, d_ffn), dtype=np.float32)
    for rank, gid in enumerate(selected):
        layer, channel = layer_channel(gid, d_ffn)
        scores[layer, channel] = float(len(selected) - rank)
    np.savez_compressed(path, mlp_scores=scores)


def select_repair(parent_rank: list[int], donor_rank: list[int], *, budget: int, replace: int) -> list[int]:
    replace = min(max(replace, 0), budget)
    selected: list[int] = []
    seen: set[int] = set()
    for gid in parent_rank[: budget - replace]:
        if gid in seen:
            continue
        selected.append(gid)
        seen.add(gid)
    for gid in donor_rank:
        if gid in seen:
            continue
        selected.append(gid)
        seen.add(gid)
        if len(selected) >= budget:
            return selected
    for gid in parent_rank:
        if gid in seen:
            continue
        selected.append(gid)
        seen.add(gid)
        if len(selected) >= budget:
            return selected
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", action="append", type=parse_named_path, required=True)
    p.add_argument("--donor", action="append", type=parse_named_path, default=[])
    p.add_argument("--donor-candidate-jsonl", type=Path)
    p.add_argument("--donor-candidate-root", type=Path)
    p.add_argument("--donor-candidate-id", action="append", default=[])
    p.add_argument("--budgets", default="120000,140000,160000,180000,200000")
    p.add_argument("--replace-counts", default="5000,10000,20000,30000,40000")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "candidate_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]
    replace_counts = [int(item) for item in args.replace_counts.split(",") if item.strip()]

    parents: dict[str, dict[str, Any]] = {}
    for name, path in args.parent:
        scores = load_scores(path)
        parents[name] = {"path": path, "scores": scores, "ranking": ranking(scores)}

    donors: dict[str, dict[str, Any]] = {}
    for name, path in args.donor:
        scores = load_scores(path)
        donors[name] = {"path": path, "scores": scores, "ranking": ranking(scores), "source": "named_donor"}

    if args.donor_candidate_jsonl:
        if not args.donor_candidate_root:
            raise ValueError("--donor-candidate-root is required with --donor-candidate-jsonl")
        wanted = set(args.donor_candidate_id)
        for row in read_jsonl(args.donor_candidate_jsonl):
            cid = row["candidate_id"]
            if wanted and cid not in wanted:
                continue
            path = args.donor_candidate_root / row["mask_path"]
            scores = load_scores(path)
            donors[cid] = {"path": path, "scores": scores, "ranking": ranking(scores), "source": "issue12_dense_candidate"}

    if not donors:
        raise ValueError("no donors loaded")

    first_parent = next(iter(parents.values()))
    n_layers, d_ffn = first_parent["scores"].shape
    total_channels = n_layers * d_ffn
    candidates: list[dict[str, Any]] = []

    def add_candidate(candidate_id: str, kind: str, selected: list[int], lineage: dict[str, Any]) -> None:
        selected = list(dict.fromkeys(selected))
        if not selected:
            return
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
                "lineage": lineage,
            }
        )

    for parent_name, parent in parents.items():
        parent_rank = parent["ranking"]
        for budget in budgets:
            add_candidate(
                f"{parent_name}_parent_k{budget}",
                "trained_parent_baseline",
                parent_rank[:budget],
                {"parent": parent_name, "parent_path": str(parent["path"]), "budget": budget},
            )
            for donor_name, donor in donors.items():
                for replace in replace_counts:
                    if replace >= budget:
                        continue
                    selected = select_repair(parent_rank, donor["ranking"], budget=budget, replace=replace)
                    add_candidate(
                        f"{parent_name}_repair_{donor_name}_r{replace}_k{budget}",
                        "parent_suffix_replaced_by_donor",
                        selected,
                        {
                            "parent": parent_name,
                            "parent_path": str(parent["path"]),
                            "donor": donor_name,
                            "donor_path": str(donor["path"]),
                            "donor_source": donor["source"],
                            "budget": budget,
                            "replace": replace,
                            "parent_prefix_kept": budget - replace,
                        },
                    )

    manifest = {
        "issue": 12,
        "artifact": "bfcl_issue12_hybrid_repair_candidates",
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_channels": total_channels,
        "budgets": budgets,
        "replace_counts": replace_counts,
        "parents": {
            name: {"path": str(item["path"]), "sha256": sha256_file(item["path"])}
            for name, item in parents.items()
        },
        "donors": {
            name: {"path": str(item["path"]), "source": item["source"], "sha256": sha256_file(item["path"])}
            for name, item in donors.items()
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
