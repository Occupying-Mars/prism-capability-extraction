#!/usr/bin/env python3
"""Build BFCL issue #12 co-activation MACE candidate masks.

This script consumes the issue #9 activation atlas inspection artifacts and
emits selected-MLP-channel candidate masks in the existing `mlp_scores` NPZ
format used by `bfcl_direct_qwen3.py eval-mask`.

The graph is intentionally sparse: top channels per query/segment/stat become
query-channel evidence, then channels that co-occur inside the same query form
a weighted graph. Thresholded connected components are used as deterministic
co-activation communities.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SEGMENT_WEIGHT = {
    "prompt": 0.85,
    "target": 1.20,
    "full": 1.00,
}

STAT_WEIGHT = {
    "mean_abs": 1.00,
    "rms": 1.05,
    "max_abs": 0.75,
}

SPLIT_SELECT = {"train", "calibration", "validation"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def channel_id(layer: int, channel: int, d_ffn: int) -> int:
    return int(layer) * d_ffn + int(channel)


def layer_channel(gid: int, d_ffn: int) -> tuple[int, int]:
    return int(gid // d_ffn), int(gid % d_ffn)


def channel_weight(row: dict[str, Any], item: dict[str, Any]) -> float:
    seg_w = SEGMENT_WEIGHT.get(str(row.get("segment")), 1.0)
    stat_w = STAT_WEIGHT.get(str(row.get("stat")), 1.0)
    rank = max(int(item.get("rank", 1)), 1)
    local = float(item.get("local_score", 1))
    global_score = float(item.get("global_score", 1))
    value = max(float(item.get("value", 0.0)), 0.0)
    rank_w = 1.0 / math.sqrt(rank)
    return seg_w * stat_w * rank_w * (0.6 * local + 0.4 * global_score) * math.log1p(value)


def failure_weight(meta: dict[str, Any]) -> float:
    primary = meta.get("primary_failure_types") or []
    repair = meta.get("repair_buckets") or []
    split = meta.get("split_role")
    weight = 1.0
    weight += 0.15 * min(len(primary), 5)
    weight += 0.10 * min(len(repair), 5)
    if "wrong_function" in primary or "function_name_disambiguation" in repair:
        weight += 0.25
    if "wrong_arg_value" in primary or "arg_value_exactness" in repair:
        weight += 0.20
    if "schema_completion" in repair:
        weight += 0.15
    if split == "heldout":
        # Heldout evidence is retained for audit tables but not selection.
        return 0.0
    return weight


class UnionFind:
    def __init__(self, items: set[int]):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x: int) -> int:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def top_items(scores: dict[int, float], k: int) -> list[int]:
    if k <= 0:
        return []
    return [gid for gid, _score in heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))]


def top_items_array(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []
    k = min(int(k), int(scores.size))
    if k == scores.size:
        idx = np.arange(scores.size)
    else:
        idx = np.argpartition(scores, -k)[-k:]
    ordered = idx[np.lexsort((idx, -scores[idx]))]
    return [int(gid) for gid in ordered[:k]]


def fill_to_budget(core: list[int], ranking: list[int], budget: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for gid in core:
        if gid in seen:
            continue
        selected.append(gid)
        seen.add(gid)
        if len(selected) >= budget:
            return selected
    for gid in ranking:
        if gid in seen:
            continue
        selected.append(gid)
        seen.add(gid)
        if len(selected) >= budget:
            return selected
    return selected


def build_dense_global_scores(
    global_scores_path: Path,
    query_rows: list[dict[str, Any]],
    atlas_manifest: dict[str, Any],
    *,
    n_layers: int,
    d_ffn: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    scores = np.load(global_scores_path, mmap_mode="r")
    expected_shape = tuple(atlas_manifest["score_shape"])
    if tuple(scores.shape) != expected_shape:
        raise ValueError(f"dense global score shape mismatch: {scores.shape} != {expected_shape}")

    total_channels = n_layers * d_ffn
    dense_hotness = np.zeros(total_channels, dtype=np.float32)
    dense_failure_pressure = np.zeros(total_channels, dtype=np.float32)
    dense_category_scores: dict[str, np.ndarray] = {}
    segments = list(atlas_manifest["segments"])
    stats = list(atlas_manifest["stats"])

    for row_idx, meta in enumerate(query_rows):
        split = meta.get("split_role")
        if split not in SPLIT_SELECT:
            continue
        query_index = int(meta.get("global_index", row_idx))
        category = str(meta.get("category", "unknown"))
        category_scores = dense_category_scores.setdefault(category, np.zeros(total_channels, dtype=np.float32))
        fw = failure_weight(meta)
        for seg_idx, segment in enumerate(segments):
            seg_w = SEGMENT_WEIGHT.get(str(segment), 1.0)
            for stat_idx, stat in enumerate(stats):
                plane_w = seg_w * STAT_WEIGHT.get(str(stat), 1.0)
                plane = scores[query_index, seg_idx, stat_idx].reshape(total_channels).astype(np.float32, copy=False)
                dense_hotness += plane_w * plane
                dense_failure_pressure += plane_w * fw * plane
                category_scores += plane_w * fw * plane

    return dense_hotness, dense_failure_pressure, dense_category_scores


def select_by_communities(
    communities: list[dict[str, Any]],
    channel_scores: dict[int, float],
    budget: int,
    *,
    prefer_failure: bool,
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    key = "failure_score" if prefer_failure else "hotness_score"
    for community in sorted(communities, key=lambda row: (row[key], row["size"]), reverse=True):
        members = sorted(
            community["members"],
            key=lambda gid: (channel_scores.get(gid, 0.0), -gid),
            reverse=True,
        )
        for gid in members:
            if gid in seen:
                continue
            selected.append(gid)
            seen.add(gid)
            if len(selected) >= budget:
                return selected
    return selected


def write_mask(path: Path, selected: list[int], *, n_layers: int, d_ffn: int) -> None:
    scores = np.zeros((n_layers, d_ffn), dtype=np.float32)
    # Assign strictly positive descending scores, so eval-mask --topk exactly
    # recovers the selected set even when many non-selected entries are zero.
    for rank, gid in enumerate(selected):
        layer, channel = layer_channel(gid, d_ffn)
        scores[layer, channel] = float(len(selected) - rank)
    np.savez_compressed(path, mlp_scores=scores)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-channels", type=Path, required=True)
    p.add_argument("--query-manifest", type=Path, required=True)
    p.add_argument("--atlas-manifest", type=Path, required=True)
    p.add_argument("--global-scores", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--budgets", default="80000,120000,160000,200000,240000")
    p.add_argument("--top-per-plane", type=int, default=32)
    p.add_argument("--coactivation-top-per-query", type=int, default=96)
    p.add_argument("--edge-min-count", type=int, default=3)
    p.add_argument("--edge-max", type=int, default=500000)
    p.add_argument("--min-community-size", type=int, default=2)
    p.add_argument("--ego-community-seeds", type=int, default=256)
    p.add_argument("--ego-community-max-size", type=int, default=2048)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "candidate_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    atlas_manifest = json.loads(args.atlas_manifest.read_text())
    n_layers = int(atlas_manifest["layers"])
    d_ffn = int(atlas_manifest["d_ffn"])
    total_channels = n_layers * d_ffn
    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]

    query_rows = read_jsonl(args.query_manifest)
    meta_by_eval_id = {row["eval_id"]: row for row in query_rows}
    dense_hotness: np.ndarray | None = None
    dense_failure_pressure: np.ndarray | None = None
    dense_category_scores: dict[str, np.ndarray] = {}
    if args.global_scores is not None:
        dense_hotness, dense_failure_pressure, dense_category_scores = build_dense_global_scores(
            args.global_scores,
            query_rows,
            atlas_manifest,
            n_layers=n_layers,
            d_ffn=d_ffn,
        )
    query_feature_scores: dict[str, Counter[int]] = defaultdict(Counter)
    query_segment_scores: dict[str, dict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    hotness: Counter[int] = Counter()
    failure_pressure: Counter[int] = Counter()
    category_scores: dict[str, Counter[int]] = defaultdict(Counter)
    split_scores: dict[str, Counter[int]] = defaultdict(Counter)

    top_rows = 0
    top_entries = 0
    selection_entries = 0
    with args.top_channels.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            top_rows += 1
            eval_id = row["eval_id"]
            meta = meta_by_eval_id.get(eval_id)
            if meta is None:
                continue
            split = meta.get("split_role")
            category = meta.get("category", "unknown")
            fw = failure_weight(meta)
            for item in row.get("top", [])[: args.top_per_plane]:
                gid = channel_id(item["layer"], item["channel"], d_ffn)
                weight = channel_weight(row, item)
                top_entries += 1
                query_feature_scores[eval_id][gid] += weight
                query_segment_scores[eval_id][str(row.get("segment"))][gid] += weight
                if split in SPLIT_SELECT:
                    selection_entries += 1
                    hotness[gid] += weight
                    failure_pressure[gid] += weight * fw
                    category_scores[category][gid] += weight * fw
                    split_scores[split][gid] += weight * fw

    # Sparse co-activation graph over top channels per query.
    edge_counts: Counter[tuple[int, int]] = Counter()
    for eval_id, scores in query_feature_scores.items():
        meta = meta_by_eval_id.get(eval_id, {})
        if meta.get("split_role") not in SPLIT_SELECT:
            continue
        nodes = top_items(dict(scores), args.coactivation_top_per_query)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                if a == b:
                    continue
                edge = (a, b) if a < b else (b, a)
                edge_counts[edge] += 1

    retained_edges = [
        (a, b, count)
        for (a, b), count in edge_counts.most_common(args.edge_max)
        if count >= args.edge_min_count
    ]
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b, count in retained_edges:
        adjacency[a].append((b, count))
        adjacency[b].append((a, count))
    nodes = set(hotness) | set(failure_pressure)
    uf = UnionFind(nodes)
    for a, b, _count in retained_edges:
        if a in nodes and b in nodes:
            uf.union(a, b)

    grouped: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        grouped[uf.find(node)].append(node)

    communities: list[dict[str, Any]] = []
    for idx, members in enumerate(grouped.values()):
        if len(members) < args.min_community_size:
            continue
        members = sorted(members)
        communities.append(
            {
                "community_id": f"c{idx:05d}",
                "size": len(members),
                "hotness_score": float(sum(hotness.get(gid, 0.0) for gid in members)),
                "failure_score": float(sum(failure_pressure.get(gid, 0.0) for gid in members)),
                "members": members,
                "sample_layer_channels": [
                    {"layer": layer_channel(gid, d_ffn)[0], "channel": layer_channel(gid, d_ffn)[1]}
                    for gid in members[:20]
                ],
            }
        )
    communities.sort(key=lambda row: (row["failure_score"], row["hotness_score"], row["size"]), reverse=True)

    seen_ego_sets: set[tuple[int, ...]] = set()
    for seed_rank, seed in enumerate(top_items(dict(failure_pressure), args.ego_community_seeds)):
        neighbors = sorted(
            adjacency.get(seed, []),
            key=lambda pair: (pair[1], failure_pressure.get(pair[0], 0.0), hotness.get(pair[0], 0.0)),
            reverse=True,
        )
        members = [seed] + [gid for gid, _count in neighbors[: max(args.ego_community_max_size - 1, 0)]]
        members = sorted(set(members))
        if len(members) < args.min_community_size:
            continue
        key = tuple(members)
        if key in seen_ego_sets:
            continue
        seen_ego_sets.add(key)
        communities.append(
            {
                "community_id": f"e{seed_rank:05d}",
                "size": len(members),
                "hotness_score": float(sum(hotness.get(gid, 0.0) for gid in members)),
                "failure_score": float(sum(failure_pressure.get(gid, 0.0) for gid in members)),
                "members": members,
                "sample_layer_channels": [
                    {"layer": layer_channel(gid, d_ffn)[0], "channel": layer_channel(gid, d_ffn)[1]}
                    for gid in members[:20]
                ],
                "seed": {
                    "global_channel": seed,
                    "layer": layer_channel(seed, d_ffn)[0],
                    "channel": layer_channel(seed, d_ffn)[1],
                },
            }
        )
    communities.sort(key=lambda row: (row["failure_score"], row["hotness_score"], row["size"]), reverse=True)

    candidates: list[dict[str, Any]] = []
    hotness_ranking = top_items_array(dense_hotness, total_channels) if dense_hotness is not None else top_items(dict(hotness), total_channels)
    failure_ranking = (
        top_items_array(dense_failure_pressure, total_channels)
        if dense_failure_pressure is not None
        else top_items(dict(failure_pressure), total_channels)
    )

    def add_candidate(candidate_id: str, kind: str, selected: list[int], lineage: dict[str, Any]) -> None:
        selected = list(dict.fromkeys(selected))
        if not selected:
            return
        mask_path = mask_dir / f"{candidate_id}.npz"
        write_mask(mask_path, selected, n_layers=n_layers, d_ffn=d_ffn)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "kind": kind,
                "mask_path": str(mask_path.relative_to(args.out_dir)),
                "selected_mlp_channels": len(selected),
                "mlp_fraction": len(selected) / total_channels,
                "topk_for_eval": len(selected),
                "lineage": lineage,
            }
        )

    for budget in budgets:
        add_candidate(
            f"top_hot_k{budget}",
            "top_hot_activation",
            hotness_ranking[:budget],
            {"source": "aggregate_hotness", "budget": budget, "dense_budget_fill": dense_hotness is not None},
        )
        add_candidate(
            f"failure_pressure_k{budget}",
            "failure_pressure",
            failure_ranking[:budget],
            {"source": "aggregate_failure_pressure", "budget": budget, "dense_budget_fill": dense_failure_pressure is not None},
        )
        add_candidate(
            f"coactivation_union_k{budget}",
            "coactivation_community_union",
            fill_to_budget(
                select_by_communities(communities, dict(hotness), budget, prefer_failure=False),
                hotness_ranking,
                budget,
            ),
            {"source": "community_union_hotness", "budget": budget, "dense_budget_fill": dense_hotness is not None},
        )
        add_candidate(
            f"failure_community_union_k{budget}",
            "failure_pressure_community_union",
            fill_to_budget(
                select_by_communities(communities, dict(failure_pressure), budget, prefer_failure=True),
                failure_ranking,
                budget,
            ),
            {"source": "community_union_failure_pressure", "budget": budget, "dense_budget_fill": dense_failure_pressure is not None},
        )

    category_source = dense_category_scores if dense_category_scores else {key: None for key in category_scores}
    for category, dense_scores in sorted(category_source.items()):
        category_ranking = (
            top_items_array(dense_scores, total_channels)
            if dense_scores is not None
            else top_items(dict(category_scores[category]), total_channels)
        )
        for budget in budgets:
            add_candidate(
                f"category_{category}_failure_k{budget}",
                "category_failure_pressure",
                category_ranking[:budget],
                {
                    "source": "category_failure_pressure",
                    "category": category,
                    "budget": budget,
                    "dense_budget_fill": dense_scores is not None,
                },
            )

    # Child and leave-one-out proposals around the strongest community union.
    top_communities = communities[:12]
    for budget in budgets:
        for depth, limit in (("top4", 4), ("top8", 8), ("top12", 12)):
            pool: list[int] = []
            for community in top_communities[:limit]:
                pool.extend(community["members"])
            pool = sorted(set(pool), key=lambda gid: failure_pressure.get(gid, 0.0), reverse=True)[:budget]
            add_candidate(
                f"child_combo_{depth}_k{budget}",
                "child_community_combination",
                fill_to_budget(pool, failure_ranking, budget),
                {
                    "source": "top_failure_communities",
                    "communities": limit,
                    "budget": budget,
                    "dense_budget_fill": dense_failure_pressure is not None,
                },
            )
        for leave_idx, community in enumerate(top_communities[:4]):
            pool = []
            leave_id = community["community_id"]
            leave = set(community["members"])
            for other in top_communities[:12]:
                if other["community_id"] == leave_id:
                    continue
                pool.extend(gid for gid in other["members"] if gid not in leave)
            pool = sorted(set(pool), key=lambda gid: failure_pressure.get(gid, 0.0), reverse=True)[:budget]
            add_candidate(
                f"loo_c{leave_idx:02d}_k{budget}",
                "leave_one_community_out",
                fill_to_budget(pool, failure_ranking, budget),
                {
                    "source": "top12_failure_communities_minus_one",
                    "left_out_community": community["community_id"],
                    "budget": budget,
                    "dense_budget_fill": dense_failure_pressure is not None,
                },
            )

    manifest = {
        "issue": 12,
        "artifact": "bfcl_issue12_recursive_coactivation_mace_candidates",
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_channels": total_channels,
        "selection_splits": sorted(SPLIT_SELECT),
        "heldout_policy": "heldout metadata retained for audit only; heldout rows not used for hotness/failure-pressure selection",
        "inputs": {
            "top_channels": str(args.top_channels),
            "top_channels_sha256": sha256_file(args.top_channels),
            "query_manifest": str(args.query_manifest),
            "query_manifest_sha256": sha256_file(args.query_manifest),
            "atlas_manifest": str(args.atlas_manifest),
            "atlas_manifest_sha256": sha256_file(args.atlas_manifest),
            "global_scores": str(args.global_scores) if args.global_scores is not None else None,
            "global_scores_sha256": sha256_file(args.global_scores) if args.global_scores is not None else None,
        },
        "build_params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {"budgets": budgets},
        "top_rows": top_rows,
        "top_entries": top_entries,
        "selection_entries": selection_entries,
        "query_count": len(query_rows),
        "edge_count_raw": len(edge_counts),
        "edge_count_retained": len(retained_edges),
        "community_count": len(communities),
        "candidate_count": len(candidates),
    }

    (args.out_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.out_dir / "candidate_masks.jsonl").open("w") as f:
        for row in candidates:
            f.write(json.dumps(row) + "\n")
    with (args.out_dir / "coactivation_communities.jsonl").open("w") as f:
        for row in communities:
            serial = dict(row)
            serial["members"] = [
                {"global_channel": gid, "layer": layer_channel(gid, d_ffn)[0], "channel": layer_channel(gid, d_ffn)[1]}
                for gid in row["members"]
            ]
            f.write(json.dumps(serial) + "\n")

    query_scores_path = args.out_dir / "query_community_scores.jsonl"
    community_sets = [(row["community_id"], set(row["members"])) for row in communities[:200]]
    with query_scores_path.open("w") as f:
        for eval_id, scores in query_feature_scores.items():
            meta = meta_by_eval_id.get(eval_id, {})
            hot = set(top_items(dict(scores), args.coactivation_top_per_query))
            rows = []
            for cid, members in community_sets:
                overlap = hot & members
                if overlap:
                    rows.append({"community_id": cid, "overlap": len(overlap)})
            f.write(
                json.dumps(
                    {
                        "eval_id": eval_id,
                        "split_role": meta.get("split_role"),
                        "category": meta.get("category"),
                        "communities": rows[:32],
                    }
                )
                + "\n"
            )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
