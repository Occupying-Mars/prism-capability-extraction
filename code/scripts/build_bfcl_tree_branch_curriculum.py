#!/usr/bin/env python3
"""Build one leak-audited BFCL tree-search branch dataset for issue #6.

The branch builder reuses the issue #5 near-miss generators, but adds branch
profiles. Profiles change the repair-bucket weighting and augmentation ratio so
the tree can explore conservative, balanced, compression-biased, and hard-case
directions from the same parent failure map.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "issue6_tree_branch_curriculum_v1"
DEFAULT_RATIO = 0.20
DEFAULT_MAX_RATIO = 0.20

PROFILE_RATIO = {
    "conservative_nearmiss": 0.15,
    "bucket_balanced": 0.20,
    "teacher_ranked": 0.20,
    "schema_stratified": 0.18,
    "compression_biased": 0.15,
    "hardcase_replay": 0.20,
    "epsilon_repair": 0.15,
    "pareto_trim": 0.10,
}

PROFILE_WEIGHTS = {
    "conservative_nearmiss": {
        "arg_value_exactness": 1.35,
        "schema_completion": 1.25,
        "function_name_disambiguation": 0.75,
        "misc_failure": 0.60,
    },
    "bucket_balanced": {},
    "teacher_ranked": {
        "arg_value_exactness": 1.50,
        "schema_completion": 1.35,
        "time_normalization": 1.20,
        "live_slot_values": 1.20,
        "misc_failure": 0.70,
    },
    "schema_stratified": {
        "schema_completion": 1.70,
        "sql_schema_discipline": 1.55,
        "function_name_disambiguation": 1.25,
        "arg_value_exactness": 0.85,
    },
    "compression_biased": {
        "arg_value_exactness": 1.35,
        "function_name_disambiguation": 1.25,
        "json_wrapper_stability": 1.20,
        "schema_completion": 1.10,
        "misc_failure": 0.65,
    },
    "hardcase_replay": {
        "misc_failure": 1.65,
        "function_name_disambiguation": 1.45,
        "sql_schema_discipline": 1.35,
        "live_slot_values": 1.25,
    },
    "epsilon_repair": {
        "schema_completion": 1.45,
        "arg_value_exactness": 1.35,
        "unit_default_normalization": 1.25,
        "time_normalization": 1.20,
    },
    "pareto_trim": {
        "arg_value_exactness": 1.40,
        "json_wrapper_stability": 1.30,
        "function_name_disambiguation": 1.20,
        "schema_completion": 0.90,
        "misc_failure": 0.55,
    },
}


def load_nearmiss_module() -> Any:
    path = Path(__file__).with_name("build_bfcl_nearmiss_curriculum.py")
    spec = importlib.util.spec_from_file_location("issue5_nearmiss_curriculum", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


near = load_nearmiss_module()
edge = near.edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--failure-bucket-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--edge-output", type=Path, required=True)
    parser.add_argument("--mixed-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--branch-profile", choices=sorted(PROFILE_RATIO), required=True)
    parser.add_argument("--augmentation-ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--max-augmentation-ratio", type=float, default=DEFAULT_MAX_RATIO)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--strict-count", type=int, default=0)
    parser.add_argument("--edge-count", type=int, default=0)
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--fail-on-leak", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def adjusted_counts(counts: Counter[str], profile: str) -> Counter[str]:
    if profile == "bucket_balanced":
        return Counter({bucket: 1 for bucket, n in counts.items() if n > 0}) or Counter({"schema_completion": 1})
    weights = PROFILE_WEIGHTS.get(profile, {})
    adjusted: Counter[str] = Counter()
    for bucket, value in counts.items():
        if value <= 0:
            continue
        adjusted[bucket] = max(1, round(value * weights.get(bucket, 1.0)))
    return adjusted or Counter({"schema_completion": 1, "arg_value_exactness": 1})


def rewrite_branch_metadata(row: dict[str, Any], idx: int, args: argparse.Namespace) -> None:
    row["id"] = f"issue6_{args.branch_id}_{args.branch_profile}_{idx:06d}"
    row["mix_id"] = row["id"]
    row["source"] = "issue6_tree_search_synthetic_nearmiss"
    row["origin"] = GENERATOR_VERSION
    row["issue"] = 6
    row["branch_id"] = args.branch_id
    row["parent_id"] = args.parent_id
    row["branch_profile"] = args.branch_profile
    row["gold_policy"] = "synthetic_by_construction_verified_schema_target_failure_bucket_weighted"
    row["teacher_policy"] = (
        "full_model_teacher_allowed_for_proposals_ranking_or_soft_targets_only_"
        "no_unverified_hard_labels"
    )


def main() -> None:
    args = parse_args()
    profile_ratio = PROFILE_RATIO[args.branch_profile]
    ratio = min(args.augmentation_ratio or profile_ratio, profile_ratio, args.max_augmentation_ratio)
    if not (0 < ratio <= args.max_augmentation_ratio <= 1.0):
        raise ValueError("augmentation ratio must be positive and <= max augmentation ratio <= 1")

    base_rows = read_jsonl(args.base_train_jsonl)
    eval_rows = read_jsonl(args.eval_jsonl)
    repair_counts_raw = near.load_repair_counts(args.failure_bucket_dirs)
    repair_counts = adjusted_counts(repair_counts_raw, args.branch_profile)

    strict_count = args.strict_count or len(base_rows)
    strict_rows = list(base_rows[:strict_count])
    edge_count = args.edge_count or max(1, round(len(strict_rows) * ratio))
    max_edge_count = max(1, round(len(strict_rows) * args.max_augmentation_ratio))
    if edge_count > max_edge_count:
        raise ValueError(f"edge_count {edge_count} exceeds max allowed {max_edge_count}")

    edge_rows, generation_stats = near.build_nearmiss_rows(
        count=edge_count,
        repair_counts=repair_counts,
        eval_rows=eval_rows,
        round_id=args.branch_id,
        seed=args.seed,
    )
    for idx, row in enumerate(edge_rows):
        rewrite_branch_metadata(row, idx, args)
        edge.validate_row(row)

    audit = edge.audit_edge_rows(
        edge_rows,
        eval_rows,
        near_threshold=args.near_threshold,
        shingle_size=args.shingle_size,
    )
    if args.fail_on_leak and not audit["passed"]:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"leak_audit": audit}, indent=2, sort_keys=True) + "\n")
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit("issue6 branch leak audit failed")

    mixed_rows = strict_rows + edge_rows
    random.Random(args.seed + 1).shuffle(mixed_rows)
    write_jsonl(args.edge_output, edge_rows)
    write_jsonl(args.mixed_output, mixed_rows)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "branch_id": args.branch_id,
        "parent_id": args.parent_id,
        "branch_profile": args.branch_profile,
        "seed": args.seed,
        "base_train_jsonl": str(args.base_train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "failure_bucket_dirs": [str(path) for path in args.failure_bucket_dirs],
        "edge_output": str(args.edge_output),
        "mixed_output": str(args.mixed_output),
        "rows": {
            "base_available": len(base_rows),
            "strict": len(strict_rows),
            "edge": len(edge_rows),
            "mixed": len(mixed_rows),
        },
        "augmentation_ratio_requested": args.augmentation_ratio,
        "augmentation_ratio_profile_default": profile_ratio,
        "augmentation_ratio_actual": len(edge_rows) / len(strict_rows),
        "max_augmentation_ratio": args.max_augmentation_ratio,
        "source_repair_bucket_counts": repair_counts_raw,
        "adjusted_repair_bucket_counts": repair_counts,
        "generation_stats": generation_stats,
        "gold_policy": "hard labels are generated by construction and schema-verified",
        "teacher_policy": "teacher may rank or propose but is not accepted as hard gold without verification",
        "leak_audit": audit,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
