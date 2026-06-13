#!/usr/bin/env python3
"""Build verified synthetic BFCL near-miss rows from failure-bucket counts.

This generator is for issue #5. It does not copy BFCL eval prompts, tools, or
targets. It uses the abstract repair-bucket distribution from a previous round's
failure buckets to choose which verified synthetic row families to generate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


GENERATOR_VERSION = "issue5_nearmiss_curriculum_v1"
DEFAULT_RATIO = 0.20
DEFAULT_MAX_RATIO = 0.20
REPAIR_TO_EDGE_BUCKET = {
    "arg_value_exactness": "type_coercion",
    "formula_normalization": "formula_normalization",
    "function_name_disambiguation": "function_name_disambiguation",
    "live_slot_values": "live_slot_values",
    "misc_failure": "json_wrapper_stability",
    "schema_completion": "schema_completion",
    "sql_schema_discipline": "sql_schema_discipline",
    "time_normalization": "time_normalization",
    "unit_default_normalization": "optional_default_arguments",
}


def load_edge_module() -> Any:
    path = Path(__file__).with_name("build_bfcl_edge_curriculum.py")
    spec = importlib.util.spec_from_file_location("issue4_edge_curriculum", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


edge = load_edge_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--failure-bucket-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--edge-output", type=Path, required=True)
    parser.add_argument("--mixed-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--round-id", default="r1")
    parser.add_argument("--augmentation-ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--max-augmentation-ratio", type=float, default=DEFAULT_MAX_RATIO)
    parser.add_argument("--seed", type=int, default=55)
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


def load_repair_counts(paths: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        manifest = path / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"missing failure bucket manifest: {manifest}")
        data = json.loads(manifest.read_text())
        counts.update(data.get("repair_bucket_counts") or {})
    return counts


def function_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return edge.function_tool(name, description, properties, required)


def make_live_slot_values(idx: int, rng: random.Random) -> dict[str, Any]:
    city = rng.choice(["edge5_ion_city", "edge5_kappa_harbor", "edge5_mira_basin", "edge5_terra_gate"])
    station = f"edge5_station_{rng.randrange(1000, 9999)}"
    unit = rng.choice(["edge5_metric", "edge5_imperial"])
    day = rng.choice(["edge5_today", "edge5_tomorrow", "edge5_friday", "edge5_monday"])
    name = "edge5_report_station_weather"
    tools = [
        function_tool(
            name,
            "Report weather-like status for a synthetic station slot request.",
            {
                "edge5_station_id": edge.string_prop("Synthetic station id."),
                "edge5_city": edge.string_prop("Synthetic city.", enum=[
                    "edge5_ion_city",
                    "edge5_kappa_harbor",
                    "edge5_mira_basin",
                    "edge5_terra_gate",
                ]),
                "edge5_day": edge.string_prop("Synthetic day token.", enum=[
                    "edge5_today",
                    "edge5_tomorrow",
                    "edge5_friday",
                    "edge5_monday",
                ]),
                "edge5_unit": edge.string_prop("Synthetic unit system.", enum=["edge5_metric", "edge5_imperial"]),
            },
            ["edge5_station_id", "edge5_city", "edge5_day", "edge5_unit"],
        )
    ]
    prompt = f"For {station} in {city}, report the status for {day} using {unit} units."
    call = {
        "name": name,
        "arguments": {
            "edge5_station_id": station,
            "edge5_city": city,
            "edge5_day": day,
            "edge5_unit": unit,
        },
    }
    return edge.base_row(idx=idx, bucket="live_slot_values", prompt=prompt, tools=tools, call=call)


def make_time_normalization(idx: int, rng: random.Random) -> dict[str, Any]:
    job = f"edge5_alarm_{rng.randrange(1000, 9999)}"
    hour = rng.choice([6, 8, 13, 17, 21, 23])
    minute = rng.choice([0, 5, 15, 30, 45])
    tz = rng.choice(["edge5_utc", "edge5_pst", "edge5_ist", "edge5_cet"])
    repeat = rng.choice(["edge5_once", "edge5_daily", "edge5_weekly"])
    time_24 = f"{hour:02d}:{minute:02d}"
    name = "edge5_schedule_alarm"
    tools = [
        function_tool(
            name,
            "Schedule a synthetic alarm using normalized 24-hour time.",
            {
                "edge5_alarm_id": edge.string_prop("Alarm id."),
                "edge5_time_24h": edge.string_prop("Time in HH:MM 24-hour format."),
                "edge5_timezone": edge.string_prop("Synthetic timezone.", enum=["edge5_utc", "edge5_pst", "edge5_ist", "edge5_cet"]),
                "edge5_repeat": edge.string_prop("Repeat policy.", enum=["edge5_once", "edge5_daily", "edge5_weekly"]),
            },
            ["edge5_alarm_id", "edge5_time_24h", "edge5_timezone", "edge5_repeat"],
        )
    ]
    prompt = f"Schedule {job} for {time_24} in {tz}; repeat policy is {repeat}."
    call = {
        "name": name,
        "arguments": {
            "edge5_alarm_id": job,
            "edge5_time_24h": time_24,
            "edge5_timezone": tz,
            "edge5_repeat": repeat,
        },
    }
    return edge.base_row(idx=idx, bucket="time_normalization", prompt=prompt, tools=tools, call=call)


GENERATORS: dict[str, Callable[[int, random.Random], dict[str, Any]]] = dict(edge.GENERATORS)
GENERATORS["live_slot_values"] = make_live_slot_values
GENERATORS["time_normalization"] = make_time_normalization


def weighted_bucket_plan(count: int, repair_counts: Counter[str], rng: random.Random) -> list[tuple[str, str]]:
    mapped: list[tuple[str, str, int]] = []
    for repair_bucket, n in repair_counts.items():
        edge_bucket = REPAIR_TO_EDGE_BUCKET.get(repair_bucket)
        if edge_bucket and n > 0:
            mapped.append((repair_bucket, edge_bucket, int(n)))
    if not mapped:
        mapped = [
            ("schema_completion", "schema_completion", 1),
            ("arg_value_exactness", "type_coercion", 1),
            ("function_name_disambiguation", "function_name_disambiguation", 1),
            ("live_slot_values", "live_slot_values", 1),
        ]

    total = sum(n for _, _, n in mapped)
    raw_counts = [(repair, edge_bucket, max(1, round(count * n / total))) for repair, edge_bucket, n in mapped]
    while sum(n for _, _, n in raw_counts) > count:
        i = max(range(len(raw_counts)), key=lambda j: raw_counts[j][2])
        repair, edge_bucket, n = raw_counts[i]
        raw_counts[i] = (repair, edge_bucket, n - 1)
    while sum(n for _, _, n in raw_counts) < count:
        i = max(range(len(raw_counts)), key=lambda j: mapped[j][2])
        repair, edge_bucket, n = raw_counts[i]
        raw_counts[i] = (repair, edge_bucket, n + 1)

    plan = [(repair, edge_bucket) for repair, edge_bucket, n in raw_counts for _ in range(n)]
    rng.shuffle(plan)
    return plan


def rewrite_metadata(row: dict[str, Any], idx: int, round_id: str, repair_bucket: str, generator_bucket: str) -> None:
    row["id"] = f"issue5_{round_id}_nearmiss_{idx:06d}"
    row["mix_id"] = row["id"]
    row["source"] = "issue5_failure_derived_synthetic_nearmiss"
    row["origin"] = GENERATOR_VERSION
    row["issue"] = 5
    row["round_id"] = round_id
    row["repair_bucket"] = repair_bucket
    row["edge_bucket"] = generator_bucket
    row["gold_policy"] = "synthetic_by_construction_verified_schema_target_failure_bucket_weighted"
    row["teacher_policy"] = "full_model_teacher_allowed_for_proposals_or_soft_targets_only_no_unverified_hard_labels"


def build_nearmiss_rows(
    *,
    count: int,
    repair_counts: Counter[str],
    eval_rows: list[dict[str, Any]],
    round_id: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    plan = weighted_bucket_plan(count, repair_counts, rng)
    forbidden_literals = set()
    for row in eval_rows:
        forbidden_literals |= edge.target_literals(row)

    rows: list[dict[str, Any]] = []
    skipped_forbidden_literal = 0
    for idx, (repair_bucket, generator_bucket) in enumerate(plan):
        generator = GENERATORS[generator_bucket]
        for _attempt in range(1000):
            row = generator(idx, rng)
            rewrite_metadata(row, idx, round_id, repair_bucket, generator_bucket)
            edge.validate_row(row)
            if not (edge.target_literals(row) & forbidden_literals):
                break
            skipped_forbidden_literal += 1
        else:
            raise RuntimeError(f"could not generate leak-free row for {repair_bucket}/{generator_bucket}")
        rows.append(row)
    return rows, {
        "skipped_forbidden_literal": skipped_forbidden_literal,
        "repair_bucket_plan_counts": Counter(repair for repair, _ in plan),
        "generator_bucket_plan_counts": Counter(bucket for _, bucket in plan),
    }


def main() -> None:
    args = parse_args()
    if not (0 < args.augmentation_ratio <= args.max_augmentation_ratio <= 1.0):
        raise ValueError("augmentation ratio must be positive and <= max augmentation ratio <= 1")

    base_rows = read_jsonl(args.base_train_jsonl)
    eval_rows = read_jsonl(args.eval_jsonl)
    repair_counts = load_repair_counts(args.failure_bucket_dirs)
    strict_count = args.strict_count or len(base_rows)
    strict_rows = list(base_rows[:strict_count])
    edge_count = args.edge_count or max(1, round(len(strict_rows) * args.augmentation_ratio))
    max_edge_count = max(1, round(len(strict_rows) * args.max_augmentation_ratio))
    if edge_count > max_edge_count:
        raise ValueError(f"edge_count {edge_count} exceeds max allowed {max_edge_count}")

    edge_rows, generation_stats = build_nearmiss_rows(
        count=edge_count,
        repair_counts=repair_counts,
        eval_rows=eval_rows,
        round_id=args.round_id,
        seed=args.seed,
    )
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
        raise SystemExit("near-miss edge leak audit failed")

    mixed_rows = strict_rows + edge_rows
    random.Random(args.seed + 1).shuffle(mixed_rows)
    write_jsonl(args.edge_output, edge_rows)
    write_jsonl(args.mixed_output, mixed_rows)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "round_id": args.round_id,
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
        "augmentation_ratio_actual": len(edge_rows) / len(strict_rows),
        "max_augmentation_ratio": args.max_augmentation_ratio,
        "source_repair_bucket_counts": repair_counts,
        "generation_stats": generation_stats,
        "gold_policy": "hard labels are generated by construction and schema-verified",
        "teacher_policy": "teacher may support fuzzy synthesis or soft-policy pressure but is not accepted as hard gold without verification",
        "leak_audit": audit,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
