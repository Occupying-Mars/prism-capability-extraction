#!/usr/bin/env python3
"""Bucket BFCL eval failures for targeted repair-data generation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SLOT_FIELDS = {
    "unit",
    "location",
    "city",
    "county",
    "country",
    "device_id",
    "time",
    "date",
    "team",
    "season",
    "format",
    "timeout",
    "cast",
}
SQL_FIELDS = {"columns", "conditions", "insert_values", "update_values", "table_name", "sql_keyword"}
FORMULA_FIELDS = {"function", "expression", "equation", "formula"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def call_name(call: Any) -> str | None:
    return call.get("name") if isinstance(call, dict) else None


def call_args(call: Any) -> dict[str, Any]:
    if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
        return call["arguments"]
    return {}


def classify(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pred = row.get("prediction_calls") or []
    refs = row.get("reference_calls") or []
    if not pred:
        return "no_parse_or_no_call", {"pred_name": None, "ref_names": sorted({call_name(r) for r in refs if call_name(r)})}
    if len(pred) != 1:
        return "multi_call_or_extra_call", {"pred_count": len(pred)}

    p = pred[0]
    pred_name = call_name(p)
    ref_names = sorted({call_name(r) for r in refs if call_name(r)})
    if pred_name not in ref_names:
        return "wrong_function", {"pred_name": pred_name, "ref_names": ref_names}

    same_name_refs = [r for r in refs if call_name(r) == pred_name]
    pred_args = call_args(p)
    pred_keys = set(pred_args)
    ref_keys = set().union(*(set(call_args(r)) for r in same_name_refs)) if same_name_refs else set()
    missing = sorted(ref_keys - pred_keys)
    extra = sorted(pred_keys - ref_keys)

    detail = {
        "pred_name": pred_name,
        "ref_names": ref_names,
        "missing_keys": missing,
        "extra_keys": extra,
        "pred_keys": sorted(pred_keys),
        "ref_keys": sorted(ref_keys),
    }
    if pred_keys != ref_keys:
        if missing and not extra:
            return "missing_arg", detail
        if extra and not missing:
            return "extra_arg", detail
        return "arg_key_mismatch", detail

    wrong_value_keys: set[str] = set()
    for key, value in pred_args.items():
        allowed = [call_args(r).get(key) for r in same_name_refs if key in call_args(r)]
        if allowed and value not in allowed:
            wrong_value_keys.add(key)
    detail["wrong_value_keys"] = sorted(wrong_value_keys)
    return "wrong_arg_value", detail


def repair_buckets(category: str, failure_type: str, detail: dict[str, Any], pair: dict[str, Any]) -> list[str]:
    keys = set(detail.get("missing_keys", [])) | set(detail.get("extra_keys", [])) | set(detail.get("wrong_value_keys", []))
    buckets: set[str] = set()

    if failure_type == "wrong_arg_value":
        buckets.add("arg_value_exactness")
    if failure_type in {"missing_arg", "extra_arg", "arg_key_mismatch"}:
        buckets.add("schema_completion")
    if failure_type == "wrong_function":
        buckets.add("function_name_disambiguation")
    if category == "sql" or keys & SQL_FIELDS:
        buckets.add("sql_schema_discipline")
    if category == "live_simple" or keys & SLOT_FIELDS:
        buckets.add("live_slot_values")
    if "unit" in keys or "format" in keys or "cast" in keys:
        buckets.add("unit_default_normalization")
    if keys & FORMULA_FIELDS:
        buckets.add("formula_normalization")

    prompt = " ".join((m.get("content") or "") for m in pair.get("messages", []))
    if re.search(r"\b(am|pm)\b|\d+\s*(am|pm)\b", prompt, re.I):
        buckets.add("time_normalization")
    if re.search(r"\bx\s*\^|\^2|\^3", prompt):
        buckets.add("formula_normalization")

    return sorted(buckets or {"misc_failure"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    pairs = {row["id"]: row for row in load_jsonl(args.pairs_jsonl)}
    eval_rows = load_jsonl(args.eval_jsonl)
    failures: list[dict[str, Any]] = []
    by_failure_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_repair_bucket: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in eval_rows:
        if row.get("correct"):
            continue
        pair = pairs.get(row["id"], {})
        category = pair.get("category", "unknown")
        failure_type, detail = classify(row)
        buckets = repair_buckets(category, failure_type, detail, pair)
        record = {
            "id": row["id"],
            "category": category,
            "failure_type": failure_type,
            "repair_buckets": buckets,
            "prompt": (pair.get("messages") or [{}])[-1].get("content", ""),
            "tools": pair.get("tools", []),
            "reference_calls": row.get("reference_calls", []),
            "prediction_calls": row.get("prediction_calls", []),
            "prediction_text": row.get("prediction_text", ""),
            "detail": detail,
        }
        failures.append(record)
        by_failure_type[failure_type].append(record)
        by_category[category].append(record)
        for bucket in buckets:
            by_repair_bucket[bucket].append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.out_dir / "all_failures.jsonl", failures)
    for name, rows in by_failure_type.items():
        dump_jsonl(args.out_dir / "by_failure_type" / f"{name}.jsonl", rows)
    for name, rows in by_category.items():
        dump_jsonl(args.out_dir / "by_category" / f"{name}.jsonl", rows)
    for name, rows in by_repair_bucket.items():
        dump_jsonl(args.out_dir / "repair_buckets" / f"{name}.jsonl", rows)

    manifest = {
        "run_name": args.run_name or args.eval_jsonl.stem,
        "eval_jsonl": str(args.eval_jsonl),
        "pairs_jsonl": str(args.pairs_jsonl),
        "total_examples": len(eval_rows),
        "correct": sum(1 for row in eval_rows if row.get("correct")),
        "failures": len(failures),
        "failure_type_counts": Counter(row["failure_type"] for row in failures),
        "category_failure_counts": Counter(row["category"] for row in failures),
        "repair_bucket_counts": Counter(bucket for row in failures for bucket in row["repair_buckets"]),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
