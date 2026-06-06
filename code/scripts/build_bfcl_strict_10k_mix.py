#!/usr/bin/env python3
"""Build a capped 10k strict BFCL-format mix from filtered tool-call datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


BFCL_HOT_KEYS = {
    "location",
    "date",
    "time",
    "unit",
    "units",
    "format",
    "city",
    "country",
    "county",
    "conditions",
    "columns",
    "insert_values",
    "update_values",
    "table_name",
    "sql_keyword",
    "timezone",
    "language",
    "text",
    "email",
    "title",
    "name",
    "id",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_key(row: dict[str, Any]) -> str:
    payload = {
        "messages": row.get("messages", []),
        "tools": row.get("tools", []),
        "target_call": row.get("target_call", {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tool_name(row: dict[str, Any]) -> str:
    return str((row.get("target_call") or {}).get("name", ""))


def arg_keys(row: dict[str, Any]) -> set[str]:
    args = (row.get("target_call") or {}).get("arguments") or {}
    return set(args) if isinstance(args, dict) else set()


def parameters(row: dict[str, Any]) -> dict[str, Any]:
    return row["tools"][0]["function"]["parameters"]


def has_hot_key(row: dict[str, Any]) -> bool:
    return bool(arg_keys(row) & BFCL_HOT_KEYS)


def has_optional_or_default(row: dict[str, Any]) -> bool:
    params = parameters(row)
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    optional = set(props) - required
    defaults = {k for k, v in props.items() if isinstance(v, dict) and "default" in v}
    emitted = arg_keys(row)
    return bool((optional | defaults) & emitted)


def validate_bfcl_shape(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len([m for m in messages if m.get("role") == "user"]) != 1:
        return "bad_messages"
    tools = row.get("tools")
    if not isinstance(tools, list) or len(tools) != 1:
        return "bad_tools"
    tool = tools[0]
    if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
        return "bad_tool_type"
    fn = tool["function"]
    params = fn.get("parameters")
    if not isinstance(params, dict) or params.get("type") != "object" or not isinstance(params.get("properties"), dict):
        return "bad_parameters"
    call = row.get("target_call")
    if not isinstance(call, dict) or call.get("name") != fn.get("name") or not isinstance(call.get("arguments"), dict):
        return "bad_target_call"
    if not set(call["arguments"]) <= set(params["properties"]):
        return "target_args_not_in_schema"
    if not set(params.get("required") or []) <= set(call["arguments"]):
        return "required_args_missing"
    target_text = row.get("target_text", "")
    if "<tool_call>" not in target_text or "</tool_call>" not in target_text:
        return "bad_target_text"
    return None


def select_with_caps(
    buckets: list[list[dict[str, Any]]],
    *,
    target_rows: int,
    per_tool_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_tool: Counter[str] = Counter()

    def try_add(row: dict[str, Any], cap: int) -> bool:
        if len(selected) >= target_rows:
            return False
        key = stable_key(row)
        name = tool_name(row)
        if key in seen or per_tool[name] >= cap:
            return False
        seen.add(key)
        per_tool[name] += 1
        selected.append(row)
        return True

    for bucket in buckets:
        rows = list(bucket)
        rng.shuffle(rows)
        for row in rows:
            try_add(row, per_tool_cap)

    cap = per_tool_cap * 2
    while len(selected) < target_rows and cap <= max(per_tool_cap * 8, per_tool_cap + 1):
        for bucket in buckets:
            rows = list(bucket)
            rng.shuffle(rows)
            for row in rows:
                try_add(row, cap)
                if len(selected) >= target_rows:
                    break
            if len(selected) >= target_rows:
                break
        cap *= 2

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            Path("data/toolmind_bfcl_strict/train.jsonl"),
            Path("data/argilla_apigen_bfcl_strict/train.jsonl"),
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("data/bfcl_strict_10k_mix/train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/bfcl_strict_10k_mix/manifest.json"))
    parser.add_argument("--target-rows", type=int, default=10_000)
    parser.add_argument("--per-tool-cap", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    input_counts = {}
    for path in args.inputs:
        loaded = read_jsonl(path)
        input_counts[str(path)] = len(loaded)
        rows.extend(loaded)

    validation_failures = Counter(reason for row in rows if (reason := validate_bfcl_shape(row)))
    rows = [row for row in rows if validate_bfcl_shape(row) is None]

    hot = [row for row in rows if has_hot_key(row)]
    optional = [row for row in rows if has_optional_or_default(row) and not has_hot_key(row)]
    rest = [row for row in rows if not has_hot_key(row) and not has_optional_or_default(row)]
    selected = select_with_caps(
        [hot, optional, rest],
        target_rows=args.target_rows,
        per_tool_cap=args.per_tool_cap,
        seed=args.seed,
    )
    if len(selected) < args.target_rows:
        raise RuntimeError(f"only selected {len(selected)} rows from {len(rows)} valid rows")

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    for idx, row in enumerate(selected):
        row["mix_id"] = f"bfcl_strict_10k_{idx:05d}"

    write_jsonl(args.output, selected)
    manifest = {
        "inputs": input_counts,
        "output": str(args.output),
        "target_rows": args.target_rows,
        "rows": len(selected),
        "seed": args.seed,
        "per_tool_cap_initial": args.per_tool_cap,
        "valid_input_rows": len(rows),
        "validation_failures": validation_failures,
        "source_counts": Counter(row.get("source", "unknown") for row in selected),
        "hot_key_rows": sum(has_hot_key(row) for row in selected),
        "optional_or_default_rows": sum(has_optional_or_default(row) for row in selected),
        "unique_tool_names": len({tool_name(row) for row in selected}),
        "top_tool_names": Counter(tool_name(row) for row in selected).most_common(25),
        "arg_count_distribution": Counter(len(arg_keys(row)) for row in selected),
        "top_arg_keys": Counter(key for row in selected for key in arg_keys(row)).most_common(50),
        "filters": {
            "single_user_prefix": True,
            "single_tool": True,
            "single_call": True,
            "target_name_matches_tool": True,
            "target_args_subset_schema": True,
            "required_args_present": True,
            "target_text_tool_call_wrapped": True,
            "schema_openai_qwen_parameters": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
