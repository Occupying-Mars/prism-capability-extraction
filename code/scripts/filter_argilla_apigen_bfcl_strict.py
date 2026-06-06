#!/usr/bin/env python3
"""Filter argilla/apigen-function-calling into strict BFCL-format single-tool rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


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

WEIRD_PROMPT_RE = re.compile(
    r"Role definition|Historical dialog|</after>|Response assistant|Inquirer:|"
    r"^System:|^Assistant:|^User:",
    re.I | re.M,
)
SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_type(value: Any) -> str:
    if not isinstance(value, str):
        return "string"
    value = value.lower()
    if value in {"str", "string"}:
        return "string"
    if value in {"int", "integer"}:
        return "integer"
    if value in {"float", "number"}:
        return "number"
    if value in {"bool", "boolean"}:
        return "boolean"
    if value.startswith("list") or value in {"array", "tuple"}:
        return "array"
    if value in {"dict", "object"}:
        return "object"
    return "string"


def normalize_property(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "string"}
    out: dict[str, Any] = {}
    if "description" in value:
        out["description"] = str(value["description"])
    out["type"] = parse_type(value.get("type", "string"))
    if "enum" in value and isinstance(value["enum"], list):
        out["enum"] = value["enum"]
    if "default" in value:
        out["default"] = value["default"]
    if out["type"] == "array" and "items" not in out:
        out["items"] = {"type": "string"}
    if out["type"] == "object" and "properties" in value and isinstance(value["properties"], dict):
        out["properties"] = {str(k): normalize_property(v) for k, v in value["properties"].items()}
    return out


def normalize_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    if not isinstance(name, str) or not SAFE_NAME_RE.match(name):
        return None
    params = tool.get("parameters")
    if not isinstance(params, dict):
        return None

    # APIGen stores the properties directly under `parameters`.
    properties = {str(k): normalize_property(v) for k, v in params.items()}
    required = [k for k, v in params.items() if not (isinstance(v, dict) and "default" in v)]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(tool.get("description", "")),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [str(k) for k in required if str(k) in properties],
            },
        },
    }


def prop_type(tool: dict[str, Any], key: str) -> str | None:
    props = tool["function"]["parameters"]["properties"]
    prop = props.get(key)
    return prop.get("type") if isinstance(prop, dict) else None


def value_matches_schema(value: Any, schema_type: str | None) -> bool:
    if schema_type in {None, "string"}:
        return isinstance(value, (str, int, float, bool)) or value is None
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return True


def keep_row(row: dict[str, Any], *, max_prompt_chars: int, max_args: int) -> tuple[dict[str, Any] | None, str]:
    prompt = row.get("query")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > max_prompt_chars:
        return None, "bad_prompt"
    if WEIRD_PROMPT_RE.search(prompt):
        return None, "bad_prompt"

    tools = parse_jsonish(row.get("tools"))
    if not isinstance(tools, list) or len(tools) != 1:
        return None, "not_single_tool"
    tool = normalize_tool(tools[0])
    if tool is None:
        return None, "bad_tool_schema"

    answers = parse_jsonish(row.get("answers"))
    if not isinstance(answers, list) or len(answers) != 1:
        return None, "not_single_call"
    call = answers[0]
    if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
        return None, "bad_target_call"
    if call.get("name") != tool["function"]["name"]:
        return None, "tool_target_name_mismatch"
    args = call["arguments"]
    if len(args) > max_args:
        return None, "too_many_args"

    props = tool["function"]["parameters"]["properties"]
    required = set(tool["function"]["parameters"].get("required", []))
    arg_keys = set(args)
    if not arg_keys <= set(props):
        return None, "target_args_not_in_schema"
    if not required <= arg_keys:
        return None, "missing_required_target_arg"
    for key, value in args.items():
        if not value_matches_schema(value, prop_type(tool, key)):
            return None, "value_type_mismatch"

    out = {
        "id": f"argilla_apigen_{row.get('id')}",
        "source": "argilla_apigen_bfcl_strict",
        "origin": row.get("origin"),
        "messages": [{"role": "user", "content": prompt}],
        "tools": [tool],
        "target_call": {"name": call["name"], "arguments": args},
        "target_text": "<tool_call>\n"
        + json.dumps({"name": call["name"], "arguments": args}, ensure_ascii=False)
        + "\n</tool_call>",
    }
    return out, "kept"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="argilla/apigen-function-calling")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/argilla_apigen_bfcl_strict/train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/argilla_apigen_bfcl_strict/manifest.json"))
    parser.add_argument("--max-prompt-chars", type=int, default=1500)
    parser.add_argument("--max-args", type=int, default=12)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen = 0
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    for row in ds:
        if args.max_rows and seen >= args.max_rows:
            break
        seen += 1
        out, reason = keep_row(row, max_prompt_chars=args.max_prompt_chars, max_args=args.max_args)
        reasons[reason] += 1
        if out is not None:
            kept.append(out)
        if seen % 10000 == 0:
            print(f"seen={seen} kept={len(kept)}", flush=True)

    write_jsonl(args.output, kept)
    hot_key_rows = sum(
        bool(set((row["target_call"].get("arguments") or {}).keys()) & BFCL_HOT_KEYS)
        for row in kept
    )
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "output": str(args.output),
        "seen": seen,
        "kept": len(kept),
        "rejection_counts": reasons,
        "origin_counts": Counter(row.get("origin", "unknown") for row in kept),
        "unique_tool_names": len({row["target_call"]["name"] for row in kept}),
        "tool_name_top20": Counter(row["target_call"]["name"] for row in kept).most_common(20),
        "arg_count_distribution": Counter(len(row["target_call"]["arguments"]) for row in kept),
        "arg_key_top40": Counter(
            key for row in kept for key in row["target_call"]["arguments"]
        ).most_common(40),
        "bfcl_hot_key_rows": hot_key_rows,
        "bfcl_hot_key_row_fraction": hot_key_rows / len(kept) if kept else 0.0,
        "filters": {
            "single_user_prefix": True,
            "single_tool": True,
            "single_call": True,
            "target_name_matches_tool": True,
            "target_args_subset_schema": True,
            "required_args_present": True,
            "schema_normalized_to_openai_parameters": True,
            "max_prompt_chars": args.max_prompt_chars,
            "max_args": args.max_args,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
