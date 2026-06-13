#!/usr/bin/env python3
"""Build leak-audited synthetic BFCL edge-curriculum rows.

This generator is for issue #4. It creates verified tool-call rows by
construction, optionally mixes them with the existing strict BFCL-style train
set, and writes a manifest that records leak-audit and bucket counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable


GENERATOR_VERSION = "issue4_edge_curriculum_v1"
DEFAULT_BUCKETS = (
    "function_name_disambiguation",
    "schema_completion",
    "enum_disambiguation",
    "json_wrapper_stability",
    "type_coercion",
    "optional_default_arguments",
    "sql_schema_discipline",
    "formula_normalization",
)
GENERIC_LITERALS = {
    "",
    "0",
    "1",
    "2",
    "3",
    "true",
    "false",
    "yes",
    "no",
    "none",
    "null",
    "en",
    "fr",
    "us",
    "id",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-train-jsonl", type=Path)
    p.add_argument("--eval-jsonl", type=Path)
    p.add_argument("--edge-output", type=Path, required=True)
    p.add_argument("--mixed-output", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--edge-count", type=int, default=3000)
    p.add_argument("--strict-count", type=int, default=9000)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--buckets", default=",".join(DEFAULT_BUCKETS))
    p.add_argument("--near-threshold", type=float, default=0.85)
    p.add_argument("--shingle-size", type=int, default=5)
    p.add_argument("--fail-on-leak", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = stable_json(value)
    value = re.sub(r"\s+", " ", value.lower()).strip()
    return value


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_shingles(text: str, size: int) -> set[str]:
    tokens = re.findall(r"[a-z0-9_./:-]+|[{}()[\],:=<>]", text)
    if not tokens:
        return set()
    if len(tokens) <= size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def row_parts(row: dict[str, Any]) -> dict[str, str]:
    prompt_obj = {
        "messages": row.get("messages"),
        "tools": row.get("tools"),
        "question": row.get("question"),
        "prompt": row.get("prompt"),
    }
    target_obj = {
        "target_text": row.get("target_text"),
        "target": row.get("target"),
        "target_call": row.get("target_call"),
        "reference_calls": row.get("reference_calls"),
        "answer": row.get("answer"),
    }
    prompt = normalize_text(prompt_obj)
    target = normalize_text(target_obj)
    return {"prompt": prompt, "target": target, "combined": f"{prompt}\n{target}"}


def target_text(call: dict[str, Any]) -> str:
    return "<tool_call>\n" + json.dumps(call, sort_keys=True, ensure_ascii=True) + "\n</tool_call>"


def function_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def string_prop(description: str, *, default: str | None = None, enum: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "string", "description": description}
    if default is not None:
        out["default"] = default
    if enum:
        out["enum"] = enum
    return out


def int_prop(description: str, *, default: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "integer", "description": description}
    if default is not None:
        out["default"] = default
    return out


def number_prop(description: str, *, default: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "number", "description": description}
    if default is not None:
        out["default"] = default
    return out


def bool_prop(description: str, *, default: bool | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "boolean", "description": description}
    if default is not None:
        out["default"] = default
    return out


def array_prop(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def object_prop(description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": properties,
        "required": required,
    }


def nonce(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randrange(10000, 99999)}"


def base_row(
    *,
    idx: int,
    bucket: str,
    prompt: str,
    tools: list[dict[str, Any]],
    call: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"issue4_edge_{idx:06d}",
        "mix_id": f"issue4_edge_{idx:06d}",
        "source": "issue4_synthetic_edge_curriculum",
        "origin": GENERATOR_VERSION,
        "edge_bucket": bucket,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "target_call": call,
        "target_text": target_text(call),
        "gold_policy": "synthetic_by_construction_verified_schema_target",
    }


def make_function_name_disambiguation(idx: int, rng: random.Random) -> dict[str, Any]:
    case_id = nonce(rng, "case")
    destination = rng.choice(["edge_northdock", "edge_southvault", "edge_eastbay", "edge_westspire"])
    priority = rng.choice(["edge_low", "edge_normal", "edge_high", "edge_urgent"])
    notify = rng.choice([True, False])
    name = "edgev4_route_incident"
    tools = [
        function_tool(
            name,
            "Route an incident case to an operations destination.",
            {
                "edge_case_id": string_prop("Unique incident case id."),
                "edge_destination": string_prop(
                    "Operations destination.",
                    enum=["edge_northdock", "edge_southvault", "edge_eastbay", "edge_westspire"],
                ),
                "edge_priority": string_prop("Routing priority.", enum=["edge_low", "edge_normal", "edge_high", "edge_urgent"]),
                "edge_notify": bool_prop("Whether to send a notification.", default=False),
            },
            ["edge_case_id", "edge_destination", "edge_priority"],
        ),
        function_tool(
            "edgev4_route_invoice",
            "Route an invoice to a finance queue.",
            {"edge_invoice_id": string_prop("Invoice id."), "edge_finance_queue": string_prop("Finance queue.")},
            ["edge_invoice_id", "edge_finance_queue"],
        ),
        function_tool(
            "edgev4_close_incident",
            "Close an incident after resolution.",
            {"edge_case_id": string_prop("Unique incident case id."), "edge_resolution": string_prop("Resolution note.")},
            ["edge_case_id", "edge_resolution"],
        ),
    ]
    prompt = f"Send incident {case_id} to {destination} with {priority} priority"
    if notify:
        prompt += " and notify the destination lead."
    else:
        prompt += " without notifying anyone."
    call = {
        "name": name,
        "arguments": {
            "edge_case_id": case_id,
            "edge_destination": destination,
            "edge_priority": priority,
            "edge_notify": notify,
        },
    }
    return base_row(idx=idx, bucket="function_name_disambiguation", prompt=prompt, tools=tools, call=call)


def make_schema_completion(idx: int, rng: random.Random) -> dict[str, Any]:
    ticket = nonce(rng, "ticket")
    region = rng.choice(["edge_aurora", "edge_cobalt", "edge_delta", "edge_ember"])
    severity = rng.choice(["edge_s1", "edge_s2", "edge_s3"])
    owner = rng.choice(["edge_ops_alpha", "edge_ops_beta", "edge_ops_gamma"])
    name = "edgev4_schedule_repair_window"
    tools = [
        function_tool(
            name,
            "Schedule a repair window and assign it to an owner.",
            {
                "edge_ticket_id": string_prop("Repair ticket id."),
                "edge_region": string_prop(
                    "Affected service region.",
                    enum=["edge_aurora", "edge_cobalt", "edge_delta", "edge_ember"],
                ),
                "edge_severity": string_prop("Incident severity.", enum=["edge_s1", "edge_s2", "edge_s3"]),
                "edge_owner": string_prop("Owning team."),
                "edge_duration_minutes": int_prop("Repair duration in minutes.", default=30),
            },
            ["edge_ticket_id", "edge_region", "edge_severity", "edge_owner"],
        )
    ]
    minutes = rng.choice([20, 30, 45, 60])
    prompt = (
        f"Create a repair window for {ticket} in {region}. It is severity {severity}, "
        f"owned by {owner}, and should last {minutes} minutes."
    )
    call = {
        "name": name,
        "arguments": {
            "edge_ticket_id": ticket,
            "edge_region": region,
            "edge_severity": severity,
            "edge_owner": owner,
            "edge_duration_minutes": minutes,
        },
    }
    return base_row(idx=idx, bucket="schema_completion", prompt=prompt, tools=tools, call=call)


def make_enum_disambiguation(idx: int, rng: random.Random) -> dict[str, Any]:
    packet = nonce(rng, "packet")
    color = rng.choice(["edge_cyan", "edge_magenta", "edge_amber", "edge_violet"])
    mode = rng.choice(["edge_archive", "edge_mirror", "edge_discard", "edge_forward"])
    name = "edgev4_apply_packet_policy"
    tools = [
        function_tool(
            name,
            "Apply a packet policy based on color and routing mode.",
            {
                "edge_packet_id": string_prop("Packet id."),
                "edge_color": string_prop("Packet color.", enum=["edge_cyan", "edge_magenta", "edge_amber", "edge_violet"]),
                "edge_mode": string_prop("Policy mode.", enum=["edge_archive", "edge_mirror", "edge_discard", "edge_forward"]),
            },
            ["edge_packet_id", "edge_color", "edge_mode"],
        )
    ]
    prompt = f"For packet {packet}, choose the {mode} policy for the {color} packet class."
    call = {"name": name, "arguments": {"edge_packet_id": packet, "edge_color": color, "edge_mode": mode}}
    return base_row(idx=idx, bucket="enum_disambiguation", prompt=prompt, tools=tools, call=call)


def make_json_wrapper_stability(idx: int, rng: random.Random) -> dict[str, Any]:
    batch = nonce(rng, "batch")
    labels = rng.sample(["edge_zircon", "edge_quartz", "edge_opal", "edge_garnet", "edge_basalt"], k=3)
    reviewer = rng.choice(["edge_review_north", "edge_review_south", "edge_review_east"])
    name = "edgev4_submit_label_batch"
    tools = [
        function_tool(
            name,
            "Submit a reviewed batch of labels.",
            {
                "edge_batch_id": string_prop("Batch id."),
                "edge_labels": array_prop("Labels to submit."),
                "edge_review": object_prop(
                    "Review metadata.",
                    {
                        "edge_reviewer": string_prop("Reviewer id."),
                        "edge_approved": bool_prop("Approval flag."),
                    },
                    ["edge_reviewer", "edge_approved"],
                ),
            },
            ["edge_batch_id", "edge_labels", "edge_review"],
        )
    ]
    prompt = f"Submit label batch {batch} with labels {', '.join(labels)}. Reviewer {reviewer} approved it."
    call = {
        "name": name,
        "arguments": {
            "edge_batch_id": batch,
            "edge_labels": labels,
            "edge_review": {"edge_reviewer": reviewer, "edge_approved": True},
        },
    }
    return base_row(idx=idx, bucket="json_wrapper_stability", prompt=prompt, tools=tools, call=call)


def make_type_coercion(idx: int, rng: random.Random) -> dict[str, Any]:
    sensor = nonce(rng, "sensor")
    retries = rng.choice([2, 3, 4, 5])
    threshold = rng.choice([0.15, 0.25, 0.4, 0.75])
    enabled = rng.choice([True, False])
    name = "edgev4_configure_sensor_guard"
    tools = [
        function_tool(
            name,
            "Configure guard parameters for a sensor.",
            {
                "edge_sensor_id": string_prop("Sensor id."),
                "edge_retries": int_prop("Retry count."),
                "edge_threshold": number_prop("Alert threshold."),
                "edge_enabled": bool_prop("Whether the guard is enabled."),
            },
            ["edge_sensor_id", "edge_retries", "edge_threshold", "edge_enabled"],
        )
    ]
    state = "enabled" if enabled else "disabled"
    prompt = f"Set sensor {sensor} guard to {state}, threshold {threshold}, with {retries} retries."
    call = {
        "name": name,
        "arguments": {
            "edge_sensor_id": sensor,
            "edge_retries": retries,
            "edge_threshold": threshold,
            "edge_enabled": enabled,
        },
    }
    return base_row(idx=idx, bucket="type_coercion", prompt=prompt, tools=tools, call=call)


def make_optional_default_arguments(idx: int, rng: random.Random) -> dict[str, Any]:
    job = nonce(rng, "job")
    lane = rng.choice(["edge_blue_lane", "edge_green_lane", "edge_silver_lane"])
    include_notes = rng.choice([True, False])
    name = "edgev4_enqueue_review_job"
    tools = [
        function_tool(
            name,
            "Enqueue a review job and include defaultable options.",
            {
                "edge_job_id": string_prop("Job id."),
                "edge_lane": string_prop("Review lane."),
                "edge_locale": string_prop("Output locale.", default="edge_en"),
                "edge_include_notes": bool_prop("Whether notes should be included.", default=False),
            },
            ["edge_job_id", "edge_lane"],
        )
    ]
    prompt = f"Queue review job {job} on {lane} in English"
    if include_notes:
        prompt += " and include notes."
    else:
        prompt += " without notes."
    call = {
        "name": name,
        "arguments": {
            "edge_job_id": job,
            "edge_lane": lane,
            "edge_locale": "edge_en",
            "edge_include_notes": include_notes,
        },
    }
    return base_row(idx=idx, bucket="optional_default_arguments", prompt=prompt, tools=tools, call=call)


def make_sql_schema_discipline(idx: int, rng: random.Random) -> dict[str, Any]:
    table = rng.choice(["edgev4_shipments", "edgev4_ledgers", "edgev4_assets"])
    column = rng.choice(["edge_status", "edge_region_code", "edge_batch_code"])
    value = nonce(rng, "sqlval")
    keyword = rng.choice(["FETCH_ROWS", "CHANGE_ROWS", "REMOVE_ROWS"])
    name = "edgev4_build_sql_plan"
    tools = [
        function_tool(
            name,
            "Build a constrained SQL plan without executing it.",
            {
                "edge_sql_action": string_prop("SQL-like operation.", enum=["FETCH_ROWS", "CHANGE_ROWS", "REMOVE_ROWS"]),
                "edge_table_ref": string_prop("Table reference.", enum=["edgev4_shipments", "edgev4_ledgers", "edgev4_assets"]),
                "edge_column_refs": array_prop("Columns involved in the plan."),
                "edge_filter_expr": string_prop("Filter expression."),
            },
            ["edge_sql_action", "edge_table_ref", "edge_column_refs", "edge_filter_expr"],
        )
    ]
    prompt = f"Draft a {keyword} plan for table {table} using column {column} where {column} equals {value}."
    call = {
        "name": name,
        "arguments": {
            "edge_sql_action": keyword,
            "edge_table_ref": table,
            "edge_column_refs": [column],
            "edge_filter_expr": f"{column} == '{value}'",
        },
    }
    return base_row(idx=idx, bucket="sql_schema_discipline", prompt=prompt, tools=tools, call=call)


def make_formula_normalization(idx: int, rng: random.Random) -> dict[str, Any]:
    formula_id = nonce(rng, "formula")
    power = rng.choice([2, 3, 4])
    variable = rng.choice(["edge_x", "edge_y", "edge_z"])
    expression = f"{variable}**{power} + {power}"
    name = "edgev4_register_formula"
    tools = [
        function_tool(
            name,
            "Register a formula using Python-style exponent syntax.",
            {
                "edge_formula_id": string_prop("Formula id."),
                "edge_expression": string_prop("Formula expression."),
                "edge_variable": string_prop("Primary variable.", enum=["edge_x", "edge_y", "edge_z"]),
            },
            ["edge_formula_id", "edge_expression", "edge_variable"],
        )
    ]
    prompt = f"Register {formula_id} as {variable}^{power} + {power}; use Python exponent syntax."
    call = {
        "name": name,
        "arguments": {
            "edge_formula_id": formula_id,
            "edge_expression": expression,
            "edge_variable": variable,
        },
    }
    return base_row(idx=idx, bucket="formula_normalization", prompt=prompt, tools=tools, call=call)


GENERATORS: dict[str, Callable[[int, random.Random], dict[str, Any]]] = {
    "function_name_disambiguation": make_function_name_disambiguation,
    "schema_completion": make_schema_completion,
    "enum_disambiguation": make_enum_disambiguation,
    "json_wrapper_stability": make_json_wrapper_stability,
    "type_coercion": make_type_coercion,
    "optional_default_arguments": make_optional_default_arguments,
    "sql_schema_discipline": make_sql_schema_discipline,
    "formula_normalization": make_formula_normalization,
}


def parse_target_text(text: str) -> Any:
    match = re.fullmatch(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", text, re.DOTALL)
    if not match:
        raise ValueError("target_text is not wrapped in one tool_call block")
    return json.loads(match.group(1))


def validate_type(value: Any, schema: dict[str, Any], path: str) -> None:
    typ = schema.get("type")
    if typ == "string" and not isinstance(value, str):
        raise ValueError(f"{path} must be string")
    if typ == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{path} must be integer")
    if typ == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ValueError(f"{path} must be number")
    if typ == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    if typ == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be array")
        item_schema = schema.get("items") or {}
        for i, item in enumerate(value):
            validate_type(item, item_schema, f"{path}[{i}]")
    if typ == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be object")
        validate_args(value, schema, path)
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise ValueError(f"{path} must be one of {enum}")


def validate_args(args: dict[str, Any], params: dict[str, Any], path: str) -> None:
    properties = params.get("properties") or {}
    required = set(params.get("required") or [])
    missing = sorted(required - set(args))
    extra = sorted(set(args) - set(properties))
    if missing:
        raise ValueError(f"{path} missing required keys {missing}")
    if extra:
        raise ValueError(f"{path} has extra keys {extra}")
    for key, value in args.items():
        validate_type(value, properties[key], f"{path}.{key}")


def validate_row(row: dict[str, Any]) -> None:
    call = row["target_call"]
    parsed = parse_target_text(row["target_text"])
    if parsed != call:
        raise ValueError(f"{row['id']} target_text does not match target_call")
    tools = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in row.get("tools", [])
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    if call["name"] not in tools:
        raise ValueError(f"{row['id']} target function missing from tool schema")
    validate_args(call.get("arguments") or {}, tools[call["name"]], row["id"])


def function_names(row: dict[str, Any]) -> set[str]:
    names = set()
    for tool in row.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
            if name:
                names.add(str(name).lower())
    for key in ("target_call",):
        call = row.get(key)
        if isinstance(call, dict) and call.get("name"):
            names.add(str(call["name"]).lower())
    for call in row.get("reference_calls") or []:
        if isinstance(call, dict) and call.get("name"):
            names.add(str(call["name"]).lower())
    return names


def iter_scalar_values(value: Any) -> list[str]:
    out = []
    if isinstance(value, dict):
        for item in value.values():
            out.extend(iter_literals(item))
    elif isinstance(value, list):
        for item in value:
            out.extend(iter_scalar_values(item))
    elif isinstance(value, (str, int, float, bool)):
        out.append(str(value))
    return out


def iter_literals(value: Any) -> list[str]:
    return iter_scalar_values(value)


def target_literals(row: dict[str, Any]) -> set[str]:
    values = set()
    for key in ("target_call", "reference_calls", "target"):
        for literal in iter_scalar_values(row.get(key)):
            norm = normalize_text(literal)
            if len(norm) >= 4 and norm not in GENERIC_LITERALS:
                values.add(norm)
    return values


def schema_argument_names(row: dict[str, Any]) -> set[str]:
    names = set()

    def visit_schema(schema: Any) -> None:
        if not isinstance(schema, dict):
            return
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, value in properties.items():
                names.add(str(key).lower())
                visit_schema(value)
        items = schema.get("items")
        if isinstance(items, dict):
            visit_schema(items)

    for tool in row.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            params = tool["function"].get("parameters")
            visit_schema(params)
    call = row.get("target_call")
    if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
        names |= {str(key).lower() for key in call["arguments"]}
    for call in row.get("reference_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
            names |= {str(key).lower() for key in call["arguments"]}
    return names


def audit_edge_rows(
    edge_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    near_threshold: float,
    shingle_size: int,
) -> dict[str, Any]:
    eval_records = []
    eval_names = set()
    eval_schema_names = set()
    eval_literals = set()
    for idx, row in enumerate(eval_rows):
        parts = row_parts(row)
        eval_records.append(
            {
                "idx": idx,
                "id": row.get("id", idx),
                "row_hash": sha(parts["combined"]),
                "prompt_hash": sha(parts["prompt"]),
                "target_hash": sha(parts["target"]),
                "shingles": token_shingles(parts["combined"], shingle_size),
            }
        )
        eval_names |= function_names(row)
        eval_schema_names |= schema_argument_names(row)
        eval_literals |= target_literals(row)

    eval_row_hashes = {rec["row_hash"] for rec in eval_records}
    eval_prompt_hashes = {rec["prompt_hash"] for rec in eval_records}
    eval_target_hashes = {rec["target_hash"] for rec in eval_records}
    exact_rows = []
    exact_prompts = []
    exact_targets = []
    name_overlaps = []
    schema_name_overlaps = []
    literal_overlaps = []
    near_overlaps = []
    max_near_similarity = 0.0

    for row in edge_rows:
        parts = row_parts(row)
        row_hash = sha(parts["combined"])
        prompt_hash = sha(parts["prompt"])
        target_hash = sha(parts["target"])
        if row_hash in eval_row_hashes:
            exact_rows.append(row["id"])
        if prompt_hash in eval_prompt_hashes:
            exact_prompts.append(row["id"])
        if target_hash in eval_target_hashes:
            exact_targets.append(row["id"])
        names = sorted(function_names(row) & eval_names)
        if names:
            name_overlaps.append({"id": row["id"], "names": names})
        schema_names = sorted(schema_argument_names(row) & eval_schema_names)
        if schema_names:
            schema_name_overlaps.append({"id": row["id"], "schema_names": schema_names[:20]})
        literals = sorted(target_literals(row) & eval_literals)
        if literals:
            literal_overlaps.append({"id": row["id"], "literals": literals[:20]})

        shingles = token_shingles(parts["combined"], shingle_size)
        if not shingles:
            continue
        for ev in eval_records:
            union = len(shingles | ev["shingles"])
            if union == 0:
                continue
            similarity = len(shingles & ev["shingles"]) / union
            max_near_similarity = max(max_near_similarity, similarity)
            if similarity >= near_threshold:
                near_overlaps.append({"id": row["id"], "eval_id": ev["id"], "similarity": similarity})
                break

    counts = {
        "exact_row_overlaps": len(exact_rows),
        "exact_prompt_overlaps": len(exact_prompts),
        "exact_target_overlaps": len(exact_targets),
        "tool_name_overlaps": len(name_overlaps),
        "schema_arg_name_overlaps": len(schema_name_overlaps),
        "target_literal_overlaps": len(literal_overlaps),
        "near_overlaps": len(near_overlaps),
    }
    return {
        "eval_rows": len(eval_rows),
        "edge_rows": len(edge_rows),
        "near_threshold": near_threshold,
        "shingle_size": shingle_size,
        "counts": counts,
        "max_near_similarity": max_near_similarity,
        "examples": {
            "exact_rows": exact_rows[:20],
            "exact_prompts": exact_prompts[:20],
            "exact_targets": exact_targets[:20],
            "tool_name_overlaps": name_overlaps[:20],
            "schema_arg_name_overlaps": schema_name_overlaps[:20],
            "target_literal_overlaps": literal_overlaps[:20],
            "near_overlaps": near_overlaps[:20],
        },
        "passed": all(value == 0 for value in counts.values()),
    }


def build_edges(
    count: int,
    buckets: list[str],
    seed: int,
    *,
    forbidden_literals: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    unknown = sorted(set(buckets) - set(GENERATORS))
    if unknown:
        raise ValueError(f"unknown buckets: {unknown}")
    rows = []
    rng = random.Random(seed)
    forbidden_literals = forbidden_literals or set()
    skipped_forbidden_literal = 0
    for idx in range(count):
        bucket = buckets[idx % len(buckets)]
        for attempt in range(1000):
            row = GENERATORS[bucket](idx, rng)
            validate_row(row)
            if not (target_literals(row) & forbidden_literals):
                break
            skipped_forbidden_literal += 1
        else:
            raise RuntimeError(f"could not build leak-free row for bucket {bucket} at index {idx}")
        rows.append(row)
    rng.shuffle(rows)
    return rows, {"skipped_forbidden_literal": skipped_forbidden_literal}


def main() -> None:
    args = parse_args()
    buckets = [item.strip() for item in args.buckets.split(",") if item.strip()]
    if not buckets:
        raise ValueError("at least one bucket is required")

    base_rows = read_jsonl(args.base_train_jsonl)
    eval_rows = read_jsonl(args.eval_jsonl)
    forbidden_literals = set()
    for row in eval_rows:
        forbidden_literals |= target_literals(row)

    edge_rows, generation_stats = build_edges(
        args.edge_count,
        buckets,
        args.seed,
        forbidden_literals=forbidden_literals,
    )

    audit = audit_edge_rows(
        edge_rows,
        eval_rows,
        near_threshold=args.near_threshold,
        shingle_size=args.shingle_size,
    )
    if args.fail_on_leak and not audit["passed"]:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"leak_audit": audit}, indent=2, sort_keys=True) + "\n")
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit("edge leak audit failed")

    strict_rows = []
    if base_rows and args.strict_count:
        rng = random.Random(args.seed + 1)
        strict_rows = list(base_rows)
        rng.shuffle(strict_rows)
        strict_rows = strict_rows[: min(args.strict_count, len(strict_rows))]

    mixed_rows = strict_rows + edge_rows
    random.Random(args.seed + 2).shuffle(mixed_rows)

    write_jsonl(args.edge_output, edge_rows)
    if args.mixed_output:
        write_jsonl(args.mixed_output, mixed_rows)

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "buckets": buckets,
        "bucket_counts": Counter(row["edge_bucket"] for row in edge_rows),
        "edge_output": str(args.edge_output),
        "mixed_output": str(args.mixed_output) if args.mixed_output else None,
        "base_train_jsonl": str(args.base_train_jsonl) if args.base_train_jsonl else None,
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "rows": {
            "edge": len(edge_rows),
            "strict": len(strict_rows),
            "mixed": len(mixed_rows),
            "base_available": len(base_rows),
        },
        "generation_stats": generation_stats,
        "gold_policy": "synthetic targets are generated by construction and validated against their schemas",
        "teacher_policy": "full model may propose/rank future rows, but this v1 hard-label set does not use unverifiable teacher answers",
        "leak_audit": audit,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
