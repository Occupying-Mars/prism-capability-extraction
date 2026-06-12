#!/usr/bin/env python3
"""Direct Qwen3/BFCL experiment utilities.

This is intentionally not the BFCL CLI harness. It treats BFCL as data:
question + tool schema -> expected call(s), then runs Qwen directly with
transformers so we can later add hooks, finetuning, and circuit probes.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

BFCL_CANONICALIZATION_SYSTEM_PROMPT = """When making a tool call:
- output only one tool call and no conversational answer.
- include optional/default arguments when the schema implies them; use the schema default if given, otherwise use "" for unspecified optional string-like fields.
- preserve exact user-provided strings, casing, punctuation, ids, dates, and names.
- for math formulas, use Python-style syntax such as x**2, not x^2.
- for fields whose schema describes arrays/lists, output arrays even for one value.
"""


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} did not contain a json list")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def jsonable(value: Any) -> Any:
    if value is Ellipsis:
        return "..."
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    return value


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:
            pass
    return value


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    value = first_present(row, ("messages", "question", "prompt", "input", "query"))
    value = parse_maybe_json(value)
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        messages = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = item.get("content", item.get("message", ""))
                messages.append({"role": role, "content": str(content)})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages
    if value is None:
        raise ValueError(f"could not find prompt/messages in row keys: {sorted(row)}")
    return [{"role": "user", "content": str(value)}]


def normalize_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        out = {str(k): normalize_json_schema(v) for k, v in value.items()}
        if out.get("type") == "dict":
            out["type"] = "object"
        return out
    if isinstance(value, list):
        return [normalize_json_schema(v) for v in value]
    return value


def normalize_tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = first_present(
        row,
        ("tools", "function", "functions", "function_doc", "function_docs", "tool_schema"),
    )
    value = parse_maybe_json(value)
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"tools field is not list/dict: {type(value)}")

    tools = []
    for tool in value:
        tool = parse_maybe_json(tool)
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            tools.append(normalize_json_schema(tool))
        else:
            tools.append({"type": "function", "function": normalize_json_schema(tool)})
    return tools


def answer_key(row: dict[str, Any]) -> str:
    return str(first_present(row, ("id", "question_id", "test_category_id", "test_id")))


def load_answers(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    answers = {}
    for row in read_records(path):
        key = answer_key(row)
        answers[key] = first_present(
            row,
            ("answer", "answers", "ground_truth", "possible_answer", "function_call", "target"),
        )
    return answers


def expand_bfcl_ground_truth(target: Any) -> list[dict[str, Any]]:
    target = parse_maybe_json(target)
    if not isinstance(target, list):
        return canonical(target)

    calls = []
    for item in target:
        item = parse_maybe_json(item)
        if not isinstance(item, dict):
            continue
        if "name" in item and "arguments" in item:
            calls.append(canonical(item))
            continue
        for name, params in item.items():
            params = parse_maybe_json(params)
            if not isinstance(params, dict):
                calls.append({"name": name, "arguments": params})
                continue
            keys = list(params)
            value_lists = []
            for key in keys:
                values = parse_maybe_json(params[key])
                if not isinstance(values, list):
                    values = [values]
                elif not values:
                    values = [[]]
                value_lists.append(values)
            for vals in itertools.product(*value_lists):
                calls.append({"name": name, "arguments": dict(zip(keys, vals))})
    return [canonical(call) for call in calls]


def make_pairs(args: argparse.Namespace) -> None:
    answers = load_answers(args.answers)
    rows = []
    for row in read_records(args.questions):
        key = answer_key(row)
        target = answers.get(key)
        if target is None:
            target = first_present(
                row,
                ("answer", "answers", "ground_truth", "possible_answer", "function_call", "target"),
            )
        rows.append(
            {
                "id": key,
                "category": args.category,
                "messages": normalize_messages(row),
                "tools": normalize_tools(row),
                "target": parse_maybe_json(target),
                "reference_calls": expand_bfcl_ground_truth(target),
                "raw": row if args.keep_raw else None,
            }
        )
    if not args.keep_raw:
        for row in rows:
            row.pop("raw", None)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} pairs -> {args.output}")


BFCL_SIMPLE_QUESTIONS_URL = "https://raw.githubusercontent.com/ShishirPatil/gorilla/70b6a4a2144597b1f99d1f4d3185d35d7ee532a4/berkeley-function-call-leaderboard/data/BFCL_v3_simple.json"
BFCL_SIMPLE_ANSWERS_URL = "https://raw.githubusercontent.com/ShishirPatil/gorilla/70b6a4a2144597b1f99d1f4d3185d35d7ee532a4/berkeley-function-call-leaderboard/data/possible_answer/BFCL_v3_simple.json"
BFCL_RAW_HOST = "raw.githubusercontent.com"
BFCL_RAW_REPO_PATH = "ShishirPatil/gorilla/70b6a4a2144597b1f99d1f4d3185d35d7ee532a4/berkeley-function-call-leaderboard/data"
BFCL_SINGLE_CALL_FILES = (
    "BFCL_v3_simple.json",
    "BFCL_v3_live_simple.json",
    "BFCL_v3_exec_simple.json",
    "BFCL_v3_java.json",
    "BFCL_v3_javascript.json",
    "BFCL_v3_sql.json",
)


def bfcl_raw_url(*parts: str) -> str:
    suffix = "/".join(part.strip("/") for part in parts if part)
    return "https" + f"://{BFCL_RAW_HOST}/{BFCL_RAW_REPO_PATH}/{suffix}"


def download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def download_bfcl_simple(args: argparse.Namespace) -> None:
    questions = args.output_dir / "BFCL_v3_simple.json"
    answers = args.output_dir / "possible_answer" / "BFCL_v3_simple.json"
    download_url(BFCL_SIMPLE_QUESTIONS_URL, questions)
    download_url(BFCL_SIMPLE_ANSWERS_URL, answers)
    print(f"questions={questions}")
    print(f"answers={answers}")


def parse_function_invocation(value: Any) -> list[dict[str, Any]]:
    value = parse_maybe_json(value)
    if isinstance(value, list):
        calls = []
        for item in value:
            calls.extend(parse_function_invocation(item))
        return calls
    if not isinstance(value, str):
        return []
    tree = ast.parse(value.strip(), mode="eval")
    if not isinstance(tree.body, ast.Call):
        return []
    call = tree.body
    parts = []
    fn = call.func
    while isinstance(fn, ast.Attribute):
        parts.append(fn.attr)
        fn = fn.value
    if isinstance(fn, ast.Name):
        parts.append(fn.id)
    else:
        return []
    name = ".".join(reversed(parts))
    args = {kw.arg: canonical(ast.literal_eval(kw.value)) for kw in call.keywords if kw.arg}
    return [{"name": name, "arguments": args}]


def is_single_turn(row: dict[str, Any]) -> bool:
    question = row.get("question")
    return not (
        isinstance(question, list)
        and len(question) > 1
        and all(isinstance(item, list) for item in question)
    )


def target_call_count(target: Any) -> int:
    target = parse_maybe_json(target)
    if isinstance(target, list) and target and isinstance(target[0], str):
        return len(parse_function_invocation(target))
    if isinstance(target, list):
        # BFCL possible_answer stores one call as
        # [{"fn_name": {"arg": [allowed_variant, ...]}}]. The cartesian product
        # of allowed args may have many valid variants, but it is still one
        # function invocation target.
        return sum(1 for item in target if isinstance(parse_maybe_json(item), dict))
    return 0


def download_bfcl_single_call(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = {"files": {}, "filter": "single-turn rows with exactly one target call"}
    for filename in BFCL_SINGLE_CALL_FILES:
        q_path = args.output_dir / filename
        download_url(bfcl_raw_url(filename), q_path)
        questions = read_records(q_path)

        answers = {}
        a_path = args.output_dir / "possible_answer" / filename
        try:
            download_url(bfcl_raw_url("possible_answer", filename), a_path)
            answers = load_answers(a_path)
        except Exception:
            a_path = None

        kept = 0
        for row in questions:
            if not is_single_turn(row):
                continue
            key = answer_key(row)
            target = answers.get(key)
            if target is None:
                target = first_present(row, ("ground_truth", "answer", "target"))
            if target_call_count(target) != 1:
                continue
            out = {
                "id": key,
                "category": filename.removesuffix(".json").removeprefix("BFCL_v3_"),
                "messages": normalize_messages(row),
                "tools": normalize_tools(row),
                "target": parse_maybe_json(target),
                "reference_calls": parse_function_invocation(target)
                or expand_bfcl_ground_truth(target),
            }
            rows.append(out)
            kept += 1
        manifest["files"][filename] = {"raw": len(questions), "kept": kept}

    write_jsonl(args.output, rows)
    manifest["total"] = len(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def parse_tool_calls(text: str) -> list[Any]:
    calls = []
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        matches = extract_json_objects(text)
    if not matches:
        matches = [text]
    for match in matches:
        parsed = parse_maybe_json(match)
        if isinstance(parsed, list):
            calls.extend(parsed)
        else:
            calls.append(parsed)
    return calls


def extract_json_objects(text: str) -> list[str]:
    objects = []
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : i + 1])
                start = None
    return objects


def canonical(value: Any) -> Any:
    value = parse_maybe_json(value)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, set):
        return sorted(canonical(v) for v in value)
    return value


def maybe_number(value: str, target: Any) -> Any:
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        return value
    stripped = value.strip()
    try:
        if isinstance(target, int) and re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        if isinstance(target, float) and re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", stripped):
            return float(stripped)
    except Exception:
        return value
    return value


def strip_wrapping_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def normalize_string_against_target(value: str, target: Any) -> Any:
    value = strip_wrapping_quotes(value)
    value = maybe_number(value, target)
    if isinstance(value, str) and isinstance(target, str) and "**" in target and "^" in value:
        value = re.sub(r"(?<=\w)\s*\^\s*(?=[\w(+-])", "**", value)
    return value


def normalize_against_target(value: Any, target: Any) -> Any:
    value = parse_maybe_json(value)
    target = parse_maybe_json(target)
    if isinstance(value, dict) and isinstance(target, dict):
        return {
            str(key): normalize_against_target(value[key], target.get(key))
            for key in sorted(value)
        }
    if isinstance(value, list) and isinstance(target, list):
        if not target:
            return value
        if len(target) == 1:
            return [normalize_against_target(item, target[0]) for item in value]
        if len(value) == len(target):
            return [
                normalize_against_target(item, target_item)
                for item, target_item in zip(value, target)
            ]
        return [normalize_against_target(item, target[0]) for item in value]
    if not isinstance(value, list) and isinstance(target, list) and target:
        return [normalize_against_target(value, target[0])]
    if isinstance(value, str):
        return normalize_string_against_target(value, target)
    return value


def normalized_prediction_ok(prediction_calls: Any, row: dict[str, Any]) -> bool:
    pred = canonical(prediction_calls)
    for option in tool_call_options(row):
        norm_pred = canonical(normalize_against_target(pred, option))
        norm_target = canonical(normalize_against_target(option, option))
        if norm_pred == norm_target:
            return True
    return False


def tool_call_options(row: dict[str, Any]) -> list[Any]:
    refs = row.get("reference_calls")
    if refs:
        return [canonical([ref]) for ref in refs]
    target = row.get("target")
    expanded = expand_bfcl_ground_truth(target)
    if expanded:
        return [canonical([ref]) for ref in expanded]
    return [canonical(target)]


def prediction_ok(prediction_calls: Any, row: dict[str, Any]) -> bool:
    pred = canonical(prediction_calls)
    return any(pred == option for option in tool_call_options(row))


def messages_for_generation(row: dict[str, Any], *, bfcl_canonicalization_prompt: bool) -> list[dict[str, str]]:
    messages = list(row["messages"])
    if not bfcl_canonicalization_prompt:
        return messages
    if messages and messages[0].get("role") == "system":
        messages = [
            {
                "role": "system",
                "content": messages[0].get("content", "") + "\n\n" + BFCL_CANONICALIZATION_SYSTEM_PROMPT,
            }
        ] + messages[1:]
    else:
        messages = [{"role": "system", "content": BFCL_CANONICALIZATION_SYSTEM_PROMPT}] + messages
    return messages


def generate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
    )
    model.eval()

    out_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        encoded_items = [
            tokenizer.apply_chat_template(
                messages_for_generation(
                    row,
                    bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt,
                ),
                tools=row.get("tools") or None,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=args.enable_thinking,
            )
            for row in batch_rows
        ]
        encoded = tokenizer.pad(
            encoded_items,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
        with torch.inference_mode():
            output = model.generate(**encoded, **gen_kwargs)
        prompt_len = encoded["input_ids"].shape[-1]
        for row, seq in zip(batch_rows, output):
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            out_rows.append(
                {
                    "id": row["id"],
                    "category": row.get("category"),
                    "prediction_text": text,
                    "prediction_calls": parse_tool_calls(text),
                    "target": row.get("target"),
                    "reference_calls": row.get("reference_calls"),
                }
            )
        print(f"generated {len(out_rows)}/{len(rows)}", flush=True)
    write_jsonl(args.output, out_rows)
    print(f"wrote generations -> {args.output}")


def score_exact(args: argparse.Namespace) -> None:
    rows = read_records(args.generations)
    correct = 0
    normalized_correct = 0
    judged = 0
    failures = []
    for row in rows:
        prediction_calls = row.get("prediction_calls")
        if "prediction_text" in row:
            prediction_calls = parse_tool_calls(row["prediction_text"])
        pred = canonical(prediction_calls)
        target_options = tool_call_options(row)
        if not target_options or target_options == [[None]]:
            continue
        judged += 1
        ok = any(pred == target for target in target_options)
        normalized_ok = normalized_prediction_ok(prediction_calls, row)
        correct += int(ok)
        normalized_correct += int(normalized_ok)
        keep_failure = (not normalized_ok) if args.normalized else (not ok)
        if keep_failure and len(failures) < args.keep_failures:
            failures.append(
                {
                    "id": row.get("id"),
                    "prediction": pred,
                    "targets": target_options,
                    "raw_correct": ok,
                    "normalized_correct": normalized_ok,
                }
            )
    summary = {
        "generations": len(rows),
        "judged": judged,
        "exact_correct": correct,
        "exact_accuracy": correct / judged if judged else None,
        "normalized_exact_correct": normalized_correct,
        "normalized_exact_accuracy": normalized_correct / judged if judged else None,
        "reported_metric": "normalized_exact" if args.normalized else "exact",
        "note": "raw exact plus normalized strict structured match; use official BFCL scorer for final reporting",
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def format_tool_call_target(row: dict[str, Any]) -> str:
    refs = row.get("reference_calls") or expand_bfcl_ground_truth(row.get("target"))
    if not refs:
        raise ValueError(f"row {row.get('id')} has no reference call")
    return "<tool_call>\n" + json.dumps(refs[0], ensure_ascii=False) + "\n</tool_call>"


def encode_prompt(tokenizer, row: dict[str, Any], *, enable_thinking: bool):
    return tokenizer.apply_chat_template(
        row["messages"],
        tools=row.get("tools") or None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )


def build_attr_prompt_target(tokenizer, row: dict[str, Any], *, enable_thinking: bool):
    import torch

    prompt = encode_prompt(tokenizer, row, enable_thinking=enable_thinking)
    target_text = format_tool_call_target(row)
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    input_ids = torch.cat([prompt["input_ids"], target_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, int(prompt["input_ids"].shape[1]), target_ids


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def relp_attribute(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from src.circuit_tracing.relp import ReLPAttributor

    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args)
    attributor = ReLPAttributor(model, tokenizer, device=str(model.device))
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    scores = torch.zeros((n_layers, d_ffn), dtype=torch.float32)

    for i, row in enumerate(rows, start=1):
        input_ids, attention_mask, prompt_len, target_ids = build_attr_prompt_target(
            tokenizer, row, enable_thinking=args.enable_thinking
        )
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        target_ids = target_ids.to(model.device)
        answer_len = target_ids.shape[1]

        def metric_fn(logits, _prompt_len=prompt_len, _target_ids=target_ids):
            positions = torch.arange(
                _prompt_len - 1,
                _prompt_len - 1 + answer_len,
                device=logits.device,
            )
            logp = torch.log_softmax(logits[:, positions, :], dim=-1)
            gold = _target_ids[0].view(1, -1, 1)
            return logp.gather(2, gold).sum()

        attr = attributor.attribute(input_ids, lambda logits: metric_fn(logits))
        for layer, tensor in attr.items():
            scores[layer] += tensor.abs().sum(dim=(0, 1)).cpu()
        if i % args.log_every == 0:
            print(f"attributed {i}/{len(rows)}")

    scores /= max(len(rows), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        mlp_scores=scores.numpy(),
        model=args.model,
        examples=len(rows),
        objective="teacher_forced_gold_tool_call_logprob",
    )
    top = torch.topk(scores.flatten(), k=min(args.report_topk, scores.numel()))
    summary = {
        "examples": len(rows),
        "model": args.model,
        "objective": "teacher-forced gold tool-call logprob over full continuation",
        "scores": str(args.output),
        "top": [
            {
                "layer": int(idx.item() // d_ffn),
                "channel": int(idx.item() % d_ffn),
                "score": float(val.item()),
            }
            for val, idx in zip(top.values, top.indices)
        ],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def load_topk_mask(path: Path, k: int) -> dict[int, set[int]]:
    import numpy as np
    import torch

    scores = torch.tensor(np.load(path)["mlp_scores"])
    flat = scores.flatten()
    k = min(k, flat.numel())
    idx = torch.topk(flat, k=k).indices
    d_ffn = scores.shape[1]
    selected: dict[int, set[int]] = {}
    for item in idx.tolist():
        layer = item // d_ffn
        channel = item % d_ffn
        selected.setdefault(layer, set()).add(channel)
    return selected


def install_mlp_keep_hooks(model, selected: dict[int, set[int]]):
    import torch

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        keep = selected.get(layer_idx, set())
        keep_idx = torch.tensor(sorted(keep), dtype=torch.long)

        def hook(module, args, _keep_idx=keep_idx):
            x = args[0]
            if _keep_idx.numel() == 0:
                return (torch.zeros_like(x),)
            keep_device = _keep_idx.to(x.device)
            y = torch.zeros_like(x)
            y.index_copy_(-1, keep_device, x.index_select(-1, keep_device))
            return (y,)

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    return hooks


def eval_mask(args: argparse.Namespace) -> None:
    import torch

    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected = load_topk_mask(args.attribution, args.topk) if args.topk else {}
    hooks = install_mlp_keep_hooks(model, selected) if args.topk else []
    out_rows = []
    try:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            encoded_items = [
                tokenizer.apply_chat_template(
                    messages_for_generation(
                        row,
                        bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt,
                    ),
                    tools=row.get("tools") or None,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    enable_thinking=args.enable_thinking,
                )
                for row in batch_rows
            ]
            encoded = tokenizer.pad(
                encoded_items,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = encoded["input_ids"].shape[-1]
            for row, seq in zip(batch_rows, output):
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                pred = parse_tool_calls(text)
                raw_correct = prediction_ok(pred, row)
                normalized_correct = normalized_prediction_ok(pred, row)
                out_rows.append(
                    {
                        "id": row["id"],
                        "prediction_text": text,
                        "prediction_calls": pred,
                        "target": row.get("target"),
                        "reference_calls": row.get("reference_calls"),
                        "correct": normalized_correct if args.normalized else raw_correct,
                        "raw_correct": raw_correct,
                        "normalized_correct": normalized_correct,
                    }
                )
            print(f"evaluated {len(out_rows)}/{len(rows)}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    write_jsonl(args.output, out_rows)
    judged = len(out_rows)
    correct = sum(int(row["correct"]) for row in out_rows)
    raw_correct = sum(int(row["raw_correct"]) for row in out_rows)
    normalized_correct = sum(int(row["normalized_correct"]) for row in out_rows)
    summary = {
        "examples": judged,
        "exact_correct": correct,
        "exact_accuracy": correct / judged if judged else None,
        "raw_exact_correct": raw_correct,
        "raw_exact_accuracy": raw_correct / judged if judged else None,
        "normalized_exact_correct": normalized_correct,
        "normalized_exact_accuracy": normalized_correct / judged if judged else None,
        "reported_metric": "normalized_exact" if args.normalized else "raw_exact",
        "bfcl_canonicalization_prompt": args.bfcl_canonicalization_prompt,
        "mask_topk": args.topk or None,
        "attribution": str(args.attribution) if args.attribution else None,
        "generations": str(args.output),
        "note": "crude exact structured match against BFCL simple possible answers",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download-bfcl-simple")
    p.add_argument("--output-dir", type=Path, default=Path("data/bfcl"))
    p.set_defaults(func=download_bfcl_simple)

    p = sub.add_parser("download-bfcl-single-call")
    p.add_argument("--output-dir", type=Path, default=Path("data/bfcl_single_call"))
    p.add_argument("--output", type=Path, default=Path("data/bfcl_single_call/pairs.jsonl"))
    p.add_argument(
        "--manifest", type=Path, default=Path("data/bfcl_single_call/manifest.json")
    )
    p.set_defaults(func=download_bfcl_single_call)

    p = sub.add_parser("make-pairs")
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--answers", type=Path)
    p.add_argument("--category", default="bfcl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--keep-raw", action="store_true")
    p.set_defaults(func=make_pairs)

    p = sub.add_parser("generate")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.set_defaults(func=generate)

    p = sub.add_parser("score-exact")
    p.add_argument("--generations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--keep-failures", type=int, default=20)
    p.add_argument("--normalized", action="store_true")
    p.set_defaults(func=score_exact)

    p = sub.add_parser("relp-attribute")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--report-topk", type=int, default=20)
    p.set_defaults(func=relp_attribute)

    p = sub.add_parser("eval-mask")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attribution", type=Path)
    p.add_argument("--topk", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.add_argument("--normalized", action="store_true")
    p.set_defaults(func=eval_mask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
