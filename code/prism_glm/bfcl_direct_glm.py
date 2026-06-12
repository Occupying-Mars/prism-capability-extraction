#!/usr/bin/env python3
"""GLM-specific BFCL attribution/eval utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if hasattr(torch, "distributed") and not hasattr(torch.distributed, "tensor"):
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peft import PeftModel  # noqa: E402
from scripts.bfcl_direct_qwen3 import (  # noqa: E402
    format_tool_call_target,
    normalized_prediction_ok,
    parse_tool_calls,
    prediction_ok,
    read_records,
    write_jsonl,
)
from scripts.train_bfcl_masked_lora import load_topk_mask  # noqa: E402
from prism_glm.relp_glm import GLMReLPAttributor, glm_d_ffn, glm_num_layers  # noqa: E402
from prism_glm.relp_glm import glm_decoder_root  # noqa: E402


def _call_from_row(row: dict) -> dict | None:
    call = row.get("target_call")
    if isinstance(call, dict):
        return call
    refs = row.get("reference_calls") or []
    if refs and isinstance(refs[0], dict):
        return refs[0]
    return None


def format_glm_native_target(row: dict) -> str:
    """GLM-4-Chat native tool call: first line name, then JSON arguments."""
    call = _call_from_row(row)
    if not call:
        raise ValueError(f"row {row.get('id')} has no target call")
    name = call.get("name")
    arguments = call.get("arguments") or {}
    return f"{name}\n{json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))}"


def _with_metadata(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        item = dict(msg)
        item.setdefault("metadata", "")
        out.append(item)
    return out


def glm_chat_messages(row: dict) -> list[dict]:
    messages = []
    tools = row.get("tools") or []
    if tools:
        messages.append({"role": "system", "content": None, "metadata": "", "tools": tools})
    messages.extend(_with_metadata(row["messages"]))
    return messages


def glm_prompt_text(row: dict) -> str:
    parts = [
        "You are a function-calling assistant. Return exactly one tool call.",
        "Use this format only: <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>",
    ]
    if row.get("tools"):
        parts.append("Available tools:\n" + json.dumps(row["tools"], ensure_ascii=False))
    parts.append("Conversation:")
    for msg in row["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n\n".join(parts)


def glm_native_prompt_text(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        glm_chat_messages(row),
        add_generation_prompt=True,
        tokenize=False,
    )


def parse_glm_native_tool_calls(text: str) -> list[dict]:
    stripped = text.strip()
    if not stripped:
        return []
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 2:
        name = lines[0]
        json_text = "\n".join(lines[1:])
        try:
            args = json.loads(json_text)
        except Exception:
            args = None
        if isinstance(args, dict):
            return [{"name": name, "arguments": args}]
    return parse_tool_calls(stripped)


def first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def encode_prompt(tokenizer, row: dict, *, prompt_format: str):
    if prompt_format == "glm_native":
        return tokenizer(glm_native_prompt_text(tokenizer, row), add_special_tokens=False, return_tensors="pt")
    return tokenizer(glm_prompt_text(row), add_special_tokens=True, return_tensors="pt")


def build_attr_prompt_target(tokenizer, row: dict, *, prompt_format: str, target_format: str):
    prompt = encode_prompt(tokenizer, row, prompt_format=prompt_format)
    target_text = format_glm_native_target(row) if target_format == "glm_native" else format_tool_call_target(row)
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([prompt["input_ids"], target_ids], dim=1)
    return input_ids, int(prompt["input_ids"].shape[1]), target_ids


def layer_device(layer) -> torch.device:
    return next(layer.parameters()).device


def install_glm_zero_keep_hooks(model, mask):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hooks = []
    for layer_idx, layer in enumerate(glm_decoder_root(base).layers):
        keep = mask[layer_idx].to(device=layer_device(layer), dtype=torch.bool)
        keep_idx = torch.where(keep)[0].to(torch.long)

        def make_hook(idx: torch.Tensor):
            def hook_fn(module, hook_args):
                act = hook_args[0]
                if idx.numel() == 0:
                    return (torch.zeros_like(act),) + hook_args[1:]
                idx_device = idx.to(act.device)
                out = torch.zeros_like(act)
                out.index_copy_(-1, idx_device, act.index_select(-1, idx_device))
                return (out,) + hook_args[1:]

            return hook_fn

        hooks.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(make_hook(keep_idx)))
    return hooks


def load_glm_model(base_path: str, adapter_path: str | None, dtype: torch.dtype):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
    if not hasattr(config, "max_length"):
        config.max_length = getattr(config, "seq_length", 8192)
    tokenizer_path = adapter_path or base_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, adapter_path).to("cuda") if adapter_path else base
    model.eval()
    return model, tokenizer


def eval_bfcl(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_glm_model(args.model, args.adapter, dtype)
    hooks = []
    if args.topk:
        if not args.attribution:
            raise ValueError("--attribution required with --topk")
        hooks = install_glm_zero_keep_hooks(model, load_topk_mask(args.attribution, args.topk))

    out_rows = []
    try:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            prompts = [glm_native_prompt_text(tokenizer, row) if args.prompt_format == "glm_native" else glm_prompt_text(row) for row in batch_rows]
            encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
            prompt_len = encoded["input_ids"].shape[-1]
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row, seq in zip(batch_rows, output):
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                pred = parse_glm_native_tool_calls(text) if args.target_format == "glm_native" else parse_tool_calls(text)
                raw_correct = prediction_ok(pred, row)
                normalized_correct = normalized_prediction_ok(pred, row)
                out_rows.append(
                    {
                        "id": row["id"],
                        "prediction_text": text,
                        "prediction_calls": pred,
                        "target": row.get("target"),
                        "reference_calls": row.get("reference_calls"),
                        "correct": normalized_correct,
                        "raw_correct": raw_correct,
                        "normalized_correct": normalized_correct,
                    }
                )
            print(f"{args.name}: evaluated {len(out_rows)}/{len(rows)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    write_jsonl(args.output, out_rows)
    judged = len(out_rows)
    raw = sum(int(row["raw_correct"]) for row in out_rows)
    norm = sum(int(row["normalized_correct"]) for row in out_rows)
    summary = {
        "name": args.name,
        "examples": judged,
        "raw_exact_correct": raw,
        "raw_exact_accuracy": raw / judged if judged else None,
        "normalized_exact_correct": norm,
        "normalized_exact_accuracy": norm / judged if judged else None,
        "reported_metric": "normalized_exact",
        "prompt_format": args.prompt_format,
        "target_format": args.target_format,
        "mask_mode": "zero" if args.topk else "none",
        "mask_topk": args.topk or None,
        "attribution": str(args.attribution) if args.attribution else None,
        "base": args.model,
        "adapter": args.adapter,
        "generations": str(args.output),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def relp_attribute(args: argparse.Namespace) -> None:
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    device = first_param_device(model)
    attributor = GLMReLPAttributor(model, tokenizer, device=str(device))
    n_layers = glm_num_layers(model.config)
    d_ffn = glm_d_ffn(model.config)
    scores = torch.zeros((n_layers, d_ffn), dtype=torch.float32)

    for i, row in enumerate(rows, start=1):
        input_ids, prompt_len, target_ids = build_attr_prompt_target(tokenizer, row, prompt_format=args.prompt_format, target_format=args.target_format)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        answer_len = target_ids.shape[1]

        def metric_fn(logits, _prompt_len=prompt_len, _target_ids=target_ids):
            positions = torch.arange(_prompt_len - 1, _prompt_len - 1 + answer_len, device=logits.device)
            logp = torch.log_softmax(logits[:, positions, :], dim=-1)
            gold = _target_ids[0].view(1, -1, 1)
            return logp.gather(2, gold).sum()

        attr = attributor.attribute(input_ids, metric_fn)
        for layer, tensor in attr.items():
            scores[layer] += tensor.abs().sum(dim=tuple(range(tensor.ndim - 1))).cpu()
        if i % args.log_every == 0:
            print(f"attributed {i}/{len(rows)}", flush=True)

    scores /= max(len(rows), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        mlp_scores=scores.numpy(),
        model=args.model,
        examples=len(rows),
        objective=f"teacher_forced_gold_{args.target_format}_logprob_glm_mlp",
    )
    top = torch.topk(scores.flatten(), k=min(args.report_topk, scores.numel()))
    summary = {
        "examples": len(rows),
        "model": args.model,
        "objective": f"teacher-forced gold {args.target_format} logprob over full continuation; GLM MLP ReLP",
        "prompt_format": args.prompt_format,
        "target_format": args.target_format,
        "scores": str(args.output),
        "shape": [n_layers, d_ffn],
        "top": [
            {"layer": int(idx.item() // d_ffn), "channel": int(idx.item() % d_ffn), "score": float(val.item())}
            for val, idx in zip(top.values, top.indices)
        ],
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("relp-attribute")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="zai-org/glm-4-9b-chat")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--prompt-format", choices=["manual_tool_call", "glm_native"], default="manual_tool_call")
    p.add_argument("--target-format", choices=["tool_call", "glm_native"], default="tool_call")
    p.add_argument("--report-topk", type=int, default=20)
    p.set_defaults(func=relp_attribute)
    e = sub.add_parser("eval")
    e.add_argument("--name", required=True)
    e.add_argument("--pairs", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--model", default="zai-org/glm-4-9b-chat")
    e.add_argument("--adapter")
    e.add_argument("--attribution", type=Path)
    e.add_argument("--topk", type=int, default=0)
    e.add_argument("--prompt-format", choices=["manual_tool_call", "glm_native"], default="glm_native")
    e.add_argument("--target-format", choices=["tool_call", "glm_native"], default="glm_native")
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--max-new-tokens", type=int, default=256)
    e.add_argument("--dtype", default="bfloat16")
    e.add_argument("--limit", type=int)
    e.set_defaults(func=eval_bfcl)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
