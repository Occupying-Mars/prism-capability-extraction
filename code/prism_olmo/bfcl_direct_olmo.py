#!/usr/bin/env python3
"""OLMo BFCL base eval and ReLP attribution utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism_olmo.relp_olmo import OLMoReLPAttributor, olmo_d_ffn, olmo_layers, olmo_num_layers  # noqa: E402
from scripts.bfcl_direct_qwen3 import (  # noqa: E402
    format_tool_call_target,
    normalized_prediction_ok,
    parse_tool_calls,
    prediction_ok,
    read_records,
    write_jsonl,
)


def olmo_prompt_text(row: dict) -> str:
    parts = [
        "You are a function-calling assistant.",
        "Return exactly one tool call and no prose.",
        "Use this exact format:",
        '<tool_call>{"name":"function_name","arguments":{}}</tool_call>',
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


def first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def load_topk_mask(path: Path, k: int) -> torch.Tensor:
    import numpy as np

    scores = torch.tensor(np.load(path)["mlp_scores"])
    flat = scores.flatten()
    k = min(k, flat.numel())
    idx = torch.topk(flat, k=k).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.view_as(scores)


def layer_device(layer) -> torch.device:
    return next(layer.parameters()).device


def install_olmo_zero_keep_hooks(model, mask: torch.Tensor):
    hooks = []
    for layer_idx, layer in enumerate(olmo_layers(model)):
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

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(keep_idx)))
    return hooks


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def encode_prompt(tokenizer, row: dict):
    return tokenizer(olmo_prompt_text(row), add_special_tokens=True, return_tensors="pt")


def build_attr_prompt_target(tokenizer, row: dict):
    prompt = encode_prompt(tokenizer, row)
    target_text = format_tool_call_target(row)
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([prompt["input_ids"], target_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, int(prompt["input_ids"].shape[1]), target_ids


def eval_bfcl(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args)
    hooks = []
    if args.topk:
        if not args.attribution:
            raise ValueError("--attribution required with --topk")
        hooks = install_olmo_zero_keep_hooks(model, load_topk_mask(args.attribution, args.topk))

    out_rows = []
    try:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            prompts = [olmo_prompt_text(row) for row in batch_rows]
            encoded = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt").to(model.device)
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
        "prompt_format": "olmo_manual_tool_call",
        "target_format": "tool_call",
        "mask_mode": "zero" if args.topk else "none",
        "mask_topk": args.topk or None,
        "attribution": str(args.attribution) if args.attribution else None,
        "base": args.model,
        "generations": str(args.output),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def relp_attribute(args: argparse.Namespace) -> None:
    import numpy as np

    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args)
    device = first_param_device(model)
    attributor = OLMoReLPAttributor(model, tokenizer, device=str(device))
    n_layers = olmo_num_layers(model.config)
    d_ffn = olmo_d_ffn(model.config)
    scores = torch.zeros((n_layers, d_ffn), dtype=torch.float32)

    for i, row in enumerate(rows, start=1):
        input_ids, attention_mask, prompt_len, target_ids = build_attr_prompt_target(tokenizer, row)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_ids = target_ids.to(device)
        answer_len = target_ids.shape[1]

        def metric_fn(logits, _prompt_len=prompt_len, _target_ids=target_ids):
            positions = torch.arange(_prompt_len - 1, _prompt_len - 1 + answer_len, device=logits.device)
            logp = torch.log_softmax(logits[:, positions, :], dim=-1)
            gold = _target_ids[0].view(1, -1, 1)
            return logp.gather(2, gold).sum()

        attr = attributor.attribute(input_ids, attention_mask, metric_fn)
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
        objective="teacher_forced_gold_tool_call_logprob_olmo_mlp",
    )
    top = torch.topk(scores.flatten(), k=min(args.report_topk, scores.numel()))
    summary = {
        "examples": len(rows),
        "model": args.model,
        "objective": "teacher-forced gold tool-call logprob over full continuation; OLMo MLP ReLP",
        "prompt_format": "olmo_manual_tool_call",
        "target_format": "tool_call",
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
    p.add_argument("--model", default="allenai/OLMo-7B-hf")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--report-topk", type=int, default=20)
    p.set_defaults(func=relp_attribute)
    e = sub.add_parser("eval")
    e.add_argument("--name", required=True)
    e.add_argument("--pairs", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--model", default="allenai/OLMo-7B-hf")
    e.add_argument("--attribution", type=Path)
    e.add_argument("--topk", type=int, default=0)
    e.add_argument("--batch-size", type=int, default=4)
    e.add_argument("--max-new-tokens", type=int, default=256)
    e.add_argument("--dtype", default="bfloat16")
    e.add_argument("--device-map", default="auto")
    e.add_argument("--limit", type=int)
    e.set_defaults(func=eval_bfcl)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
