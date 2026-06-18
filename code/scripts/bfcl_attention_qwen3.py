#!/usr/bin/env python3
"""Qwen3 BFCL attention substrate attribution and keep-only evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_common.wandb_utils import add_wandb_args, init_wandb_run, log_wandb_receipts
from scripts.bfcl_direct_qwen3 import (
    build_attr_prompt_target,
    messages_for_generation,
    normalized_prediction_ok,
    parse_tool_calls,
    prediction_ok,
    read_records,
    write_jsonl,
)


FULL_QWEN3_BFCL_ANCHOR = 664


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{name!r} is not a torch dtype")
    return dtype


def input_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = args.tokenizer or args.adapter or args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=args.local_files_only)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=args.local_files_only)
        if args.merge_adapter:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def qwen_attention_shape(model) -> tuple[int, int, int, int]:
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(cfg.num_attention_heads)
    hidden = int(cfg.hidden_size)
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
    if n_heads * head_dim != hidden:
        raise ValueError(f"head shape mismatch: heads={n_heads} head_dim={head_dim} hidden={hidden}")
    return n_layers, n_heads, hidden, head_dim


def metric_from_gold(logits: torch.Tensor, prompt_len: int, target_ids: torch.Tensor) -> torch.Tensor:
    answer_len = target_ids.shape[1]
    positions = torch.arange(prompt_len - 1, prompt_len - 1 + answer_len, device=logits.device)
    logp = torch.log_softmax(logits[:, positions, :], dim=-1)
    gold = target_ids.to(logits.device)[0].view(1, -1, 1)
    return logp.gather(2, gold).sum()


def top_entries(scores: torch.Tensor, width: int, k: int, kind: str) -> list[dict[str, Any]]:
    flat = scores.flatten()
    top = torch.topk(flat, k=min(k, flat.numel()))
    out = []
    for val, idx in zip(top.values, top.indices):
        item = int(idx.item())
        layer = item // width
        unit = item % width
        key = "head" if kind == "head" else "channel"
        out.append({"layer": layer, key: unit, "score": float(val.item())})
    return out


def actgrad_attribute(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args)
    device = input_device(model)
    n_layers, n_heads, hidden, head_dim = qwen_attention_shape(model)
    head_scores = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    ov_scores = torch.zeros((n_layers, hidden), dtype=torch.float32)

    run = init_wandb_run(
        args,
        default_project="prism-bfcl-attention",
        default_group=args.run_name,
        default_job_type="attention-actgrad",
        default_name=args.run_name,
        default_tags="bfcl,qwen3,attention,actgrad",
        config={
            "cmd": "actgrad-attribute",
            "model": args.model,
            "adapter": args.adapter,
            "examples": len(rows),
            "objective": "teacher_forced_gold_tool_call_logprob",
            "shape": {
                "layers": n_layers,
                "heads": n_heads,
                "hidden": hidden,
                "head_dim": head_dim,
            },
        },
    )

    activations: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, hook_args):
            act = hook_args[0]
            act.retain_grad()
            activations[layer_idx] = act

        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_idx)))

    started = time.time()
    try:
        for i, row in enumerate(rows, start=1):
            activations.clear()
            input_ids, attention_mask, prompt_len, target_ids = build_attr_prompt_target(
                tokenizer, row, enable_thinking=args.enable_thinking
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            model.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            metric = metric_from_gold(logits, prompt_len, target_ids)
            metric.backward()

            for layer_idx, act in activations.items():
                grad = act.grad if act.grad is not None else torch.zeros_like(act)
                attr = (act * grad).detach().abs().float()
                ov_scores[layer_idx] += attr.sum(dim=(0, 1)).cpu()
                head_attr = attr.view(attr.shape[0], attr.shape[1], n_heads, head_dim)
                head_scores[layer_idx] += head_attr.sum(dim=(0, 1, 3)).cpu()

            if i % args.log_every == 0 or i == len(rows):
                elapsed = time.time() - started
                print(f"attributed {i}/{len(rows)} elapsed_s={elapsed:.1f}", flush=True)
                if run is not None:
                    run.log(
                        {
                            "attribution/examples": i,
                            "attribution/elapsed_s": elapsed,
                            "attribution/head_score_mass": float(head_scores.sum().item()),
                            "attribution/ov_score_mass": float(ov_scores.sum().item()),
                        },
                        step=i,
                    )
            del logits, metric
            if torch.cuda.is_available() and i % args.empty_cache_every == 0:
                torch.cuda.empty_cache()
    finally:
        for hook in hooks:
            hook.remove()

    denom = max(len(rows), 1)
    head_scores /= denom
    ov_scores /= denom
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        head_scores=head_scores.numpy(),
        ov_scores=ov_scores.numpy(),
        model=args.model,
        adapter=args.adapter or "",
        examples=len(rows),
        objective="teacher_forced_gold_tool_call_logprob",
        attribution="act_times_grad_o_proj_input",
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_size=hidden,
        head_dim=head_dim,
    )
    summary = {
        "run_name": args.run_name,
        "examples": len(rows),
        "model": args.model,
        "adapter": args.adapter,
        "scores": str(args.output),
        "objective": "teacher-forced gold tool-call continuation logprob",
        "attribution": "act x grad on self_attn.o_proj input; heads are sums over their OV channel block",
        "shape": {"layers": n_layers, "heads": n_heads, "hidden": hidden, "head_dim": head_dim},
        "top_heads": top_entries(head_scores, n_heads, args.report_topk, "head"),
        "top_ov_channels": top_entries(ov_scores, hidden, args.report_topk, "ov"),
        "elapsed_s": time.time() - started,
    }
    summary_path = args.output.with_suffix(".summary.json")
    json_dump(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if run is not None:
        run.log(
            {
                "attribution/final_examples": len(rows),
                "attribution/final_head_score_mass": float(head_scores.sum().item()),
                "attribution/final_ov_score_mass": float(ov_scores.sum().item()),
            }
        )
        log_wandb_receipts(run, name=f"{args.run_name}-attention-actgrad", files=[args.output, summary_path])
        run.finish()


def build_mean_cache(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args)
    device = input_device(model)
    n_layers, _n_heads, hidden, _head_dim = qwen_attention_shape(model)
    sums = [torch.zeros(hidden, dtype=torch.float64) for _ in range(n_layers)]
    counts = [0 for _ in range(n_layers)]
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, hook_args):
            act = hook_args[0].detach().float().reshape(-1, hidden).cpu()
            sums[layer_idx] += act.sum(dim=0, dtype=torch.float64)
            counts[layer_idx] += int(act.shape[0])

        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_idx)))

    run = init_wandb_run(
        args,
        default_project="prism-bfcl-attention",
        default_group=args.run_name,
        default_job_type="attention-mean-cache",
        default_name=args.run_name,
        default_tags="bfcl,qwen3,attention,mean-cache",
        config={"cmd": "build-mean-cache", "model": args.model, "adapter": args.adapter, "examples": len(rows)},
    )

    started = time.time()
    try:
        with torch.inference_mode():
            for i, row in enumerate(rows, start=1):
                input_ids, attention_mask, _prompt_len, _target_ids = build_attr_prompt_target(
                    tokenizer, row, enable_thinking=args.enable_thinking
                )
                model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                if i % args.log_every == 0 or i == len(rows):
                    elapsed = time.time() - started
                    print(f"mean-cache {i}/{len(rows)} elapsed_s={elapsed:.1f}", flush=True)
                    if run is not None:
                        run.log({"mean_cache/examples": i, "mean_cache/elapsed_s": elapsed}, step=i)
    finally:
        for hook in hooks:
            hook.remove()

    means = torch.stack(
        [
            (sums[layer_idx] / max(counts[layer_idx], 1)).to(dtype=torch.float32)
            for layer_idx in range(n_layers)
        ],
        dim=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        means=means.numpy(),
        counts=np.array(counts, dtype=np.int64),
        model=args.model,
        adapter=args.adapter or "",
        examples=len(rows),
        activation="self_attn.o_proj_input",
    )
    summary = {
        "run_name": args.run_name,
        "examples": len(rows),
        "model": args.model,
        "adapter": args.adapter,
        "means": str(args.output),
        "activation": "self_attn.o_proj input",
        "counts_min": min(counts) if counts else 0,
        "counts_max": max(counts) if counts else 0,
        "elapsed_s": time.time() - started,
    }
    summary_path = args.output.with_suffix(".summary.json")
    json_dump(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if run is not None:
        log_wandb_receipts(run, name=f"{args.run_name}-attention-mean-cache", files=[args.output, summary_path])
        run.finish()


def load_attention_scores(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    return torch.tensor(data["head_scores"], dtype=torch.float32), torch.tensor(data["ov_scores"], dtype=torch.float32)


def make_keep_mask(
    *,
    head_scores: torch.Tensor,
    ov_scores: torch.Tensor,
    unit: str,
    topk: int,
    random_seed: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n_layers, n_heads = head_scores.shape
    hidden = ov_scores.shape[1]
    head_dim = hidden // n_heads
    keep = torch.zeros((n_layers, hidden), dtype=torch.bool)
    rng = random.Random(random_seed) if random_seed is not None else None

    if unit == "head":
        total = n_layers * n_heads
        k = min(topk, total)
        if rng is None:
            indices = torch.topk(head_scores.flatten(), k=k).indices.tolist()
        else:
            indices = rng.sample(range(total), k)
        for item in indices:
            layer = item // n_heads
            head = item % n_heads
            keep[layer, head * head_dim : (head + 1) * head_dim] = True
        return keep, {"unit": "head", "requested_topk": topk, "kept_heads": k, "kept_ov_channels": int(keep.sum().item())}

    if unit == "ov":
        total = ov_scores.numel()
        k = min(topk, total)
        if rng is None:
            indices = torch.topk(ov_scores.flatten(), k=k).indices.tolist()
        else:
            indices = rng.sample(range(total), k)
        for item in indices:
            keep[item // hidden, item % hidden] = True
        kept_heads_touched = int(sum(keep[layer].view(n_heads, head_dim).any(dim=1).sum().item() for layer in range(n_layers)))
        return keep, {
            "unit": "ov",
            "requested_topk": topk,
            "kept_heads_touched": kept_heads_touched,
            "kept_ov_channels": int(keep.sum().item()),
        }

    raise ValueError(f"unknown unit: {unit}")


def load_means(path: Path | None, expected_shape: tuple[int, int]) -> torch.Tensor | None:
    if path is None:
        return None
    means = torch.tensor(np.load(path)["means"], dtype=torch.float32)
    if tuple(means.shape) != expected_shape:
        raise ValueError(f"mean cache shape {tuple(means.shape)} != expected {expected_shape}")
    return means


def install_attention_keep_hooks(
    model,
    keep: torch.Tensor,
    *,
    ablation: str,
    means: torch.Tensor | None,
):
    hooks = []
    n_layers, hidden = keep.shape
    if ablation == "mean" and means is None:
        raise ValueError("mean ablation requires --mean-cache")

    for layer_idx, layer in enumerate(model.model.layers):
        layer_keep = keep[layer_idx].clone()
        layer_mean = means[layer_idx].clone() if means is not None else None

        def hook(_module, hook_args, _keep=layer_keep, _mean=layer_mean):
            x = hook_args[0]
            keep_dev = _keep.to(device=x.device).view(1, 1, hidden)
            if ablation == "zero":
                replacement = torch.zeros_like(x)
            else:
                replacement = _mean.to(device=x.device, dtype=x.dtype).view(1, 1, hidden).expand_as(x)
            y = torch.where(keep_dev, x, replacement)
            return (y,) + hook_args[1:]

        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(hook))
    if len(hooks) != n_layers:
        raise RuntimeError(f"installed {len(hooks)} hooks for {n_layers} layers")
    return hooks


def summarize_eval_rows(rows: list[dict[str, Any]], pairs_by_id: dict[str, dict[str, Any]], normalized: bool) -> dict[str, Any]:
    judged = len(rows)
    key = "normalized_correct" if normalized else "raw_correct"
    correct = sum(int(row.get(key, False)) for row in rows)
    raw_correct = sum(int(row.get("raw_correct", False)) for row in rows)
    normalized_correct = sum(int(row.get("normalized_correct", False)) for row in rows)
    by_category: dict[str, dict[str, Any]] = {}
    counts: defaultdict[str, Counter] = defaultdict(Counter)
    for row in rows:
        category = pairs_by_id.get(row["id"], {}).get("category", "unknown")
        counts[category]["examples"] += 1
        counts[category]["raw_correct"] += int(row.get("raw_correct", False))
        counts[category]["normalized_correct"] += int(row.get("normalized_correct", False))
    for category, counter in sorted(counts.items()):
        examples = counter["examples"]
        by_category[category] = {
            "examples": examples,
            "raw_exact_correct": counter["raw_correct"],
            "raw_exact_accuracy": counter["raw_correct"] / examples if examples else None,
            "normalized_exact_correct": counter["normalized_correct"],
            "normalized_exact_accuracy": counter["normalized_correct"] / examples if examples else None,
        }
    return {
        "examples": judged,
        "exact_correct": correct,
        "exact_accuracy": correct / judged if judged else None,
        "raw_exact_correct": raw_correct,
        "raw_exact_accuracy": raw_correct / judged if judged else None,
        "normalized_exact_correct": normalized_correct,
        "normalized_exact_accuracy": normalized_correct / judged if judged else None,
        "reported_metric": "normalized_exact" if normalized else "raw_exact",
        "recovery_vs_full_anchor": (correct / FULL_QWEN3_BFCL_ANCHOR) if FULL_QWEN3_BFCL_ANCHOR else None,
        "full_anchor_normalized_exact": FULL_QWEN3_BFCL_ANCHOR,
        "by_category": by_category,
    }


def evaluate_once(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    pairs_by_id: dict[str, dict[str, Any]],
    model,
    tokenizer,
    output: Path,
) -> dict[str, Any]:
    device = input_device(model)
    out_rows = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            encoded_items = [
                tokenizer.apply_chat_template(
                    messages_for_generation(row, bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt),
                    tools=row.get("tools") or None,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    enable_thinking=args.enable_thinking,
                )
                for row in batch_rows
            ]
            encoded = tokenizer.pad(encoded_items, padding=True, return_tensors="pt").to(device)
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = encoded["input_ids"].shape[-1]
            for pair, seq in zip(batch_rows, output_ids):
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                pred = parse_tool_calls(text)
                raw_correct = prediction_ok(pred, pair)
                normalized_correct = normalized_prediction_ok(pred, pair)
                out_rows.append(
                    {
                        "id": pair["id"],
                        "category": pair.get("category"),
                        "prediction_text": text,
                        "prediction_calls": pred,
                        "target": pair.get("target"),
                        "reference_calls": pair.get("reference_calls"),
                        "correct": normalized_correct if args.normalized else raw_correct,
                        "raw_correct": raw_correct,
                        "normalized_correct": normalized_correct,
                    }
                )
            print(f"{output.stem}: evaluated {len(out_rows)}/{len(rows)}", flush=True)
    write_jsonl(output, out_rows)
    return summarize_eval_rows(out_rows, pairs_by_id, args.normalized)


def eval_ladder(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    pairs_by_id = {row["id"]: row for row in rows}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    _n_layers, _n_heads, hidden, _head_dim = qwen_attention_shape(model)
    head_scores, ov_scores = load_attention_scores(args.attribution) if args.attribution else (None, None)
    means = load_means(args.mean_cache, (model.config.num_hidden_layers, hidden))
    topks = parse_int_list(args.topks)
    if args.include_unmasked:
        topks = [0] + topks

    run = init_wandb_run(
        args,
        default_project="prism-bfcl-attention",
        default_group=args.run_name,
        default_job_type="attention-eval",
        default_name=args.run_name,
        default_tags=f"bfcl,qwen3,attention,eval,{args.unit},{args.ablation}",
        config={
            "cmd": "eval-ladder",
            "model": args.model,
            "adapter": args.adapter,
            "unit": args.unit,
            "topks": topks,
            "ablation": args.ablation,
            "attribution": str(args.attribution) if args.attribution else None,
            "mean_cache": str(args.mean_cache) if args.mean_cache else None,
            "random_seed": args.random_seed,
        },
    )

    aggregate = {
        "run_name": args.run_name,
        "model": args.model,
        "adapter": args.adapter,
        "pairs": str(args.pairs),
        "unit": args.unit,
        "ablation": args.ablation,
        "attribution": str(args.attribution) if args.attribution else None,
        "mean_cache": str(args.mean_cache) if args.mean_cache else None,
        "random_seed": args.random_seed,
        "results": [],
    }
    receipt_files: list[Path] = []
    for topk in topks:
        hooks = []
        mask_info: dict[str, Any]
        if topk == 0:
            name = "unmasked"
            mask_info = {"unit": "none", "requested_topk": 0, "kept_ov_channels": hidden * model.config.num_hidden_layers}
        else:
            if head_scores is None or ov_scores is None:
                raise ValueError("--attribution is required for masked eval")
            keep, mask_info = make_keep_mask(
                head_scores=head_scores,
                ov_scores=ov_scores,
                unit=args.unit,
                topk=topk,
                random_seed=args.random_seed,
            )
            hooks = install_attention_keep_hooks(model, keep, ablation=args.ablation, means=means)
            random_tag = f"_random{args.random_seed}" if args.random_seed is not None else ""
            name = f"{args.unit}_{args.ablation}_k{topk}{random_tag}"
        try:
            jsonl_path = output_dir / f"{name}.jsonl"
            summary = evaluate_once(
                args=args,
                rows=rows,
                pairs_by_id=pairs_by_id,
                model=model,
                tokenizer=tokenizer,
                output=jsonl_path,
            )
        finally:
            for hook in hooks:
                hook.remove()
        summary.update(
            {
                "name": name,
                "mask": mask_info,
                "generations": str(jsonl_path),
                "summary": str(jsonl_path.with_suffix(".summary.json")),
            }
        )
        json_dump(jsonl_path.with_suffix(".summary.json"), summary)
        receipt_files.extend([jsonl_path, jsonl_path.with_suffix(".summary.json")])
        aggregate["results"].append(summary)
        if run is not None:
            run.log(
                {
                    "eval/topk": topk,
                    "eval/exact_correct": summary["exact_correct"],
                    "eval/exact_accuracy": summary["exact_accuracy"],
                    "eval/raw_exact_correct": summary["raw_exact_correct"],
                    "eval/normalized_exact_correct": summary["normalized_exact_correct"],
                    "eval/recovery_vs_full_anchor": summary["recovery_vs_full_anchor"],
                    "eval/kept_ov_channels": mask_info.get("kept_ov_channels"),
                    "eval/kept_heads": mask_info.get("kept_heads"),
                },
                step=max(topk, len(aggregate["results"])),
            )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    aggregate_path = output_dir / "ladder_summary.json"
    json_dump(aggregate_path, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if run is not None:
        log_wandb_receipts(run, name=f"{args.run_name}-attention-eval", files=[aggregate_path, *receipt_files])
        run.finish()


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokenizer")
    parser.add_argument("--adapter")
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("actgrad-attribute")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-name", default="issue3-attn-actgrad")
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--report-topk", type=int, default=30)
    p.add_argument("--empty-cache-every", type=int, default=25)
    add_common_model_args(p)
    add_wandb_args(p, default_project="prism-bfcl-attention", default_job_type="attention-actgrad")
    p.set_defaults(func=actgrad_attribute)

    p = sub.add_parser("build-mean-cache")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-name", default="issue3-attn-mean-cache")
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    add_common_model_args(p)
    add_wandb_args(p, default_project="prism-bfcl-attention", default_job_type="attention-mean-cache")
    p.set_defaults(func=build_mean_cache)

    p = sub.add_parser("eval-ladder")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--attribution", type=Path)
    p.add_argument("--mean-cache", type=Path)
    p.add_argument("--run-name", default="issue3-attn-eval")
    p.add_argument("--unit", choices=["head", "ov"], default="head")
    p.add_argument("--topks", default="8,16,32,64,128,256,512,1024")
    p.add_argument("--ablation", choices=["zero", "mean"], default="zero")
    p.add_argument("--random-seed", type=int)
    p.add_argument("--include-unmasked", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.add_argument("--normalized", action="store_true")
    add_common_model_args(p)
    add_wandb_args(p, default_project="prism-bfcl-attention", default_job_type="attention-eval")
    p.set_defaults(func=eval_ladder)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
