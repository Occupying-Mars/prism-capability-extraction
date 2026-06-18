#!/usr/bin/env python3
"""Qwen3 BFCL attention substrate attribution and keep-only evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from math import ceil
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


def load_resume_summary(enabled: bool, jsonl_path: Path) -> dict[str, Any] | None:
    summary_path = jsonl_path.with_suffix(".summary.json")
    if not enabled:
        return None
    if not summary_path.exists() or not jsonl_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    print(f"resume: using existing {summary_path}", flush=True)
    return summary


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


def decoder_root(model):
    cur = model
    for _ in range(8):
        if hasattr(cur, "layers"):
            return cur
        for attr in ("model", "base_model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    raise AttributeError("could not locate decoder .layers")


def decoder_layers(model):
    return list(decoder_root(model).layers)


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


def pick_indices(
    scores: torch.Tensor,
    *,
    k: int,
    random_seed: int | None,
    already_chosen: set[int] | None = None,
) -> list[int]:
    total = scores.numel()
    if k <= 0 or total == 0:
        return []
    chosen = already_chosen or set()
    flat = scores.flatten()
    candidates = [idx for idx in range(total) if idx not in chosen and bool(torch.isfinite(flat[idx]).item())]
    if not candidates:
        return []
    k = min(k, len(candidates))
    if random_seed is not None:
        return random.Random(random_seed).sample(candidates, k)

    masked = flat.clone()
    if chosen:
        masked[list(chosen)] = -torch.inf
    return torch.topk(masked, k=k).indices.tolist()


def pick_layer_balanced_indices(
    scores: torch.Tensor,
    *,
    k: int,
    layer_floor: int,
    random_seed: int | None,
) -> list[int]:
    n_layers, width = scores.shape
    if k <= 0:
        return []
    if layer_floor <= 0:
        return pick_indices(scores, k=k, random_seed=random_seed)

    rng = random.Random(random_seed) if random_seed is not None else None
    chosen: set[int] = set()
    floor = min(layer_floor, width)
    for layer in range(n_layers):
        remaining = k - len(chosen)
        if remaining <= 0:
            break
        take = min(floor, width, remaining)
        if rng is None:
            layer_items = torch.topk(scores[layer], k=take).indices.tolist()
        else:
            layer_items = rng.sample(range(width), take)
        chosen.update(layer * width + item for item in layer_items)

    remaining = k - len(chosen)
    if remaining > 0:
        chosen.update(pick_indices(scores, k=remaining, random_seed=random_seed, already_chosen=chosen))
    return list(chosen)


def actgrad_attribute(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args)
    device = input_device(model)
    n_layers, n_heads, hidden, head_dim = qwen_attention_shape(model)
    mlp_keep, mlp_info = load_mlp_keep_mask(args.mlp_attribution, args.mlp_topk, (n_layers, int(model.config.intermediate_size)))
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
            "mlp_mask": mlp_info,
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
            replacement_args = None
            if not act.requires_grad:
                act = act.detach().requires_grad_(True)
                replacement_args = (act,) + hook_args[1:]
            act.retain_grad()
            activations[layer_idx] = act
            return replacement_args

        return hook

    for layer_idx, layer in enumerate(decoder_layers(model)):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_idx)))
    mlp_hooks = install_mlp_keep_hooks(model, mlp_keep)

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
        for hook in mlp_hooks:
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
        "mlp_mask": mlp_info,
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
    mlp_keep, mlp_info = load_mlp_keep_mask(
        args.mlp_attribution,
        args.mlp_topk,
        (n_layers, int(model.config.intermediate_size)),
    )
    sums = [torch.zeros(hidden, dtype=torch.float64) for _ in range(n_layers)]
    counts = [0 for _ in range(n_layers)]
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, hook_args):
            act = hook_args[0].detach().float().reshape(-1, hidden).cpu()
            sums[layer_idx] += act.sum(dim=0, dtype=torch.float64)
            counts[layer_idx] += int(act.shape[0])

        return hook

    for layer_idx, layer in enumerate(decoder_layers(model)):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_idx)))
    mlp_hooks = install_mlp_keep_hooks(model, mlp_keep)

    run = init_wandb_run(
        args,
        default_project="prism-bfcl-attention",
        default_group=args.run_name,
        default_job_type="attention-mean-cache",
        default_name=args.run_name,
        default_tags="bfcl,qwen3,attention,mean-cache",
        config={
            "cmd": "build-mean-cache",
            "model": args.model,
            "adapter": args.adapter,
            "examples": len(rows),
            "mlp_mask": mlp_info,
        },
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
        for hook in mlp_hooks:
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
        mlp_mask=json.dumps(mlp_info, sort_keys=True) if mlp_info else "",
    )
    summary = {
        "run_name": args.run_name,
        "examples": len(rows),
        "model": args.model,
        "adapter": args.adapter,
        "means": str(args.output),
        "activation": "self_attn.o_proj input",
        "mlp_mask": mlp_info,
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


def load_mlp_keep_mask(path: Path | None, topk: int | None, expected_shape: tuple[int, int]) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    data = np.load(path)
    if "mlp_keep" in data.files:
        keep = torch.tensor(data["mlp_keep"], dtype=torch.bool)
        source_key = "mlp_keep"
    elif "mask" in data.files and data["mask"].dtype == np.bool_:
        keep = torch.tensor(data["mask"], dtype=torch.bool)
        source_key = "mask"
    elif "mlp_scores" in data.files:
        scores = torch.tensor(data["mlp_scores"], dtype=torch.float32)
        if tuple(scores.shape) != expected_shape:
            raise ValueError(f"mlp score shape {tuple(scores.shape)} != expected {expected_shape}")
        if topk is None:
            raise ValueError("--mlp-topk is required when --mlp-attribution contains mlp_scores")
        flat = scores.flatten()
        k = min(int(topk), flat.numel())
        idx = torch.topk(flat, k=k).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        d_ffn = scores.shape[1]
        for item in idx.tolist():
            keep[item // d_ffn, item % d_ffn] = True
        source_key = "mlp_scores"
    else:
        raise ValueError(f"{path} has no supported MLP mask key; keys={data.files}")
    if tuple(keep.shape) != expected_shape:
        raise ValueError(f"mlp mask shape {tuple(keep.shape)} != expected {expected_shape}")
    kept = int(keep.sum().item())
    return keep, {
        "mlp_attribution": str(path),
        "mlp_source_key": source_key,
        "mlp_requested_topk": topk,
        "mlp_kept": kept,
        "mlp_total": int(keep.numel()),
        "mlp_keep_fraction": kept / max(int(keep.numel()), 1),
        "mlp_ablation": "zero",
    }


def make_keep_mask(
    *,
    head_scores: torch.Tensor,
    ov_scores: torch.Tensor,
    unit: str,
    topk: int,
    random_seed: int | None,
    mask_strategy: str = "global",
    layer_floor: int = 0,
    head_scaffold_topk: int | None = None,
    head_scaffold_layer_floor: int = 0,
    head_scaffold_multiplier: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n_layers, n_heads = head_scores.shape
    hidden = ov_scores.shape[1]
    head_dim = hidden // n_heads
    keep = torch.zeros((n_layers, hidden), dtype=torch.bool)

    if mask_strategy not in {"global", "layer-balanced", "head-scaffold-ov"}:
        raise ValueError(f"unknown mask strategy: {mask_strategy}")
    if layer_floor < 0:
        raise ValueError("--layer-floor must be non-negative")
    if head_scaffold_layer_floor < 0:
        raise ValueError("--head-scaffold-layer-floor must be non-negative")

    if unit == "head":
        if mask_strategy == "head-scaffold-ov":
            raise ValueError("head-scaffold-ov requires --unit ov")
        total = n_layers * n_heads
        k = min(topk, total)
        if mask_strategy == "layer-balanced":
            indices = pick_layer_balanced_indices(
                head_scores,
                k=k,
                layer_floor=layer_floor,
                random_seed=random_seed,
            )
        else:
            indices = pick_indices(head_scores, k=k, random_seed=random_seed)
        for item in indices:
            layer = item // n_heads
            head = item % n_heads
            keep[layer, head * head_dim : (head + 1) * head_dim] = True
        return keep, {
            "unit": "head",
            "mask_strategy": mask_strategy,
            "requested_topk": topk,
            "layer_floor": layer_floor if mask_strategy == "layer-balanced" else 0,
            "kept_heads": int(len(indices)),
            "kept_ov_channels": int(keep.sum().item()),
        }

    if unit == "ov":
        total = ov_scores.numel()
        k = min(topk, total)
        if mask_strategy == "layer-balanced":
            indices = pick_layer_balanced_indices(
                ov_scores,
                k=k,
                layer_floor=layer_floor,
                random_seed=random_seed,
            )
            scaffold_info: dict[str, Any] = {}
        elif mask_strategy == "head-scaffold-ov":
            if head_scaffold_topk is None:
                min_heads_for_capacity = ceil(k / max(head_dim, 1))
                head_scaffold_topk = int(ceil(min_heads_for_capacity * head_scaffold_multiplier))
            scaffold_heads = pick_layer_balanced_indices(
                head_scores,
                k=min(head_scaffold_topk, head_scores.numel()),
                layer_floor=head_scaffold_layer_floor,
                random_seed=random_seed,
            )
            candidate_scores = torch.full_like(ov_scores, -torch.inf)
            for item in scaffold_heads:
                layer = item // n_heads
                head = item % n_heads
                start = head * head_dim
                candidate_scores[layer, start : start + head_dim] = ov_scores[layer, start : start + head_dim]
            candidates = int(torch.isfinite(candidate_scores).sum().item())
            indices = pick_indices(candidate_scores, k=min(k, candidates), random_seed=random_seed)
            scaffold_info = {
                "head_scaffold_topk": int(head_scaffold_topk),
                "head_scaffold_kept_heads": int(len(scaffold_heads)),
                "head_scaffold_layer_floor": head_scaffold_layer_floor,
                "head_scaffold_multiplier": head_scaffold_multiplier,
                "head_scaffold_candidate_ov_channels": candidates,
            }
        else:
            indices = pick_indices(ov_scores, k=k, random_seed=random_seed)
            scaffold_info = {}
        for item in indices:
            keep[item // hidden, item % hidden] = True
        kept_heads_touched = int(sum(keep[layer].view(n_heads, head_dim).any(dim=1).sum().item() for layer in range(n_layers)))
        info = {
            "unit": "ov",
            "mask_strategy": mask_strategy,
            "requested_topk": topk,
            "layer_floor": layer_floor if mask_strategy == "layer-balanced" else 0,
            "kept_heads_touched": kept_heads_touched,
            "kept_ov_channels": int(keep.sum().item()),
        }
        info.update(scaffold_info)
        return keep, info

    raise ValueError(f"unknown unit: {unit}")


def mask_output_name(args: argparse.Namespace, topk: int, mask_info: dict[str, Any], mlp_info: dict[str, Any] | None) -> str:
    strategy = mask_info.get("mask_strategy", "global")
    strategy_tag = ""
    if strategy == "layer-balanced":
        strategy_tag = f"_layer_balanced_lf{mask_info.get('layer_floor', 0)}"
    elif strategy == "head-scaffold-ov":
        strategy_tag = (
            f"_head_scaffold_ov_hs{mask_info.get('head_scaffold_kept_heads', 0)}"
            f"_hlf{mask_info.get('head_scaffold_layer_floor', 0)}"
        )
    random_tag = f"_random{args.random_seed}" if args.random_seed is not None else ""
    mlp_tag = "_mlp" if mlp_info else ""
    return f"{args.unit}_{args.ablation}{strategy_tag}_k{topk}{mlp_tag}{random_tag}"


def make_boundary_swap_keep_mask(
    *,
    ov_scores: torch.Tensor,
    topk: int,
    swap_batch_size: int,
    remove_batch: int,
    add_batch: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if topk <= 0:
        raise ValueError("--topk must be positive")
    if swap_batch_size <= 0:
        raise ValueError("--swap-batch-size must be positive")
    if remove_batch < 0 or add_batch < 0:
        raise ValueError("swap batch indexes must be non-negative")
    n_layers, hidden = ov_scores.shape
    total = ov_scores.numel()
    k = min(topk, total)
    order = torch.argsort(ov_scores.flatten(), descending=True)
    selected = order[:k]
    outside = order[k:]
    remove_end = k - remove_batch * swap_batch_size
    remove_start = max(remove_end - swap_batch_size, 0)
    add_start = add_batch * swap_batch_size
    add_end = min(add_start + (remove_end - remove_start), outside.numel())
    if remove_start >= remove_end:
        raise ValueError(f"remove batch {remove_batch} is outside selected topk={k}")
    if add_start >= add_end:
        raise ValueError(f"add batch {add_batch} is outside remaining channels")
    removed = selected[remove_start:remove_end]
    added = outside[add_start:add_end]
    kept = torch.cat([selected[:remove_start], selected[remove_end:], added])
    keep = torch.zeros(total, dtype=torch.bool)
    keep[kept] = True
    keep = keep.view(n_layers, hidden)
    return keep, {
        "unit": "ov",
        "mask_strategy": "boundary-swap",
        "requested_topk": topk,
        "kept_ov_channels": int(keep.sum().item()),
        "swap_batch_size": swap_batch_size,
        "remove_batch": remove_batch,
        "add_batch": add_batch,
        "removed_rank_start": int(remove_start),
        "removed_rank_end_exclusive": int(remove_end),
        "added_rank_start": int(k + add_start),
        "added_rank_end_exclusive": int(k + add_end),
        "removed_score_min": float(ov_scores.flatten()[removed].min().item()),
        "removed_score_max": float(ov_scores.flatten()[removed].max().item()),
        "added_score_min": float(ov_scores.flatten()[added].min().item()),
        "added_score_max": float(ov_scores.flatten()[added].max().item()),
        "kept_heads_touched": int(
            sum(keep[layer].view(int(hidden // 128), 128).any(dim=1).sum().item() for layer in range(n_layers))
        )
        if hidden % 128 == 0
        else None,
    }


def boundary_swap_output_name(
    *,
    topk: int,
    swap_batch_size: int,
    remove_batch: int,
    add_batch: int,
    mlp_info: dict[str, Any] | None,
) -> str:
    mlp_tag = "_mlp" if mlp_info else ""
    return f"ov_zero_boundary_swap_k{topk}_b{swap_batch_size}_rm{remove_batch}_add{add_batch}{mlp_tag}"


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

    for layer_idx, layer in enumerate(decoder_layers(model)):
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


def install_mlp_keep_hooks(model, keep: torch.Tensor | None):
    if keep is None:
        return []
    hooks = []
    n_layers, d_ffn = keep.shape
    layers = decoder_layers(model)
    if len(layers) != n_layers:
        raise RuntimeError(f"mlp mask has {n_layers} layers but model has {len(layers)}")
    for layer_idx, layer in enumerate(layers):
        layer_keep = keep[layer_idx].clone()

        def hook(_module, hook_args, _keep=layer_keep):
            x = hook_args[0]
            keep_dev = _keep.to(device=x.device).view(1, 1, d_ffn)
            return (torch.where(keep_dev, x, torch.zeros_like(x)),) + hook_args[1:]

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
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
        "recovery_vs_full_anchor": (correct / FULL_QWEN3_BFCL_ANCHOR) if judged == 1007 else None,
        "full_anchor_normalized_exact": FULL_QWEN3_BFCL_ANCHOR,
        "recovery_note": "score/664 is reported only for full 1007-row evals",
        "by_category": by_category,
    }


def summarize_teacher_forced(
    rows: list[dict[str, Any]],
    pairs_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    examples = len(rows)
    total_tokens = sum(int(row["target_tokens"]) for row in rows)
    token_correct = sum(int(row["target_token_correct"]) for row in rows)
    seq_correct = sum(int(row["target_sequence_argmax_exact"]) for row in rows)
    total_logprob = sum(float(row["target_logprob"]) for row in rows)
    by_category_counts: defaultdict[str, Counter] = defaultdict(Counter)
    by_category_logprob: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        category = pairs_by_id.get(row["id"], {}).get("category", "unknown")
        by_category_counts[category]["examples"] += 1
        by_category_counts[category]["tokens"] += int(row["target_tokens"])
        by_category_counts[category]["token_correct"] += int(row["target_token_correct"])
        by_category_counts[category]["sequence_exact"] += int(row["target_sequence_argmax_exact"])
        by_category_logprob[category] += float(row["target_logprob"])
    by_category: dict[str, dict[str, Any]] = {}
    for category, counts in sorted(by_category_counts.items()):
        tokens = int(counts["tokens"])
        cat_examples = int(counts["examples"])
        by_category[category] = {
            "examples": cat_examples,
            "target_tokens": tokens,
            "target_token_accuracy": counts["token_correct"] / tokens if tokens else None,
            "sequence_argmax_exact_correct": int(counts["sequence_exact"]),
            "sequence_argmax_exact_accuracy": counts["sequence_exact"] / cat_examples if cat_examples else None,
            "mean_target_logprob": by_category_logprob[category] / cat_examples if cat_examples else None,
            "mean_target_token_logprob": by_category_logprob[category] / tokens if tokens else None,
        }
    return {
        "examples": examples,
        "target_tokens": total_tokens,
        "target_token_correct": token_correct,
        "target_token_accuracy": token_correct / total_tokens if total_tokens else None,
        "sequence_argmax_exact_correct": seq_correct,
        "sequence_argmax_exact_accuracy": seq_correct / examples if examples else None,
        "mean_target_logprob": total_logprob / examples if examples else None,
        "mean_target_token_logprob": total_logprob / total_tokens if total_tokens else None,
        "by_category": by_category,
    }


def teacher_forced_once(
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
        for i, row in enumerate(rows, start=1):
            input_ids, attention_mask, prompt_len, target_ids = build_attr_prompt_target(
                tokenizer, row, enable_thinking=args.enable_thinking
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            answer_len = target_ids.shape[1]
            positions = torch.arange(prompt_len - 1, prompt_len - 1 + answer_len, device=logits.device)
            target_logits = logits[:, positions, :]
            logp = torch.log_softmax(target_logits, dim=-1)
            gold = target_ids[0].view(1, -1, 1)
            gold_logp = logp.gather(2, gold).squeeze(0).squeeze(-1)
            pred = target_logits.argmax(dim=-1)
            token_correct = int((pred == target_ids).sum().item())
            out_rows.append(
                {
                    "id": row["id"],
                    "category": row.get("category"),
                    "target_tokens": int(answer_len),
                    "target_token_correct": token_correct,
                    "target_token_accuracy": token_correct / answer_len if answer_len else None,
                    "target_sequence_argmax_exact": bool(token_correct == answer_len),
                    "target_logprob": float(gold_logp.sum().item()),
                    "mean_target_token_logprob": float(gold_logp.mean().item()) if answer_len else None,
                }
            )
            if i % args.log_every == 0 or i == len(rows):
                print(f"{output.stem}: teacher-forced {i}/{len(rows)}", flush=True)
    write_jsonl(output, out_rows)
    return summarize_teacher_forced(out_rows, pairs_by_id)


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
    n_layers, _n_heads, hidden, _head_dim = qwen_attention_shape(model)
    head_scores, ov_scores = load_attention_scores(args.attribution) if args.attribution else (None, None)
    means = load_means(args.mean_cache, (model.config.num_hidden_layers, hidden))
    mlp_keep, mlp_info = load_mlp_keep_mask(args.mlp_attribution, args.mlp_topk, (n_layers, int(model.config.intermediate_size)))
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
            "mask_strategy": args.mask_strategy,
            "topks": topks,
            "ablation": args.ablation,
            "attribution": str(args.attribution) if args.attribution else None,
            "mean_cache": str(args.mean_cache) if args.mean_cache else None,
            "mlp_mask": mlp_info,
            "random_seed": args.random_seed,
            "layer_floor": args.layer_floor,
            "head_scaffold_topk": args.head_scaffold_topk,
            "head_scaffold_layer_floor": args.head_scaffold_layer_floor,
            "head_scaffold_multiplier": args.head_scaffold_multiplier,
        },
    )

    aggregate = {
        "run_name": args.run_name,
        "model": args.model,
        "adapter": args.adapter,
        "pairs": str(args.pairs),
        "unit": args.unit,
        "mask_strategy": args.mask_strategy,
        "ablation": args.ablation,
        "attribution": str(args.attribution) if args.attribution else None,
        "mean_cache": str(args.mean_cache) if args.mean_cache else None,
        "mlp_mask": mlp_info,
        "random_seed": args.random_seed,
        "layer_floor": args.layer_floor,
        "head_scaffold_topk": args.head_scaffold_topk,
        "head_scaffold_layer_floor": args.head_scaffold_layer_floor,
        "head_scaffold_multiplier": args.head_scaffold_multiplier,
        "results": [],
    }
    receipt_files: list[Path] = []
    mlp_hooks = install_mlp_keep_hooks(model, mlp_keep)
    try:
        for topk in topks:
            hooks = []
            mask_info: dict[str, Any]
            keep = None
            if topk == 0:
                name = "mlp_substrate_only" if mlp_info else "unmasked"
                mask_info = {
                    "unit": "none",
                    "requested_topk": 0,
                    "kept_ov_channels": hidden * model.config.num_hidden_layers,
                    "mlp": mlp_info,
                }
            else:
                if head_scores is None or ov_scores is None:
                    raise ValueError("--attribution is required for masked eval")
                keep, mask_info = make_keep_mask(
                    head_scores=head_scores,
                    ov_scores=ov_scores,
                    unit=args.unit,
                    topk=topk,
                    random_seed=args.random_seed,
                    mask_strategy=args.mask_strategy,
                    layer_floor=args.layer_floor,
                    head_scaffold_topk=args.head_scaffold_topk,
                    head_scaffold_layer_floor=args.head_scaffold_layer_floor,
                    head_scaffold_multiplier=args.head_scaffold_multiplier,
                )
                mask_info["mlp"] = mlp_info
                name = mask_output_name(args, topk, mask_info, mlp_info)
            jsonl_path = output_dir / f"{name}.jsonl"
            summary = load_resume_summary(args.resume, jsonl_path)
            if summary is None:
                try:
                    if keep is not None:
                        hooks = install_attention_keep_hooks(model, keep, ablation=args.ablation, means=means)
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
                        "eval/ladder_index": len(aggregate["results"]),
                        "eval/topk": topk,
                        "eval/exact_correct": summary["exact_correct"],
                        "eval/exact_accuracy": summary["exact_accuracy"],
                        "eval/raw_exact_correct": summary["raw_exact_correct"],
                        "eval/normalized_exact_correct": summary["normalized_exact_correct"],
                        "eval/recovery_vs_full_anchor": summary["recovery_vs_full_anchor"],
                        "eval/kept_ov_channels": mask_info.get("kept_ov_channels"),
                        "eval/kept_heads": mask_info.get("kept_heads"),
                        "eval/kept_heads_touched": mask_info.get("kept_heads_touched"),
                        "eval/mlp_kept": mlp_info.get("mlp_kept") if mlp_info else None,
                        "eval/layer_floor": mask_info.get("layer_floor"),
                        "eval/head_scaffold_kept_heads": mask_info.get("head_scaffold_kept_heads"),
                    },
                    step=len(aggregate["results"]),
                )
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    finally:
        for hook in mlp_hooks:
            hook.remove()

    aggregate_path = output_dir / "ladder_summary.json"
    json_dump(aggregate_path, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if run is not None:
        log_wandb_receipts(run, name=f"{args.run_name}-attention-eval", files=[aggregate_path, *receipt_files])
        run.finish()


def eval_boundary_swap(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    pairs_by_id = {row["id"]: row for row in rows}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    n_layers, _n_heads, hidden, _head_dim = qwen_attention_shape(model)
    _head_scores, ov_scores = load_attention_scores(args.attribution)
    means = load_means(args.mean_cache, (model.config.num_hidden_layers, hidden))
    mlp_keep, mlp_info = load_mlp_keep_mask(
        args.mlp_attribution,
        args.mlp_topk,
        (n_layers, int(model.config.intermediate_size)),
    )
    remove_batches = parse_int_list(args.remove_batches)
    add_batches = parse_int_list(args.add_batches)

    run = init_wandb_run(
        args,
        default_project="prism-bfcl-attention",
        default_group=args.run_name,
        default_job_type="attention-boundary-swap-eval",
        default_name=args.run_name,
        default_tags="bfcl,qwen3,attention,boundary-swap,eval",
        config={
            "cmd": "eval-boundary-swap",
            "model": args.model,
            "adapter": args.adapter,
            "pairs": str(args.pairs),
            "attribution": str(args.attribution),
            "topk": args.topk,
            "swap_batch_size": args.swap_batch_size,
            "remove_batches": remove_batches,
            "add_batches": add_batches,
            "ablation": args.ablation,
            "mean_cache": str(args.mean_cache) if args.mean_cache else None,
            "mlp_mask": mlp_info,
        },
    )

    aggregate = {
        "run_name": args.run_name,
        "model": args.model,
        "adapter": args.adapter,
        "pairs": str(args.pairs),
        "attribution": str(args.attribution),
        "topk": args.topk,
        "swap_batch_size": args.swap_batch_size,
        "remove_batches": remove_batches,
        "add_batches": add_batches,
        "ablation": args.ablation,
        "mean_cache": str(args.mean_cache) if args.mean_cache else None,
        "mlp_mask": mlp_info,
        "results": [],
    }
    receipt_files: list[Path] = []
    mlp_hooks = install_mlp_keep_hooks(model, mlp_keep)
    try:
        for remove_batch in remove_batches:
            for add_batch in add_batches:
                keep, mask_info = make_boundary_swap_keep_mask(
                    ov_scores=ov_scores,
                    topk=args.topk,
                    swap_batch_size=args.swap_batch_size,
                    remove_batch=remove_batch,
                    add_batch=add_batch,
                )
                mask_info["mlp"] = mlp_info
                name = boundary_swap_output_name(
                    topk=args.topk,
                    swap_batch_size=args.swap_batch_size,
                    remove_batch=remove_batch,
                    add_batch=add_batch,
                    mlp_info=mlp_info,
                )
                jsonl_path = output_dir / f"{name}.jsonl"
                summary = load_resume_summary(args.resume, jsonl_path)
                if summary is None:
                    hooks = []
                    try:
                        hooks = install_attention_keep_hooks(model, keep, ablation=args.ablation, means=means)
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
                            "swap/ladder_index": len(aggregate["results"]),
                            "swap/topk": args.topk,
                            "swap/remove_batch": remove_batch,
                            "swap/add_batch": add_batch,
                            "swap/swap_batch_size": args.swap_batch_size,
                            "swap/exact_correct": summary["exact_correct"],
                            "swap/raw_exact_correct": summary["raw_exact_correct"],
                            "swap/normalized_exact_correct": summary["normalized_exact_correct"],
                            "swap/kept_ov_channels": mask_info.get("kept_ov_channels"),
                            "swap/mlp_kept": mlp_info.get("mlp_kept") if mlp_info else None,
                        },
                        step=len(aggregate["results"]),
                    )
                print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    finally:
        for hook in mlp_hooks:
            hook.remove()

    aggregate_path = output_dir / "boundary_swap_summary.json"
    json_dump(aggregate_path, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if run is not None:
        log_wandb_receipts(run, name=f"{args.run_name}-attention-boundary-swap", files=[aggregate_path, *receipt_files])
        run.finish()


def teacher_forced_ladder(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    pairs_by_id = {row["id"]: row for row in rows}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    n_layers, _n_heads, hidden, _head_dim = qwen_attention_shape(model)
    head_scores, ov_scores = load_attention_scores(args.attribution) if args.attribution else (None, None)
    means = load_means(args.mean_cache, (model.config.num_hidden_layers, hidden))
    mlp_keep, mlp_info = load_mlp_keep_mask(args.mlp_attribution, args.mlp_topk, (n_layers, int(model.config.intermediate_size)))
    topks = parse_int_list(args.topks)
    if args.include_unmasked:
        topks = [0] + topks

    run = init_wandb_run(
        args,
        default_project="prism-bfcl-attention",
        default_group=args.run_name,
        default_job_type="attention-teacher-forced",
        default_name=args.run_name,
        default_tags=f"bfcl,qwen3,attention,teacher-forced,{args.unit},{args.ablation}",
        config={
            "cmd": "teacher-forced-ladder",
            "model": args.model,
            "adapter": args.adapter,
            "unit": args.unit,
            "mask_strategy": args.mask_strategy,
            "topks": topks,
            "ablation": args.ablation,
            "attribution": str(args.attribution) if args.attribution else None,
            "mean_cache": str(args.mean_cache) if args.mean_cache else None,
            "mlp_mask": mlp_info,
            "random_seed": args.random_seed,
            "layer_floor": args.layer_floor,
            "head_scaffold_topk": args.head_scaffold_topk,
            "head_scaffold_layer_floor": args.head_scaffold_layer_floor,
            "head_scaffold_multiplier": args.head_scaffold_multiplier,
        },
    )

    aggregate = {
        "run_name": args.run_name,
        "model": args.model,
        "adapter": args.adapter,
        "pairs": str(args.pairs),
        "unit": args.unit,
        "mask_strategy": args.mask_strategy,
        "ablation": args.ablation,
        "attribution": str(args.attribution) if args.attribution else None,
        "mean_cache": str(args.mean_cache) if args.mean_cache else None,
        "mlp_mask": mlp_info,
        "random_seed": args.random_seed,
        "layer_floor": args.layer_floor,
        "head_scaffold_topk": args.head_scaffold_topk,
        "head_scaffold_layer_floor": args.head_scaffold_layer_floor,
        "head_scaffold_multiplier": args.head_scaffold_multiplier,
        "metric": "teacher_forced_gold_tool_call_target",
        "results": [],
    }
    receipt_files: list[Path] = []
    mlp_hooks = install_mlp_keep_hooks(model, mlp_keep)
    try:
        for topk in topks:
            hooks = []
            keep = None
            if topk == 0:
                name = "mlp_substrate_only" if mlp_info else "unmasked"
                mask_info = {
                    "unit": "none",
                    "requested_topk": 0,
                    "kept_ov_channels": hidden * model.config.num_hidden_layers,
                    "mlp": mlp_info,
                }
            else:
                if head_scores is None or ov_scores is None:
                    raise ValueError("--attribution is required for masked teacher-forced eval")
                keep, mask_info = make_keep_mask(
                    head_scores=head_scores,
                    ov_scores=ov_scores,
                    unit=args.unit,
                    topk=topk,
                    random_seed=args.random_seed,
                    mask_strategy=args.mask_strategy,
                    layer_floor=args.layer_floor,
                    head_scaffold_topk=args.head_scaffold_topk,
                    head_scaffold_layer_floor=args.head_scaffold_layer_floor,
                    head_scaffold_multiplier=args.head_scaffold_multiplier,
                )
                mask_info["mlp"] = mlp_info
                name = mask_output_name(args, topk, mask_info, mlp_info)
            jsonl_path = output_dir / f"{name}.teacher_forced.jsonl"
            summary = load_resume_summary(args.resume, jsonl_path)
            if summary is None:
                try:
                    if keep is not None:
                        hooks = install_attention_keep_hooks(model, keep, ablation=args.ablation, means=means)
                    summary = teacher_forced_once(
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
                        "records": str(jsonl_path),
                        "summary": str(jsonl_path.with_suffix(".summary.json")),
                    }
                )
                json_dump(jsonl_path.with_suffix(".summary.json"), summary)
            receipt_files.extend([jsonl_path, jsonl_path.with_suffix(".summary.json")])
            aggregate["results"].append(summary)
            if run is not None:
                run.log(
                    {
                        "tf/ladder_index": len(aggregate["results"]),
                        "tf/topk": topk,
                        "tf/target_token_accuracy": summary["target_token_accuracy"],
                        "tf/sequence_argmax_exact_accuracy": summary["sequence_argmax_exact_accuracy"],
                        "tf/mean_target_logprob": summary["mean_target_logprob"],
                        "tf/mean_target_token_logprob": summary["mean_target_token_logprob"],
                        "tf/kept_ov_channels": mask_info.get("kept_ov_channels"),
                        "tf/kept_heads": mask_info.get("kept_heads"),
                        "tf/kept_heads_touched": mask_info.get("kept_heads_touched"),
                        "tf/mlp_kept": mlp_info.get("mlp_kept") if mlp_info else None,
                        "tf/layer_floor": mask_info.get("layer_floor"),
                        "tf/head_scaffold_kept_heads": mask_info.get("head_scaffold_kept_heads"),
                    },
                    step=len(aggregate["results"]),
                )
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    finally:
        for hook in mlp_hooks:
            hook.remove()

    aggregate_path = output_dir / "teacher_forced_ladder_summary.json"
    json_dump(aggregate_path, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if run is not None:
        log_wandb_receipts(run, name=f"{args.run_name}-attention-teacher-forced", files=[aggregate_path, *receipt_files])
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
    p.add_argument("--mlp-attribution", type=Path)
    p.add_argument("--mlp-topk", type=int)
    add_common_model_args(p)
    add_wandb_args(p, default_project="prism-bfcl-attention", default_job_type="attention-actgrad")
    p.set_defaults(func=actgrad_attribute)

    p = sub.add_parser("build-mean-cache")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-name", default="issue3-attn-mean-cache")
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--mlp-attribution", type=Path)
    p.add_argument("--mlp-topk", type=int)
    add_common_model_args(p)
    add_wandb_args(p, default_project="prism-bfcl-attention", default_job_type="attention-mean-cache")
    p.set_defaults(func=build_mean_cache)

    p = sub.add_parser("eval-ladder")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--attribution", type=Path)
    p.add_argument("--mean-cache", type=Path)
    p.add_argument("--mlp-attribution", type=Path)
    p.add_argument("--mlp-topk", type=int)
    p.add_argument("--run-name", default="issue3-attn-eval")
    p.add_argument("--unit", choices=["head", "ov"], default="head")
    p.add_argument("--mask-strategy", choices=["global", "layer-balanced", "head-scaffold-ov"], default="global")
    p.add_argument("--topks", default="8,16,32,64,128,256,512,1024")
    p.add_argument("--ablation", choices=["zero", "mean"], default="zero")
    p.add_argument("--layer-floor", type=int, default=0)
    p.add_argument("--head-scaffold-topk", type=int)
    p.add_argument("--head-scaffold-layer-floor", type=int, default=0)
    p.add_argument("--head-scaffold-multiplier", type=float, default=2.0)
    p.add_argument("--random-seed", type=int)
    p.add_argument("--include-unmasked", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.add_argument("--normalized", action="store_true")
    p.add_argument("--resume", action="store_true")
    add_common_model_args(p)
    add_wandb_args(p, default_project="prism-bfcl-attention", default_job_type="attention-eval")
    p.set_defaults(func=eval_ladder)

    p = sub.add_parser("eval-boundary-swap")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--mean-cache", type=Path)
    p.add_argument("--mlp-attribution", type=Path)
    p.add_argument("--mlp-topk", type=int)
    p.add_argument("--run-name", default="issue3-attn-boundary-swap")
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--swap-batch-size", type=int, default=4096)
    p.add_argument("--remove-batches", default="0")
    p.add_argument("--add-batches", default="0,1,2,3")
    p.add_argument("--ablation", choices=["zero", "mean"], default="zero")
    p.add_argument("--include-unmasked", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.add_argument("--normalized", action="store_true")
    p.add_argument("--resume", action="store_true")
    add_common_model_args(p)
    add_wandb_args(p, default_project="prism-bfcl-attention", default_job_type="attention-boundary-swap-eval")
    p.set_defaults(func=eval_boundary_swap)

    p = sub.add_parser("teacher-forced-ladder")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--attribution", type=Path)
    p.add_argument("--mean-cache", type=Path)
    p.add_argument("--mlp-attribution", type=Path)
    p.add_argument("--mlp-topk", type=int)
    p.add_argument("--run-name", default="issue3-attn-teacher-forced")
    p.add_argument("--unit", choices=["head", "ov"], default="head")
    p.add_argument("--mask-strategy", choices=["global", "layer-balanced", "head-scaffold-ov"], default="global")
    p.add_argument("--topks", default="8,16,32,64,128,256,512,1024")
    p.add_argument("--ablation", choices=["zero", "mean"], default="zero")
    p.add_argument("--layer-floor", type=int, default=0)
    p.add_argument("--head-scaffold-topk", type=int)
    p.add_argument("--head-scaffold-layer-floor", type=int, default=0)
    p.add_argument("--head-scaffold-multiplier", type=float, default=2.0)
    p.add_argument("--random-seed", type=int)
    p.add_argument("--include-unmasked", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    add_common_model_args(p)
    add_wandb_args(p, default_project="prism-bfcl-attention", default_job_type="attention-teacher-forced")
    p.set_defaults(func=teacher_forced_ladder)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
