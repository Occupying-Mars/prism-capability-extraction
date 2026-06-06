#!/usr/bin/env python3
"""GKD/AOPD hybrid for BFCL masked LoRA.

This runner keeps the sampled-token Prime-RL OPD contract but anchors it with
offline gold-trajectory distillation:

1. The current masked student policy generates completions.
2. The unmasked base teacher scores those exact sampled tokens.
3. Positive-advantage sampled tokens receive the online AOPD update.
4. Every update also gets gold-prefix teacher KL plus CE on the reference tool
   call, so syntax/value grounding does not drift.

The online loss mirrors the Prime-RL shape:

    advantage = teacher_logprob(sampled_token) - rollout_logprob(sampled_token)
    loss      = - advantage.detach() * exp(current_logprob - rollout_logprob)
                + kl_tau * (current_logprob - rollout_logprob)^2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bfcl_direct_qwen3 import messages_for_generation, normalized_prediction_ok, parse_tool_calls  # noqa: E402
from scripts.train_bfcl_masked_lora import (  # noqa: E402
    ToolMindDataset,
    answer_ce_loss,
    build_mean_cache,
    collate,
    decoder_root,
    install_mean_ablation_hooks,
    kl_loss,
    layer_device,
    load_topk_mask,
    model_input_device,
    read_jsonl,
)


@dataclass
class RolloutSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    rollout_logprobs: torch.Tensor
    teacher_logprobs: torch.Tensor
    source_id: str
    parsed_call: bool
    normalized_correct: bool
    completion_tokens: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--init-adapter", type=Path)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-jsonl", type=Path, default=Path("data/bfcl_strict_10k_mix_len1024/train.jsonl"))
    p.add_argument("--attribution", type=Path, default=Path("runs/issue49_bfcl_single_call/relp_full.npz"))
    p.add_argument("--topk", type=int, default=160000)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=1000)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--rollout-batch-size", type=int, default=8)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--offline-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--updates-per-rollout", type=int, default=1)
    p.add_argument("--max-updates", type=int, default=125)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adv-scale", type=float, default=1.0)
    p.add_argument("--kl-tau", type=float, default=1e-3)
    p.add_argument("--online-opd-beta", type=float, default=0.2)
    p.add_argument("--offline-kl-beta", type=float, default=0.8)
    p.add_argument("--ce-beta", type=float, default=0.05)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--positive-opd-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dppo-threshold", type=float, default=0.2)
    p.add_argument("--mask-mode", choices=["zero", "mean"], default="zero")
    p.add_argument("--eval-rollout-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--save-merged", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bfcl-canonicalization-prompt", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def row_for_scoring(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("target") is not None or row.get("reference_calls") is not None:
        return row
    out = dict(row)
    if "target_call" in out:
        out["target"] = out["target_call"]
    return out


def encode_prompt(row: dict[str, Any], tokenizer, args: argparse.Namespace) -> torch.Tensor | None:
    row = row_for_scoring(row)
    encoded = tokenizer.apply_chat_template(
        messages_for_generation(row, bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt),
        tools=row.get("tools") or None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        enable_thinking=False,
    )
    ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    if ids.numel() >= args.max_seq_length:
        return None
    return ids


def token_logprobs_for_sequence(logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return log p(input_ids[t] | input_ids[:t]) for positions selected by mask."""
    shifted_logits = logits[:, :-1, :]
    shifted_ids = input_ids[:, 1:]
    shifted_mask = mask[:, 1:]
    logp = torch.log_softmax(shifted_logits, dim=-1)
    token_logp = logp.gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp[shifted_mask]


def pad_samples(samples: list[RolloutSample], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(sample.input_ids.numel()) for sample in samples)
    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    completion_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    rollout_logprobs = torch.zeros((batch_size, max_len), dtype=torch.float32)
    teacher_logprobs = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for idx, sample in enumerate(samples):
        n = int(sample.input_ids.numel())
        input_ids[idx, :n] = sample.input_ids
        attention_mask[idx, :n] = sample.attention_mask
        completion_mask[idx, :n] = sample.completion_mask
        rollout_logprobs[idx, :n] = sample.rollout_logprobs
        teacher_logprobs[idx, :n] = sample.teacher_logprobs
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "rollout_logprobs": rollout_logprobs,
        "teacher_logprobs": teacher_logprobs,
    }


def install_zero_keep_hooks(model, mask):
    hooks = []
    for layer_idx, layer in enumerate(decoder_root(model).layers):
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


def install_mask_hooks(model, mask, means, dtype: torch.dtype, mask_mode: str):
    if mask_mode == "zero":
        return install_zero_keep_hooks(model, mask)
    if means is None:
        raise ValueError("mean mask mode requires means")
    return install_mean_ablation_hooks(model, mask, means, dtype=dtype)


def masked_forward(model, batch: dict[str, torch.Tensor], mask, means, dtype: torch.dtype, mask_mode: str):
    hooks = install_mask_hooks(model, mask, means, dtype, mask_mode)
    try:
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    finally:
        for hook in hooks:
            hook.remove()


def collect_rollouts(
    *,
    model,
    rows: list[dict[str, Any]],
    row_cursor: int,
    tokenizer,
    args: argparse.Namespace,
    mask,
    means,
    dtype: torch.dtype,
    mask_mode: str,
    input_device: torch.device,
) -> tuple[list[RolloutSample], int, dict[str, Any]]:
    model.eval()
    samples: list[RolloutSample] = []
    stats = {
        "attempted": 0,
        "kept": 0,
        "empty": 0,
        "too_long": 0,
        "parsed_calls": 0,
        "normalized_correct": 0,
        "tokens": 0,
    }
    while len(samples) < args.rollout_batch_size and stats["attempted"] < len(rows):
        row = rows[row_cursor % len(rows)]
        row_cursor += 1
        stats["attempted"] += 1
        score_row = row_for_scoring(row)
        prompt_ids = encode_prompt(score_row, tokenizer, args)
        if prompt_ids is None:
            stats["too_long"] += 1
            continue
        prompt_len = int(prompt_ids.numel())
        prompt_batch = {
            "input_ids": prompt_ids.unsqueeze(0).to(input_device),
            "attention_mask": torch.ones(1, prompt_len, dtype=torch.long, device=input_device),
        }
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
        hooks = install_mask_hooks(model, mask, means, dtype, mask_mode)
        try:
            with torch.inference_mode():
                output = model.generate(**prompt_batch, **gen_kwargs)[0].detach().cpu()
        finally:
            for hook in hooks:
                hook.remove()
        completion_ids = output[prompt_len:]
        if tokenizer.eos_token_id is not None:
            eos = (completion_ids == int(tokenizer.eos_token_id)).nonzero(as_tuple=False)
            if eos.numel():
                completion_ids = completion_ids[: int(eos[0].item()) + 1]
        if completion_ids.numel() == 0:
            stats["empty"] += 1
            continue
        input_ids = torch.cat([prompt_ids, completion_ids], dim=0)
        if input_ids.numel() > args.max_seq_length:
            stats["too_long"] += 1
            continue
        attention_mask = torch.ones_like(input_ids)
        completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        completion_mask[prompt_len:] = True
        batch = {
            "input_ids": input_ids.unsqueeze(0).to(input_device),
            "attention_mask": attention_mask.unsqueeze(0).to(input_device),
        }
        completion_mask_batch = completion_mask.unsqueeze(0).to(input_device)
        with torch.inference_mode():
            student_logits = masked_forward(model, batch, mask, means, dtype, mask_mode)
            rollout_lp = token_logprobs_for_sequence(student_logits, batch["input_ids"], completion_mask_batch)
            with model.disable_adapter():
                teacher_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
            teacher_lp = token_logprobs_for_sequence(teacher_logits, batch["input_ids"], completion_mask_batch)
        rollout_full = torch.zeros_like(input_ids, dtype=torch.float32)
        teacher_full = torch.zeros_like(input_ids, dtype=torch.float32)
        rollout_full[completion_mask] = rollout_lp.detach().float().cpu()
        teacher_full[completion_mask] = teacher_lp.detach().float().cpu()
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        parsed_calls = parse_tool_calls(text)
        parsed_call = bool(parsed_calls)
        normalized_correct = normalized_prediction_ok(parsed_calls, score_row)
        stats["parsed_calls"] += int(parsed_call)
        stats["normalized_correct"] += int(normalized_correct)
        stats["tokens"] += int(completion_ids.numel())
        stats["kept"] += 1
        samples.append(
            RolloutSample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                completion_mask=completion_mask,
                rollout_logprobs=rollout_full,
                teacher_logprobs=teacher_full,
                source_id=str(score_row.get("id", score_row.get("mix_id", row_cursor))),
                parsed_call=parsed_call,
                normalized_correct=normalized_correct,
                completion_tokens=int(completion_ids.numel()),
            )
        )
    return samples, row_cursor, stats


def prime_opd_loss(
    current_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    adv_scale: float,
    kl_tau: float,
    dppo_threshold: float,
    positive_only: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_ratio = current_logprobs - rollout_logprobs
    ratio = torch.exp(log_ratio.clamp(min=-10.0, max=10.0))
    advantage = adv_scale * (teacher_logprobs - rollout_logprobs)
    probs_diff = torch.exp(current_logprobs) - torch.exp(rollout_logprobs)
    invalid_high = probs_diff > dppo_threshold
    invalid_low = probs_diff < -dppo_threshold
    invalid = torch.where(advantage > 0, invalid_high, invalid_low)
    keep = ~invalid
    if positive_only:
        keep = keep & (advantage > 0)
    if int(keep.sum().item()) == 0:
        pg_loss = current_logprobs.sum() * 0.0
    else:
        pg_loss = -(advantage.detach() * ratio)[keep].mean()
    kl_loss = (log_ratio.square()).mean()
    loss = pg_loss + kl_tau * kl_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "pg_loss": float(pg_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "teacher_adv": float(advantage.detach().mean().cpu()),
        "teacher_adv_pos_frac": float((advantage.detach() > 0).float().mean().cpu()),
        "opd_selected_frac": float(keep.float().mean().cpu()),
        "mean_log_ratio": float(log_ratio.detach().mean().cpu()),
        "masked_frac": float(invalid.float().mean().cpu()),
        "tokens": float(current_logprobs.numel()),
    }
    return loss, metrics


def next_offline_batch(loader: DataLoader, state: dict[str, Any]) -> dict[str, torch.Tensor]:
    try:
        return next(state["iterator"])
    except StopIteration:
        state["iterator"] = iter(loader)
        return next(state["iterator"])


def offline_gkd_loss(
    *,
    model,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    mask,
    means,
    dtype: torch.dtype,
    mask_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad(), model.disable_adapter():
        teacher_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    student_logits = masked_forward(model, batch, mask, means, dtype, mask_mode)
    offline_kl = kl_loss(student_logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
    ce = answer_ce_loss(student_logits, batch["labels"])
    loss = args.offline_kl_beta * offline_kl + args.ce_beta * ce
    return loss, {
        "offline_loss": float(loss.detach().cpu()),
        "offline_kl": float(offline_kl.detach().cpu()),
        "ce": float(ce.detach().cpu()),
    }


def train_on_samples(
    *,
    model,
    samples: list[RolloutSample],
    offline_loader: DataLoader,
    offline_state: dict[str, Any],
    tokenizer,
    args: argparse.Namespace,
    mask,
    means,
    dtype: torch.dtype,
    mask_mode: str,
    input_device: torch.device,
    optimizer,
    scheduler,
    global_update: int,
) -> tuple[int, list[dict[str, float]]]:
    model.train()
    metrics: list[dict[str, float]] = []
    random.shuffle(samples)
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.updates_per_rollout):
        for start in range(0, len(samples), args.train_batch_size):
            chunk = samples[start : start + args.train_batch_size]
            raw_batch = pad_samples(chunk, tokenizer.pad_token_id)
            batch = {key: value.to(input_device) for key, value in raw_batch.items()}
            logits = masked_forward(model, batch, mask, means, dtype, mask_mode)
            current = token_logprobs_for_sequence(logits, batch["input_ids"], batch["completion_mask"])
            rollout = batch["rollout_logprobs"][batch["completion_mask"]]
            teacher = batch["teacher_logprobs"][batch["completion_mask"]]
            online_loss, row = prime_opd_loss(
                current,
                rollout,
                teacher,
                adv_scale=args.adv_scale,
                kl_tau=args.kl_tau,
                dppo_threshold=args.dppo_threshold,
                positive_only=args.positive_opd_only,
            )
            offline_raw = next_offline_batch(offline_loader, offline_state)
            offline_batch = {key: value.to(input_device) for key, value in offline_raw.items()}
            offline_loss, offline_row = offline_gkd_loss(
                model=model,
                batch=offline_batch,
                args=args,
                mask=mask,
                means=means,
                dtype=dtype,
                mask_mode=mask_mode,
            )
            loss = args.online_opd_beta * online_loss + offline_loss
            row.update(offline_row)
            row["weighted_online_loss"] = float((args.online_opd_beta * online_loss).detach().cpu())
            row["total_loss"] = float(loss.detach().cpu())
            (loss / args.grad_accum).backward()
            accum += 1
            metrics.append(row)
            if accum % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            if global_update >= args.max_updates:
                return global_update, metrics
    if accum % args.grad_accum:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_update += 1
    return global_update, metrics


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    vals = [row[key] for row in rows if key in row]
    return sum(vals) / max(len(vals), 1)


def save_checkpoint(model, tokenizer, out_dir: Path, update: int) -> str:
    path = out_dir / "checkpoints" / f"update_{update:06d}" / "adapter"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.train_jsonl)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("no training rows loaded")
    print(f"[data] rows={len(rows)} train={args.train_jsonl}", flush=True)
    offline_dataset = ToolMindDataset(rows, tokenizer, args.max_seq_length)
    if len(offline_dataset) == 0:
        raise ValueError("no offline gold rows encoded")
    offline_loader = DataLoader(
        offline_dataset,
        batch_size=args.offline_batch_size,
        shuffle=True,
        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id),
    )
    offline_state = {"iterator": iter(offline_loader)}

    init_desc = str(args.init_adapter) if args.init_adapter else f"fresh_lora_r{args.lora_r}"
    print(f"[model] base={args.base_model} init={init_desc}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    base.config.use_cache = False
    input_device = model_input_device(base)
    n_layers = int(base.config.num_hidden_layers)
    d_ffn = int(base.config.intermediate_size)

    mask = load_topk_mask(args.attribution, args.topk)
    kept = sum(int(v.sum().item()) for v in mask.values())
    print(f"[mask] topk={args.topk} kept={kept} attribution={args.attribution}", flush=True)

    means = None
    if args.mask_mode == "mean":
        print(f"[mean] building cache n={args.n_calib}", flush=True)
        means = build_mean_cache(base, rows, tokenizer, args, n_layers=n_layers, d_ffn=d_ffn, dtype=dtype)

    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        adapter_init = {"mode": "init_adapter", "path": str(args.init_adapter)}
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            use_rslora=args.use_rslora,
            bias="none",
        )
        model = get_peft_model(base, lora_config)
        adapter_init = {
            "mode": "fresh_lora",
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "use_rslora": args.use_rslora,
        }
    model.print_trainable_parameters()
    model.train()

    total_updates = args.max_updates
    warmup_steps = max(int(total_updates * args.warmup_ratio), 0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    summary = {
        "method": "gkd_aopd_hybrid_sampled_token_v0",
        "contract": {
            "rollout": f"current LoRA student under MLP {args.mask_mode}-isolated keep mask",
            "teacher": "same base model with adapter disabled and no mask",
            "loss": "offline gold-prefix teacher KL/CE plus positive-advantage sampled-token AOPD",
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "n_rows": len(rows),
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "mask_kept": kept,
        "adapter_init": adapter_init,
        "warmup_steps": warmup_steps,
        "logs": [],
        "checkpoints": [],
    }
    (args.out_dir / "config.json").write_text(json.dumps(summary, indent=2))

    row_cursor = 0
    global_update = 0
    rollout_idx = 0
    start_time = time.time()
    while global_update < args.max_updates:
        samples, row_cursor, rollout_stats = collect_rollouts(
            model=model,
            rows=rows,
            row_cursor=row_cursor,
            tokenizer=tokenizer,
            args=args,
            mask=mask,
            means=means,
            dtype=dtype,
            mask_mode=args.mask_mode,
            input_device=input_device,
        )
        if not samples:
            raise ValueError(f"no samples collected: {rollout_stats}")
        rollout_idx += 1
        before_update = global_update
        global_update, train_metrics = train_on_samples(
            model=model,
            samples=samples,
            offline_loader=offline_loader,
            offline_state=offline_state,
            tokenizer=tokenizer,
            args=args,
            mask=mask,
            means=means,
            dtype=dtype,
            mask_mode=args.mask_mode,
            input_device=input_device,
            optimizer=optimizer,
            scheduler=scheduler,
            global_update=global_update,
        )
        log_row = {
            "rollout": rollout_idx,
            "update": global_update,
            "updates_added": global_update - before_update,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": time.time() - start_time,
            "rollout_stats": rollout_stats,
            "train": {
                "loss": mean_metric(train_metrics, "loss"),
                "pg_loss": mean_metric(train_metrics, "pg_loss"),
                "kl_loss": mean_metric(train_metrics, "kl_loss"),
                "teacher_adv": mean_metric(train_metrics, "teacher_adv"),
                "teacher_adv_pos_frac": mean_metric(train_metrics, "teacher_adv_pos_frac"),
                "opd_selected_frac": mean_metric(train_metrics, "opd_selected_frac"),
                "mean_log_ratio": mean_metric(train_metrics, "mean_log_ratio"),
                "masked_frac": mean_metric(train_metrics, "masked_frac"),
                "offline_loss": mean_metric(train_metrics, "offline_loss"),
                "offline_kl": mean_metric(train_metrics, "offline_kl"),
                "ce": mean_metric(train_metrics, "ce"),
                "weighted_online_loss": mean_metric(train_metrics, "weighted_online_loss"),
                "total_loss": mean_metric(train_metrics, "total_loss"),
                "tokens": mean_metric(train_metrics, "tokens"),
            },
        }
        summary["logs"].append(log_row)
        (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(log_row), flush=True)

        if args.save_every and global_update and global_update % args.save_every == 0:
            checkpoint = save_checkpoint(model, tokenizer, args.out_dir, global_update)
            summary["checkpoints"].append({"update": global_update, "adapter_dir": checkpoint})
            (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
            print(f"[checkpoint] update={global_update} adapter={checkpoint}", flush=True)

    model.eval()
    summary["elapsed_s"] = time.time() - start_time
    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary["adapter_dir"] = str(adapter_dir)
    print(f"[done] adapter={adapter_dir}", flush=True)

    if args.save_merged:
        merged_dir = args.out_dir / "merged"
        print(f"[merge] saving {merged_dir}", flush=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        summary["merged_dir"] = str(merged_dir)

    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
