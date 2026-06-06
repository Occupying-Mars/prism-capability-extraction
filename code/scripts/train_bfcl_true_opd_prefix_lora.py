#!/usr/bin/env python3
"""Train BFCL masked LoRA with true prefix-state OPD.

Loop implemented by this script:

1. Roll out a masked policy on the train prompt pool.
2. Keep prefixes that the masked policy actually generated.
3. Train a fresh masked LoRA student to match the unmasked teacher policy at
   those exact prefix states.

This is closer to DAgger-style on-policy distillation than the offline
gold-prefix KL baseline in train_bfcl_masked_policy_distill.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bfcl_direct_qwen3 import (  # noqa: E402
    messages_for_generation,
    normalized_prediction_ok,
    parse_tool_calls,
)
from scripts.train_bfcl_masked_lora import (  # noqa: E402
    ToolMindDataset,
    build_mean_cache,
    collate,
    install_mean_ablation_hooks,
    load_topk_mask,
    model_input_device,
    move_batch,
    read_jsonl,
    save_adapter_checkpoint,
)


@dataclass
class PrefixState:
    input_ids: torch.Tensor
    kl_logit_mask: torch.Tensor
    source_id: str
    prefix_len: int
    failed_rollout: bool
    parsed_call: bool
    weight: float


class PrefixStateDataset(Dataset):
    def __init__(self, states: list[PrefixState]):
        self.states = states

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> PrefixState:
        return self.states[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-model", default="Qwen/Qwen3-8B")
    p.add_argument("--rollout-model", required=True, help="Merged model path used to visit OPD states.")
    p.add_argument("--train-jsonl", type=Path, default=Path("data/bfcl_strict_10k_mix_len1024/train.jsonl"))
    p.add_argument("--attribution", type=Path, default=Path("runs/issue49_bfcl_single_call/relp_full.npz"))
    p.add_argument("--topk", type=int, default=240000)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--rollout-batch-size", type=int, default=1)
    p.add_argument("--prefixes-per-row", type=int, default=4)
    p.add_argument("--success-sample-rate", type=float, default=0.10)
    p.add_argument("--include-successes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-parsed-call", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-generated-prefix-tokens", type=int, default=1)
    p.add_argument("--failed-prefix-weight", type=float, default=0.5)
    p.add_argument("--parsed-prefix-weight", type=float, default=1.0)
    p.add_argument("--success-prefix-weight", type=float, default=1.0)
    p.add_argument("--bfcl-canonicalization-prompt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--offline-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--opd-loss", choices=["fkl", "rkl", "jsd", "skl", "srkl"], default="fkl")
    p.add_argument("--offline-loss", choices=["fkl", "rkl", "jsd", "skl", "srkl"], default="fkl")
    p.add_argument("--opd-beta", type=float, default=1.0)
    p.add_argument("--offline-beta", type=float, default=1.0)
    p.add_argument("--skew-alpha", type=float, default=0.1)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--teacher-confidence-power", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--save-merged", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def row_for_scoring(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("target") is not None or row.get("reference_calls") is not None:
        return row
    out = dict(row)
    if "target_call" in out:
        out["target"] = out["target_call"]
    return out


def choose_prefix_lengths(n_tokens: int, limit: int) -> list[int]:
    if n_tokens <= 0:
        return [0]
    candidates = [0, 1, n_tokens // 4, n_tokens // 2, (3 * n_tokens) // 4, max(n_tokens - 1, 0)]
    out = []
    for item in candidates:
        item = max(0, min(int(item), max(n_tokens - 1, 0)))
        if item not in out:
            out.append(item)
    return out[: max(limit, 1)]


def collect_prefix_states(args, rows: list[dict[str, Any]], tokenizer, mask) -> tuple[list[PrefixState], dict[str, Any]]:
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[rollout] loading {args.rollout_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.rollout_model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    input_device = model_input_device(model)

    hooks = install_zero_keep_hooks(model, mask)
    states: list[PrefixState] = []
    stats = {
        "rows": 0,
        "failures": 0,
        "successes": 0,
        "parsed_calls": 0,
        "kept_successes": 0,
        "skipped_unparsed": 0,
        "states": 0,
    }
    try:
        for idx, row in enumerate(rows):
            score_row = row_for_scoring(row)
            encoded = tokenizer.apply_chat_template(
                messages_for_generation(score_row, bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt),
                tools=score_row.get("tools") or None,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=False,
            )
            prompt_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            if prompt_ids.numel() >= args.max_seq_length:
                continue
            with torch.inference_mode():
                output = model.generate(
                    input_ids=prompt_ids.unsqueeze(0).to(input_device),
                    attention_mask=torch.ones(1, prompt_ids.numel(), dtype=torch.long, device=input_device),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )[0].detach().cpu()
            gen_ids = output[prompt_ids.numel() :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            parsed_calls = parse_tool_calls(text)
            parsed_call = bool(parsed_calls)
            failed = not normalized_prediction_ok(parsed_calls, score_row)
            stats["rows"] += 1
            stats["failures" if failed else "successes"] += 1
            stats["parsed_calls"] += int(parsed_call)
            if failed and args.require_parsed_call and not parsed_call:
                stats["skipped_unparsed"] += 1
                continue
            if not failed:
                if not args.include_successes or random.random() > args.success_sample_rate:
                    continue
                stats["kept_successes"] += 1
            weight = args.success_prefix_weight if not failed else args.failed_prefix_weight
            if parsed_call:
                weight *= args.parsed_prefix_weight
            for prefix_len in choose_prefix_lengths(int(gen_ids.numel()), args.prefixes_per_row):
                if prefix_len < args.min_generated_prefix_tokens:
                    continue
                input_ids = torch.cat([prompt_ids, gen_ids[:prefix_len]], dim=0)
                if input_ids.numel() < 2 or input_ids.numel() > args.max_seq_length:
                    continue
                kl_mask = torch.zeros(input_ids.numel(), dtype=torch.bool)
                kl_mask[-1] = True
                states.append(
                    PrefixState(
                        input_ids=input_ids,
                        kl_logit_mask=kl_mask,
                        source_id=str(score_row.get("id", score_row.get("mix_id", idx))),
                        prefix_len=int(prefix_len),
                        failed_rollout=failed,
                        parsed_call=parsed_call,
                        weight=float(weight),
                    )
                )
            stats["states"] = len(states)
            if stats["rows"] % 100 == 0:
                print(json.dumps({"rollout_rows": stats["rows"], "states": len(states), "failures": stats["failures"]}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return states, stats


def install_zero_keep_hooks(model, mask):
    hooks = []
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        keep = mask[layer_idx].to(dtype=torch.bool)
        keep_idx = torch.where(keep)[0].to(torch.long)

        def make_hook(idx):
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


def collate_prefix_states(rows: list[PrefixState], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(row.input_ids.numel()) for row in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    kl_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    opd_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    offline_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    kl_weight = torch.zeros((len(rows), max_len), dtype=torch.float32)
    for idx, row in enumerate(rows):
        n = int(row.input_ids.numel())
        input_ids[idx, :n] = row.input_ids
        attention_mask[idx, :n] = 1
        kl_mask[idx, :n] = row.kl_logit_mask
        opd_mask[idx, :n] = row.kl_logit_mask
        kl_weight[idx, :n][row.kl_logit_mask] = float(row.weight)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kl_logit_mask": kl_mask,
        "opd_kl_mask": opd_mask,
        "offline_kl_mask": offline_mask,
        "kl_weight": kl_weight,
    }


def combine_kl_batches(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], pad_id: int) -> dict[str, torch.Tensor]:
    left = dict(left)
    right = dict(right)
    for batch, is_opd in ((left, True), (right, False)):
        if "opd_kl_mask" not in batch:
            batch["opd_kl_mask"] = batch["kl_logit_mask"] if is_opd else torch.zeros_like(batch["kl_logit_mask"])
        if "offline_kl_mask" not in batch:
            batch["offline_kl_mask"] = torch.zeros_like(batch["kl_logit_mask"]) if is_opd else batch["kl_logit_mask"]
        if "kl_weight" not in batch:
            batch["kl_weight"] = batch["kl_logit_mask"].to(torch.float32)
    keys = ("input_ids", "attention_mask", "kl_logit_mask", "opd_kl_mask", "offline_kl_mask", "kl_weight")
    max_len = max(int(left["input_ids"].shape[1]), int(right["input_ids"].shape[1]))
    out = {}
    for key in keys:
        chunks = []
        for batch in (left, right):
            value = batch[key]
            pad_len = max_len - int(value.shape[1])
            if pad_len:
                pad_value = pad_id if key == "input_ids" else 0
                pad = torch.full((value.shape[0], pad_len), pad_value, dtype=value.dtype)
                value = torch.cat([value, pad], dim=1)
            chunks.append(value)
        out[key] = torch.cat(chunks, dim=0)
    return out


def next_from(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def prefix_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    kind: str,
    temperature: float,
    skew_alpha: float,
    teacher_confidence_power: float,
) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return student_logits.new_zeros(())
    s = student_logits[mask] / temperature
    t = teacher_logits[mask] / temperature
    w = weights[mask].to(dtype=s.dtype)
    log_s = torch.nn.functional.log_softmax(s, dim=-1)
    log_t = torch.nn.functional.log_softmax(t, dim=-1)
    p_s = log_s.exp()
    p_t = log_t.exp()
    eps = torch.finfo(s.dtype).tiny

    if kind == "fkl":
        per_token = (p_t * (log_t - log_s)).sum(dim=-1)
    elif kind == "rkl":
        per_token = (p_s * (log_s - log_t)).sum(dim=-1)
    elif kind == "jsd":
        mix = 0.5 * (p_t + p_s)
        log_mix = mix.clamp_min(eps).log()
        per_token = 0.5 * (p_t * (log_t - log_mix)).sum(dim=-1) + 0.5 * (p_s * (log_s - log_mix)).sum(dim=-1)
    elif kind == "skl":
        mix = (1.0 - skew_alpha) * p_t + skew_alpha * p_s.detach()
        log_mix = mix.clamp_min(eps).log()
        per_token = (mix * (log_mix - log_s)).sum(dim=-1)
    elif kind == "srkl":
        mix = (1.0 - skew_alpha) * p_t + skew_alpha * p_s.detach()
        log_mix = mix.clamp_min(eps).log()
        per_token = (p_s * (log_s - log_mix)).sum(dim=-1)
    else:
        raise ValueError(f"unknown distillation loss: {kind}")

    if teacher_confidence_power > 0:
        entropy = -(p_t * log_t).sum(dim=-1)
        confidence = (1.0 - entropy / math.log(float(student_logits.shape[-1]))).clamp(min=0.0, max=1.0)
        w = w * confidence.pow(teacher_confidence_power)
    return (per_token * w).sum() / w.sum().clamp_min(1.0) * (temperature**2)


def save_state_manifest(path: Path, states: list[PrefixState], stats: dict[str, Any]) -> None:
    rows = [
        {
            "source_id": state.source_id,
            "prefix_len": state.prefix_len,
            "failed_rollout": state.failed_rollout,
            "parsed_call": state.parsed_call,
            "weight": state.weight,
            "n_tokens": int(state.input_ids.numel()),
        }
        for state in states
    ]
    path.write_text(json.dumps({"stats": stats, "states": rows}, indent=2))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.train_jsonl)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[data] rows={len(rows)} train={args.train_jsonl}", flush=True)

    mask = load_topk_mask(args.attribution, args.topk)
    kept = sum(int(v.sum().item()) for v in mask.values())
    print(f"[mask] topk={args.topk} kept={kept} attribution={args.attribution}", flush=True)

    states, rollout_stats = collect_prefix_states(args, rows, tokenizer, mask)
    if not states:
        raise ValueError("no OPD prefix states collected")
    save_state_manifest(args.out_dir / "opd_prefix_states_manifest.json", states, rollout_stats)
    print(f"[opd] states={len(states)} stats={rollout_stats}", flush=True)

    dataset = PrefixStateDataset(states)
    opd_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda xs: collate_prefix_states(xs, tokenizer.pad_token_id),
    )
    offline_dataset = ToolMindDataset(rows, tokenizer, args.max_seq_length)
    offline_loader = DataLoader(
        offline_dataset,
        batch_size=args.offline_batch_size,
        shuffle=True,
        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id),
    )

    print(f"[teacher/student] loading base {args.teacher_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    base.config.use_cache = False
    input_device = model_input_device(base)
    n_layers = int(base.config.num_hidden_layers)
    d_ffn = int(base.config.intermediate_size)

    print(f"[mean] building cache n={args.n_calib}", flush=True)
    means = build_mean_cache(base, rows, tokenizer, args, n_layers=n_layers, d_ffn=d_ffn, dtype=dtype)

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
    model.print_trainable_parameters()
    model.train()

    total_batches = math.ceil(len(opd_loader) * args.epochs)
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps * args.grad_accum)
    total_steps = math.ceil(total_batches / args.grad_accum)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    summary = {
        "method": "true_opd_prefix_weighted_source_loss_v1",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rollout_stats": rollout_stats,
        "n_states": len(states),
        "n_offline_rows": len(offline_dataset),
        "mix": "each training microbatch concatenates OPD prefix states with offline v2 gold-prefix states",
        "losses": {
            "opd_loss": args.opd_loss,
            "offline_loss": args.offline_loss,
            "opd_beta": args.opd_beta,
            "offline_beta": args.offline_beta,
            "skew_alpha": args.skew_alpha,
            "teacher_confidence_power": args.teacher_confidence_power,
        },
        "mask_kept": kept,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "logs": [],
        "checkpoints": [],
    }
    (args.out_dir / "config.json").write_text(json.dumps(summary, indent=2))

    start = time.time()
    global_step = 0
    seen_batches = 0
    running = {"loss": 0.0, "opd_loss": 0.0, "offline_loss": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)
    offline_iter = iter(offline_loader)
    while global_step < total_steps:
        for raw_opd_batch in opd_loader:
            if global_step >= total_steps:
                break
            raw_offline_batch, offline_iter = next_from(offline_iter, offline_loader)
            raw_batch = combine_kl_batches(raw_opd_batch, raw_offline_batch, tokenizer.pad_token_id)
            batch = move_batch(raw_batch, str(input_device))
            with torch.no_grad(), model.disable_adapter():
                teacher_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
            hooks = install_mean_ablation_hooks(model, mask, means, dtype=dtype)
            try:
                student_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
            finally:
                for hook in hooks:
                    hook.remove()
            opd_loss = prefix_distill_loss(
                student_logits,
                teacher_logits,
                batch["opd_kl_mask"],
                batch["kl_weight"],
                kind=args.opd_loss,
                temperature=args.kl_temperature,
                skew_alpha=args.skew_alpha,
                teacher_confidence_power=args.teacher_confidence_power,
            )
            offline_loss = prefix_distill_loss(
                student_logits,
                teacher_logits,
                batch["offline_kl_mask"],
                batch["kl_weight"],
                kind=args.offline_loss,
                temperature=args.kl_temperature,
                skew_alpha=args.skew_alpha,
                teacher_confidence_power=args.teacher_confidence_power,
            )
            loss = args.opd_beta * opd_loss + args.offline_beta * offline_loss
            (loss / args.grad_accum).backward()
            seen_batches += 1
            running["loss"] += float(loss.detach().cpu())
            running["opd_loss"] += float(opd_loss.detach().cpu())
            running["offline_loss"] += float(offline_loss.detach().cpu())
            running["n"] += 1
            if seen_batches % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step == 1 or global_step % args.eval_every == 0 or global_step == total_steps:
                denom = max(running["n"], 1)
                row = {
                    "step": global_step,
                    "loss": running["loss"] / denom,
                    "opd_loss": running["opd_loss"] / denom,
                    "offline_loss": running["offline_loss"] / denom,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - start,
                }
                summary["logs"].append(row)
                (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(row), flush=True)
                running = {"loss": 0.0, "opd_loss": 0.0, "offline_loss": 0.0, "n": 0}
            if args.save_every and global_step % args.save_every == 0:
                checkpoint_dir = save_adapter_checkpoint(model, tokenizer, args.out_dir, global_step)
                summary["checkpoints"].append({"step": global_step, "adapter_dir": str(checkpoint_dir)})
                (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
                print(f"[checkpoint] step={global_step} adapter={checkpoint_dir}", flush=True)

    model.eval()
    summary["elapsed_s"] = time.time() - start
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
