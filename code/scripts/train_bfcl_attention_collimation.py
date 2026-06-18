#!/usr/bin/env python3
"""Collimate a BFCL adapter under a fixed MLP + attention substrate."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_common.wandb_utils import add_wandb_args, init_wandb_run, log_wandb_receipts
from scripts.bfcl_attention_qwen3 import (
    install_attention_keep_hooks,
    install_mlp_keep_hooks,
    load_attention_scores,
    load_mlp_keep_mask,
    make_keep_mask,
    qwen_attention_shape,
)
from scripts.bfcl_direct_qwen3 import (
    format_tool_call_target,
    messages_for_generation,
    read_records,
)
from scripts.train_bfcl_masked_lora import answer_ce_loss, kl_loss, save_adapter_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--init-adapter", type=Path)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--attention-attribution", type=Path, required=True)
    p.add_argument("--attention-topk", type=int, required=True)
    p.add_argument("--attention-unit", choices=["head", "ov"], default="ov")
    p.add_argument("--attention-ablation", choices=["zero"], default="zero")
    p.add_argument("--mlp-attribution", type=Path, required=True)
    p.add_argument("--mlp-topk", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--bfcl-canonicalization-prompt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--policy-kl-beta", type=float, default=1.0)
    p.add_argument("--ce-beta", type=float, default=0.1)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument(
        "--teacher-mode",
        choices=["fixed", "dynamic"],
        default="fixed",
        help="fixed loads a frozen full-attn teacher; dynamic preserves the original moving-teacher behavior.",
    )
    p.add_argument("--teacher-model", default=None, help="Teacher base model; defaults to --model.")
    p.add_argument("--teacher-adapter", type=Path, help="Teacher adapter; defaults to --init-adapter.")
    p.add_argument("--teacher-device", default=None, help="Teacher device when --teacher-device-map is unset; defaults to --device.")
    p.add_argument("--teacher-device-map")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--tf-eval-rows", type=int, default=64)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-name", default="issue3-attention-collimation")
    add_wandb_args(
        p,
        default_project="prism-bfcl-attention",
        default_job_type="attention-collimation",
        default_tags="bfcl,qwen3,attention,collimation",
    )
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def encode_pair(row: dict[str, Any], tokenizer, args: argparse.Namespace) -> dict[str, Any] | None:
    prompt = tokenizer.apply_chat_template(
        messages_for_generation(row, bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt),
        tools=row.get("tools") or None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        enable_thinking=False,
    )
    prompt_ids = list(prompt["input_ids"])
    target_text = format_tool_call_target(row)
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    if not target_ids:
        return None
    input_ids = prompt_ids + target_ids
    if len(input_ids) > args.max_seq_length:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    kl_mask = [False] * len(input_ids)
    for idx in range(max(len(prompt_ids) - 1, 0), len(input_ids) - 1):
        kl_mask[idx] = True
    return {
        "id": row.get("id"),
        "category": row.get("category", "unknown"),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "kl_logit_mask": torch.tensor(kl_mask, dtype=torch.bool),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "prompt_len": len(prompt_ids),
    }


class BfclPairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, args: argparse.Namespace):
        self.rows = []
        dropped = Counter()
        for row in rows:
            try:
                enc = encode_pair(row, tokenizer, args)
            except Exception:
                dropped["encode_error"] += 1
                continue
            if enc is None:
                dropped["too_long_or_empty"] += 1
                continue
            self.rows.append(enc)
        print(f"[data] kept={len(self.rows)} dropped={dict(dropped)}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate(rows: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(int(row["input_ids"].shape[0]) for row in rows)
    batch: dict[str, Any] = {
        "ids": [row["id"] for row in rows],
        "categories": [row["category"] for row in rows],
        "prompt_lens": torch.tensor([row["prompt_len"] for row in rows], dtype=torch.long),
    }
    for key, pad_val, dtype in (
        ("input_ids", pad_id, torch.long),
        ("labels", -100, torch.long),
        ("attention_mask", 0, torch.long),
    ):
        value = torch.full((len(rows), max_len), pad_val, dtype=dtype)
        for idx, row in enumerate(rows):
            value[idx, : row[key].shape[0]] = row[key]
        batch[key] = value
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for idx, row in enumerate(rows):
        mask[idx, : row["kl_logit_mask"].shape[0]] = row["kl_logit_mask"]
    batch["kl_logit_mask"] = mask
    return batch


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def input_device(model) -> torch.device:
    return next(model.parameters()).device


def load_causal_lm(model_name: str, args: argparse.Namespace, dtype: torch.dtype, *, device: str, device_map: str | None):
    load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": "eager",
        "local_files_only": args.local_files_only,
    }
    if device_map:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if not device_map:
        model = model.to(device)
    model.config.use_cache = False
    return model


def masked_student_forward(
    model,
    batch: dict[str, Any],
    *,
    attention_keep: torch.Tensor | None,
    mlp_keep: torch.Tensor,
    args: argparse.Namespace,
):
    mlp_hooks = install_mlp_keep_hooks(model, mlp_keep)
    attn_hooks = []
    if attention_keep is not None:
        attn_hooks = install_attention_keep_hooks(model, attention_keep, ablation=args.attention_ablation, means=None)
    try:
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    finally:
        for hook in [*attn_hooks, *mlp_hooks]:
            hook.remove()


def teacher_forward(model, batch: dict[str, Any], *, mlp_keep: torch.Tensor):
    hooks = install_mlp_keep_hooks(model, mlp_keep)
    try:
        with torch.no_grad():
            return model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def teacher_forced_probe(
    model,
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    attention_keep: torch.Tensor | None,
    mlp_keep: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    prefix: str = "tf_probe",
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    token_correct = 0
    token_total = 0
    seq_correct = 0
    examples = 0
    for row in rows[: args.tf_eval_rows]:
        enc = encode_pair(row, tokenizer, args)
        if enc is None:
            continue
        batch = move_tensor_batch(collate([enc], tokenizer.pad_token_id), device)
        logits = masked_student_forward(
            model,
            batch,
            attention_keep=attention_keep,
            mlp_keep=mlp_keep,
            args=args,
        )
        logit_mask = batch["kl_logit_mask"][:, :-1]
        target = batch["input_ids"][:, 1:][logit_mask]
        pred = logits[:, :-1, :].argmax(dim=-1)[logit_mask]
        n_correct = int((pred == target).sum().item())
        n_tokens = int(target.numel())
        token_correct += n_correct
        token_total += n_tokens
        seq_correct += int(n_correct == n_tokens)
        examples += 1
    if was_training:
        model.train()
    return {
        f"{prefix}_examples": examples,
        f"{prefix}_target_tokens": token_total,
        f"{prefix}_target_token_accuracy": token_correct / token_total if token_total else None,
        f"{prefix}_sequence_argmax_exact": seq_correct,
        f"{prefix}_sequence_argmax_exact_accuracy": seq_correct / examples if examples else None,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)

    rows = read_records(args.pairs)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[data] raw_rows={len(rows)} pairs={args.pairs}", flush=True)

    tokenizer_name = args.init_adapter or args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=args.local_files_only)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = BfclPairDataset(rows, tokenizer, args)
    if len(dataset) == 0:
        raise ValueError("no train rows survived encoding")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id),
    )

    base = load_causal_lm(args.model, args, dtype, device=args.device, device_map=args.device_map)

    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, local_files_only=args.local_files_only, is_trainable=True)
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
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()

    teacher_model = None
    if args.teacher_mode == "fixed":
        teacher_model_name = args.teacher_model or args.model
        teacher_adapter = args.teacher_adapter or args.init_adapter
        teacher_device = args.teacher_device or args.device
        teacher_base = load_causal_lm(
            teacher_model_name,
            args,
            dtype,
            device=teacher_device,
            device_map=args.teacher_device_map,
        )
        if teacher_adapter:
            teacher_model = PeftModel.from_pretrained(
                teacher_base,
                teacher_adapter,
                local_files_only=args.local_files_only,
                is_trainable=False,
            )
        else:
            teacher_model = teacher_base
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    n_layers, _n_heads, hidden, _head_dim = qwen_attention_shape(model)
    mlp_keep, mlp_info = load_mlp_keep_mask(
        args.mlp_attribution,
        args.mlp_topk,
        (n_layers, int(model.config.intermediate_size)),
    )
    if mlp_keep is None:
        raise ValueError("--mlp-attribution is required")
    head_scores, ov_scores = load_attention_scores(args.attention_attribution)
    attention_keep, attention_info = make_keep_mask(
        head_scores=head_scores,
        ov_scores=ov_scores,
        unit=args.attention_unit,
        topk=args.attention_topk,
        random_seed=None,
    )
    attention_info["attention_ablation"] = args.attention_ablation

    total_batches = math.ceil(len(loader) * args.epochs)
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps * args.grad_accum)
    total_steps = max(math.ceil(total_batches / args.grad_accum), 1)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    summary: dict[str, Any] = {
        "method": f"bfcl_attention_collimation_v2_{args.teacher_mode}_teacher",
        "contract": {
            "teacher": (
                "frozen teacher adapter, fixed MLP mask, full attention"
                if args.teacher_mode == "fixed"
                else "same trainable adapter, fixed MLP mask, full attention"
            ),
            "student": "trainable adapter, fixed MLP mask, fixed attention keep mask",
            "behavior_gate": "must evaluate with full autoregressive BFCL normalized exact after training",
        },
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_rows": len(dataset),
        "n_layers": n_layers,
        "hidden": hidden,
        "mlp_mask": mlp_info,
        "attention_mask": attention_info,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "logs": [],
        "checkpoints": [],
    }
    config_path = args.out_dir / "config.json"
    train_summary_path = args.out_dir / "train_summary.json"
    config_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    train_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    run = init_wandb_run(
        args,
        default_project="prism-bfcl-attention",
        default_group=args.run_name,
        default_job_type="attention-collimation",
        default_name=args.run_name,
        default_tags="bfcl,qwen3,attention,collimation",
        config={
            "cmd": "train_bfcl_attention_collimation",
            "model": args.model,
            "init_adapter": str(args.init_adapter) if args.init_adapter else None,
            "pairs": str(args.pairs),
            "n_rows": len(dataset),
            "mlp_mask": mlp_info,
            "attention_mask": attention_info,
            "loss": {"policy_kl_beta": args.policy_kl_beta, "ce_beta": args.ce_beta},
            "teacher": {
                "mode": args.teacher_mode,
                "model": args.teacher_model or args.model,
                "adapter": str(args.teacher_adapter or args.init_adapter) if (args.teacher_adapter or args.init_adapter) else None,
            },
        },
    )

    device = input_device(model)
    teacher_device = input_device(teacher_model) if teacher_model is not None else device
    mlp_keep = mlp_keep.to(device="cpu")
    attention_keep = attention_keep.to(device="cpu")
    start = time.time()
    global_step = 0
    seen_batches = 0
    running = Counter()
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        for raw_batch in loader:
            if global_step >= total_steps:
                break
            batch = move_tensor_batch(raw_batch, device)
            if teacher_model is None:
                teacher_logits = teacher_forward(model, batch, mlp_keep=mlp_keep)
            else:
                teacher_batch = batch if teacher_device == device else move_tensor_batch(raw_batch, teacher_device)
                teacher_logits = teacher_forward(teacher_model, teacher_batch, mlp_keep=mlp_keep)
            student_logits = masked_student_forward(
                model,
                batch,
                attention_keep=attention_keep,
                mlp_keep=mlp_keep,
                args=args,
            )
            if teacher_logits.device != student_logits.device:
                teacher_logits = teacher_logits.to(student_logits.device)
            policy_kl = kl_loss(student_logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
            ce = answer_ce_loss(student_logits, batch["labels"])
            loss = args.policy_kl_beta * policy_kl + args.ce_beta * ce
            (loss / args.grad_accum).backward()

            running["loss"] += float(loss.detach().cpu())
            running["policy_kl"] += float(policy_kl.detach().cpu())
            running["ce"] += float(ce.detach().cpu())
            running["n"] += 1
            seen_batches += 1
            if seen_batches % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % args.eval_every == 0 or global_step == total_steps:
                denom = max(int(running["n"]), 1)
                row = {
                    "step": global_step,
                    "loss": running["loss"] / denom,
                    "policy_kl": running["policy_kl"] / denom,
                    "ce": running["ce"] / denom,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - start,
                }
                row.update(
                    teacher_forced_probe(
                        model,
                        rows,
                        tokenizer,
                        attention_keep=attention_keep,
                        mlp_keep=mlp_keep,
                        args=args,
                        device=device,
                        prefix="tf_probe",
                    )
                )
                row.update(
                    teacher_forced_probe(
                        model,
                        rows,
                        tokenizer,
                        attention_keep=None,
                        mlp_keep=mlp_keep,
                        args=args,
                        device=device,
                        prefix="tf_full_attn_probe",
                    )
                )
                summary["logs"].append(row)
                train_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                if run is not None:
                    run.log({f"train/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=global_step)
                running = Counter()

            if args.save_every and global_step % args.save_every == 0:
                checkpoint_dir = save_adapter_checkpoint(model, tokenizer, args.out_dir, global_step)
                summary["checkpoints"].append({"step": global_step, "adapter_dir": str(checkpoint_dir)})
                train_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                print(f"[checkpoint] step={global_step} adapter={checkpoint_dir}", flush=True)

    model.eval()
    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary["adapter_dir"] = str(adapter_dir)
    summary["elapsed_s"] = time.time() - start
    train_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[done] adapter={adapter_dir}", flush=True)

    if run is not None:
        log_wandb_receipts(run, name=f"{args.run_name}-attention-collimation", files=[config_path, train_summary_path])
        run.finish()


if __name__ == "__main__":
    main()
