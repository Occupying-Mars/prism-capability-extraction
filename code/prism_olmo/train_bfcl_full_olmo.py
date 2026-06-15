#!/usr/bin/env python3
"""Train OLMo BFCL full-substrate LoRA baselines."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism_olmo.bfcl_direct_olmo import format_olmo_target, olmo_prompt_text  # noqa: E402
from scripts.train_bfcl_masked_lora import answer_ce_loss, collate, kl_loss, move_batch, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMo-7B-0724-Instruct-hf")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-seq-length", type=int, default=1536)
    p.add_argument("--call-tag", choices=["tool_call", "function_call"], default="function_call")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--policy-kl-beta", type=float, default=1.0)
    p.add_argument("--ce-beta", type=float, default=0.2)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def encode_row(row: dict[str, Any], tokenizer, max_seq_length: int, call_tag: str) -> dict[str, torch.Tensor] | None:
    try:
        target_text = format_olmo_target(row, call_tag=call_tag)
    except ValueError:
        return None
    prompt_ids = tokenizer(olmo_prompt_text(row, call_tag=call_tag), add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_length:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    kl_mask = [False] * len(input_ids)
    for idx in range(max(len(prompt_ids) - 1, 0), len(input_ids) - 1):
        kl_mask[idx] = True
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "kl_logit_mask": torch.tensor(kl_mask, dtype=torch.bool),
    }


class OLMOSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_seq_length: int, call_tag: str):
        self.rows = []
        dropped = 0
        for row in rows:
            enc = encode_row(row, tokenizer, max_seq_length, call_tag)
            if enc is None:
                dropped += 1
            else:
                self.rows.append(enc)
        print(f"[data] kept={len(self.rows)} dropped={dropped}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.rows[idx]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = read_jsonl(args.train_jsonl)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[data] raw_rows={len(rows)} train={args.train_jsonl} call_tag={args.call_tag}", flush=True)
    dataset = OLMOSFTDataset(rows, tokenizer, args.max_seq_length, args.call_tag)
    if not dataset:
        raise ValueError("no train rows survived encoding")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device)
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        use_rslora=args.use_rslora,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]

    total_batches = math.ceil(len(loader) * args.epochs)
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps * args.grad_accum)
    total_steps = math.ceil(total_batches / args.grad_accum)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 0)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    summary_path = args.out_dir / "train_summary.json"
    summary: dict[str, Any] = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_mode": "lora",
        "n_rows": len(dataset),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "logs": [],
        "checkpoints": [],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    start = time.time()
    global_step = 0
    seen_batches = 0
    running = {"loss": 0.0, "policy_kl": 0.0, "ce": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)
    while global_step < total_steps:
        for raw_batch in loader:
            if global_step >= total_steps:
                break
            batch = move_batch(raw_batch, args.device)
            teacher_logits = None
            if args.policy_kl_beta:
                with torch.no_grad(), model.disable_adapter():
                    teacher_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    ).logits
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            ce = answer_ce_loss(out.logits, batch["labels"])
            if teacher_logits is None:
                policy_kl = torch.zeros((), device=ce.device, dtype=ce.dtype)
            else:
                policy_kl = kl_loss(out.logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
            loss = args.policy_kl_beta * policy_kl + args.ce_beta * ce
            (loss / args.grad_accum).backward()
            seen_batches += 1
            running["loss"] += float(loss.detach().cpu())
            running["policy_kl"] += float(policy_kl.detach().cpu())
            running["ce"] += float(ce.detach().cpu())
            running["n"] += 1

            if seen_batches % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % args.log_every == 0 or global_step == total_steps:
                denom = max(running["n"], 1)
                row = {
                    "step": global_step,
                    "loss": running["loss"] / denom,
                    "policy_kl": running["policy_kl"] / denom,
                    "ce": running["ce"] / denom,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - start,
                }
                summary["logs"].append(row)
                summary_path.write_text(json.dumps(summary, indent=2))
                print(json.dumps(row), flush=True)
                running = {"loss": 0.0, "policy_kl": 0.0, "ce": 0.0, "n": 0}

            if args.save_every and global_step % args.save_every == 0:
                ckpt = args.out_dir / f"checkpoint_step_{global_step:06d}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                summary["checkpoints"].append({"step": global_step, "path": str(ckpt)})
                summary_path.write_text(json.dumps(summary, indent=2))

    final_dir = args.out_dir / "adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    summary["final_dir"] = str(final_dir)
    summary["elapsed_s"] = time.time() - start
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"done": True, "out_dir": str(args.out_dir), "final_dir": str(final_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
