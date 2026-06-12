#!/usr/bin/env python3
"""Train GLM BFCL full-substrate baselines.

Two modes are supported:

* ``lora``: supervised LoRA on the full model, no MLP masking.
* ``full_sft``: full-parameter supervised finetuning, no MLP masking.
"""

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
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

if hasattr(torch, "distributed") and not hasattr(torch.distributed, "tensor"):
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        pass

from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism_glm.bfcl_direct_glm import format_glm_native_target, glm_native_prompt_text, glm_prompt_text  # noqa: E402
from scripts.train_bfcl_masked_lora import answer_ce_loss, collate, move_batch, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="zai-org/glm-4-9b-chat")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-mode", choices=["lora", "full_sft"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--prompt-format", choices=["manual_tool_call", "glm_native"], default="glm_native")
    p.add_argument("--target-format", choices=["tool_call", "glm_native"], default="glm_native")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--optimizer", choices=["adamw", "adafactor"], default=None)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def target_text_for_row(row: dict[str, Any], target_format: str) -> str:
    if target_format == "glm_native":
        try:
            return format_glm_native_target(row)
        except ValueError:
            return ""
    if row.get("target_text"):
        return str(row["target_text"]).strip()
    call = row.get("target_call")
    if isinstance(call, dict):
        return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
    return ""


def encode_row(row: dict[str, Any], tokenizer, max_seq_length: int, prompt_format: str, target_format: str):
    target_text = target_text_for_row(row, target_format)
    if not target_text:
        return None
    if prompt_format == "glm_native":
        prompt_ids = tokenizer(glm_native_prompt_text(tokenizer, row), add_special_tokens=False)["input_ids"]
    else:
        prompt_ids = tokenizer(glm_prompt_text(row), add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    if eos is not None:
        target_ids = target_ids + [int(eos)]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_length:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "kl_logit_mask": torch.ones(len(input_ids), dtype=torch.bool),
    }


class GLMSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_seq_length: int, prompt_format: str, target_format: str):
        self.rows = []
        dropped = 0
        for row in rows:
            enc = encode_row(row, tokenizer, max_seq_length, prompt_format, target_format)
            if enc is None:
                dropped += 1
            else:
                self.rows.append(enc)
        print(f"[data] kept={len(self.rows)} dropped={dropped}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_optimizer(args: argparse.Namespace, params):
    optimizer_name = args.optimizer or ("adamw" if args.train_mode == "lora" else "adafactor")
    lr = args.lr if args.lr is not None else (2e-4 if args.train_mode == "lora" else 2e-5)
    if optimizer_name == "adafactor":
        return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=args.weight_decay), optimizer_name, lr
    return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay), optimizer_name, lr


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = read_jsonl(args.train_jsonl)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(
        f"[data] raw_rows={len(rows)} train={args.train_jsonl} "
        f"prompt_format={args.prompt_format} target_format={args.target_format}",
        flush=True,
    )
    dataset = GLMSFTDataset(rows, tokenizer, args.max_seq_length, args.prompt_format, args.target_format)
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
        trust_remote_code=True,
    ).to(args.device)
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.train_mode == "lora":
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
    else:
        for param in model.parameters():
            param.requires_grad_(True)

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_batches = math.ceil(len(loader) * args.epochs)
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps * args.grad_accum)
    total_steps = math.ceil(total_batches / args.grad_accum)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 0)
    optimizer, optimizer_name, lr = make_optimizer(args, trainable)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    summary: dict[str, Any] = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_mode": args.train_mode,
        "optimizer": optimizer_name,
        "lr": lr,
        "n_rows": len(dataset),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "logs": [],
        "checkpoints": [],
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    start = time.time()
    global_step = 0
    seen_batches = 0
    running_loss = 0.0
    running_n = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < total_steps:
        for raw_batch in loader:
            if global_step >= total_steps:
                break
            batch = move_batch(raw_batch, args.device)
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = answer_ce_loss(out.logits, batch["labels"])
            (loss / args.grad_accum).backward()
            seen_batches += 1
            running_loss += float(loss.detach().cpu())
            running_n += 1
            if seen_batches % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % args.log_every == 0 or global_step == total_steps:
                row = {
                    "step": global_step,
                    "loss": running_loss / max(running_n, 1),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - start,
                }
                summary["logs"].append(row)
                (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(row), flush=True)
                running_loss = 0.0
                running_n = 0

            if args.save_every and global_step % args.save_every == 0:
                ckpt = args.out_dir / f"checkpoint_step_{global_step:06d}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                summary["checkpoints"].append({"step": global_step, "path": str(ckpt)})
                (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    summary["elapsed_s"] = time.time() - start
    if args.save_final:
        final_dir = args.out_dir / ("adapter" if args.train_mode == "lora" else "model")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        summary["final_dir"] = str(final_dir)
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"done": True, "out_dir": str(args.out_dir), "final_dir": summary.get("final_dir")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
