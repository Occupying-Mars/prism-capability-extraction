#!/usr/bin/env python3
"""Train a masked BFCL student by distilling the unmasked base policy.

This is the simplest OPD-shaped conditioning run for issue #49:

- teacher: base model, unmasked, adapters disabled
- student: same base model + LoRA, evaluated under an MLP keep mask
- data: BFCL-style prompt/tool/target rows from the v2 strict mix

The rollout part is intentionally not implemented here. This script is the
first offline policy-distillation baseline before we spend GPU time on an
iterative on-policy collector.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_bfcl_masked_lora import (  # noqa: E402
    ToolMindDataset,
    answer_ce_loss,
    build_mean_cache,
    collate,
    install_mean_ablation_hooks,
    kl_loss,
    load_topk_mask,
    model_input_device,
    move_batch,
    read_jsonl,
    save_adapter_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--train-jsonl", type=Path, default=Path("data/bfcl_strict_10k_mix_len1024/train.jsonl"))
    p.add_argument("--attribution", type=Path, default=Path("runs/issue49_bfcl_single_call/relp_full.npz"))
    p.add_argument("--topk", type=int, default=240000)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    p.add_argument("--max-memory", default=None, help="Optional comma list, e.g. 0:46GiB,1:46GiB")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int, default=1)
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
    p.add_argument("--policy-kl-beta", type=float, default=1.0)
    p.add_argument("--ce-beta", type=float, default=0.2)
    p.add_argument("--unmasked-kl-beta", type=float, default=0.0)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-merged", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def parse_max_memory(value: str | None) -> dict[int, str] | None:
    if not value:
        return None
    out = {}
    for item in value.split(","):
        key, mem = item.split(":", 1)
        out[int(key.strip())] = mem.strip()
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.train_jsonl)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[data] raw_rows={len(rows)} train={args.train_jsonl}", flush=True)
    dataset = ToolMindDataset(rows, tokenizer, args.max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id),
    )
    if not dataset:
        raise ValueError("no train rows survived encoding")

    print(
        f"[model] {args.model} teacher=unmasked_base student=masked_lora topk={args.topk} "
        f"device={args.device} device_map={args.device_map} dtype={args.dtype}",
        flush=True,
    )
    load_kwargs = {"torch_dtype": dtype, "attn_implementation": "eager"}
    max_memory = parse_max_memory(args.max_memory)
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
        if max_memory:
            load_kwargs["max_memory"] = max_memory

    base = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not args.device_map:
        base = base.to(args.device)
    base.config.use_cache = False
    input_device = model_input_device(base)
    n_layers = int(base.config.num_hidden_layers)
    d_ffn = int(base.config.intermediate_size)

    mask = load_topk_mask(args.attribution, args.topk)
    kept = sum(int(v.sum().item()) for v in mask.values())
    print(f"[mask] topk={args.topk} kept={kept} attribution={args.attribution}", flush=True)

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

    total_batches = math.ceil(len(loader) * args.epochs)
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps * args.grad_accum)
    total_steps = math.ceil(total_batches / args.grad_accum)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    summary = {
        "method": "masked_policy_distillation_offline_v0",
        "contract": {
            "teacher": "same base model, adapters disabled, no MLP mask",
            "student": "same base model plus LoRA, MLP mean-ablation keep mask active",
            "policy_states": "chat-template prompt plus gold tool-call prefix positions from train_jsonl",
        },
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_rows": len(dataset),
        "n_layers": n_layers,
        "d_ffn": d_ffn,
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
    running = {"loss": 0.0, "policy_kl": 0.0, "ce": 0.0, "unmasked_kl": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        for raw_batch in loader:
            if global_step >= total_steps:
                break
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

            policy_kl = kl_loss(student_logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
            ce = answer_ce_loss(student_logits, batch["labels"])
            if args.unmasked_kl_beta > 0:
                unmasked_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                unmasked_kl = kl_loss(unmasked_logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
            else:
                unmasked_kl = student_logits.new_zeros(())

            loss = args.policy_kl_beta * policy_kl + args.ce_beta * ce + args.unmasked_kl_beta * unmasked_kl
            (loss / args.grad_accum).backward()
            seen_batches += 1
            running["loss"] += float(loss.detach().cpu())
            running["policy_kl"] += float(policy_kl.detach().cpu())
            running["ce"] += float(ce.detach().cpu())
            running["unmasked_kl"] += float(unmasked_kl.detach().cpu())
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
                    "policy_kl": running["policy_kl"] / denom,
                    "ce": running["ce"] / denom,
                    "unmasked_kl": running["unmasked_kl"] / denom,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - start,
                }
                summary["logs"].append(row)
                (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(row), flush=True)
                running = {"loss": 0.0, "policy_kl": 0.0, "ce": 0.0, "unmasked_kl": 0.0, "n": 0}

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
