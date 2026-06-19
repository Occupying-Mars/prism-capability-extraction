#!/usr/bin/env python3
"""Train BFCL masked LoRA conditioners on ToolMind-style rows."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    p.add_argument("--max-memory", default=None, help="Optional comma list, e.g. 0:46GiB,1:46GiB")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--masked-kl-beta", type=float, default=1.0)
    p.add_argument("--ce-beta", type=float, default=0.2)
    p.add_argument("--unmasked-kl-beta", type=float, default=0.05)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0, help="Save LoRA adapter checkpoint every N optimizer steps.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-merged", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_topk_mask(path: Path, k: int) -> dict[int, torch.Tensor]:
    scores = torch.tensor(np.load(path)["mlp_scores"])
    flat = scores.flatten()
    k = min(k, flat.numel())
    idx = torch.topk(flat, k=k).indices
    d_ffn = scores.shape[1]
    out = {layer: torch.zeros(d_ffn, dtype=torch.bool) for layer in range(scores.shape[0])}
    for item in idx.tolist():
        out[item // d_ffn][item % d_ffn] = True
    return out


def encode_row(row: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, Any] | None:
    target_text = (row.get("target_text") or "").strip()
    if not target_text:
        return None
    prompt = tokenizer.apply_chat_template(
        row["messages"],
        tools=row.get("tools") or None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        enable_thinking=False,
    )
    prompt_ids = list(prompt["input_ids"])
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    if not target_ids:
        return None
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_length:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    kl_mask = [False] * len(input_ids)
    for idx in range(max(len(prompt_ids) - 1, 0), len(input_ids) - 1):
        kl_mask[idx] = True
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "kl_logit_mask": torch.tensor(kl_mask, dtype=torch.bool),
    }


class ToolMindDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_seq_length: int):
        self.rows = []
        dropped = 0
        for row in rows:
            enc = encode_row(row, tokenizer, max_seq_length)
            if enc is None:
                dropped += 1
                continue
            self.rows.append(enc)
        print(f"[data] kept={len(self.rows)} dropped={dropped}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate(rows: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(r["input_ids"].shape[0]) for r in rows)
    batch = {}
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


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def first_param_device(module) -> torch.device:
    return next(module.parameters()).device


def model_input_device(model) -> torch.device:
    try:
        return first_param_device(model.get_input_embeddings())
    except Exception:
        return first_param_device(model)


def layer_device(layer) -> torch.device:
    return first_param_device(layer)


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


def build_mean_cache(model, rows, tokenizer, args, *, n_layers: int, d_ffn: int, dtype):
    layers = list(decoder_root(model).layers)
    sums = {i: torch.zeros(d_ffn, device=layer_device(layer), dtype=torch.float32) for i, layer in enumerate(layers)}
    counts = {i: 0 for i in range(n_layers)}
    hooks = []
    input_device = model_input_device(model)

    def make_hook(layer_idx: int):
        def hook_fn(module, hook_args, output):
            act = hook_args[0].detach().to(torch.float32)
            flat = act.reshape(-1, act.shape[-1])
            sums[layer_idx] += flat.sum(dim=0)
            counts[layer_idx] += flat.shape[0]
        return hook_fn

    for i, layer in enumerate(layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            for row in rows[: args.n_calib]:
                enc = encode_row(row, tokenizer, args.max_seq_length)
                if enc is None:
                    continue
                model(enc["prompt_ids"].unsqueeze(0).to(input_device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {i: (sums[i] / max(counts[i], 1)).to(dtype=dtype) for i in range(n_layers)}


def install_mean_ablation_hooks(model, mask, means, *, dtype):
    hooks = []
    for layer_idx, layer in enumerate(decoder_root(model).layers):
        device = layer_device(layer)
        keep = mask[layer_idx].to(device=device).to(dtype=dtype).view(1, 1, -1)
        mean = means[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)

        def make_hook(keep_tensor, mean_tensor):
            def hook_fn(module, hook_args):
                act = hook_args[0]
                return (act * keep_tensor + mean_tensor * (1.0 - keep_tensor),) + hook_args[1:]
            return hook_fn

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(keep, mean)))
    return hooks


def answer_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor, *, temperature: float) -> torch.Tensor:
    logit_mask = mask[:, :-1]
    if int(logit_mask.sum().item()) == 0:
        return student_logits.new_zeros(())
    s = student_logits[:, :-1, :][logit_mask] / temperature
    t = teacher_logits[:, :-1, :][logit_mask] / temperature
    return F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def save_adapter_checkpoint(model, tokenizer, out_dir: Path, step: int) -> Path:
    checkpoint_dir = out_dir / "checkpoints" / f"step_{step:06d}" / "adapter"
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


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

    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(","):
            key, value = item.split(":", 1)
            max_memory[int(key.strip())] = value.strip()

    print(
        f"[model] {args.model} device={args.device} device_map={args.device_map} dtype={args.dtype}",
        flush=True,
    )
    load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": "eager",
    }
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
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    summary = {
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
    running = {"loss": 0.0, "masked_kl": 0.0, "ce": 0.0, "unmasked_kl": 0.0, "n": 0}
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
                masked_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
            finally:
                for h in hooks:
                    h.remove()

            masked_kl = kl_loss(masked_logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
            ce = answer_ce_loss(masked_logits, batch["labels"])
            if args.unmasked_kl_beta > 0:
                unmasked_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                unmasked_kl = kl_loss(unmasked_logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
            else:
                unmasked_kl = masked_logits.new_zeros(())

            loss = args.masked_kl_beta * masked_kl + args.ce_beta * ce + args.unmasked_kl_beta * unmasked_kl
            (loss / args.grad_accum).backward()
            seen_batches += 1
            running["loss"] += float(loss.detach().cpu())
            running["masked_kl"] += float(masked_kl.detach().cpu())
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
                    "masked_kl": running["masked_kl"] / denom,
                    "ce": running["ce"] / denom,
                    "unmasked_kl": running["unmasked_kl"] / denom,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - start,
                }
                summary["logs"].append(row)
                (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(row), flush=True)
                running = {"loss": 0.0, "masked_kl": 0.0, "ce": 0.0, "unmasked_kl": 0.0, "n": 0}
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
