#!/usr/bin/env python3
"""Fixed-mask masked-KL conditioning for EN->PT MVC experiments.

Teacher: unmasked base model.
Student: same base with a LoRA conditioner.
Bottleneck: fixed MLP-channel mask; non-mask channels are mean-ablated during
the masked student forward.
"""

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

from translation_io import DEFAULT_PROMPT_STYLE, apply_chat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--jsonl", required=True,
                   help="jsonl rows with en plus a teacher/reference target field")
    p.add_argument("--fixed-mask", required=True,
                   help=".full.npz mask from attribute_translation.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-field", default="model_hyp",
                   help="CE target field; falls back to pt when missing")
    p.add_argument("--target-language", default="Portuguese")
    p.add_argument("--prompt-style", default=DEFAULT_PROMPT_STYLE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--mean-on", default="full", choices=["prompt", "full"])
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
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
    p.add_argument("--kl-on", default="answer", choices=["prompt", "answer", "all"])
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-merged", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def decoder_root(model):
    cur = model
    for attr in ("base_model", "model", "language_model"):
        nxt = getattr(cur, attr, None)
        if nxt is not None and nxt is not cur and hasattr(nxt, "layers"):
            return nxt
        if nxt is not None and nxt is not cur:
            cur = nxt
    cur = model
    for _ in range(5):
        if hasattr(cur, "layers"):
            return cur
        if hasattr(cur, "model") and getattr(cur, "model") is not cur:
            cur = cur.model
            continue
        if hasattr(cur, "base_model") and getattr(cur, "base_model") is not cur:
            cur = cur.base_model
            continue
        break
    raise AttributeError("could not locate decoder .layers")


def text_config(model):
    return getattr(model.config, "text_config", model.config)


def load_mask_npz(path: str | Path, n_layers: int, d_ffn: int) -> dict[int, torch.Tensor]:
    arr = np.load(path)
    out = {}
    for layer in range(n_layers):
        key = f"layer_{layer}"
        if key in arr:
            out[layer] = torch.from_numpy(arr[key]).bool()
        else:
            out[layer] = torch.zeros(d_ffn, dtype=torch.bool)
    return out


def load_rows(path: str, *, target_field: str) -> list[dict[str, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            src = (raw.get("en") or raw.get("src") or "").strip()
            tgt = (raw.get(target_field) or raw.get("pt") or raw.get("tgt") or "").strip()
            if src and tgt:
                rows.append({"en": src, "target": tgt})
    return rows


def encode_row(row: dict[str, str], tokenizer, args) -> dict[str, Any] | None:
    prompt = apply_chat(
        tokenizer, row["en"], target_language=args.target_language,
        add_generation_prompt=True, prompt_style=args.prompt_style,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    if not target_ids:
        return None
    full_ids = prompt_ids + target_ids
    if len(full_ids) > args.max_seq_length:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    mask = [False] * len(full_ids)
    if args.kl_on == "prompt":
        lo, hi = 0, max(len(prompt_ids) - 1, 0)
    elif args.kl_on == "answer":
        lo, hi = max(len(prompt_ids) - 1, 0), len(full_ids) - 1
    else:
        lo, hi = 0, len(full_ids) - 1
    for idx in range(lo, hi):
        mask[idx] = True
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "kl_logit_mask": torch.tensor(mask, dtype=torch.bool),
    }


class JsonlTranslationDataset(Dataset):
    def __init__(self, rows, tokenizer, args):
        self.rows = []
        dropped = 0
        for row in rows:
            enc = encode_row(row, tokenizer, args)
            if enc is None:
                dropped += 1
                continue
            self.rows.append(enc)
        print(f"[data] kept={len(self.rows)} dropped={dropped}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate(rows: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(int(r["input_ids"].shape[0]) for r in rows)
    batch = {}
    for key, pad_val, dtype in (
        ("input_ids", pad_id, torch.long),
        ("labels", -100, torch.long),
        ("attention_mask", 0, torch.long),
    ):
        value = torch.full((len(rows), max_len), pad_val, dtype=dtype)
        for idx, row in enumerate(rows):
            item = row[key]
            value[idx, : item.shape[0]] = item
        batch[key] = value
    kl_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for idx, row in enumerate(rows):
        item = row["kl_logit_mask"]
        kl_mask[idx, : item.shape[0]] = item
    batch["kl_logit_mask"] = kl_mask
    return batch


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_mean_cache(model, rows, tokenizer, args, *, n_layers: int, d_ffn: int,
                     dtype: torch.dtype) -> dict[int, torch.Tensor]:
    sums = {i: torch.zeros(d_ffn, device=args.device, dtype=torch.float32)
            for i in range(n_layers)}
    counts = {i: 0 for i in range(n_layers)}
    hooks = []
    root = decoder_root(model)

    def make_hook(layer_idx: int):
        def hook_fn(module, hook_args, output):
            act = hook_args[0].detach().to(torch.float32)
            flat = act.reshape(-1, act.shape[-1])
            sums[layer_idx] += flat.sum(dim=0)
            counts[layer_idx] += flat.shape[0]
        return hook_fn

    for i, layer in enumerate(root.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            for row in rows[: args.n_calib]:
                enc = encode_row(row, tokenizer, args)
                if enc is None:
                    continue
                ids = enc["prompt_ids"] if args.mean_on == "prompt" else enc["input_ids"]
                model(ids.unsqueeze(0).to(args.device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {
        i: (sums[i] / max(counts[i], 1)).to(dtype=dtype)
        for i in range(n_layers)
    }


def install_mean_ablation_hooks(model, mask, means, *, device: str,
                                dtype: torch.dtype):
    hooks = []
    root = decoder_root(model)
    for layer_idx, layer in enumerate(root.layers):
        keep_bool = mask[layer_idx].to(device)
        keep = keep_bool.to(dtype=dtype).view(1, 1, -1)
        mean = means[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)

        def make_hook(keep, mean):
            def hook_fn(module, hook_args):
                act = hook_args[0]
                patched = act * keep + mean * (1.0 - keep)
                return (patched,) + hook_args[1:]
            return hook_fn

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(keep, mean)))
    return hooks


def answer_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            mask: torch.Tensor, *, temperature: float) -> torch.Tensor:
    logit_mask = mask[:, :-1]
    if int(logit_mask.sum().item()) == 0:
        return student_logits.new_zeros(())
    s = student_logits[:, :-1, :][logit_mask] / temperature
    t = teacher_logits[:, :-1, :][logit_mask] / temperature
    return F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(args.jsonl, target_field=args.target_field)
    random.shuffle(rows)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    print(f"[data] raw_rows={len(rows)} jsonl={args.jsonl}", flush=True)

    dataset = JsonlTranslationDataset(rows, tokenizer, args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id),
    )

    print(f"[model] {args.model} device={args.device} dtype={args.dtype}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(args.device)
    base.config.use_cache = False
    cfg = text_config(base)
    n_layers = int(cfg.num_hidden_layers)
    d_ffn = int(cfg.intermediate_size)
    mask = load_mask_npz(args.fixed_mask, n_layers, d_ffn)
    kept = sum(int(v.sum().item()) for v in mask.values())
    print(f"[mask] kept={kept} path={args.fixed_mask}", flush=True)

    print(f"[mean] building cache n={args.n_calib} mean_on={args.mean_on}", flush=True)
    means = build_mean_cache(
        base, rows, tokenizer, args, n_layers=n_layers, d_ffn=d_ffn, dtype=dtype,
    )

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
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    summary = {
        "args": vars(args),
        "n_rows": len(dataset),
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "mask_kept": kept,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "logs": [],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(summary, f, indent=2)

    start = time.time()
    global_step = 0
    seen_batches = 0
    running = {"loss": 0.0, "masked_kl": 0.0, "ce": 0.0,
               "unmasked_kl": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        for raw_batch in loader:
            if global_step >= total_steps:
                break
            batch = move_batch(raw_batch, args.device)

            with torch.no_grad(), model.disable_adapter():
                teacher_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits

            hooks = install_mean_ablation_hooks(
                model, mask, means, device=args.device, dtype=dtype,
            )
            try:
                masked_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
            finally:
                for h in hooks:
                    h.remove()

            masked_kl = kl_loss(
                masked_logits, teacher_logits, batch["kl_logit_mask"],
                temperature=args.kl_temperature,
            )
            ce = answer_ce_loss(masked_logits, batch["labels"])

            if args.unmasked_kl_beta > 0:
                unmasked_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                unmasked_kl = kl_loss(
                    unmasked_logits, teacher_logits, batch["kl_logit_mask"],
                    temperature=args.kl_temperature,
                )
            else:
                unmasked_kl = masked_logits.new_zeros(())

            loss = (
                args.masked_kl_beta * masked_kl
                + args.ce_beta * ce
                + args.unmasked_kl_beta * unmasked_kl
            )
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

            if (global_step == 1 or global_step % args.eval_every == 0
                    or global_step == total_steps):
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
                with open(out_dir / "train_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                print(json.dumps(row), flush=True)
                running = {"loss": 0.0, "masked_kl": 0.0, "ce": 0.0,
                           "unmasked_kl": 0.0, "n": 0}

    model.eval()
    summary["elapsed_s"] = time.time() - start
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[done] adapter={adapter_dir}", flush=True)

    if args.save_merged:
        merged_dir = out_dir / "merged"
        print(f"[merge] saving {merged_dir}", flush=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        summary["merged_dir"] = str(merged_dir)

    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
