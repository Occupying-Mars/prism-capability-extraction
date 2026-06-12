#!/usr/bin/env python3
"""Train GLM BFCL masked LoRA conditioners."""

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
    answer_ce_loss,
    collate,
    kl_loss,
    load_topk_mask,
    model_input_device,
    move_batch,
    read_jsonl,
    save_adapter_checkpoint,
)
from prism_glm.bfcl_direct_glm import format_glm_native_target, glm_native_prompt_text, glm_prompt_text  # noqa: E402
from prism_glm.relp_glm import glm_d_ffn, glm_decoder_root, glm_num_layers  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="zai-org/glm-4-9b-chat")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--prompt-format", choices=["manual_tool_call", "glm_native"], default="manual_tool_call")
    p.add_argument("--target-format", choices=["tool_call", "glm_native"], default="tool_call")
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
    p.add_argument("--masked-kl-beta", type=float, default=1.0)
    p.add_argument("--ce-beta", type=float, default=0.2)
    p.add_argument("--unmasked-kl-beta", type=float, default=0.05)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-merged", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def target_text_for_row(row: dict, target_format: str) -> str:
    call = row.get("target_call")
    if target_format == "glm_native":
        try:
            return format_glm_native_target(row)
        except ValueError:
            return ""
    if row.get("target_text"):
        return str(row["target_text"]).strip()
    if isinstance(call, dict):
        return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
    return ""


def encode_row(row: dict, tokenizer, max_seq_length: int, prompt_format: str, target_format: str):
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


class GLMDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_seq_length: int, prompt_format: str, target_format: str):
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


def layer_device(layer) -> torch.device:
    return next(layer.parameters()).device


def build_mean_cache(model, rows, tokenizer, args, *, n_layers: int, d_ffn: int, dtype):
    layers = list(glm_decoder_root(model).layers)
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
        hooks.append(layer.mlp.dense_4h_to_h.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            for row in rows[: args.n_calib]:
                enc = encode_row(row, tokenizer, args.max_seq_length, args.prompt_format, args.target_format)
                if enc is None:
                    continue
                model(enc["prompt_ids"].unsqueeze(0).to(input_device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {i: (sums[i] / max(counts[i], 1)).to(dtype=dtype) for i in range(n_layers)}


def install_mean_ablation_hooks(model, mask, means, *, dtype):
    hooks = []
    for layer_idx, layer in enumerate(glm_decoder_root(model).layers):
        device = layer_device(layer)
        keep = mask[layer_idx].to(device=device).to(dtype=dtype).view(1, 1, -1)
        mean = means[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)

        def make_hook(keep_tensor, mean_tensor):
            def hook_fn(module, hook_args):
                act = hook_args[0]
                return (act * keep_tensor + mean_tensor * (1.0 - keep_tensor),) + hook_args[1:]
            return hook_fn

        hooks.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(make_hook(keep, mean)))
    return hooks


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(args.train_jsonl)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[data] raw_rows={len(rows)} train={args.train_jsonl} prompt_format={args.prompt_format} target_format={args.target_format}", flush=True)
    dataset = GLMDataset(rows, tokenizer, args.max_seq_length, args.prompt_format, args.target_format)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id))
    if not dataset:
        raise ValueError("no train rows survived encoding")

    load_kwargs = {"torch_dtype": dtype, "attn_implementation": "eager", "trust_remote_code": True}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    base = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not args.device_map:
        base = base.to(args.device)
    base.config.use_cache = False
    input_device = model_input_device(base)
    n_layers = glm_num_layers(base.config)
    d_ffn = glm_d_ffn(base.config)
    mask = load_topk_mask(args.attribution, args.topk)
    kept = sum(int(v.sum().item()) for v in mask.values())
    print(f"[mask] topk={args.topk} kept={kept} attribution={args.attribution}", flush=True)

    print(f"[mean] building cache n={args.n_calib}", flush=True)
    means = build_mean_cache(base, rows, tokenizer, args, n_layers=n_layers, d_ffn=d_ffn, dtype=dtype)

    lora_config = LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=args.target_modules, use_rslora=args.use_rslora, bias="none")
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
    summary = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "n_rows": len(dataset), "n_layers": n_layers, "d_ffn": d_ffn, "mask_kept": kept, "total_steps": total_steps, "warmup_steps": warmup_steps, "logs": [], "checkpoints": []}
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
                teacher_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            hooks = install_mean_ablation_hooks(model, mask, means, dtype=dtype)
            try:
                masked_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            finally:
                for h in hooks:
                    h.remove()
            masked_kl = kl_loss(masked_logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
            ce = answer_ce_loss(masked_logits, batch["labels"])
            if args.unmasked_kl_beta > 0:
                unmasked_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
                unmasked_kl = kl_loss(unmasked_logits, teacher_logits, batch["kl_logit_mask"], temperature=args.kl_temperature)
            else:
                unmasked_kl = masked_logits.new_zeros(())
            loss = args.masked_kl_beta * masked_kl + args.ce_beta * ce + args.unmasked_kl_beta * unmasked_kl
            (loss / args.grad_accum).backward()
            seen_batches += 1
            for key, value in (("loss", loss), ("masked_kl", masked_kl), ("ce", ce), ("unmasked_kl", unmasked_kl)):
                running[key] += float(value.detach().cpu())
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
                row = {k: running[k] / denom for k in ("loss", "masked_kl", "ce", "unmasked_kl")}
                row.update({"step": global_step, "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - start})
                summary["logs"].append(row)
                (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(row), flush=True)
                running = {"loss": 0.0, "masked_kl": 0.0, "ce": 0.0, "unmasked_kl": 0.0, "n": 0}
            if args.save_every and global_step % args.save_every == 0:
                checkpoint_dir = save_adapter_checkpoint(model, tokenizer, args.out_dir, global_step)
                summary["checkpoints"].append({"step": global_step, "adapter_dir": str(checkpoint_dir)})

    model.eval()
    summary["elapsed_s"] = time.time() - start
    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary["adapter_dir"] = str(adapter_dir)
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] adapter={adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
