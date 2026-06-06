#!/usr/bin/env python3
"""Sentence-aligned MLP attribution for HY-MT EN->PT translation.

This scores MLP intermediate channels against a teacher-forced full-target
Portuguese objective instead of a first-token objective. It is intentionally
shardable: run one shard per GPU, then merge the resulting score tensors.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from translation_io import DEFAULT_PROMPT_STYLE, encode_chat_with_target
from translation_region_student import decoder_root, text_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-language", default="Portuguese")
    p.add_argument("--prompt-style", default=DEFAULT_PROMPT_STYLE)
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-lang", default="por_Latn")
    p.add_argument("--split", default="dev")
    p.add_argument("--n-attr", type=int, default=512)
    p.add_argument("--seed", type=int, default=38)
    p.add_argument("--max-target-tokens", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--save-every", type=int, default=25)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_pairs(args: argparse.Namespace) -> list[dict[str, str]]:
    ds = load_dataset("mteb/flores", split=args.split)
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(ds):
        src = row.get(args.src_lang)
        tgt = row.get(args.tgt_lang)
        if src and tgt:
            rows.append({"id": str(idx), "en": src, "pt": tgt})
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_attr]
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index out of range")
    return rows[args.shard_index :: args.num_shards]


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    dtype = dtype_from_name(args.dtype)
    rows = load_pairs(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    cfg = text_config(model)
    n_layers = int(cfg.num_hidden_layers)
    d_ffn = int(cfg.intermediate_size)
    layer_scores = {i: torch.zeros(d_ffn, dtype=torch.float64) for i in range(n_layers)}
    layer_abs_act = {i: torch.zeros(d_ffn, dtype=torch.float64) for i in range(n_layers)}
    layer_abs_grad = {i: torch.zeros(d_ffn, dtype=torch.float64) for i in range(n_layers)}
    token_count = 0
    used_rows = 0
    skipped_rows = 0
    start = time.time()

    root = decoder_root(model)
    acts: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, hook_args):
            act = hook_args[0].detach().requires_grad_(True)
            act.retain_grad()
            acts[layer_idx] = act
            return (act,) + hook_args[1:]
        return hook_fn

    for layer_idx, layer in enumerate(root.layers):
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_idx)))

    try:
        for row_idx, row in enumerate(rows, start=1):
            acts.clear()
            full_ids, prompt_len = encode_chat_with_target(
                tokenizer,
                row["en"],
                row["pt"],
                target_language=args.target_language,
                prompt_style=args.prompt_style,
            )
            target_len = int(full_ids.shape[0]) - int(prompt_len)
            if target_len <= 0:
                skipped_rows += 1
                continue
            if target_len > args.max_target_tokens:
                full_ids = full_ids[: prompt_len + args.max_target_tokens]
                target_len = args.max_target_tokens
            if int(full_ids.shape[0]) > args.max_seq_length:
                skipped_rows += 1
                continue

            input_ids = full_ids.unsqueeze(0).to(args.device)
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            model.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, use_cache=False).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            loss.backward()
            valid_tokens = int(labels[:, 1:].ne(-100).sum().item())
            token_count += valid_tokens
            used_rows += 1
            for layer_idx, act in acts.items():
                grad = act.grad
                if grad is None:
                    continue
                flat_act = act.detach().to(torch.float32).reshape(-1, act.shape[-1])
                flat_grad = grad.detach().to(torch.float32).reshape(-1, grad.shape[-1])
                layer_scores[layer_idx] += (flat_act * flat_grad).abs().sum(dim=0).cpu().to(torch.float64)
                layer_abs_act[layer_idx] += flat_act.abs().sum(dim=0).cpu().to(torch.float64)
                layer_abs_grad[layer_idx] += flat_grad.abs().sum(dim=0).cpu().to(torch.float64)
            if args.save_every and row_idx % args.save_every == 0:
                save_payload(out_path, {
                    "layer_scores": layer_scores,
                    "layer_abs_act": layer_abs_act,
                    "layer_abs_grad": layer_abs_grad,
                    "used_rows": used_rows,
                    "skipped_rows": skipped_rows,
                    "token_count": token_count,
                    "partial": True,
                    "args": vars(args),
                })
                print(json.dumps({"rows_seen": row_idx, "used_rows": used_rows, "elapsed_s": time.time() - start}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    payload = {
        "layer_scores": layer_scores,
        "layer_abs_act": layer_abs_act,
        "layer_abs_grad": layer_abs_grad,
        "used_rows": used_rows,
        "skipped_rows": skipped_rows,
        "token_count": token_count,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "elapsed_s": time.time() - start,
        "partial": False,
        "args": vars(args),
    }
    save_payload(out_path, payload)
    print(json.dumps({
        "out": str(out_path),
        "used_rows": used_rows,
        "skipped_rows": skipped_rows,
        "token_count": token_count,
        "elapsed_s": payload["elapsed_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
