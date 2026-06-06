#!/usr/bin/env python3
"""Evaluate masks by autoregressively generating full answers under CF patching."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_position_composed_full_answer import build_full_answer_records
from full_answer_group_search import count_mask, load_mlp_final_npz
from position_interface_decomposition import split_pairs
from union_attribution import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-pair-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--n-prune-test", type=int, default=500)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--positions", nargs="+", default=["hundreds", "tens", "ones"])
    parser.add_argument("--mask", action="append", default=[], metavar="NAME:PATH", help="mlp_final NPZ mask.")
    return parser.parse_args()


def parse_mask_specs(values: list[str]) -> list[tuple[str, Path]]:
    out = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"--mask must be NAME:PATH, got {value!r}")
        name, raw_path = value.split(":", 1)
        out.append((name, Path(raw_path)))
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def load_records(out_dir: Path, tokenizer, positions: list[str], n_prune_test: int) -> tuple[list[dict], dict, dict]:
    records = []
    n_pairs_by_position = {}
    skipped_by_position = {}
    for pos in positions:
        payload = json.load(open(out_dir / f"{pos}_pairs.json"))
        _attr_pairs, test_pairs = split_pairs(payload["pairs"], n_prune_test)
        pos_records, skipped = build_full_answer_records(tokenizer, test_pairs, pos)
        records.extend(pos_records)
        n_pairs_by_position[pos] = len(test_pairs)
        skipped_by_position[pos] = skipped
    return records, n_pairs_by_position, skipped_by_position


def generate_one(
    model,
    input_ids: torch.Tensor,
    cf_prompt_ids: torch.Tensor,
    answer_token_ids: torch.Tensor,
    *,
    mlp_keep: dict[int, torch.Tensor],
    head_keep: dict[int, torch.Tensor],
    device: str,
    num_heads: int,
) -> list[int]:
    dtype = next(model.parameters()).dtype
    prompt_ids = input_ids[: -int(answer_token_ids.shape[0])].to(device)
    cf_ids = cf_prompt_ids.to(device)
    generated: list[int] = []

    cache_mlp: dict[int, torch.Tensor] = {}
    cache_attn: dict[int, torch.Tensor] = {}

    def make_mlp_cache(layer_idx):
        def hook(_module, hook_args):
            cache_mlp[layer_idx] = hook_args[0].detach().clone()

        return hook

    def make_attn_cache(layer_idx):
        def hook(_module, hook_args):
            cache_attn[layer_idx] = hook_args[0].detach().clone()

        return hook

    cache_hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        cache_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_mlp_cache(layer_idx)))
        cache_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_attn_cache(layer_idx)))
    try:
        with torch.no_grad():
            model(cf_ids.unsqueeze(0))
    finally:
        for handle in cache_hooks:
            handle.remove()

    for _step in range(int(answer_token_ids.shape[0])):
        cur_ids = torch.cat(
            [prompt_ids, torch.tensor(generated, dtype=torch.long, device=device)],
            dim=0,
        ).unsqueeze(0)
        patch_hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            mlp_mask = mlp_keep[layer_idx].to(device=device, dtype=dtype)
            head_mask = head_keep[layer_idx].to(device=device, dtype=dtype)
            neg_mlp = cache_mlp[layer_idx]
            neg_attn = cache_attn[layer_idx]
            prompt_len = int(neg_mlp.shape[1])

            def mlp_patch(mask, neg_act, pl=prompt_len):
                def hook(_module, hook_args):
                    x = hook_args[0]
                    keep = mask.view(1, 1, -1)
                    out = x.clone()
                    out[:, :pl, :] = x[:, :pl, :] * keep + neg_act[:, :pl, :] * (1.0 - keep)
                    out[:, pl:, :] = x[:, pl:, :] * keep
                    return (out,) + hook_args[1:]

                return hook

            def attn_patch(mask, neg_act, heads=num_heads, pl=prompt_len):
                def hook(_module, hook_args):
                    x = hook_args[0]
                    batch, seq, dim = x.shape
                    per = x.view(batch, seq, heads, -1)
                    neg_per = neg_act.view(batch, int(neg_act.shape[1]), heads, -1)
                    keep = mask.view(1, 1, -1, 1)
                    out = per.clone()
                    out[:, :pl, :, :] = per[:, :pl, :, :] * keep + neg_per[:, :pl, :, :] * (1.0 - keep)
                    out[:, pl:, :, :] = per[:, pl:, :, :] * keep
                    return (out.reshape(batch, seq, dim),) + hook_args[1:]

                return hook

            patch_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mlp_patch(mlp_mask, neg_mlp)))
            patch_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(attn_patch(head_mask, neg_attn)))
        try:
            with torch.no_grad():
                logits = model(cur_ids).logits
            next_id = int(logits[0, -1, :].argmax().item())
            generated.append(next_id)
        finally:
            for handle in patch_hooks:
                handle.remove()
    return generated


def evaluate_mask(model, records: list[dict], mask, *, device: str, num_heads: int, full_head) -> dict:
    correct = 0
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})
    wrong = []
    for idx, rec in enumerate(records):
        pred_ids = generate_one(
            model,
            rec["input_ids"],
            rec["cf_prompt_ids"],
            rec["answer_token_ids"],
            mlp_keep=mask,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
        )
        gold_ids = rec["answer_token_ids"].tolist()
        ok = pred_ids == gold_ids
        correct += int(ok)
        source = rec["source_position"]
        by_source[source]["correct"] += int(ok)
        by_source[source]["total"] += 1
        if not ok and len(wrong) < 25:
            wrong.append({"index": idx, "source_position": source, "pred_ids": pred_ids, "gold_ids": gold_ids})
    total = len(records)
    return {
        "correct": correct,
        "n": total,
        "accuracy": correct / max(total, 1),
        "by_source_position": {
            key: {"correct": vals["correct"], "n": vals["total"], "accuracy": vals["correct"] / max(vals["total"], 1)}
            for key, vals in sorted(by_source.items())
        },
        "wrong_sample": wrong,
    }


def main() -> None:
    args = parse_args()
    if not args.mask:
        raise ValueError("at least one --mask is required")
    out_path = Path(args.out)
    device = pick_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records, n_pairs_by_position, skipped_by_position = load_records(
        Path(args.position_pair_dir), tokenizer, args.positions, args.n_prune_test
    )
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard settings")
    if args.num_shards > 1:
        records = [rec for idx, rec in enumerate(records) if idx % args.num_shards == args.shard_index]
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        raise RuntimeError("no generation records")
    print(f"[records] {len(records):,} skipped={sum(skipped_by_position.values()):,}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    total_mlp = n_layers * d_ffn
    full_head = {layer: torch.ones(num_heads, dtype=torch.bool, device=device) for layer in range(n_layers)}

    result = {
        "position_pair_dir": args.position_pair_dir,
        "model": args.model,
        "model_key": args.model_key or args.model,
        "positions": args.positions,
        "n_prune_test": args.n_prune_test,
        "max_records": args.max_records,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "n_records": len(records),
        "n_pairs_by_position": n_pairs_by_position,
        "skipped_by_position": skipped_by_position,
        "by_mask": {},
    }
    write_json(out_path, result)

    for name, path in parse_mask_specs(args.mask):
        mask = load_mlp_final_npz(path, n_layers=n_layers, d_ffn=d_ffn, device=device)
        check = evaluate_mask(model, records, mask, device=device, num_heads=num_heads, full_head=full_head)
        mlp_kept = count_mask(mask)
        result["by_mask"][name] = {
            **check,
            "mlp_kept": mlp_kept,
            "mlp_keep_fraction": mlp_kept / max(total_mlp, 1),
        }
        write_json(out_path, result)
        print(f"  {name:>28} mlp={mlp_kept:>7d} gen_exact={check['accuracy']:.2%}", flush=True)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
