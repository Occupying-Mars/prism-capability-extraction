#!/usr/bin/env python3
"""Evaluate simple route-conditioned full-answer mask choices."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_position_composed_full_answer import load_composed_masks
from full_answer_group_search import count_mask, load_mlp_final_npz, parse_extra_masks
from position_interface_decomposition import carry_positions, resolve_batch_size, split_pairs
from union_attribution import pick_device


ROUTE_KEYS = [
    "source_position",
    "answer_len",
    "leading_carry",
    "n_carries",
    "carry_signature",
    "source_x_answer_len",
    "source_x_carry_signature",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-out-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-prune-test", type=int, default=500)
    parser.add_argument("--positions", nargs="+", default=["hundreds", "tens", "ones"])
    parser.add_argument("--mode-masks", nargs="+", default=["topk_500", "rel_0.01", "rel_0.001"])
    parser.add_argument("--extra-mask", action="append", default=[], metavar="NAME:PATH")
    parser.add_argument("--route-keys", nargs="+", default=ROUTE_KEYS)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def route_fields(ex: dict, source_position: str) -> dict[str, str]:
    answer = str(ex["answer"])
    signature = carry_positions(int(ex["a"]), int(ex["b"]))
    sig_text = "none" if not signature else "carry_" + "_".join(str(v) for v in signature)
    answer_len = f"len_{len(answer)}"
    leading = "leading_carry" if len(answer) == 3 else "no_leading_carry"
    n_carries = f"n_carries_{len(signature)}"
    return {
        "source_position": source_position,
        "answer_len": answer_len,
        "leading_carry": leading,
        "n_carries": n_carries,
        "carry_signature": sig_text,
        "source_x_answer_len": f"{source_position}|{answer_len}",
        "source_x_carry_signature": f"{source_position}|{sig_text}",
    }


def build_records(tokenizer, pairs: list[dict], source_position: str) -> tuple[list[dict], int]:
    records = []
    skipped = 0
    for pair in pairs:
        ex = pair["target"]
        cf = pair["cf"]
        ans = str(ex["answer"])
        prompt_spaced = ex["prompt"] + " "
        cf_prompt_spaced = cf["prompt"] + " "
        prompt_ids = tokenizer(prompt_spaced, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        cf_prompt_ids = tokenizer(cf_prompt_spaced, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if int(prompt_ids.shape[0]) != int(cf_prompt_ids.shape[0]):
            skipped += 1
            continue
        full_ids = tokenizer(prompt_spaced + ans, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ans_start = int(prompt_ids.shape[0])
        gold_ids = full_ids[ans_start:]
        gold_text = [tokenizer.decode([tid]).strip() for tid in gold_ids.tolist()]
        if len(gold_text) != len(ans) or any(len(tok) != 1 or not tok.isdigit() for tok in gold_text):
            skipped += 1
            continue
        records.append(
            {
                "input_ids": full_ids,
                "cf_prompt_ids": cf_prompt_ids,
                "answer_start": ans_start,
                "answer_token_ids": gold_ids.clone().long(),
                "routes": route_fields(ex, source_position),
            }
        )
    return records, skipped


def load_records(out_dir: Path, tokenizer, positions: list[str], n_prune_test: int):
    records = []
    skipped_by_position = {}
    n_pairs_by_position = {}
    for position in positions:
        payload = json.load(open(out_dir / f"{position}_pairs.json"))
        _attr_pairs, test_pairs = split_pairs(payload["pairs"], n_prune_test)
        pos_records, skipped = build_records(tokenizer, test_pairs, position)
        records.extend(pos_records)
        skipped_by_position[position] = skipped
        n_pairs_by_position[position] = len(test_pairs)
    return records, n_pairs_by_position, skipped_by_position


def empty_counts() -> dict[str, int]:
    return {"correct": 0, "total": 0}


def finish_counts(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    return {
        name: {"correct": vals["correct"], "total": vals["total"], "accuracy": vals["correct"] / max(vals["total"], 1)}
        for name, vals in sorted(counts.items())
    }


def evaluate_mask_matches(
    model,
    records: list[dict],
    *,
    mlp_keep: dict[int, torch.Tensor],
    head_keep: dict[int, torch.Tensor],
    device: str,
    num_heads: int,
    batch_size: int,
    route_keys: list[str],
):
    dtype = next(model.parameters()).dtype
    matches_by_index = [False] * len(records)
    correct = 0
    total = 0
    by_route = {key: defaultdict(empty_counts) for key in route_keys}
    idx = sorted(
        range(len(records)),
        key=lambda i: (
            int(records[i]["input_ids"].shape[0]),
            int(records[i]["cf_prompt_ids"].shape[0]),
            int(records[i]["answer_token_ids"].shape[0]),
        ),
    )
    group = 0
    while group < len(idx):
        end = group
        key = (
            int(records[idx[group]]["input_ids"].shape[0]),
            int(records[idx[group]]["cf_prompt_ids"].shape[0]),
            int(records[idx[group]]["answer_token_ids"].shape[0]),
        )
        while end < len(idx):
            rec = records[idx[end]]
            probe = (
                int(rec["input_ids"].shape[0]),
                int(rec["cf_prompt_ids"].shape[0]),
                int(rec["answer_token_ids"].shape[0]),
            )
            if probe != key:
                break
            end += 1
        group_idx = idx[group:end]
        group = end
        for j in range(0, len(group_idx), batch_size):
            chunk = group_idx[j : j + batch_size]
            t_ids = torch.stack([records[k]["input_ids"] for k in chunk]).to(device)
            c_ids = torch.stack([records[k]["cf_prompt_ids"] for k in chunk]).to(device)
            gold = torch.stack([records[k]["answer_token_ids"] for k in chunk]).to(device)
            answer_start = int(records[chunk[0]]["answer_start"])
            answer_len = int(gold.shape[1])

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
                    model(c_ids)
            finally:
                for handle in cache_hooks:
                    handle.remove()

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
                    logits = model(t_ids).logits
                pos = torch.arange(answer_start - 1, answer_start - 1 + answer_len, device=device)
                preds = logits[:, pos, :].argmax(dim=-1)
                batch_matches = (preds == gold).all(dim=1)
                correct += int(batch_matches.sum().item())
                total += int(t_ids.shape[0])
                for offset, rec_idx in enumerate(chunk):
                    ok = bool(batch_matches[offset].item())
                    matches_by_index[rec_idx] = ok
                    for route_key in route_keys:
                        value = records[rec_idx]["routes"].get(route_key, "missing")
                        by_route[route_key][value]["correct"] += int(ok)
                        by_route[route_key][value]["total"] += 1
            finally:
                for handle in patch_hooks:
                    handle.remove()
                cache_mlp.clear()
                cache_attn.clear()

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "matches": matches_by_index,
        "by_route": {route_key: finish_counts(counts) for route_key, counts in by_route.items()},
    }


def route_conditioned_summary(mask_results: dict[str, dict], records: list[dict], route_key: str) -> dict:
    values = sorted({rec["routes"].get(route_key, "missing") for rec in records})
    selected = {}
    correct = 0
    total = 0
    for value in values:
        candidates = []
        for mask_name, result in mask_results.items():
            row = result["by_route"].get(route_key, {}).get(value)
            if not row:
                continue
            candidates.append((row["accuracy"], -result["mlp_kept"], mask_name, row))
        if not candidates:
            continue
        _acc, _neg_mlp, mask_name, row = max(candidates)
        selected[value] = {"mask": mask_name, **row}
        correct += int(row["correct"])
        total += int(row["total"])
    return {
        "route_key": route_key,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "selected_by_value": selected,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records, n_pairs_by_position, skipped_by_position = load_records(
        Path(args.position_out_dir), tokenizer, args.positions, args.n_prune_test
    )
    if not records:
        raise RuntimeError("no evaluation records")
    print(f"[records] records={len(records):,} skipped={sum(skipped_by_position.values()):,}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    total_mlp = n_layers * d_ffn
    full_head = {layer: torch.ones(num_heads, dtype=torch.bool, device=device) for layer in range(n_layers)}

    by_mode, _head_by_mode = load_composed_masks(
        Path(args.position_out_dir) / "composed_union.full.npz",
        n_layers=n_layers,
        d_ffn=d_ffn,
        num_heads=num_heads,
        device=device,
    )
    masks = {name: by_mode[name] for name in args.mode_masks if name in by_mode}
    for name, path in parse_extra_masks(args.extra_mask):
        masks[name] = load_mlp_final_npz(path, n_layers=n_layers, d_ffn=d_ffn, device=device)
    if not masks:
        raise RuntimeError("no masks requested")

    result = {
        "position_out_dir": args.position_out_dir,
        "model": args.model,
        "positions": args.positions,
        "n_prune_test": args.n_prune_test,
        "n_records": len(records),
        "n_pairs_by_position": n_pairs_by_position,
        "skipped_by_position": skipped_by_position,
        "route_keys": args.route_keys,
        "by_mask": {},
        "route_conditioned": {},
    }
    write_json(out_path, result)

    for mask_name, mask in masks.items():
        print(f"[mask] {mask_name}", flush=True)
        check = evaluate_mask_matches(
            model,
            records,
            mlp_keep=mask,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            route_keys=args.route_keys,
        )
        mlp_kept = count_mask(mask)
        result["by_mask"][mask_name] = {
            "accuracy": check["accuracy"],
            "correct": check["correct"],
            "total": check["total"],
            "mlp_kept": mlp_kept,
            "mlp_keep_fraction": mlp_kept / max(total_mlp, 1),
            "by_route": check["by_route"],
        }
        write_json(out_path, result)
        print(f"  exact={check['accuracy']:.2%} mlp={mlp_kept:,}", flush=True)

    for route_key in args.route_keys:
        result["route_conditioned"][route_key] = route_conditioned_summary(result["by_mask"], records, route_key)
    write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
