#!/usr/bin/env python3
"""Evaluate composed position-interface masks on full-answer recovery."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from position_interface_decomposition import resolve_batch_size, split_pairs
from union_attribution import TRACKED_MODES, _tensor_union_size, pick_device


def build_full_answer_records(tokenizer, pairs: list[dict], source_position: str):
    records = []
    skipped = 0
    for pair in pairs:
        ex = pair["target"]
        cf = pair["cf"]
        ans = str(ex["answer"])
        prompt_spaced = ex["prompt"] + " "
        prompt_ids = tokenizer(
            prompt_spaced, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        full_ids = tokenizer(
            prompt_spaced + ans, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        cf_prompt_ids = tokenizer(
            cf["prompt"] + " ", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        ans_start = int(prompt_ids.shape[0])
        gold_ids = full_ids[ans_start:]
        gold_text = [tokenizer.decode([tid]).strip() for tid in gold_ids.tolist()]
        if len(gold_text) != len(ans) or any(
            len(tok) != 1 or not tok.isdigit() for tok in gold_text
        ):
            skipped += 1
            continue
        if int(prompt_ids.shape[0]) != int(cf_prompt_ids.shape[0]):
            skipped += 1
            continue
        records.append(
            {
                "input_ids": full_ids,
                "cf_prompt_ids": cf_prompt_ids,
                "answer_start": ans_start,
                "answer_token_ids": gold_ids.clone().long(),
                "source_position": source_position,
            }
        )
    return records, skipped


def load_eval_records(out_dir: Path, tokenizer, positions: list[str], n_prune_test: int):
    records = []
    skipped_by_position = {}
    n_pairs_by_position = {}
    for pos in positions:
        payload = json.load(open(out_dir / f"{pos}_pairs.json"))
        pairs = payload["pairs"]
        _, test_pairs = split_pairs(pairs, n_prune_test)
        pos_records, skipped = build_full_answer_records(tokenizer, test_pairs, pos)
        records.extend(pos_records)
        skipped_by_position[pos] = skipped
        n_pairs_by_position[pos] = len(test_pairs)
    return records, n_pairs_by_position, skipped_by_position


def load_composed_masks(
    npz_path: Path,
    *,
    n_layers: int,
    d_ffn: int,
    num_heads: int,
    device: str,
):
    data = np.load(npz_path)
    mlp_by_mode = {
        mode: {l: torch.zeros(d_ffn, dtype=torch.bool, device=device) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }
    head_by_mode = {
        mode: {l: torch.ones(num_heads, dtype=torch.bool, device=device) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }
    for mode in TRACKED_MODES:
        mlp_key = f"mlp_{mode}"
        if mlp_key in data:
            for layer, idx in data[mlp_key].tolist():
                mlp_by_mode[mode][int(layer)][int(idx)] = True
        head_key = f"heads_{mode}"
        if head_key in data:
            head_by_mode[mode] = {
                l: torch.zeros(num_heads, dtype=torch.bool, device=device) for l in range(n_layers)
            }
            for layer, idx in data[head_key].tolist():
                head_by_mode[mode][int(layer)][int(idx)] = True
    return mlp_by_mode, head_by_mode


def run_full_answer_cf_patch(
    model,
    records,
    *,
    mlp_keep,
    head_keep,
    device,
    num_heads,
    batch_size,
    label,
):
    dtype = next(model.parameters()).dtype
    correct = 0
    total = 0
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})
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

            def _mk_mlp_cache(li):
                def hook(mod, args):
                    cache_mlp[li] = args[0].detach().clone()

                return hook

            def _mk_attn_cache(li):
                def hook(mod, args):
                    cache_attn[li] = args[0].detach().clone()

                return hook

            cache_hooks = []
            for li, layer in enumerate(model.model.layers):
                cache_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mk_mlp_cache(li)))
                cache_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_mk_attn_cache(li)))
            try:
                with torch.no_grad():
                    model(c_ids)
            finally:
                for h in cache_hooks:
                    h.remove()

            patch_hooks = []
            for li, layer in enumerate(model.model.layers):
                mlp_mask = mlp_keep[li].to(device=device, dtype=dtype)
                head_mask = head_keep[li].to(device=device, dtype=dtype)
                neg_mlp = cache_mlp[li]
                neg_attn = cache_attn[li]
                prompt_len = int(neg_mlp.shape[1])

                def _mlp_patch(mask, neg_act, pl=prompt_len):
                    def hook(mod, args):
                        x = args[0]
                        keep = mask.view(1, 1, -1)
                        out = x.clone()
                        out[:, :pl, :] = x[:, :pl, :] * keep + neg_act[:, :pl, :] * (1.0 - keep)
                        out[:, pl:, :] = x[:, pl:, :] * keep
                        return (out,) + args[1:]

                    return hook

                def _attn_patch(mask, neg_act, H=num_heads, pl=prompt_len):
                    def hook(mod, args):
                        x = args[0]
                        b, s, d = x.shape
                        per = x.view(b, s, H, -1)
                        neg_per = neg_act.view(b, int(neg_act.shape[1]), H, -1)
                        keep = mask.view(1, 1, -1, 1)
                        out = per.clone()
                        out[:, :pl, :, :] = per[:, :pl, :, :] * keep + neg_per[:, :pl, :, :] * (1.0 - keep)
                        out[:, pl:, :, :] = per[:, pl:, :, :] * keep
                        return (out.reshape(b, s, d),) + args[1:]

                    return hook

                patch_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mlp_patch(mlp_mask, neg_mlp)))
                patch_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_attn_patch(head_mask, neg_attn)))

            try:
                with torch.no_grad():
                    logits = model(t_ids).logits
                pos = torch.arange(
                    answer_start - 1,
                    answer_start - 1 + answer_len,
                    device=device,
                )
                preds = logits[:, pos, :].argmax(dim=-1)
                matches = (preds == gold).all(dim=1)
                correct += int(matches.sum().item())
                total += int(t_ids.shape[0])
                for offset, rec_idx in enumerate(chunk):
                    source = records[rec_idx]["source_position"]
                    by_source[source]["correct"] += int(matches[offset].item())
                    by_source[source]["total"] += 1
            finally:
                for h in patch_hooks:
                    h.remove()
                cache_mlp.clear()
                cache_attn.clear()

    return {
        "label": label,
        "n": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "by_source_position": {
            pos: {
                "n": vals["total"],
                "correct": vals["correct"],
                "accuracy": vals["correct"] / max(vals["total"], 1),
            }
            for pos, vals in sorted(by_source.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--position-out-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--n-prune-test", type=int, default=500)
    ap.add_argument("--positions", nargs="+", default=["overflow", "hundreds", "tens", "ones"])
    ap.add_argument(
        "--extra-mask",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="Additional mlp_final NPZ mask to evaluate.",
    )
    args = ap.parse_args()

    out_dir = Path(args.position_out_dir)
    output = Path(args.output) if args.output else out_dir / "composed_full_answer_check.json"
    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    print(f"[records] {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records, n_pairs_by_position, skipped_by_position = load_eval_records(
        out_dir, tokenizer, args.positions, args.n_prune_test
    )
    print(f"  records={len(records):,} skipped={sum(skipped_by_position.values()):,}")

    print(f"[model] {args.model}")
    print(f"  device={device} dtype={args.dtype} batch_size={batch_size}")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads

    mlp_by_mode, head_by_mode = load_composed_masks(
        out_dir / "composed_union.full.npz",
        n_layers=n_layers,
        d_ffn=d_ffn,
        num_heads=num_heads,
        device=device,
    )
    full_mlp = {l: torch.ones(d_ffn, dtype=torch.bool, device=device) for l in range(n_layers)}
    full_head = {l: torch.ones(num_heads, dtype=torch.bool, device=device) for l in range(n_layers)}

    extra_masks = {}
    for spec in args.extra_mask:
        if ":" not in spec:
            raise ValueError(f"--extra-mask must be NAME:PATH, got {spec!r}")
        name, raw_path = spec.split(":", 1)
        if not name:
            raise ValueError(f"empty extra mask name in {spec!r}")
        data = np.load(raw_path)
        if "mlp_final" not in data:
            raise KeyError(f"mlp_final missing from {raw_path}")
        mask = {l: torch.zeros(d_ffn, dtype=torch.bool, device=device) for l in range(n_layers)}
        for layer, idx in data["mlp_final"].tolist():
            mask[int(layer)][int(idx)] = True
        extra_masks[name] = mask

    result = {
        "position_out_dir": str(out_dir),
        "model": args.model,
        "positions": args.positions,
        "n_records": len(records),
        "n_pairs_by_position": n_pairs_by_position,
        "skipped_by_position": skipped_by_position,
        "baseline": run_full_answer_cf_patch(
            model,
            records,
            mlp_keep=full_mlp,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            label="baseline_full_mask_cf",
        ),
        "by_threshold": {},
        "extra_masks": {},
    }
    print(f"  baseline={result['baseline']['accuracy']:.2%}")

    total_mlp = n_layers * d_ffn
    total_heads = n_layers * num_heads
    for mode in TRACKED_MODES:
        check = run_full_answer_cf_patch(
            model,
            records,
            mlp_keep=mlp_by_mode[mode],
            head_keep=head_by_mode[mode],
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            label=f"composed_{mode}_full_answer_cf",
        )
        mlp_kept = _tensor_union_size(mlp_by_mode[mode])
        heads_kept = _tensor_union_size(head_by_mode[mode])
        result["by_threshold"][mode] = {
            "mlp_kept": mlp_kept,
            "mlp_keep_fraction": mlp_kept / max(total_mlp, 1),
            "heads_kept": heads_kept,
            "heads_keep_fraction": heads_kept / max(total_heads, 1),
            "full_answer_acc_cf_patch": check["accuracy"],
            "n_test": check["n"],
            "by_source_position": check["by_source_position"],
        }
        print(
            f"  {mode:>10} mlp={mlp_kept:>7d} "
            f"frac={mlp_kept / max(total_mlp, 1):.2%} "
            f"full={check['accuracy']:.2%}"
        )

    for name, mask in extra_masks.items():
        check = run_full_answer_cf_patch(
            model,
            records,
            mlp_keep=mask,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            label=f"extra_{name}_full_answer_cf",
        )
        mlp_kept = _tensor_union_size(mask)
        result["extra_masks"][name] = {
            "mlp_kept": mlp_kept,
            "mlp_keep_fraction": mlp_kept / max(total_mlp, 1),
            "heads_kept": total_heads,
            "heads_keep_fraction": 1.0,
            "full_answer_acc_cf_patch": check["accuracy"],
            "n_test": check["n"],
            "by_source_position": check["by_source_position"],
        }
        print(
            f"  {name:>10} mlp={mlp_kept:>7d} "
            f"frac={mlp_kept / max(total_mlp, 1):.2%} "
            f"full={check['accuracy']:.2%}"
        )

    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
