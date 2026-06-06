#!/usr/bin/env python3
"""Evaluate arbitrary MLP masks on full-answer CF-patch recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_position_composed_full_answer import load_composed_masks, load_eval_records, run_full_answer_cf_patch
from full_answer_group_search import count_mask, load_mlp_final_npz
from position_interface_decomposition import resolve_batch_size
from union_attribution import TRACKED_MODES, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-pair-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-prune-test", type=int, default=500)
    parser.add_argument("--positions", nargs="+", default=["hundreds", "tens", "ones"])
    parser.add_argument("--composed-mask", default=None, help="Optional composed_union.full.npz to evaluate by mode.")
    parser.add_argument("--modes", nargs="+", default=["topk_500", "rel_0.001"], choices=TRACKED_MODES)
    parser.add_argument("--mask", action="append", default=[], metavar="NAME:PATH", help="Extra mlp_final NPZ mask.")
    return parser.parse_args()


def parse_mask_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"--mask must be NAME:PATH, got {value!r}")
        name, raw_path = value.split(":", 1)
        if not name:
            raise ValueError(f"empty mask name in {value!r}")
        specs.append((name, Path(raw_path)))
    return specs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[records] {args.position_pair_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records, n_pairs_by_position, skipped_by_position = load_eval_records(
        Path(args.position_pair_dir), tokenizer, args.positions, args.n_prune_test
    )
    if not records:
        raise RuntimeError("no evaluation records")
    print(f"  records={len(records):,} skipped={sum(skipped_by_position.values()):,}", flush=True)

    print(f"[model] {args.model} device={device} dtype={args.dtype} batch={batch_size}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    total_mlp = n_layers * d_ffn
    total_heads = n_layers * num_heads
    full_head = {layer: torch.ones(num_heads, dtype=torch.bool, device=device) for layer in range(n_layers)}

    masks: dict[str, dict[int, torch.Tensor]] = {}
    if args.composed_mask:
        by_mode, _head_by_mode = load_composed_masks(
            Path(args.composed_mask),
            n_layers=n_layers,
            d_ffn=d_ffn,
            num_heads=num_heads,
            device=device,
        )
        for mode in args.modes:
            masks[mode] = by_mode[mode]
    for name, path in parse_mask_specs(args.mask):
        masks[name] = load_mlp_final_npz(path, n_layers=n_layers, d_ffn=d_ffn, device=device)
    if not masks:
        raise RuntimeError("no masks requested")

    result = {
        "position_pair_dir": args.position_pair_dir,
        "composed_mask": args.composed_mask,
        "model": args.model,
        "positions": args.positions,
        "n_prune_test": args.n_prune_test,
        "n_records": len(records),
        "n_pairs_by_position": n_pairs_by_position,
        "skipped_by_position": skipped_by_position,
        "dtype": args.dtype,
        "device": device,
        "batch_size": batch_size,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "num_heads": num_heads,
        "by_mask": {},
    }
    write_json(out_path, result)

    for name, mask in masks.items():
        check = run_full_answer_cf_patch(
            model,
            records,
            mlp_keep=mask,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            label=name,
        )
        mlp_kept = count_mask(mask)
        row = {
            "accuracy": check["accuracy"],
            "correct": check["correct"],
            "n": check["n"],
            "mlp_kept": mlp_kept,
            "mlp_keep_fraction": mlp_kept / max(total_mlp, 1),
            "heads_kept": total_heads,
            "heads_keep_fraction": 1.0,
            "by_source_position": check.get("by_source_position", {}),
        }
        result["by_mask"][name] = row
        write_json(out_path, result)
        print(f"  {name:>28} mlp={mlp_kept:>7d} exact={row['accuracy']:.2%}", flush=True)

    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
