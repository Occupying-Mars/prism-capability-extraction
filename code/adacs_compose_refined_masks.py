#!/usr/bin/env python3
"""Evaluate compatibility of refined ADACS primitive masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adacs_trace_primitives import PRIMITIVE_POSITION, load_adacs_pairs
from position_interface_decomposition import build_position_eval_records, resolve_batch_size, run_position_check_cf_patch
from union_attribution import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ones-mask", required=True)
    parser.add_argument("--tens-mask", required=True)
    parser.add_argument(
        "--extra-mask",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="Additional mask NPZ to evaluate, for example balanced:path/to/mask.npz.",
    )
    parser.add_argument("--ones-pair-dir", required=True)
    parser.add_argument("--tens-pair-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def load_final_mask(path: Path, n_layers: int, d_ffn: int) -> dict[int, torch.Tensor]:
    data = np.load(path)
    if "mlp_final" not in data:
        raise KeyError(f"mlp_final missing from {path}")
    mask = {layer: torch.zeros(d_ffn, dtype=torch.bool) for layer in range(n_layers)}
    for layer, idx in data["mlp_final"].tolist():
        mask[int(layer)][int(idx)] = True
    return mask


def union_masks(*masks: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    out = {layer: values.clone() for layer, values in masks[0].items()}
    for mask in masks[1:]:
        for layer in out:
            out[layer] |= mask[layer]
    return out


def intersect_masks(*masks: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    out = {layer: values.clone() for layer, values in masks[0].items()}
    for mask in masks[1:]:
        for layer in out:
            out[layer] &= mask[layer]
    return out


def subtract_mask(mask: dict[int, torch.Tensor], minus: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {layer: mask[layer] & ~minus[layer] for layer in mask}


def count_mask(mask: dict[int, torch.Tensor]) -> int:
    return int(sum(int(values.sum().item()) for values in mask.values()))


def parse_extra_mask(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise ValueError(f"--extra-mask must be NAME:PATH, got {value!r}")
    name, path = value.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty extra-mask name in {value!r}")
    return name, Path(path)


def load_records(tokenizer, pair_dir: Path, primitive: str, n_eval: int | None):
    pairs, stats = load_adacs_pairs(pair_dir, primitive, None)
    eval_pairs = pairs if n_eval is None else pairs[:n_eval]
    records, skipped = build_position_eval_records(tokenizer, eval_pairs, PRIMITIVE_POSITION[primitive])
    return records, skipped, stats, len(pairs), len(eval_pairs)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists; pass --force to overwrite")

    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    datasets = {
        "carry_ones_to_tens": load_records(tokenizer, Path(args.ones_pair_dir), "carry_ones_to_tens", args.n_eval),
        "carry_tens_to_hundreds": load_records(tokenizer, Path(args.tens_pair_dir), "carry_tens_to_hundreds", args.n_eval),
    }

    print(f"[model] {args.model} device={device} dtype={args.dtype} batch={batch_size}", flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager")
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    total_mlp = n_layers * d_ffn
    total_heads = n_layers * num_heads
    masks = {
        "ones_refined": load_final_mask(Path(args.ones_mask), n_layers, d_ffn),
        "tens_refined": load_final_mask(Path(args.tens_mask), n_layers, d_ffn),
    }
    masks["union_refined"] = union_masks(masks["ones_refined"], masks["tens_refined"])
    masks["intersection_refined"] = intersect_masks(masks["ones_refined"], masks["tens_refined"])
    masks["ones_only"] = subtract_mask(masks["ones_refined"], masks["tens_refined"])
    masks["tens_only"] = subtract_mask(masks["tens_refined"], masks["ones_refined"])
    for spec in args.extra_mask:
        name, path = parse_extra_mask(spec)
        if name in masks:
            raise ValueError(f"duplicate mask name {name!r}")
        masks[name] = load_final_mask(path, n_layers, d_ffn)
    full_head = {layer: torch.ones(num_heads, dtype=torch.bool, device=device) for layer in range(n_layers)}

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "device": device,
        "batch_size": batch_size,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "num_heads": num_heads,
        "total_mlp_neurons": total_mlp,
        "total_attention_heads": total_heads,
        "masks": {
            name: {
                "mlp_kept": count_mask(mask),
                "mlp_keep_fraction": count_mask(mask) / max(total_mlp, 1),
            }
            for name, mask in masks.items()
        },
        "datasets": {},
        "evals": {},
    }

    for primitive, (records, skipped, stats, n_pairs, n_eval_pairs) in datasets.items():
        result["datasets"][primitive] = {
            "pair_stats": stats,
            "n_pairs": n_pairs,
            "n_eval_pairs": n_eval_pairs,
            "n_eval_records": len(records),
            "skipped": skipped,
        }
        result["evals"][primitive] = {}
        for mask_name, mask in masks.items():
            check = run_position_check_cf_patch(
                model,
                records,
                mlp_keep=mask,
                head_keep=full_head,
                device=device,
                num_heads=num_heads,
                batch_size=batch_size,
                label=f"{primitive}_{mask_name}",
            )
            row = {
                "accuracy": check["accuracy"],
                "correct": check["correct"],
                "n": check["n"],
                "mlp_kept": count_mask(mask),
                "mlp_keep_fraction": count_mask(mask) / max(total_mlp, 1),
                "heads_kept": total_heads,
                "heads_keep_fraction": 1.0,
            }
            result["evals"][primitive][mask_name] = row
            print(f"  {primitive} {mask_name} mlp={row['mlp_kept']:,} cf={row['accuracy']:.2%}", flush=True)
            write_json(out_path, result)

    write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
