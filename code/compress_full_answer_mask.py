#!/usr/bin/env python3
"""Greedily compress a full-answer MLP mask by dropping layer groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_position_composed_full_answer import load_eval_records, run_full_answer_cf_patch
from full_answer_group_search import clone_mask, count_mask, load_mlp_final_npz, save_mask_npz
from position_interface_decomposition import resolve_batch_size
from union_attribution import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-pair-dir", required=True)
    parser.add_argument("--base-mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-prune-test", type=int, default=500)
    parser.add_argument("--positions", nargs="+", default=["hundreds", "tens", "ones"])
    parser.add_argument("--min-accuracy", type=float, default=0.995)
    parser.add_argument("--max-step-drop", type=float, default=0.005)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def drop_layer(mask: dict[int, torch.Tensor], layer: int) -> dict[int, torch.Tensor]:
    out = clone_mask(mask)
    out[layer] = torch.zeros_like(out[layer], dtype=torch.bool)
    return out


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists; pass --force to overwrite")

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
    current_mask = load_mlp_final_npz(Path(args.base_mask), n_layers=n_layers, d_ffn=d_ffn, device=device)

    def eval_mask(label: str, mask: dict[int, torch.Tensor]) -> dict:
        check = run_full_answer_cf_patch(
            model,
            records,
            mlp_keep=mask,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            label=label,
        )
        kept = count_mask(mask)
        row = {
            "label": label,
            "accuracy": check["accuracy"],
            "correct": check["correct"],
            "n": check["n"],
            "mlp_kept": kept,
            "mlp_keep_fraction": kept / max(total_mlp, 1),
            "heads_kept": total_heads,
            "heads_keep_fraction": 1.0,
            "by_source_position": check.get("by_source_position", {}),
        }
        print(f"  {label:>28} mlp={kept:>7d} exact={row['accuracy']:.2%}", flush=True)
        return row

    initial = eval_mask("initial", current_mask)
    result = {
        "position_pair_dir": args.position_pair_dir,
        "base_mask": args.base_mask,
        "model": args.model,
        "positions": args.positions,
        "n_prune_test": args.n_prune_test,
        "n_records": len(records),
        "n_pairs_by_position": n_pairs_by_position,
        "skipped_by_position": skipped_by_position,
        "dtype": args.dtype,
        "device": device,
        "batch_size": batch_size,
        "min_accuracy": args.min_accuracy,
        "max_step_drop": args.max_step_drop,
        "max_steps": args.max_steps,
        "initial": initial,
        "steps": [],
    }
    write_json(out_path, result)

    current = initial
    dropped_layers: set[int] = set()
    candidate_layers = {layer for layer in current_mask if int(current_mask[layer].sum().item()) > 0}
    for step in range(1, args.max_steps + 1):
        trials = []
        for layer in sorted(candidate_layers - dropped_layers):
            if int(current_mask[layer].sum().item()) == 0:
                continue
            trial_mask = drop_layer(current_mask, layer)
            trial = eval_mask(f"step_{step}_drop_layer_{layer:02d}", trial_mask)
            trials.append({**trial, "drop_layer": layer, "removed_mlp": current["mlp_kept"] - trial["mlp_kept"]})
        acceptable = [
            row
            for row in trials
            if row["accuracy"] >= args.min_accuracy and row["accuracy"] >= current["accuracy"] - args.max_step_drop
        ]
        step_row = {
            "step": step,
            "current_before": current,
            "accepted": False,
            "top_trials": sorted(trials, key=lambda row: (row["accuracy"], row["removed_mlp"]), reverse=True)[:8],
        }
        if not acceptable:
            result["steps"].append(step_row)
            write_json(out_path, result)
            break
        best = min(acceptable, key=lambda row: (row["mlp_kept"], -row["accuracy"]))
        dropped_layers.add(int(best["drop_layer"]))
        current_mask = drop_layer(current_mask, int(best["drop_layer"]))
        current = eval_mask(f"accepted_step_{step}_drop_layer_{int(best['drop_layer']):02d}", current_mask)
        step_row.update({"accepted": True, "accepted_trial": best, "current_after": current})
        result["steps"].append(step_row)
        write_json(out_path, result)

    final_mask_path = out_path.with_suffix(".final_mask.full.npz")
    save_mask_npz(final_mask_path, current_mask)
    result["final"] = current
    result["dropped_layers"] = sorted(dropped_layers)
    result["final_mask_npz"] = str(final_mask_path)
    write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
