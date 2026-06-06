#!/usr/bin/env python3
"""Find the smallest full-answer mask that reaches a target recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_position_composed_full_answer import load_composed_masks, load_eval_records, run_full_answer_cf_patch
from full_answer_group_search import apply_group, clone_mask, count_mask, make_group_specs, save_mask_npz
from position_interface_decomposition import resolve_batch_size
from union_attribution import TRACKED_MODES, pick_device


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
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--base-mask", default="topk_500")
    parser.add_argument("--baseline-masks", nargs="+", default=["topk_100", "rel_0.05", "topk_500", "rel_0.01", "topk_2000", "rel_0.001"])
    parser.add_argument("--drop-from", nargs="+", default=["topk_500"])
    parser.add_argument("--shells", nargs="+", default=["rel_0.01:topk_500", "topk_2000:topk_500", "rel_0.001:topk_500"])
    parser.add_argument("--ranking-json", default=None)
    parser.add_argument("--max-add-candidates", type=int, default=48)
    parser.add_argument("--max-add-steps", type=int, default=4)
    parser.add_argument("--max-compress-steps", type=int, default=12)
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


def spec_key(spec: dict) -> tuple:
    return (spec.get("kind"), spec.get("source_mask"), spec.get("minus_mask"), int(spec.get("layer", -1)))


def ordered_add_specs(args: argparse.Namespace, masks: dict[str, dict[int, torch.Tensor]]) -> list[dict]:
    ns = SimpleNamespace(
        drop_from=args.drop_from,
        shells=args.shells,
        num_shards=1,
        shard_index=0,
        max_groups=None,
    )
    specs = [spec for spec in make_group_specs(ns, masks) if spec["kind"] == "add_shell_layer"]
    by_key = {spec_key(spec): spec for spec in specs}
    if not args.ranking_json:
        return specs
    ranking = json.load(open(args.ranking_json))
    ranked_rows = [row for row in ranking.get("groups", []) if row.get("kind") == "add_shell_layer"]
    ranked_rows.sort(
        key=lambda row: (
            row.get("accuracy", 0.0),
            row.get("delta_accuracy_per_1k_neurons", 0.0),
            -row.get("test_mlp_kept", 10**12),
        ),
        reverse=True,
    )
    ordered = []
    seen = set()
    for row in ranked_rows:
        key = spec_key(row)
        spec = by_key.get(key)
        if spec is None or key in seen:
            continue
        ordered.append(spec)
        seen.add(key)
    ordered.extend(spec for spec in specs if spec_key(spec) not in seen)
    return ordered


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists; pass --force")

    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records, n_pairs_by_position, skipped_by_position = load_eval_records(
        Path(args.position_out_dir), tokenizer, args.positions, args.n_prune_test
    )
    if not records:
        raise RuntimeError("no evaluation records")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    total_mlp = n_layers * d_ffn
    total_heads = n_layers * num_heads
    full_head = {layer: torch.ones(num_heads, dtype=torch.bool, device=device) for layer in range(n_layers)}
    by_mode, _head_by_mode = load_composed_masks(
        Path(args.position_out_dir) / "composed_union.full.npz",
        n_layers=n_layers,
        d_ffn=d_ffn,
        num_heads=num_heads,
        device=device,
    )
    masks = {mode: by_mode[mode] for mode in TRACKED_MODES if mode in by_mode}
    if args.base_mask not in masks:
        raise KeyError(f"unknown base mask {args.base_mask!r}")

    result = {
        "position_out_dir": args.position_out_dir,
        "model": args.model,
        "positions": args.positions,
        "n_prune_test": args.n_prune_test,
        "n_records": len(records),
        "n_pairs_by_position": n_pairs_by_position,
        "skipped_by_position": skipped_by_position,
        "target_accuracy": args.target_accuracy,
        "base_mask": args.base_mask,
        "baseline_masks": args.baseline_masks,
        "shells": args.shells,
        "ranking_json": args.ranking_json,
        "dtype": args.dtype,
        "device": device,
        "batch_size": batch_size,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_neurons": total_mlp,
        "total_attention_heads": total_heads,
        "baselines": {},
        "paths": [],
        "compressed_candidates": [],
    }
    write_json(out_path, result)

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
        print(f"  {label:>34} mlp={kept:>7d} exact={row['accuracy']:.2%}", flush=True)
        return row

    def compress_candidate(label: str, start_mask: dict[int, torch.Tensor], start_row: dict) -> tuple[dict, dict[int, torch.Tensor]]:
        current_mask = clone_mask(start_mask)
        current = start_row
        steps = []
        for step in range(1, args.max_compress_steps + 1):
            trials = []
            for layer in sorted(current_mask):
                if int(current_mask[layer].sum().item()) == 0:
                    continue
                trial_mask = drop_layer(current_mask, layer)
                trial = eval_mask(f"{label}_compress_{step}_drop_{layer:02d}", trial_mask)
                if trial["accuracy"] >= args.target_accuracy:
                    trials.append({**trial, "drop_layer": layer})
            if not trials:
                steps.append({"step": step, "accepted": False})
                break
            best = min(trials, key=lambda row: (row["mlp_kept"], -row["accuracy"]))
            current_mask = drop_layer(current_mask, int(best["drop_layer"]))
            current = eval_mask(f"{label}_compress_{step}_accepted", current_mask)
            steps.append({"step": step, "accepted": True, "accepted_trial": best, "current": current})
        return {"label": label, "initial": start_row, "steps": steps, "final": current}, current_mask

    candidate_masks: list[tuple[str, dict, dict[int, torch.Tensor]]] = []
    for name in args.baseline_masks:
        if name not in masks:
            continue
        row = eval_mask(name, masks[name])
        result["baselines"][name] = row
        write_json(out_path, result)
        if row["accuracy"] >= args.target_accuracy:
            candidate_masks.append((f"baseline_{name}", row, masks[name]))

    add_specs = ordered_add_specs(args, masks)[: args.max_add_candidates]
    base_row = result["baselines"].get(args.base_mask) or eval_mask(args.base_mask, masks[args.base_mask])
    current_mask = clone_mask(masks[args.base_mask])
    current = base_row
    remaining = {spec["id"]: spec for spec in add_specs}
    path = {"label": f"add_from_{args.base_mask}", "initial": current, "steps": []}
    for step in range(1, args.max_add_steps + 1):
        trials = []
        for group_id, spec in list(remaining.items()):
            trial_mask = apply_group(current_mask, spec, masks)
            if count_mask(trial_mask) == count_mask(current_mask):
                continue
            trial = eval_mask(f"add_step_{step}_{group_id}", trial_mask)
            trials.append({**trial, "group_id": group_id, "group": spec})
        if not trials:
            path["steps"].append({"step": step, "accepted": False})
            break
        target_trials = [row for row in trials if row["accuracy"] >= args.target_accuracy]
        if target_trials:
            best = min(target_trials, key=lambda row: (row["mlp_kept"], -row["accuracy"]))
        else:
            best = max(trials, key=lambda row: (row["accuracy"], -row["mlp_kept"]))
        current_mask = apply_group(current_mask, best["group"], masks)
        current = eval_mask(f"add_step_{step}_accepted", current_mask)
        remaining.pop(best["group_id"], None)
        path["steps"].append({"step": step, "accepted": True, "accepted_trial": best, "current": current})
        if current["accuracy"] >= args.target_accuracy:
            candidate_masks.append((f"add_from_{args.base_mask}", current, current_mask))
            break
    result["paths"].append(path)
    write_json(out_path, result)

    best_row = None
    best_mask = None
    for label, row, mask in candidate_masks:
        compressed, compressed_mask = compress_candidate(label, mask, row)
        result["compressed_candidates"].append(compressed)
        final = compressed["final"]
        if final["accuracy"] >= args.target_accuracy and (
            best_row is None or (final["mlp_kept"], -final["accuracy"]) < (best_row["mlp_kept"], -best_row["accuracy"])
        ):
            best_row = final
            best_mask = compressed_mask
        write_json(out_path, result)

    if best_row is not None and best_mask is not None:
        best_mask_path = out_path.with_suffix(".best_mask.full.npz")
        save_mask_npz(best_mask_path, best_mask)
        result["best"] = best_row
        result["best_mask_npz"] = str(best_mask_path)
    else:
        result["best"] = None
        result["best_mask_npz"] = None
    write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
