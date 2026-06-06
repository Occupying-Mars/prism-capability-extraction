#!/usr/bin/env python3
"""Greedy search over ADACS causal layer groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adacs_causal_group_compose import add_group_, empty_like, group_mask
from adacs_causal_group_rank import clone_mask, count_mask, full_mlp_mask
from adacs_trace_primitives import PRIMITIVE_POSITION, load_adacs_pairs
from position_interface_decomposition import (
    build_position_eval_records,
    load_union_npz,
    resolve_batch_size,
    run_position_check_cf_patch,
    split_pairs,
)
from union_attribution import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-json", required=True)
    parser.add_argument("--primitive", default="carry_tens_to_hundreds", choices=sorted(PRIMITIVE_POSITION))
    parser.add_argument("--search-pair-dir", required=True)
    parser.add_argument("--validation-pair-dir", default=None)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--search-n-pairs", type=int, default=20000)
    parser.add_argument("--search-n-prune-test", type=int, default=1000)
    parser.add_argument("--validation-n-pairs", type=int, default=None)
    parser.add_argument("--validation-n-eval", type=int, default=None)
    parser.add_argument("--base-mode", default="rel_0.05")
    parser.add_argument("--candidate-source-mode", default="topk_500")
    parser.add_argument("--candidate-minus-mode", default="rel_0.05")
    parser.add_argument("--max-forward-steps", type=int, default=16)
    parser.add_argument("--min-forward-delta", type=float, default=0.001)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--backward-tolerance", type=float, default=0.0)
    parser.add_argument("--max-backward-passes", type=int, default=6)
    parser.add_argument(
        "--start-from-base",
        action="store_true",
        help="Start greedy additions from the full base-mode mask instead of the positive-drop core.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def load_search_records(tokenizer, pair_dir: Path, primitive: str, n_pairs: int, n_prune_test: int):
    pairs, stats = load_adacs_pairs(pair_dir, primitive, n_pairs)
    _attr_pairs, test_pairs = split_pairs(pairs, n_prune_test)
    records, skipped = build_position_eval_records(tokenizer, test_pairs, PRIMITIVE_POSITION[primitive])
    return records, skipped, stats, len(pairs), len(test_pairs)


def load_validation_records(
    tokenizer,
    pair_dir: Path,
    primitive: str,
    n_pairs: int | None,
    n_eval: int | None,
):
    pairs, stats = load_adacs_pairs(pair_dir, primitive, n_pairs)
    eval_pairs = pairs if n_eval is None else pairs[:n_eval]
    records, skipped = build_position_eval_records(tokenizer, eval_pairs, PRIMITIVE_POSITION[primitive])
    return records, skipped, stats, len(pairs), len(eval_pairs)


def positive_drop_groups(ranking: dict, base_mode: str) -> list[dict]:
    return [
        row
        for row in ranking["groups"]
        if row["kind"] == "drop_layer"
        and row["base_mode"] == base_mode
        and row.get("delta_accuracy", 0.0) > 0
    ]


def shell_groups(ranking: dict, source_mode: str, minus_mode: str) -> list[dict]:
    rows = [
        row
        for row in ranking["groups"]
        if row["kind"] == "add_shell_layer"
        and row["source_mode"] == source_mode
        and row["minus_mode"] == minus_mode
    ]
    return sorted(
        rows,
        key=lambda row: (
            row.get("delta_accuracy", 0.0),
            row.get("delta_accuracy_per_1k_neurons", 0.0),
        ),
        reverse=True,
    )


def rebuild_mask(selected: list[dict], masks_by_mode: dict[str, dict[int, torch.Tensor]], base_mode: str):
    mask = empty_like(masks_by_mode[base_mode])
    for row in selected:
        add_group_(mask, group_mask(row, masks_by_mode))
    return mask


def save_mask_npz(path: Path, mask: dict[int, torch.Tensor]) -> None:
    import numpy as np

    pairs: list[tuple[int, int]] = []
    for layer in sorted(mask):
        idx = torch.nonzero(mask[layer].cpu(), as_tuple=False).flatten().tolist()
        pairs.extend((int(layer), int(i)) for i in idx)
    arr = np.array(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mlp_final=arr)


def subtract_group_(base: dict[int, torch.Tensor], removal: dict[int, torch.Tensor]) -> None:
    for layer in base:
        base[layer] &= ~removal[layer]


def summarize_eval(label: str, check: dict, mlp_keep: dict[int, torch.Tensor], total_mlp: int, total_heads: int) -> dict:
    kept = count_mask(mlp_keep)
    return {
        "label": label,
        "accuracy": check["accuracy"],
        "correct": check["correct"],
        "n": check["n"],
        "mlp_kept": kept,
        "mlp_keep_fraction": kept / max(total_mlp, 1),
        "heads_kept": total_heads,
        "heads_keep_fraction": 1.0,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists; pass --force to overwrite")

    ranking = json.load(open(args.ranking_json))
    position = PRIMITIVE_POSITION[args.primitive]
    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    search_records, search_skipped, search_pair_stats, search_n_pairs, search_n_eval_pairs = load_search_records(
        tokenizer,
        Path(args.search_pair_dir),
        args.primitive,
        args.search_n_pairs,
        args.search_n_prune_test,
    )
    if not search_records:
        raise RuntimeError("no search records")
    validation_records = None
    validation_meta = None
    if args.validation_pair_dir:
        validation_records, val_skipped, val_pair_stats, val_n_pairs, val_n_eval_pairs = load_validation_records(
            tokenizer,
            Path(args.validation_pair_dir),
            args.primitive,
            args.validation_n_pairs,
            args.validation_n_eval,
        )
        if not validation_records:
            raise RuntimeError("no validation records")
        validation_meta = {
            "pair_dir": args.validation_pair_dir,
            "pair_stats": val_pair_stats,
            "n_pairs": val_n_pairs,
            "n_eval_pairs": val_n_eval_pairs,
            "n_eval_records": len(validation_records),
            "skipped": val_skipped,
        }

    print(
        f"[model] {args.model} device={device} dtype={args.dtype} batch={batch_size} "
        f"search_records={len(search_records):,}",
        flush=True,
    )
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
    masks_by_mode = load_union_npz(
        Path(args.trace_dir) / f"{args.primitive}_union.full.npz",
        n_layers=n_layers,
        d_ffn=d_ffn,
    )
    full_head = {layer: torch.ones(num_heads, dtype=torch.bool, device=device) for layer in range(n_layers)}

    result = {
        "primitive": args.primitive,
        "position_probe": position,
        "ranking_json": args.ranking_json,
        "search_pair_dir": args.search_pair_dir,
        "trace_dir": args.trace_dir,
        "model": args.model,
        "dtype": args.dtype,
        "device": device,
        "batch_size": batch_size,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "num_heads": num_heads,
        "total_mlp_neurons": total_mlp,
        "total_attention_heads": total_heads,
        "search": {
            "pair_stats": search_pair_stats,
            "n_pairs": search_n_pairs,
            "n_eval_pairs": search_n_eval_pairs,
            "n_eval_records": len(search_records),
            "skipped": search_skipped,
            "base_mode": args.base_mode,
            "candidate_source_mode": args.candidate_source_mode,
            "candidate_minus_mode": args.candidate_minus_mode,
            "max_forward_steps": args.max_forward_steps,
            "min_forward_delta": args.min_forward_delta,
            "backward": args.backward,
            "backward_tolerance": args.backward_tolerance,
            "forward_steps": [],
            "backward_steps": [],
        },
        "validation": validation_meta,
        "baselines": {},
    }

    def eval_records(label: str, records: list[dict], mlp_keep: dict[int, torch.Tensor]) -> dict:
        check = run_position_check_cf_patch(
            model,
            records,
            mlp_keep=mlp_keep,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            label=label,
        )
        row = summarize_eval(label, check, mlp_keep, total_mlp, total_heads)
        print(f"  {label} mlp={row['mlp_kept']:,} cf={row['accuracy']:.2%}", flush=True)
        return row

    for name, mask in [
        ("full", full_mlp_mask(n_layers, d_ffn)),
        (args.base_mode, masks_by_mode[args.base_mode]),
        (args.candidate_source_mode, masks_by_mode[args.candidate_source_mode]),
    ]:
        result["baselines"][name] = eval_records(f"search_{name}", search_records, mask)
        write_json(out_path, result)

    selected = [] if args.start_from_base else sorted(positive_drop_groups(ranking, args.base_mode), key=lambda row: row["layer"])
    candidates = shell_groups(ranking, args.candidate_source_mode, args.candidate_minus_mode)
    candidate_by_id = {row["id"]: row for row in candidates}
    current_mask = clone_mask(masks_by_mode[args.base_mode]) if args.start_from_base else rebuild_mask(selected, masks_by_mode, args.base_mode)
    current = eval_records("search_init_base" if args.start_from_base else "search_init_positive_drop_core", search_records, current_mask)
    result["search"]["initial_selected_group_ids"] = [row["id"] for row in selected]
    result["search"]["initial"] = current
    write_json(out_path, result)

    remaining_ids = [row["id"] for row in candidates]
    for step_idx in range(args.max_forward_steps):
        trials = []
        for group_id in remaining_ids:
            group = candidate_by_id[group_id]
            if args.start_from_base:
                trial_mask = clone_mask(current_mask)
                add_group_(trial_mask, group_mask(group, masks_by_mode))
            else:
                trial_selected = selected + [group]
                trial_mask = rebuild_mask(trial_selected, masks_by_mode, args.base_mode)
            trial = eval_records(f"forward_{step_idx + 1}_{group_id}", search_records, trial_mask)
            trials.append({**trial, "group_id": group_id, "group": group})
        if not trials:
            break
        best = max(trials, key=lambda row: (row["accuracy"], -row["mlp_kept"]))
        accepted = best["accuracy"] >= current["accuracy"] + args.min_forward_delta
        result["search"]["forward_steps"].append(
            {
                "step": step_idx + 1,
                "accepted": accepted,
                "current_before": current,
                "best_trial": best,
                "top_trials": sorted(trials, key=lambda row: row["accuracy"], reverse=True)[:8],
            }
        )
        write_json(out_path, result)
        if not accepted:
            break
        selected.append(candidate_by_id[best["group_id"]])
        remaining_ids.remove(best["group_id"])
        current = best
        if args.start_from_base:
            add_group_(current_mask, group_mask(candidate_by_id[best["group_id"]], masks_by_mode))
        else:
            current_mask = rebuild_mask(selected, masks_by_mode, args.base_mode)

    if args.backward:
        for pass_idx in range(args.max_backward_passes):
            trials = []
            for row in selected:
                trial_selected = [keep for keep in selected if keep["id"] != row["id"]]
                if args.start_from_base:
                    trial_mask = clone_mask(current_mask)
                    subtract_group_(trial_mask, group_mask(row, masks_by_mode))
                else:
                    trial_mask = rebuild_mask(trial_selected, masks_by_mode, args.base_mode)
                trial = eval_records(f"backward_{pass_idx + 1}_{row['id']}", search_records, trial_mask)
                trials.append({**trial, "remove_group_id": row["id"], "group": row})
            if not trials:
                break
            best = max(trials, key=lambda row: (row["accuracy"], -row["mlp_kept"]))
            accepted = best["accuracy"] >= current["accuracy"] - args.backward_tolerance
            result["search"]["backward_steps"].append(
                {
                    "pass": pass_idx + 1,
                    "accepted": accepted,
                    "current_before": current,
                    "best_trial": best,
                    "top_trials": sorted(trials, key=lambda row: row["accuracy"], reverse=True)[:8],
                }
            )
            write_json(out_path, result)
            if not accepted:
                break
            selected = [row for row in selected if row["id"] != best["remove_group_id"]]
            current = best
            if args.start_from_base:
                current_mask = clone_mask(current_mask)
                subtract_group_(current_mask, group_mask(best["group"], masks_by_mode))
            else:
                current_mask = rebuild_mask(selected, masks_by_mode, args.base_mode)

    final_mask = current_mask if args.start_from_base else rebuild_mask(selected, masks_by_mode, args.base_mode)
    final = eval_records("search_final", search_records, final_mask)
    result["search"]["final"] = final
    result["selected_group_ids"] = [row["id"] for row in selected]
    result["selected_groups"] = selected
    mask_path = out_path.with_suffix(".final_mask.full.npz")
    save_mask_npz(mask_path, final_mask)
    result["final_mask_npz"] = str(mask_path)

    if validation_records is not None:
        result["validation"] = dict(result["validation"] or {})
        result["validation"]["baselines"] = {
            args.base_mode: eval_records(f"validation_{args.base_mode}", validation_records, masks_by_mode[args.base_mode]),
            args.candidate_source_mode: eval_records(
                f"validation_{args.candidate_source_mode}",
                validation_records,
                masks_by_mode[args.candidate_source_mode],
            ),
        }
        result["validation"]["final"] = eval_records("validation_final", validation_records, final_mask)

    write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
