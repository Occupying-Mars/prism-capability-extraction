#!/usr/bin/env python3
"""Evaluate cumulative composites from ADACS causal group rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adacs_causal_group_rank import clone_mask, count_mask, shell_mask
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
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-pairs", type=int, default=20000)
    parser.add_argument("--n-prune-test", type=int, default=1000)
    parser.add_argument("--max-eval-records", type=int, default=None)
    parser.add_argument("--base-mode", default="rel_0.05")
    parser.add_argument("--outer-source-mode", default="topk_500")
    parser.add_argument("--outer-minus-mode", default="rel_0.05")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 5, 8, 13, 21, 28])
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def empty_like(mask: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {layer: torch.zeros_like(values, dtype=torch.bool) for layer, values in mask.items()}


def group_mask(row: dict, masks_by_mode: dict[str, dict[int, torch.Tensor]]) -> dict[int, torch.Tensor]:
    layer = int(row["layer"])
    if row["kind"] == "drop_layer":
        source = masks_by_mode[row["source_mode"]]
        out = empty_like(source)
        out[layer] = source[layer].clone()
        return out
    if row["kind"] == "add_shell_layer":
        shell = shell_mask(masks_by_mode, row["source_mode"], row["minus_mode"])
        out = empty_like(shell)
        out[layer] = shell[layer].clone()
        return out
    raise ValueError(f"unsupported row kind {row['kind']!r}")


def add_group_(base: dict[int, torch.Tensor], addition: dict[int, torch.Tensor]) -> None:
    for layer in base:
        base[layer] |= addition[layer]


def positive_outer_groups(ranking: dict, source_mode: str, minus_mode: str) -> list[dict]:
    rows = []
    for row in ranking["groups"]:
        if row["kind"] != "add_shell_layer":
            continue
        if row["source_mode"] != source_mode or row["minus_mode"] != minus_mode:
            continue
        if row.get("delta_accuracy", 0.0) <= 0:
            continue
        rows.append(row)
    return rows


def positive_drop_groups(ranking: dict, base_mode: str) -> list[dict]:
    rows = []
    for row in ranking["groups"]:
        if row["kind"] != "drop_layer" or row["base_mode"] != base_mode:
            continue
        if row.get("delta_accuracy", 0.0) <= 0:
            continue
        rows.append(row)
    return rows


def step_values(requested: list[int], max_len: int) -> list[int]:
    vals = sorted({min(s, max_len) for s in requested if s > 0 and max_len > 0})
    if max_len and max_len not in vals:
        vals.append(max_len)
    return vals


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    position = PRIMITIVE_POSITION[args.primitive]
    ranking = json.load(open(args.ranking_json))

    print(f"[records] primitive={args.primitive} position={position}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pairs, pair_stats = load_adacs_pairs(Path(args.pair_dir), args.primitive, args.n_pairs)
    _attr_pairs, test_pairs = split_pairs(pairs, args.n_prune_test)
    records, skipped = build_position_eval_records(tokenizer, test_pairs, position)
    if args.max_eval_records is not None:
        records = records[: args.max_eval_records]
    if not records:
        raise RuntimeError("no evaluation records")
    print(f"  eval_records={len(records):,} skipped={skipped:,}", flush=True)

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
        "pair_dir": args.pair_dir,
        "trace_dir": args.trace_dir,
        "pair_stats": pair_stats,
        "n_eval_records": len(records),
        "eval_records_skipped": skipped,
        "model": args.model,
        "dtype": args.dtype,
        "device": device,
        "batch_size": batch_size,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "num_heads": num_heads,
        "total_mlp_neurons": total_mlp,
        "total_attention_heads": total_heads,
        "plans": {},
    }

    def eval_mask(label: str, mlp_keep: dict[int, torch.Tensor]) -> dict:
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
        mlp_kept = count_mask(mlp_keep)
        row = {
            "label": label,
            "accuracy": check["accuracy"],
            "correct": check["correct"],
            "n": check["n"],
            "mlp_kept": mlp_kept,
            "mlp_keep_fraction": mlp_kept / max(total_mlp, 1),
            "heads_kept": total_heads,
            "heads_keep_fraction": 1.0,
        }
        print(f"  {label} mlp={mlp_kept:,} cf={row['accuracy']:.2%}", flush=True)
        return row

    result["baseline"] = {
        args.base_mode: eval_mask(args.base_mode, masks_by_mode[args.base_mode]),
        args.outer_source_mode: eval_mask(args.outer_source_mode, masks_by_mode[args.outer_source_mode]),
    }
    write_json(out_path, result)

    outer = positive_outer_groups(ranking, args.outer_source_mode, args.outer_minus_mode)
    outer_by_delta = sorted(outer, key=lambda r: r["delta_accuracy"], reverse=True)
    outer_by_per1k = sorted(outer, key=lambda r: r["delta_accuracy_per_1k_neurons"], reverse=True)
    core_by_delta = sorted(positive_drop_groups(ranking, args.base_mode), key=lambda r: r["delta_accuracy"], reverse=True)
    core_by_per1k = sorted(
        positive_drop_groups(ranking, args.base_mode),
        key=lambda r: r["delta_accuracy_per_1k_neurons"],
        reverse=True,
    )

    plans = [
        ("base_plus_outer_by_delta", clone_mask(masks_by_mode[args.base_mode]), outer_by_delta),
        ("base_plus_outer_by_per1k", clone_mask(masks_by_mode[args.base_mode]), outer_by_per1k),
        ("core_by_drop_delta", empty_like(masks_by_mode[args.base_mode]), core_by_delta),
        ("core_by_drop_per1k", empty_like(masks_by_mode[args.base_mode]), core_by_per1k),
    ]

    for plan_name, base_mask, groups in plans:
        rows = []
        cumulative = clone_mask(base_mask)
        added_ids: list[str] = []
        steps = step_values(args.steps, len(groups))
        cursor = 0
        for step in steps:
            while cursor < step:
                row = groups[cursor]
                add_group_(cumulative, group_mask(row, masks_by_mode))
                added_ids.append(row["id"])
                cursor += 1
            check = eval_mask(f"{plan_name}_top_{step}", cumulative)
            rows.append({**check, "step": step, "added_group_ids": list(added_ids)})
            result["plans"][plan_name] = rows
            write_json(out_path, result)

    hybrid_core_n = min(21, len(core_by_delta))
    if hybrid_core_n:
        for plan_name, outer_groups in [
            ("core21_drop_delta_plus_outer_delta", outer_by_delta),
            ("core21_drop_delta_plus_outer_per1k", outer_by_per1k),
        ]:
            rows = []
            cumulative = empty_like(masks_by_mode[args.base_mode])
            added_ids: list[str] = []
            for row in core_by_delta[:hybrid_core_n]:
                add_group_(cumulative, group_mask(row, masks_by_mode))
                added_ids.append(row["id"])
            steps = step_values(args.steps, len(outer_groups))
            cursor = 0
            for step in steps:
                while cursor < step:
                    row = outer_groups[cursor]
                    add_group_(cumulative, group_mask(row, masks_by_mode))
                    added_ids.append(row["id"])
                    cursor += 1
                check = eval_mask(f"{plan_name}_outer_top_{step}", cumulative)
                rows.append(
                    {
                        **check,
                        "core_step": hybrid_core_n,
                        "outer_step": step,
                        "added_group_ids": list(added_ids),
                    }
                )
                result["plans"][plan_name] = rows
                write_json(out_path, result)

    write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
