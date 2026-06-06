#!/usr/bin/env python3
"""Causal group ranking for ADACS primitive masks.

Given a saved ADACS trace mask, evaluate layer-sized MLP groups under the same
position-level CF-patching probe used by `adacs_trace_primitives.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adacs_trace_primitives import PRIMITIVE_POSITION, load_adacs_pairs
from position_interface_decomposition import (
    build_position_eval_records,
    load_union_npz,
    resolve_batch_size,
    run_position_check_cf_patch,
    split_pairs,
)
from union_attribution import TRACKED_MODES, _tensor_union_size, pick_device


DEFAULT_BASELINES = ["topk_100", "rel_0.05", "topk_500", "rel_0.01"]
DEFAULT_SHELLS = ["rel_0.05:topk_100", "topk_500:rel_0.05", "rel_0.01:topk_500"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive", default="carry_tens_to_hundreds", choices=sorted(PRIMITIVE_POSITION))
    parser.add_argument("--pair-dir")
    parser.add_argument("--trace-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-pairs", type=int, default=20000)
    parser.add_argument("--n-prune-test", type=int, default=1000)
    parser.add_argument("--max-eval-records", type=int, default=None)
    parser.add_argument("--base-mode", default="rel_0.05", choices=TRACKED_MODES)
    parser.add_argument("--baseline-modes", nargs="+", default=DEFAULT_BASELINES, choices=TRACKED_MODES)
    parser.add_argument(
        "--shells",
        nargs="+",
        default=DEFAULT_SHELLS,
        help="Shell specs as SOURCE:MINUS. Each layer of SOURCE & ~MINUS is added to MINUS.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--merge-shards",
        nargs="*",
        default=None,
        help="Merge shard JSON outputs and exit. The --out path receives the merged JSON.",
    )
    return parser.parse_args()


def count_mask(mask: dict[int, torch.Tensor]) -> int:
    return _tensor_union_size(mask)


def clone_mask(mask: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {layer: values.clone() for layer, values in mask.items()}


def full_mlp_mask(n_layers: int, d_ffn: int) -> dict[int, torch.Tensor]:
    return {layer: torch.ones(d_ffn, dtype=torch.bool) for layer in range(n_layers)}


def layer_only_mask(source: dict[int, torch.Tensor], layer: int, d_ffn: int) -> dict[int, torch.Tensor]:
    out = {li: torch.zeros(d_ffn, dtype=torch.bool) for li in source}
    out[layer] = source[layer].clone()
    return out


def shell_mask(
    masks_by_mode: dict[str, dict[int, torch.Tensor]],
    source_mode: str,
    minus_mode: str,
) -> dict[int, torch.Tensor]:
    source = masks_by_mode[source_mode]
    minus = masks_by_mode[minus_mode]
    return {layer: source[layer] & ~minus[layer] for layer in source}


def make_group_specs(
    masks_by_mode: dict[str, dict[int, torch.Tensor]],
    *,
    base_mode: str,
    shell_specs: list[str],
    d_ffn: int,
) -> list[dict]:
    specs: list[dict] = []
    base = masks_by_mode[base_mode]
    for layer in sorted(base):
        group_size = int(base[layer].sum().item())
        if group_size == 0:
            continue
        specs.append(
            {
                "id": f"drop__{base_mode}__layer_{layer:02d}",
                "kind": "drop_layer",
                "base_mode": base_mode,
                "source_mode": base_mode,
                "minus_mode": None,
                "add_base_mode": None,
                "layer": layer,
                "group_size": group_size,
            }
        )

    for spec in shell_specs:
        if ":" not in spec:
            raise ValueError(f"invalid shell spec {spec!r}; expected SOURCE:MINUS")
        source_mode, minus_mode = spec.split(":", 1)
        if source_mode not in masks_by_mode or minus_mode not in masks_by_mode:
            raise ValueError(f"unknown shell mode in {spec!r}")
        shell = shell_mask(masks_by_mode, source_mode, minus_mode)
        for layer in sorted(shell):
            group_size = int(shell[layer].sum().item())
            if group_size == 0:
                continue
            specs.append(
                {
                    "id": f"add__{source_mode}_minus_{minus_mode}__to_{minus_mode}__layer_{layer:02d}",
                    "kind": "add_shell_layer",
                    "base_mode": None,
                    "source_mode": source_mode,
                    "minus_mode": minus_mode,
                    "add_base_mode": minus_mode,
                    "layer": layer,
                    "group_size": group_size,
                }
            )
    return specs


def mask_for_group(
    spec: dict,
    masks_by_mode: dict[str, dict[int, torch.Tensor]],
    *,
    d_ffn: int,
) -> dict[int, torch.Tensor]:
    layer = int(spec["layer"])
    if spec["kind"] == "drop_layer":
        out = clone_mask(masks_by_mode[spec["base_mode"]])
        out[layer] &= ~masks_by_mode[spec["source_mode"]][layer]
        return out
    if spec["kind"] == "add_shell_layer":
        out = clone_mask(masks_by_mode[spec["add_base_mode"]])
        shell = shell_mask(masks_by_mode, spec["source_mode"], spec["minus_mode"])
        out[layer] |= shell[layer]
        return out
    if spec["kind"] == "only_layer":
        return layer_only_mask(masks_by_mode[spec["source_mode"]], layer, d_ffn)
    raise ValueError(f"unknown group kind {spec['kind']!r}")


def add_scores(row: dict, baselines: dict[str, dict]) -> dict:
    if row["kind"] == "drop_layer":
        ref_name = row["base_mode"]
        ref_acc = baselines[ref_name]["accuracy"]
        delta = ref_acc - row["accuracy"]
    elif row["kind"] == "add_shell_layer":
        ref_name = row["add_base_mode"]
        ref_acc = baselines[ref_name]["accuracy"]
        delta = row["accuracy"] - ref_acc
    else:
        ref_name = None
        ref_acc = None
        delta = row["accuracy"]
    group_size = max(int(row["group_size"]), 1)
    row["reference"] = ref_name
    row["reference_accuracy"] = ref_acc
    row["delta_accuracy"] = delta
    row["delta_accuracy_per_1k_neurons"] = delta * 1000.0 / group_size
    return row


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def rank_groups(groups: list[dict]) -> dict:
    positive = [g for g in groups if g.get("delta_accuracy", 0.0) > 0]
    return {
        "top_by_delta_accuracy": sorted(positive, key=lambda g: g["delta_accuracy"], reverse=True)[:20],
        "top_by_delta_per_1k": sorted(
            positive, key=lambda g: g["delta_accuracy_per_1k_neurons"], reverse=True
        )[:20],
    }


def merge_shards(paths: list[str], out_path: Path) -> None:
    payloads = [json.load(open(p)) for p in paths]
    if not payloads:
        raise ValueError("--merge-shards requires at least one input")
    merged = dict(payloads[0])
    groups_by_id = {}
    for payload in payloads:
        for row in payload.get("groups", []):
            groups_by_id[row["id"]] = row
    merged["groups"] = sorted(groups_by_id.values(), key=lambda r: (r["kind"], r.get("source_mode") or "", r["layer"]))
    merged["num_shards_merged"] = len(payloads)
    merged["rankings"] = rank_groups(merged["groups"])
    write_json(out_path, merged)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if args.merge_shards is not None:
        merge_shards(args.merge_shards, out_path)
        print(f"[done] merged {len(args.merge_shards)} shards -> {out_path}", flush=True)
        return

    if args.pair_dir is None or args.trace_dir is None:
        raise ValueError("--pair-dir and --trace-dir are required unless --merge-shards is used")
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard settings")

    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    position = PRIMITIVE_POSITION[args.primitive]

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

    if out_path.exists() and not args.force:
        result = json.load(open(out_path))
    else:
        result = {
            "primitive": args.primitive,
            "position_probe": position,
            "pair_dir": args.pair_dir,
            "trace_dir": args.trace_dir,
            "pair_stats": pair_stats,
            "n_pairs": len(pairs),
            "n_test_pairs": len(test_pairs),
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
            "base_mode": args.base_mode,
            "baseline_modes": args.baseline_modes,
            "shells": args.shells,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "baselines": {},
            "groups": [],
        }

    def eval_mask(name: str, mlp_keep: dict[int, torch.Tensor]) -> dict:
        check = run_position_check_cf_patch(
            model,
            records,
            mlp_keep=mlp_keep,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            label=name,
        )
        mlp_kept = count_mask(mlp_keep)
        return {
            "label": name,
            "accuracy": check["accuracy"],
            "correct": check["correct"],
            "n": check["n"],
            "mlp_kept": mlp_kept,
            "mlp_keep_fraction": mlp_kept / max(total_mlp, 1),
            "heads_kept": total_heads,
            "heads_keep_fraction": 1.0,
        }

    baselines = result.setdefault("baselines", {})
    if "full" not in baselines:
        baselines["full"] = eval_mask("full", full_mlp_mask(n_layers, d_ffn))
        write_json(out_path, result)
        print(f"  full cf={baselines['full']['accuracy']:.2%}", flush=True)
    for mode in args.baseline_modes:
        if mode not in baselines:
            baselines[mode] = eval_mask(mode, masks_by_mode[mode])
            write_json(out_path, result)
        print(f"  {mode:>8} cf={baselines[mode]['accuracy']:.2%} mlp={baselines[mode]['mlp_kept']:,}", flush=True)

    specs = make_group_specs(
        masks_by_mode,
        base_mode=args.base_mode,
        shell_specs=args.shells,
        d_ffn=d_ffn,
    )
    specs = [spec for i, spec in enumerate(specs) if i % args.num_shards == args.shard_index]
    if args.max_groups is not None:
        specs = specs[: args.max_groups]
    completed = {row["id"] for row in result.get("groups", [])}
    print(f"[groups] shard={args.shard_index}/{args.num_shards} todo={sum(s['id'] not in completed for s in specs)}", flush=True)
    for spec in specs:
        if spec["id"] in completed:
            continue
        mlp_keep = mask_for_group(spec, masks_by_mode, d_ffn=d_ffn)
        check = eval_mask(spec["id"], mlp_keep)
        row = {
            **spec,
            "accuracy": check["accuracy"],
            "correct": check["correct"],
            "n": check["n"],
            "test_mlp_kept": check["mlp_kept"],
            "test_mlp_keep_fraction": check["mlp_keep_fraction"],
        }
        row = add_scores(row, baselines)
        result["groups"].append(row)
        result["rankings"] = rank_groups(result["groups"])
        write_json(out_path, result)
        print(
            f"  {row['id']} size={row['group_size']:,} acc={row['accuracy']:.2%} "
            f"delta={row['delta_accuracy']:+.2%} per1k={row['delta_accuracy_per_1k_neurons']:+.3%}",
            flush=True,
        )
    result["rankings"] = rank_groups(result["groups"])
    write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
