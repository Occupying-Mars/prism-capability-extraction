#!/usr/bin/env python3
"""Direct full-answer group search over composed position masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_position_composed_full_answer import load_composed_masks, load_eval_records, run_full_answer_cf_patch
from position_interface_decomposition import resolve_batch_size
from union_attribution import TRACKED_MODES, _tensor_union_size, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-out-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-prune-test", type=int, default=500)
    parser.add_argument("--positions", nargs="+", default=["hundreds", "tens", "ones"])
    parser.add_argument("--modes", nargs="+", default=TRACKED_MODES)
    parser.add_argument("--extra-mask", action="append", default=[], metavar="NAME:PATH")
    parser.add_argument("--baseline-masks", nargs="+", default=["topk_100", "rel_0.05", "topk_500", "rel_0.01", "topk_2000", "rel_0.001"])
    parser.add_argument("--base-mask", default="topk_500")
    parser.add_argument("--drop-from", nargs="+", default=["topk_500"])
    parser.add_argument(
        "--shells",
        nargs="+",
        default=["rel_0.01:topk_500", "topk_2000:topk_500", "rel_0.001:topk_500"],
        help="Candidate add shells as SOURCE:MINUS.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--merge-shards", nargs="*", default=None)
    parser.add_argument("--ranking-json", default=None)
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--max-greedy-candidates", type=int, default=32)
    parser.add_argument("--max-forward-steps", type=int, default=12)
    parser.add_argument("--min-forward-delta", type=float, default=0.001)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def clone_mask(mask: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {layer: values.clone() for layer, values in mask.items()}


def empty_like(mask: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {layer: torch.zeros_like(values, dtype=torch.bool) for layer, values in mask.items()}


def count_mask(mask: dict[int, torch.Tensor]) -> int:
    return _tensor_union_size(mask)


def load_mlp_final_npz(path: Path, *, n_layers: int, d_ffn: int, device: str) -> dict[int, torch.Tensor]:
    data = np.load(path)
    if "mlp_final" not in data:
        raise KeyError(f"mlp_final missing from {path}")
    mask = {layer: torch.zeros(d_ffn, dtype=torch.bool, device=device) for layer in range(n_layers)}
    for layer, idx in data["mlp_final"].tolist():
        mask[int(layer)][int(idx)] = True
    return mask


def save_mask_npz(path: Path, mask: dict[int, torch.Tensor]) -> None:
    pairs: list[tuple[int, int]] = []
    for layer in sorted(mask):
        idx = torch.nonzero(mask[layer].detach().cpu(), as_tuple=False).flatten().tolist()
        pairs.extend((int(layer), int(i)) for i in idx)
    arr = np.array(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mlp_final=arr)


def parse_extra_masks(values: list[str]) -> list[tuple[str, Path]]:
    out = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"--extra-mask must be NAME:PATH, got {value!r}")
        name, raw_path = value.split(":", 1)
        if not name:
            raise ValueError(f"empty extra mask name in {value!r}")
        out.append((name, Path(raw_path)))
    return out


def shell_mask(masks: dict[str, dict[int, torch.Tensor]], source: str, minus: str) -> dict[int, torch.Tensor]:
    if source not in masks or minus not in masks:
        raise KeyError(f"unknown shell mask {source}:{minus}")
    return {layer: masks[source][layer] & ~masks[minus][layer] for layer in masks[source]}


def group_mask(spec: dict, masks: dict[str, dict[int, torch.Tensor]]) -> dict[int, torch.Tensor]:
    layer = int(spec["layer"])
    if spec["kind"] == "drop_layer":
        source = masks[spec["source_mask"]]
        out = empty_like(source)
        out[layer] = source[layer].clone()
        return out
    if spec["kind"] == "add_shell_layer":
        shell = shell_mask(masks, spec["source_mask"], spec["minus_mask"])
        out = empty_like(shell)
        out[layer] = shell[layer].clone()
        return out
    raise ValueError(f"unknown group kind {spec['kind']!r}")


def apply_group(mask: dict[int, torch.Tensor], spec: dict, masks: dict[str, dict[int, torch.Tensor]]) -> dict[int, torch.Tensor]:
    out = clone_mask(mask)
    group = group_mask(spec, masks)
    if spec["kind"] == "drop_layer":
        for layer in out:
            out[layer] &= ~group[layer]
    elif spec["kind"] == "add_shell_layer":
        for layer in out:
            out[layer] |= group[layer]
    else:
        raise ValueError(f"unknown group kind {spec['kind']!r}")
    return out


def make_group_specs(args: argparse.Namespace, masks: dict[str, dict[int, torch.Tensor]]) -> list[dict]:
    specs: list[dict] = []
    for name in args.drop_from:
        if name not in masks:
            raise KeyError(f"unknown --drop-from mask {name!r}")
        for layer in sorted(masks[name]):
            group_size = int(masks[name][layer].sum().item())
            if group_size == 0:
                continue
            specs.append(
                {
                    "id": f"drop__{name}__layer_{layer:02d}",
                    "kind": "drop_layer",
                    "source_mask": name,
                    "minus_mask": None,
                    "layer": layer,
                    "group_size": group_size,
                }
            )
    for raw in args.shells:
        if ":" not in raw:
            raise ValueError(f"invalid shell {raw!r}; expected SOURCE:MINUS")
        source, minus = raw.split(":", 1)
        shell = shell_mask(masks, source, minus)
        for layer in sorted(shell):
            group_size = int(shell[layer].sum().item())
            if group_size == 0:
                continue
            specs.append(
                {
                    "id": f"add__{source}_minus_{minus}__layer_{layer:02d}",
                    "kind": "add_shell_layer",
                    "source_mask": source,
                    "minus_mask": minus,
                    "layer": layer,
                    "group_size": group_size,
                }
            )
    specs = [spec for i, spec in enumerate(specs) if i % args.num_shards == args.shard_index]
    if args.max_groups is not None:
        specs = specs[: args.max_groups]
    return specs


def rank_groups(groups: list[dict]) -> dict:
    positive = [row for row in groups if row.get("delta_accuracy", 0.0) > 0.0]
    return {
        "top_by_delta_accuracy": sorted(positive, key=lambda row: row["delta_accuracy"], reverse=True)[:20],
        "top_by_delta_per_1k": sorted(positive, key=lambda row: row["delta_accuracy_per_1k_neurons"], reverse=True)[:20],
    }


def merge_shards(paths: list[str], out_path: Path) -> None:
    payloads = [json.load(open(path)) for path in paths]
    if not payloads:
        raise ValueError("--merge-shards needs at least one input")
    merged = dict(payloads[0])
    groups_by_id = {}
    for payload in payloads:
        for row in payload.get("groups", []):
            groups_by_id[row["id"]] = row
    merged["groups"] = sorted(groups_by_id.values(), key=lambda row: (row["kind"], row["id"]))
    merged["num_shards_merged"] = len(payloads)
    merged["rankings"] = rank_groups(merged["groups"])
    write_json(out_path, merged)
    print(f"[done] merged {len(paths)} shards -> {out_path}", flush=True)


def summarize_eval(label: str, check: dict, mask: dict[int, torch.Tensor], *, total_mlp: int, total_heads: int) -> dict:
    kept = count_mask(mask)
    return {
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


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if args.merge_shards is not None:
        merge_shards(args.merge_shards, out_path)
        return
    if args.position_out_dir is None:
        raise ValueError("--position-out-dir is required unless --merge-shards is used")
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard settings")
    if out_path.exists() and not args.force and args.ranking_json is None:
        result = json.load(open(out_path))
    else:
        result = None

    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[records] {args.position_out_dir} positions={','.join(args.positions)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records, n_pairs_by_position, skipped_by_position = load_eval_records(
        Path(args.position_out_dir), tokenizer, args.positions, args.n_prune_test
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

    by_mode, _head_by_mode = load_composed_masks(
        Path(args.position_out_dir) / "composed_union.full.npz",
        n_layers=n_layers,
        d_ffn=d_ffn,
        num_heads=num_heads,
        device=device,
    )
    masks: dict[str, dict[int, torch.Tensor]] = {mode: by_mode[mode] for mode in args.modes if mode in by_mode}
    for name, path in parse_extra_masks(args.extra_mask):
        masks[name] = load_mlp_final_npz(path, n_layers=n_layers, d_ffn=d_ffn, device=device)
    if args.base_mask not in masks:
        raise KeyError(f"unknown --base-mask {args.base_mask!r}; available={sorted(masks)}")

    if result is None:
        result = {
            "position_out_dir": args.position_out_dir,
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
            "total_mlp_neurons": total_mlp,
            "total_attention_heads": total_heads,
            "base_mask": args.base_mask,
            "drop_from": args.drop_from,
            "shells": args.shells,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "baselines": {},
            "groups": [],
            "greedy": {},
        }

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
        row = summarize_eval(label, check, mask, total_mlp=total_mlp, total_heads=total_heads)
        print(f"  {label:>38} mlp={row['mlp_kept']:>7d} exact={row['accuracy']:.2%}", flush=True)
        return row

    baselines = result.setdefault("baselines", {})
    for name in args.baseline_masks:
        if name not in masks:
            continue
        if name not in baselines:
            baselines[name] = eval_mask(name, masks[name])
            write_json(out_path, result)
        else:
            print(f"  {name:>38} exact={baselines[name]['accuracy']:.2%} cached", flush=True)

    if args.ranking_json is None:
        specs = make_group_specs(args, masks)
        completed = {row["id"] for row in result.get("groups", [])}
        base_row = baselines.get(args.base_mask) or eval_mask(args.base_mask, masks[args.base_mask])
        baselines[args.base_mask] = base_row
        print(f"[rank] todo={sum(spec['id'] not in completed for spec in specs)} shard={args.shard_index}/{args.num_shards}", flush=True)
        for spec in specs:
            if spec["id"] in completed:
                continue
            trial_mask = apply_group(masks[args.base_mask], spec, masks)
            check = eval_mask(spec["id"], trial_mask)
            delta = check["accuracy"] - base_row["accuracy"]
            row = {
                **spec,
                "accuracy": check["accuracy"],
                "correct": check["correct"],
                "n": check["n"],
                "test_mlp_kept": check["mlp_kept"],
                "test_mlp_keep_fraction": check["mlp_keep_fraction"],
                "reference": args.base_mask,
                "reference_accuracy": base_row["accuracy"],
                "delta_accuracy": delta,
                "delta_accuracy_per_1k_neurons": delta * 1000.0 / max(int(spec["group_size"]), 1),
            }
            result["groups"].append(row)
            result["rankings"] = rank_groups(result["groups"])
            write_json(out_path, result)
            print(
                f"    delta={row['delta_accuracy']:+.2%} per1k={row['delta_accuracy_per_1k_neurons']:+.3%}",
                flush=True,
            )
    else:
        ranking = json.load(open(args.ranking_json))
        result["ranking_json"] = args.ranking_json
        result["groups"] = ranking.get("groups", [])
        result["rankings"] = rank_groups(result["groups"])
        write_json(out_path, result)

    if args.rank_only:
        print(f"[done] rank-only -> {out_path}", flush=True)
        return

    ranked = sorted(
        [row for row in result.get("groups", []) if row.get("delta_accuracy", -1.0) > 0.0],
        key=lambda row: (row["delta_accuracy"], row.get("delta_accuracy_per_1k_neurons", 0.0)),
        reverse=True,
    )[: args.max_greedy_candidates]
    current_mask = clone_mask(masks[args.base_mask])
    current = eval_mask("greedy_init", current_mask)
    remaining = {row["id"]: row for row in ranked}
    accepted: list[dict] = []
    result["greedy"] = {
        "max_greedy_candidates": args.max_greedy_candidates,
        "max_forward_steps": args.max_forward_steps,
        "min_forward_delta": args.min_forward_delta,
        "initial": current,
        "steps": [],
    }
    write_json(out_path, result)

    for step_idx in range(args.max_forward_steps):
        trials = []
        for group_id, spec in list(remaining.items()):
            trial_mask = apply_group(current_mask, spec, masks)
            if count_mask(trial_mask) == count_mask(current_mask) and spec["kind"] == "add_shell_layer":
                continue
            trial = eval_mask(f"greedy_step_{step_idx + 1}_{group_id}", trial_mask)
            trials.append({**trial, "group_id": group_id, "group": spec})
        if not trials:
            break
        best = max(trials, key=lambda row: (row["accuracy"], -row["mlp_kept"]))
        accepted_step = best["accuracy"] >= current["accuracy"] + args.min_forward_delta
        result["greedy"]["steps"].append(
            {
                "step": step_idx + 1,
                "accepted": accepted_step,
                "current_before": current,
                "best_trial": best,
                "top_trials": sorted(trials, key=lambda row: row["accuracy"], reverse=True)[:8],
            }
        )
        write_json(out_path, result)
        if not accepted_step:
            break
        spec = remaining.pop(best["group_id"])
        current_mask = apply_group(current_mask, spec, masks)
        current = best
        accepted.append(spec)

    final = eval_mask("greedy_final", current_mask)
    final_mask_path = out_path.with_suffix(".final_mask.full.npz")
    save_mask_npz(final_mask_path, current_mask)
    result["greedy"]["final"] = final
    result["greedy"]["accepted_group_ids"] = [row["id"] for row in accepted]
    result["greedy"]["accepted_groups"] = accepted
    result["greedy"]["final_mask_npz"] = str(final_mask_path)
    write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
