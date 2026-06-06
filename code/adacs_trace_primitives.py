#!/usr/bin/env python3
"""Trace ADACS primitive contrast pairs with existing position probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from carry_into_thousands import run_attribution, save_union
from position_interface_decomposition import (
    build_position_attr_records,
    build_position_eval_records,
    resolve_batch_size,
    run_position_check_cf_patch,
    run_position_check_zero,
    split_pairs,
)
from src.circuit_tracing.profiling import add_profile_args, make_profiler
from src.circuit_tracing.relp import ReLPAttributor
from union_attribution import TRACKED_MODES, _tensor_union_size, pick_device


PRIMITIVE_POSITION = {
    "ones_digit": "ones",
    "tens_digit": "tens",
    "hundreds_digit": "hundreds",
    "carry_ones_to_tens": "tens",
    "carry_tens_to_hundreds": "hundreds",
    "carry_into_thousands": "overflow",
    "answer_length": "overflow",
}


POSITION_OFFSETS = {
    "overflow": 4,
    "hundreds": 3,
    "tens": 2,
    "ones": 1,
}


def digit_at_position(answer: str, position: str) -> str | None:
    offset = POSITION_OFFSETS[position]
    if len(answer) < offset:
        return None
    return answer[len(answer) - offset]


def convert_pair(pair: dict, primitive: str) -> dict | None:
    position = PRIMITIVE_POSITION[primitive]
    target = dict(pair["target"])
    cf = dict(pair["cf"])

    if primitive == "carry_tens_to_hundreds":
        target_answer = str(target["answer"])
        cf_answer = str(cf["answer"])
        if {len(target_answer), len(cf_answer)} == {2, 3}:
            if len(target_answer) == 2:
                target, cf = cf, target
                target_answer, cf_answer = cf_answer, target_answer
            return {
                "position": position,
                "target": target,
                "cf": cf,
                "target_digit": target_answer[0],
                "cf_digit": cf_answer[0],
                "match": pair.get("match", {}),
                "adacs_score": pair.get("score"),
                "adacs_target_value": pair.get("target_value"),
                "adacs_cf_value": pair.get("cf_value"),
            }

    if position == "overflow" and len(str(target["answer"])) < 4:
        if len(str(cf["answer"])) < 4:
            return None
        target, cf = cf, target

    if position == "overflow":
        target_answer = str(target["answer"])
        cf_answer = str(cf["answer"])
        if len(target_answer) < 4 or len(cf_answer) < 3:
            return None
        target_digit = target_answer[0]
        cf_digit = cf_answer[0]
    else:
        target_digit = digit_at_position(str(target["answer"]), position)
        cf_digit = digit_at_position(str(cf["answer"]), position)
    if target_digit is None or cf_digit is None or target_digit == cf_digit:
        return None

    return {
        "position": position,
        "target": target,
        "cf": cf,
        "target_digit": target_digit,
        "cf_digit": cf_digit,
        "match": pair.get("match", {}),
        "adacs_score": pair.get("score"),
        "adacs_target_value": pair.get("target_value"),
        "adacs_cf_value": pair.get("cf_value"),
    }


def load_adacs_pairs(pair_dir: Path, primitive: str, n_pairs: int | None) -> tuple[list[dict], dict]:
    raw_path = pair_dir / f"pairs_{primitive}.json"
    payload = json.load(open(raw_path))
    converted = []
    skipped = 0
    for pair in payload["pairs"]:
        rec = convert_pair(pair, primitive)
        if rec is None:
            skipped += 1
            continue
        converted.append(rec)
        if n_pairs is not None and len(converted) >= n_pairs:
            break
    stats = {
        "source_pair_file": str(raw_path),
        "source_stats": payload.get("stats", {}),
        "converted_pairs": len(converted),
        "conversion_skipped_before_cap": skipped,
        "n_pairs_cap": n_pairs,
    }
    return converted, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace one ADACS primitive.")
    parser.add_argument("--pair-dir", default="results/adacs_milestone1_full")
    parser.add_argument("--primitive", required=True, choices=sorted(PRIMITIVE_POSITION))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-pairs", type=int, default=4000)
    parser.add_argument("--n-prune-test", type=int, default=500)
    add_profile_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_dir = Path(args.pair_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    profiler = make_profiler(args, device)

    pairs, pair_stats = load_adacs_pairs(pair_dir, args.primitive, args.n_pairs)
    attr_pairs, test_pairs = split_pairs(pairs, args.n_prune_test)
    position = PRIMITIVE_POSITION[args.primitive]
    summary = {
        "primitive": args.primitive,
        "position_probe": position,
        "pair_stats": pair_stats,
        "attr_pairs": len(attr_pairs),
        "test_pairs": len(test_pairs),
        "model": args.model,
        "dtype": args.dtype,
        "device": device,
        "batch_size": batch_size,
    }
    print(json.dumps(summary, indent=2), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    attr_records, attr_skipped = build_position_attr_records(tokenizer, attr_pairs, position)
    eval_records, eval_skipped = build_position_eval_records(tokenizer, test_pairs, position)
    summary["attr_records"] = len(attr_records)
    summary["attr_records_skipped"] = attr_skipped
    summary["eval_records"] = len(eval_records)
    summary["eval_records_skipped"] = eval_skipped
    if not attr_records:
        raise RuntimeError(f"no valid attribution records for {args.primitive}")

    print(f"[model] {args.model} device={device} dtype={args.dtype} batch={batch_size}", flush=True)
    with profiler.phase("model_load", model=args.model):
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model, dtype=dtype, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )
    for param in model.parameters():
        param.requires_grad_(True)

    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    summary["n_layers"] = n_layers
    summary["d_ffn"] = d_ffn
    summary["num_heads"] = num_heads

    attributor = ReLPAttributor(model, tokenizer, device=device)
    res = run_attribution(
        attr_records,
        model,
        attributor,
        device=device,
        n_layers=n_layers,
        d_ffn=d_ffn,
        num_heads=num_heads,
        batch_size=batch_size,
        profiler=profiler,
        label=args.primitive,
    )
    save_union(out_dir, args.primitive, res, n_layers, d_ffn, num_heads)

    pruning = {
        "probe": f"adacs_{args.primitive}_argmax_at_{position}",
        "baseline": {},
        "by_threshold": {},
    }
    if eval_records:
        full_mlp = {l: torch.ones(d_ffn, dtype=torch.bool, device=device) for l in range(n_layers)}
        full_head = {l: torch.ones(num_heads, dtype=torch.bool, device=device) for l in range(n_layers)}
        pruning["baseline"] = run_position_check_cf_patch(
            model,
            eval_records,
            mlp_keep=full_mlp,
            head_keep=full_head,
            device=device,
            num_heads=num_heads,
            batch_size=batch_size,
            label=f"{args.primitive}_baseline_cf",
        )
        for mode in TRACKED_MODES:
            mlp_keep = res["mlp_u"][mode]
            zero = run_position_check_zero(
                model,
                eval_records,
                mlp_keep=mlp_keep,
                head_keep=full_head,
                device=device,
                num_heads=num_heads,
                batch_size=batch_size,
                label=f"{args.primitive}_{mode}_zero",
            )
            cf = run_position_check_cf_patch(
                model,
                eval_records,
                mlp_keep=mlp_keep,
                head_keep=full_head,
                device=device,
                num_heads=num_heads,
                batch_size=batch_size,
                label=f"{args.primitive}_{mode}_cf",
            )
            mlp_kept = _tensor_union_size(mlp_keep)
            pruning["by_threshold"][mode] = {
                "mlp_kept": mlp_kept,
                "mlp_keep_fraction": mlp_kept / max(n_layers * d_ffn, 1),
                "heads_kept": n_layers * num_heads,
                "heads_keep_fraction": 1.0,
                "acc_zero_ablation": zero["accuracy"],
                "acc_cf_patch": cf["accuracy"],
                "n_test": cf["n"],
            }
            print(
                f"{args.primitive:>24} {mode:>12} mlp={mlp_kept:>7d} "
                f"zero={zero['accuracy']:.2%} cf={cf['accuracy']:.2%}",
                flush=True,
            )
    summary["pruning"] = pruning

    with open(out_dir / f"{args.primitive}_pruning_check.json", "w") as f:
        json.dump(pruning, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    profiler.close()
    print(f"[done] artifacts -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
