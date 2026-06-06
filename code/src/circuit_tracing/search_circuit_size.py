#!/usr/bin/env python3
"""
Top-down circuit size search.

Starts from a large circuit (all neurons or --start-k) and halves down
until generation accuracy drops below a threshold. Saves the smallest
circuit that still passes.

Strategy: binary-style halving
  k_max -> k_max/2 -> k_max/4 -> ... -> stop when acc < threshold
  Then backtrack one step and do a finer linear search in that range.

Usage:
    python -m src.circuit_tracing.search_circuit_size --name search-v1
    python -m src.circuit_tracing.search_circuit_size --name search-v1 --start-k 5000 --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .relp import ReLPAttributor
from .circuit import extract_circuit
from .tasks import FullAdditionTask
from .ablation import MeanCache


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Top-down circuit size search")
    p.add_argument("--model",       default="Qwen/Qwen3-1.7B")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--dtype",       default="float32",
                   choices=["float32", "float16", "bfloat16"])
    # dataset
    p.add_argument("--n-examples",  type=int, default=500,
                   help="Total examples to generate")
    p.add_argument("--n-attr",      type=int, default=40,
                   help="Examples for attribution")
    p.add_argument("--n-calib",     type=int, default=50,
                   help="Examples for mean cache")
    p.add_argument("--n-eval",      type=int, default=100,
                   help="Examples for generation eval at each k")
    p.add_argument("--seed",        type=int, default=42)
    # search
    p.add_argument("--start-k",     type=int, default=None,
                   help="Starting k (default: total neurons)")
    p.add_argument("--min-k",       type=int, default=10,
                   help="Smallest k to try")
    p.add_argument("--threshold",   type=float, default=0.80,
                   help="Stop when accuracy drops below this")
    # output
    p.add_argument("--name",        default="search",
                   help="Run name — outputs go to results/<name>/")
    return p.parse_args(argv)


def build_circuit_hooks(model, circuit_nodes, mean_cache, device):
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size

    masks = {}
    for layer in range(n_layers):
        m = torch.zeros(d_ffn, dtype=torch.bool, device=device)
        for l, n in circuit_nodes:
            if l == layer:
                m[n] = True
        if layer in mean_cache.means:
            masks[layer] = m

    hooks = []
    for layer_idx, mask in masks.items():
        mean = mean_cache.means[layer_idx].to(device)
        ablate = ~mask

        def make_hook(m, a):
            def hook_fn(module, args):
                act = args[0].clone()
                act[:, :, a] = m[a]
                return (act,) + args[1:]
            return hook_fn

        h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            make_hook(mean, ablate)
        )
        hooks.append(h)
    return hooks


def generate_greedy(model, tokenizer, input_ids, device, max_tokens=8):
    ids = input_ids.to(device)
    digits = []
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        next_id = logits.argmax().item()
        tok = tokenizer.decode(next_id).strip()
        if tok.isdigit():
            digits.append(tok)
        elif digits:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return "".join(digits)


def eval_at_k(model, tokenizer, agg_attr, mean_cache, eval_examples, task,
              k, device):
    """Run generation eval at circuit size k, return (accuracy, details)."""
    circuit = extract_circuit(agg_attr, k=k)
    correct = 0
    results = []

    for ex in eval_examples:
        ids = task.tokenize(ex.input_text).to(device)
        answer = ex.target_text

        hooks = build_circuit_hooks(model, circuit.nodes, mean_cache, device)
        try:
            pred = generate_greedy(model, tokenizer, ids, device)
        finally:
            for h in hooks:
                h.remove()

        ok = pred == answer
        if ok:
            correct += 1
        results.append({
            "input": ex.input_text, "answer": answer,
            "predicted": pred, "correct": ok,
        })

    acc = correct / len(eval_examples)
    return acc, correct, results, circuit


def main(argv=None):
    args = parse_args(argv)
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device

    out_dir = Path(f"results/{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────
    print(f"[1/4] Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    total_neurons = n_layers * d_ffn
    print(f"       {n_layers} layers, d_ffn={d_ffn}, total={total_neurons:,}")

    start_k = args.start_k or total_neurons

    # ── 2. Dataset ────────────────────────────────────────────────
    print(f"[2/4] Generating {args.n_examples} examples ...")
    task = FullAdditionTask(tokenizer, seed=args.seed, n=args.n_examples)
    all_ex = task.examples

    n_a = min(args.n_attr, len(all_ex) - args.n_eval)
    n_e = min(args.n_eval, len(all_ex) - n_a)
    attr_examples = all_ex[:n_a]
    eval_examples = all_ex[n_a:n_a + n_e]
    calib_examples = all_ex[:args.n_calib]

    print(f"       attr={n_a}, eval={n_e}, calib={len(calib_examples)}")

    # ── 3. Mean cache + attribution ──────────────────────────────
    print(f"[3/4] Mean cache ({len(calib_examples)} ex) + ReLP attribution ({n_a} ex) ...")
    calib_ids = [task.tokenize(ex.input_text).to(device) for ex in calib_examples]
    mean_cache = MeanCache.build(model, calib_ids, device=device)

    attributor = ReLPAttributor(model, tokenizer, device=device)
    t0 = time.time()
    agg_attr = attributor.attribute_dataset(attr_examples, task)
    print(f"       Attribution done in {time.time() - t0:.1f}s")

    # ── 4. Eval full model baseline ──────────────────────────────
    print(f"\n[4/4] Full model baseline ({n_e} examples) ...")
    full_correct = 0
    for ex in eval_examples:
        ids = task.tokenize(ex.input_text).to(device)
        pred = generate_greedy(model, tokenizer, ids, device)
        if pred == ex.target_text:
            full_correct += 1
    full_acc = full_correct / n_e
    print(f"       Full model: {full_correct}/{n_e} = {full_acc:.1%}")

    if full_acc < args.threshold:
        print(f"\n  WARNING: Full model accuracy ({full_acc:.1%}) is already below "
              f"threshold ({args.threshold:.0%}).")
        print(f"  The model can't do this task well enough to trace a circuit.")
        return

    # ── 5. Phase 1: halving search ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Halving search (start_k={start_k:,}, threshold={args.threshold:.0%})")
    print(f"{'='*60}\n")

    sweep_log = []
    k = start_k
    last_good_k = k
    first_bad_k = None

    while k >= args.min_k:
        print(f"  k={k:>6,} ...", end=" ", flush=True)
        t0 = time.time()
        acc, correct, results, circuit = eval_at_k(
            model, tokenizer, agg_attr, mean_cache,
            eval_examples, task, k, device,
        )
        elapsed = time.time() - t0
        print(f"{correct}/{n_e} = {acc:.1%}  ({elapsed:.1f}s)  "
              f"layers={len(circuit.layers())}")

        sweep_log.append({
            "k": k, "accuracy": acc, "correct": correct,
            "n_eval": n_e, "n_layers": len(circuit.layers()),
            "time_s": round(elapsed, 1),
        })

        if acc >= args.threshold:
            last_good_k = k
            k = k // 2
        else:
            first_bad_k = k
            print(f"\n  Dropped below {args.threshold:.0%} at k={k:,}")
            break
    else:
        print(f"\n  Reached min_k={args.min_k} and still above threshold!")
        first_bad_k = args.min_k // 2

    # ── 6. Phase 2: fine-grained linear search ───────────────────
    if first_bad_k is not None and last_good_k > first_bad_k:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: Fine search between k={first_bad_k:,} and k={last_good_k:,}")
        print(f"{'='*60}\n")

        # split the range into ~8 steps
        gap = last_good_k - first_bad_k
        step_size = max(gap // 8, 1)
        fine_ks = list(range(first_bad_k + step_size, last_good_k, step_size))

        for k in fine_ks:
            print(f"  k={k:>6,} ...", end=" ", flush=True)
            t0 = time.time()
            acc, correct, results, circuit = eval_at_k(
                model, tokenizer, agg_attr, mean_cache,
                eval_examples, task, k, device,
            )
            elapsed = time.time() - t0
            print(f"{correct}/{n_e} = {acc:.1%}  ({elapsed:.1f}s)")

            sweep_log.append({
                "k": k, "accuracy": acc, "correct": correct,
                "n_eval": n_e, "n_layers": len(circuit.layers()),
                "time_s": round(elapsed, 1),
            })

            if acc >= args.threshold:
                last_good_k = min(last_good_k, k)

    # sort sweep log by k for readability
    sweep_log.sort(key=lambda r: r["k"])

    # ── 7. Save best circuit ─────────────────────────────────────
    best_circuit = extract_circuit(agg_attr, k=last_good_k)

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"  Smallest circuit at >={args.threshold:.0%} accuracy: k={last_good_k:,}")
    print(f"  Neurons: {best_circuit.size}, Layers: {sorted(best_circuit.layers())}")
    print(f"  Full model: {full_acc:.1%}")
    print(f"{'='*60}")

    # print full sweep table
    print(f"\n  {'k':>8}  {'acc':>6}  {'correct':>7}  {'layers':>6}  {'time':>6}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}")
    for r in sweep_log:
        marker = " <-- best" if r["k"] == last_good_k else ""
        print(f"  {r['k']:>8,}  {r['accuracy']:>5.1%}  {r['correct']:>4}/{r['n_eval']}  "
              f"{r['n_layers']:>6}  {r['time_s']:>5.1f}s{marker}")

    # save circuit checkpoint
    circuit_path = out_dir / "best.circuit.json"
    circuit_data = {
        "model": args.model,
        "k": last_good_k,
        "accuracy": next(r["accuracy"] for r in sweep_log if r["k"] == last_good_k),
        "full_model_accuracy": full_acc,
        "threshold": args.threshold,
        "nodes": sorted([list(n) for n in best_circuit.nodes]),
        "mean_cache": {str(k): v.tolist() for k, v in mean_cache.means.items()},
    }
    with open(circuit_path, "w") as f:
        json.dump(circuit_data, f)
    print(f"\n  Circuit saved: {circuit_path}")

    # save full search results
    results_path = out_dir / "search_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "full_model_accuracy": full_acc,
            "best_k": last_good_k,
            "sweep": sweep_log,
        }, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")


if __name__ == "__main__":
    main()
