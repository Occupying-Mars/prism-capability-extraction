#!/usr/bin/env python3
"""
Trace a single unified circuit for the full 2-digit addition task.

Unlike the per-property pipeline (run.py), this finds ONE circuit
that captures the model's entire ability to do 2-digit addition.

Usage
-----
    python -m src.circuit_tracing.run_full_addition
    python -m src.circuit_tracing.run_full_addition --circuit-size 200 --n-attr 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .relp import ReLPAttributor
from .circuit import Circuit, extract_circuit, extract_circuit_with_edges, evaluate_circuit
from .tasks import FullAdditionTask, metric_input_ids
from .ablation import MeanCache
from .visualize import (
    build_dashboard, save_dashboard,
    plot_ablation_sweep, plot_top_neurons, plot_example_behaviour,
    plot_edge_sankey, plot_edge_layer_heatmap, plot_layer_attributions,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unified 2-digit addition circuit")
    p.add_argument("--model",        default="Qwen/Qwen3-1.7B")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--dtype",        default="float32",
                   choices=["float32", "float16", "bfloat16"])
    # dataset
    p.add_argument("--n-examples",   type=int, default=200,
                   help="Total examples to generate (ignored if --dataset given)")
    p.add_argument("--dataset",      default=None,
                   help="Optional JSON of pre-curated {a, b, answer} examples "
                        "(e.g. data/correct_addition_dataset.json). "
                        "Sum ≤ 99 entries are used.")
    p.add_argument("--seed",         type=int, default=42)
    # circuit
    p.add_argument("--circuit-size", type=int, default=200)
    p.add_argument("--sweep-ks",     nargs="*", type=int,
                   default=[10, 20, 50, 100, 200, 500])
    p.add_argument("--n-attr",       type=int, default=20,
                   help="Examples for attribution")
    p.add_argument("--n-calib",      type=int, default=30,
                   help="Examples for mean cache")
    p.add_argument("--n-eval",       type=int, default=15,
                   help="Examples for evaluation")
    # edges
    p.add_argument("--edge-k-nodes", type=int, default=100)
    p.add_argument("--edge-k-edges", type=int, default=500)
    p.add_argument("--n-edge-attr",  type=int, default=5)
    p.add_argument("--no-edges",     action="store_true")
    # output
    p.add_argument("--name",         default=None,
                   help="Run name — outputs go to results/<name>/")
    p.add_argument("--dashboard",    default=None)
    p.add_argument("--output",       default=None)
    args = p.parse_args(argv)

    # resolve output paths from run name
    if args.name:
        run_dir = f"results/{args.name}"
        if args.dashboard is None:
            args.dashboard = f"{run_dir}/dashboard.html"
        if args.output is None:
            args.output = f"{run_dir}/results.json"
    else:
        if args.dashboard is None:
            args.dashboard = "results/full_addition_dashboard.html"
        if args.output is None:
            args.output = "results/full_addition_results.json"

    return args


def main(argv=None):
    args = parse_args(argv)
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype  = dtype_map[args.dtype]
    device = args.device
    compute_edges = not args.no_edges
    n_steps = 7 if compute_edges else 6

    # ── 1. Load model ─────────────────────────────────────────────
    print(f"[1/{n_steps}] Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    print(f"       {n_layers} layers, d_ffn={d_ffn}")

    # ── 2. Dataset ────────────────────────────────────────────────
    if args.dataset:
        print(f"[2/{n_steps}] Loading curated dataset: {args.dataset}")
        with open(args.dataset) as f:
            ds = json.load(f)
        pairs = [(e["a"], e["b"]) for e in ds["examples"]]
        task = FullAdditionTask(tokenizer, seed=args.seed, pairs=pairs)
        print(f"       Loaded {len(pairs)} pairs; "
              f"{len(task.examples)} usable after sum≤99 filter + cf build")
    else:
        print(f"[2/{n_steps}] Generating {args.n_examples} addition examples ...")
        task = FullAdditionTask(tokenizer, seed=args.seed, n=args.n_examples)
        print(f"       Got {len(task.examples)} examples")
    all_ex = task.examples
    print(f"       Sample: {all_ex[0].input_text} -> {all_ex[0].target_text}"
          f"  (cf: {all_ex[0].cf_input_text} -> {all_ex[0].cf_target_text})")

    # split
    n_a = min(args.n_attr, len(all_ex) - args.n_eval)
    n_e = min(args.n_eval, len(all_ex) - n_a)
    attr_examples = all_ex[:n_a]
    eval_examples = all_ex[n_a:n_a + n_e]
    calib_examples = all_ex[:args.n_calib]

    # ── 3. Mean cache ─────────────────────────────────────────────
    print(f"[3/{n_steps}] Building mean activation cache ({len(calib_examples)} ex) ...")
    calib_ids = [metric_input_ids(task, ex).to(device) for ex in calib_examples]
    mean_cache = MeanCache.build(model, calib_ids, device=device)

    # ── 4. Attribution + sweep ────────────────────────────────────
    print(f"[4/{n_steps}] ReLP attribution ({n_a} examples) ...")
    attributor = ReLPAttributor(model, tokenizer, device=device)

    t0 = time.time()
    agg_attr = attributor.attribute_dataset(attr_examples, task)
    print(f"       Done in {time.time() - t0:.1f}s")

    # layer totals for dashboard
    layer_totals = {l: v.sum().item() for l, v in agg_attr.items()}

    # sweep
    print(f"       Ablation sweep k={args.sweep_ks} ...")
    sweep_rows = []
    for k in args.sweep_ks:
        c = extract_circuit(agg_attr, k=k)
        ev = evaluate_circuit(model, c, mean_cache, eval_examples, task, device=device)
        sweep_rows.append({
            "k": k,
            "faithfulness": ev["faithfulness"],
            "completeness": ev["completeness"],
            "m_full": ev["m_full"],
            "m_circuit": ev["m_circuit"],
            "m_complement": ev["m_complement"],
            "m_empty": ev["m_empty"],
        })
        print(f"       k={k:>4d}  faith={ev['faithfulness']:.3f}  "
              f"compl={ev['completeness']:.3f}  m_full={ev['m_full']:.3f}")

    # best circuit
    circuit = extract_circuit(agg_attr, k=args.circuit_size)
    print(f"       Final circuit: {circuit}")

    # per-example data
    print(f"       Collecting per-example data ...")
    example_rows = []
    for ex in eval_examples[:8]:
        ids = metric_input_ids(task, ex).to(device)
        metric_fn = task.make_metric_fn(ex)
        with torch.no_grad():
            m_full = metric_fn(model(ids).logits).item()
        from .ablation import mean_ablate_run
        m_circ = mean_ablate_run(model, ids, metric_fn, mean_cache,
                                 circuit, ablate_complement=True, device=device)
        m_comp = mean_ablate_run(model, ids, metric_fn, mean_cache,
                                 circuit, ablate_complement=False, device=device)
        example_rows.append({
            "input": ex.input_text,
            "answer": ex.target_text,
            "m_full": m_full,
            "m_circuit": m_circ,
            "m_complement": m_comp,
        })

    # ── 5. Edge attribution ───────────────────────────────────────
    edge_scores = {}
    if compute_edges:
        print(f"[5/{n_steps}] Edge attribution ({args.n_edge_attr} ex, "
              f"k_nodes={args.edge_k_nodes}) ...")
        t0 = time.time()
        edge_examples = attr_examples[:args.n_edge_attr]
        edge_scores = attributor.attribute_edges_dataset(
            edge_examples, task,
            node_attributions=agg_attr,
            k_nodes=args.edge_k_nodes,
        )
        print(f"       {len(edge_scores)} edges in {time.time() - t0:.1f}s")

        if edge_scores:
            edge_circuit = extract_circuit_with_edges(
                agg_attr, edge_scores,
                k_nodes=args.edge_k_nodes,
                k_edges=args.edge_k_edges,
            )
            print(f"       Edge-pruned circuit: {edge_circuit}")

    # ── 6. Dashboard ──────────────────────────────────────────────
    step = 6 if compute_edges else 5
    print(f"[{step}/{n_steps}] Building dashboard ...")

    tag = "full_addition"
    html = build_dashboard(
        sweep_data={tag: sweep_rows},
        layer_attrs={tag: layer_totals},
        circuits={tag: circuit.nodes},
        example_data={tag: example_rows},
        top_neurons_per_prop={tag: {l: v.clone() for l, v in agg_attr.items()}},
        edge_data={tag: edge_scores} if edge_scores else None,
        n_layers=n_layers,
        title=f"Full Addition Circuit — {args.model}",
    )
    save_dashboard(html, args.dashboard)

    # ── Generation eval ─────────────────────────────────────────
    step = step + 1
    n_gen_eval = min(50, len(eval_examples))
    print(f"\n[{step}/{n_steps}] Generation eval ({n_gen_eval} examples) ...")

    def _generate_greedy(input_ids, max_tokens=8):
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

    def _build_circuit_hooks():
        n_l = model.config.num_hidden_layers
        d = model.config.intermediate_size
        masks = {}
        for layer in range(n_l):
            m = torch.zeros(d, dtype=torch.bool, device=device)
            for l, n in circuit.nodes:
                if l == layer:
                    m[n] = True
            if layer in mean_cache.means:
                masks[layer] = m
        hooks = []
        for li, mask in masks.items():
            mean = mean_cache.means[li].to(device)
            ablate = ~mask
            def make_hook(m, a):
                def hook_fn(module, args):
                    act = args[0].clone()
                    act[:, :, a] = m[a]
                    return (act,) + args[1:]
                return hook_fn
            h = model.model.layers[li].mlp.down_proj.register_forward_pre_hook(
                make_hook(mean, ablate)
            )
            hooks.append(h)
        return hooks

    full_correct = 0
    circ_correct = 0
    gen_results = []

    for ex in eval_examples[:n_gen_eval]:
        ids = task.tokenize(ex.input_text).to(device)
        answer = ex.target_text

        full_pred = _generate_greedy(ids)

        hooks = _build_circuit_hooks()
        try:
            circ_pred = _generate_greedy(ids)
        finally:
            for h in hooks:
                h.remove()

        f_ok = full_pred == answer
        c_ok = circ_pred == answer
        if f_ok:
            full_correct += 1
        if c_ok:
            circ_correct += 1
        gen_results.append({
            "input": ex.input_text, "answer": answer,
            "full_pred": full_pred, "circ_pred": circ_pred,
            "full_ok": f_ok, "circ_ok": c_ok,
        })

    print(f"       Full model: {full_correct}/{n_gen_eval} = {full_correct/n_gen_eval:.1%}")
    print(f"       Circuit:    {circ_correct}/{n_gen_eval} = {circ_correct/n_gen_eval:.1%}")
    print(f"\n       Sample results:")
    for r in gen_results[:10]:
        f_tag = "ok" if r["full_ok"] else r["full_pred"] or "(empty)"
        c_tag = "ok" if r["circ_ok"] else r["circ_pred"] or "(empty)"
        print(f"         {r['input']} {r['answer']}  full={f_tag}  circuit={c_tag}")

    # save circuit checkpoint
    circuit_path = Path(args.output).with_suffix(".circuit.json")
    circuit_data = {
        "model": args.model,
        "nodes": sorted([list(n) for n in circuit.nodes]),
        "mean_cache": {str(k): v.tolist() for k, v in mean_cache.means.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(circuit_path, "w") as f:
        json.dump(circuit_data, f)
    print(f"\n       Circuit saved: {circuit_path} ({circuit_path.stat().st_size / 1024:.0f} KB)")

    # save JSON
    output = {
        "config": vars(args),
        "sweep": sweep_rows,
        "circuit_size": circuit.size,
        "circuit_nodes": sorted([list(n) for n in circuit.nodes]),
        "circuit_layers": sorted(circuit.layers()),
        "n_edges": len(edge_scores),
        "examples": example_rows,
        "generation_eval": {
            "full_correct": full_correct,
            "circuit_correct": circ_correct,
            "n_eval": n_gen_eval,
            "results": gen_results,
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # summary
    print(f"\n{'='*60}")
    print(f"  Circuit: {circuit.size} neurons across {len(circuit.layers())} layers")
    if edge_scores:
        print(f"  Edges: {len(edge_scores)} computed")
    best = next((r for r in sweep_rows if r["k"] == args.circuit_size), sweep_rows[-1])
    print(f"  Faithfulness: {best['faithfulness']:.3f}")
    print(f"  Completeness: {best['completeness']:.3f}")
    print(f"  m_full (avg): {best['m_full']:.3f}")
    print(f"  Generation:   full={full_correct}/{n_gen_eval} ({full_correct/n_gen_eval:.1%})  "
          f"circuit={circ_correct}/{n_gen_eval} ({circ_correct/n_gen_eval:.1%})")
    print(f"{'='*60}")
    print(f"\n  Dashboard: file://{Path(args.dashboard).resolve()}")
    print(f"  JSON:      {args.output}\n")

    return output


if __name__ == "__main__":
    main()
