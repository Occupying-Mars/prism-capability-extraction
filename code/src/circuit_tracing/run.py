#!/usr/bin/env python3
"""
End-to-end circuit tracing pipeline with W&B logging and HTML dashboard.

Loads a HuggingFace causal LM, generates a structured dataset,
runs RelP attribution per property, runs ablation sweeps,
and produces an interactive HTML dashboard for debugging circuits.

Usage
-----
    python -m src.circuit_tracing.run
    python -m src.circuit_tracing.run --model Qwen/Qwen3-1.7B --properties digit_2 digit_5
    python -m src.circuit_tracing.run --all-properties --circuit-size 100
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .relp import ReLPAttributor
from .circuit import Circuit, extract_circuit, extract_circuit_with_edges, evaluate_circuit
from .tasks import StructuredAdditionTask
from .ablation import MeanCache, mean_ablate_run
from .dataset import generate_dataset, save_dataset, load_dataset
from .sweep import ablation_sweep
from .visualize import build_dashboard, save_dashboard


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Neuron-basis circuit tracing")
    # model
    p.add_argument("--model",        default="Qwen/Qwen3-1.7B")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--dtype",        default="float32",
                   choices=["float32", "float16", "bfloat16"])
    # dataset
    p.add_argument("--dataset",      default=None,
                   help="Path to pre-generated dataset JSON")
    p.add_argument("--dataset-output", default="data/addition_dataset.json")
    p.add_argument("--max-per-subprop", type=int, default=80)
    # properties to trace
    p.add_argument("--properties",   nargs="*", default=None,
                   help="e.g. digit_2 digit_5 carry_behavior")
    p.add_argument("--subproperties", nargs="*", default=None,
                   help="e.g. a_tens sum_ones carry")
    p.add_argument("--all-properties", action="store_true")
    # circuit
    p.add_argument("--circuit-size", type=int, default=50)
    p.add_argument("--sweep-ks",     nargs="*", type=int,
                   default=[5, 10, 20, 50, 100, 200],
                   help="k values for ablation sweep")
    p.add_argument("--n-attr",       type=int, default=20)
    p.add_argument("--n-calib",      type=int, default=30)
    p.add_argument("--n-eval",       type=int, default=10)
    # edges
    p.add_argument("--edge-k-nodes", type=int, default=500,
                   help="Top neurons to consider for edge computation")
    p.add_argument("--edge-k-edges", type=int, default=2000,
                   help="Top edges to keep after pruning")
    p.add_argument("--n-edge-attr",  type=int, default=5,
                   help="Examples for edge attribution (expensive)")
    p.add_argument("--no-edges",     action="store_true",
                   help="Skip edge attribution (faster)")
    # output
    p.add_argument("--dashboard",    default="results/dashboard.html")
    p.add_argument("--output",       default=None,
                   help="path to save JSON results")
    # wandb
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-project", default="circuit-tracing")
    p.add_argument("--no-wandb",     action="store_true")
    # misc
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────
# Per-example evaluation (for dashboard)
# ──────────────────────────────────────────────────────────────────────

def _collect_example_data(
    model, circuit, mean_cache, examples, task, device,
) -> list[dict]:
    """Run each example through full/circuit/ablated and collect metrics."""
    rows = []
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    all_nodes = {(l, n) for l in range(n_layers) for n in range(d_ffn)}
    all_circuit = Circuit(nodes=all_nodes)

    for ex in examples:
        ids = task.tokenize(ex.input_text).to(device)
        metric_fn = task.make_metric_fn(ex)

        with torch.no_grad():
            m_full = metric_fn(model(ids).logits).item()

        m_circ = mean_ablate_run(
            model, ids, metric_fn, mean_cache,
            circuit, ablate_complement=True, device=device,
        )
        m_comp = mean_ablate_run(
            model, ids, metric_fn, mean_cache,
            circuit, ablate_complement=False, device=device,
        )

        rows.append({
            "input": ex.input_text,
            "answer": ex.target_text,
            "target_digit": ex.target_digit,
            "cf_digit": ex.cf_digit,
            "m_full": m_full,
            "m_circuit": m_circ,
            "m_complement": m_comp,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype  = dtype_map[args.dtype]
    device = args.device
    use_wandb = HAS_WANDB and not args.no_wandb
    compute_edges = not args.no_edges
    n_steps = 8 if compute_edges else 7

    # ── W&B ───────────────────────────────────────────────────────
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )
        print("[wandb] Run initialised")

    # ── 1. Load model ─────────────────────────────────────────────
    step = 1
    print(f"[{step}/{n_steps}] Loading {args.model} ({args.dtype}) on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_ffn    = model.config.intermediate_size
    print(f"       {n_layers} layers, d_ffn={d_ffn}, "
          f"total MLP neurons = {n_layers * d_ffn:,}")

    # ── 2. Dataset ────────────────────────────────────────────────
    print(f"[2/{n_steps}] Preparing dataset ...")
    if args.dataset and Path(args.dataset).exists():
        ds = load_dataset(args.dataset)
        print(f"       Loaded from {args.dataset}")
    else:
        ds = generate_dataset(
            max_per_subprop=args.max_per_subprop, seed=args.seed,
        )
        save_dataset(ds, args.dataset_output)

    all_props = list(ds["properties"].keys())
    if args.all_properties:
        selected_props = all_props
    elif args.properties:
        selected_props = args.properties
    else:
        selected_props = ["digit_2", "carry_behavior"]
    print(f"       Properties: {selected_props}")

    # ── 3. Mean cache ─────────────────────────────────────────────
    print(f"[3/{n_steps}] Computing mean activation cache ...")
    calib_texts = set()
    for pname in all_props:
        for spdata in ds["properties"][pname]["subproperties"].values():
            for ex in spdata["examples"][:5]:
                calib_texts.add(ex["input_text"])
            if len(calib_texts) >= args.n_calib:
                break

    calib_ids = [
        tokenizer(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        for t in list(calib_texts)[: args.n_calib]
    ]
    mean_cache = MeanCache.build(model, calib_ids, device=device)
    print(f"       {len(calib_ids)} examples, {len(mean_cache.means)} layers")

    # ── 4. Per-property: attribution + sweep + example data ───────
    attributor = ReLPAttributor(model, tokenizer, device=device)

    # dashboard data collectors
    dash_sweep:      dict[str, list[dict]]       = {}
    dash_layer_attr: dict[str, dict[int, float]]  = {}
    dash_circuits:   dict[str, set]               = {}
    dash_examples:   dict[str, list[dict]]        = {}
    dash_top_neurons: dict[str, dict]             = {}
    dash_edges:      dict[str, dict]              = {}
    all_results:     dict[str, dict]              = {}
    # store raw node attributions for edge computation
    all_node_attrs:  dict[str, dict]              = {}
    all_tasks_attr:  dict[str, object]            = {}

    total_subprops = sum(
        len(ds["properties"][p]["subproperties"])
        for p in selected_props if p in ds["properties"]
    )
    print(f"\n[4/{n_steps}] Attribution + ablation sweep ({total_subprops} subproperties) ...")

    for pname in selected_props:
        if pname not in ds["properties"]:
            print(f"  [!] '{pname}' not in dataset, skipping")
            continue

        pdata = ds["properties"][pname]
        for spname, spdata in pdata["subproperties"].items():
            if args.subproperties and spname not in args.subproperties:
                continue

            tag = f"{pname}/{spname}"
            raw_examples = spdata["examples"]
            n_avail = len(raw_examples)

            n_a = min(args.n_attr, n_avail - args.n_eval)
            n_e = min(args.n_eval, n_avail - n_a)
            if n_a < 3 or n_e < 2:
                print(f"  [{tag}] skipping — only {n_avail} examples")
                continue

            attr_raw = raw_examples[:n_a]
            eval_raw = raw_examples[n_a : n_a + n_e]

            task_attr = StructuredAdditionTask(tokenizer, attr_raw)
            task_eval = StructuredAdditionTask(tokenizer, eval_raw)

            # ── attribution ──
            print(f"  [{tag}] attributing ({n_a} ex) ...", end=" ", flush=True)
            t0 = time.time()
            agg_attr = attributor.attribute_dataset(task_attr.examples, task_attr)
            print(f"{time.time() - t0:.1f}s")

            # store for dashboard and edge computation
            layer_totals = {l: v.sum().item() for l, v in agg_attr.items()}
            dash_layer_attr[tag] = layer_totals
            dash_top_neurons[tag] = {l: v.clone() for l, v in agg_attr.items()}
            all_node_attrs[tag] = agg_attr
            all_tasks_attr[tag] = (attr_raw, task_attr)

            # ── circuit extraction (at requested k) ──
            circuit = extract_circuit(agg_attr, k=args.circuit_size)
            dash_circuits[tag] = circuit.nodes
            print(f"  [{tag}] circuit: {circuit}")

            # ── ablation sweep ──
            print(f"  [{tag}] ablation sweep k={args.sweep_ks} ...")
            sweep_rows = []
            for k in args.sweep_ks:
                c = extract_circuit(agg_attr, k=k)
                ev = evaluate_circuit(
                    model, c, mean_cache,
                    task_eval.examples, task_eval,
                    device=device,
                )
                sweep_rows.append({
                    "k": k,
                    "faithfulness": ev["faithfulness"],
                    "completeness": ev["completeness"],
                    "m_full": ev["m_full"],
                    "m_circuit": ev["m_circuit"],
                    "m_complement": ev["m_complement"],
                    "m_empty": ev["m_empty"],
                })
                print(f"    k={k:>4d}  faith={ev['faithfulness']:.3f}  "
                      f"compl={ev['completeness']:.3f}  m_full={ev['m_full']:.3f}")
            dash_sweep[tag] = sweep_rows

            # ── per-example data (for best circuit size) ──
            print(f"  [{tag}] collecting per-example data ...", end=" ", flush=True)
            ex_data = _collect_example_data(
                model, circuit, mean_cache,
                task_eval.examples[:5], task_eval, device,
            )
            dash_examples[tag] = ex_data
            print(f"{len(ex_data)} examples")

            # ── wandb: log sweep as line plot ──
            if use_wandb:
                sweep_table = wandb.Table(
                    columns=["k", "faithfulness", "completeness", "m_full"],
                    data=[[r["k"], r["faithfulness"], r["completeness"], r["m_full"]]
                          for r in sweep_rows],
                )
                wandb.log({
                    f"sweep/{tag}": wandb.plot.line_series(
                        xs=[r["k"] for r in sweep_rows],
                        ys=[
                            [r["faithfulness"] for r in sweep_rows],
                            [r["completeness"] for r in sweep_rows],
                        ],
                        keys=["faithfulness", "completeness"],
                        title=f"Ablation Sweep — {tag}",
                        xname="circuit size (k)",
                    ),
                    f"sweep_table/{tag}": sweep_table,
                })

            # store summary for final table
            best_row = next(
                (r for r in sweep_rows if r["k"] == args.circuit_size),
                sweep_rows[-1],
            )
            all_results[tag] = {**best_row, "property": pname, "subproperty": spname}

    # ── 5. Edge attribution ─────────────────────────────────────────
    if compute_edges:
        print(f"\n[5/{n_steps}] Edge attribution ...")
        for tag in list(all_node_attrs.keys()):
            agg_attr = all_node_attrs[tag]
            attr_raw, task_attr = all_tasks_attr[tag]

            # use fewer examples for edges (expensive)
            n_edge = min(args.n_edge_attr, len(task_attr.examples))
            edge_examples = task_attr.examples[:n_edge]

            print(f"  [{tag}] computing edges (k_nodes={args.edge_k_nodes}, "
                  f"{n_edge} ex) ...", end=" ", flush=True)
            t0 = time.time()
            edge_scores = attributor.attribute_edges_dataset(
                edge_examples, task_attr,
                node_attributions=agg_attr,
                k_nodes=args.edge_k_nodes,
            )
            print(f"{time.time() - t0:.1f}s, {len(edge_scores)} edges")

            dash_edges[tag] = edge_scores

            # extract edge-pruned circuit
            edge_circuit = extract_circuit_with_edges(
                agg_attr, edge_scores,
                k_nodes=args.edge_k_nodes,
                k_edges=args.edge_k_edges,
            )
            print(f"  [{tag}] edge-pruned circuit: {edge_circuit}")

    # ── 6. Cross-property overlap ─────────────────────────────────
    print(f"\n[{6 if compute_edges else 5}/{n_steps}] Cross-property circuit overlap ...")
    tags = list(dash_circuits.keys())
    for i, t1 in enumerate(tags):
        for j, t2 in enumerate(tags):
            if i >= j:
                continue
            shared = dash_circuits[t1] & dash_circuits[t2]
            union  = dash_circuits[t1] | dash_circuits[t2]
            jacc = len(shared) / len(union) if union else 0.0
            if len(shared) > 0:
                print(f"  {t1} ∩ {t2}: {len(shared)} shared (J={jacc:.2f})")

    # ── 7. Build dashboard ────────────────────────────────────────
    print(f"\n[{7 if compute_edges else 6}/{n_steps}] Building HTML dashboard ...")
    html = build_dashboard(
        sweep_data=dash_sweep,
        layer_attrs=dash_layer_attr,
        circuits=dash_circuits,
        example_data=dash_examples,
        top_neurons_per_prop=dash_top_neurons,
        edge_data=dash_edges if compute_edges else None,
        n_layers=n_layers,
        title=f"Circuit Tracing — {args.model}",
    )
    save_dashboard(html, args.dashboard)

    if use_wandb:
        wandb.log({"dashboard": wandb.Html(html)})

    # ── 8. Summary ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'Property':<30} {'k':>5} {'Faith':>8} {'Compl':>8} {'m_full':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for tag, r in all_results.items():
        print(f"  {tag:<30} {r['k']:>5d} {r['faithfulness']:>8.3f} "
              f"{r['completeness']:>8.3f} {r['m_full']:>8.3f}")
    print(f"{'='*65}")
    if compute_edges:
        print(f"  Edge computation: ON (k_nodes={args.edge_k_nodes}, k_edges={args.edge_k_edges})")
    print(f"\n  Dashboard: {args.dashboard}")
    print(f"  Open in browser: file://{Path(args.dashboard).resolve()}\n")

    # save JSON
    output = {
        "config": vars(args),
        "per_property": all_results,
        "sweeps": {k: v for k, v in dash_sweep.items()},
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  JSON results: {args.output}")

    if use_wandb:
        wandb.finish()
        print("[wandb] Run finished")

    return output


if __name__ == "__main__":
    main()
