"""
Ablation sweep — vary circuit size k and measure faithfulness / completeness.

This produces the core data for Figure 1-style plots:
how sparse can a circuit be while remaining faithful?
"""

from __future__ import annotations

import time
import torch
from typing import Dict, List, Optional

from .relp import ReLPAttributor
from .circuit import Circuit, extract_circuit, evaluate_circuit
from .tasks import StructuredAdditionTask
from .ablation import MeanCache


def ablation_sweep(
    model,
    tokenizer,
    attr_examples: List[dict],
    eval_examples: List[dict],
    mean_cache: MeanCache,
    k_values: Optional[List[int]] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> List[dict]:
    """
    Run RelP attribution once, then extract circuits at multiple sizes k
    and evaluate each.

    Returns list of dicts with keys:
        k, faithfulness, completeness, m_full, m_circuit, m_complement, m_empty
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100, 200, 500]

    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    max_k = n_layers * d_ffn

    # cap k values
    k_values = [k for k in k_values if k <= max_k]

    # ── attribution ──
    task_attr = StructuredAdditionTask(tokenizer, attr_examples)
    attributor = ReLPAttributor(model, tokenizer, device=device)

    if verbose:
        print(f"  [sweep] attributing over {len(attr_examples)} examples ...", end=" ", flush=True)
    t0 = time.time()
    agg_attr = attributor.attribute_dataset(task_attr.examples, task_attr)
    if verbose:
        print(f"{time.time() - t0:.1f}s")

    # ── sweep over k ──
    task_eval = StructuredAdditionTask(tokenizer, eval_examples)
    results = []

    for k in k_values:
        circuit = extract_circuit(agg_attr, k=k)

        if verbose:
            print(f"  [sweep] k={k:>5d} ...", end=" ", flush=True)
        t0 = time.time()
        ev = evaluate_circuit(
            model, circuit, mean_cache,
            task_eval.examples, task_eval,
            device=device,
        )
        elapsed = time.time() - t0

        row = {
            "k": k,
            "faithfulness": ev["faithfulness"],
            "completeness": ev["completeness"],
            "m_full": ev["m_full"],
            "m_circuit": ev["m_circuit"],
            "m_complement": ev["m_complement"],
            "m_empty": ev["m_empty"],
        }
        results.append(row)

        if verbose:
            print(f"faith={row['faithfulness']:.3f}  compl={row['completeness']:.3f}  ({elapsed:.1f}s)")

    return results
