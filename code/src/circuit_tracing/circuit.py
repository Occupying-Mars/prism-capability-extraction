"""
Circuit extraction and evaluation.

A *circuit* is a sparse set of MLP neurons (layer, neuron_idx) that
accounts for the model's behaviour on a task.

Evaluation follows Marks et al. (2025):
  faithfulness  — ablating the complement should preserve the metric
  completeness  — ablating the circuit itself should destroy the metric
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .ablation import MeanCache, mean_ablate_run
from .tasks import metric_input_ids


# ──────────────────────────────────────────────────────────────────────

EdgeKey = Tuple[Tuple[int, int], Tuple[int, int]]
# ((src_layer, src_neuron), (tgt_layer, tgt_neuron))


@dataclass
class Circuit:
    """A sparse set of MLP neurons (and optionally edges) that form a circuit."""

    nodes: Set[Tuple[int, int]] = field(default_factory=set)
    # each node is (layer_idx, neuron_idx)

    edges: Set[EdgeKey] = field(default_factory=set)
    # each edge is ((src_layer, src_neuron), (tgt_layer, tgt_neuron))

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def layers(self) -> Set[int]:
        return {l for l, _ in self.nodes}

    def neurons_in_layer(self, layer: int) -> List[int]:
        return sorted(n for l, n in self.nodes if l == layer)

    def incoming_edges(self, node: Tuple[int, int]) -> Set[EdgeKey]:
        return {e for e in self.edges if e[1] == node}

    def outgoing_edges(self, node: Tuple[int, int]) -> Set[EdgeKey]:
        return {e for e in self.edges if e[0] == node}

    def __repr__(self) -> str:
        parts = f"Circuit(size={self.size}, layers={sorted(self.layers())}"
        if self.edges:
            parts += f", edges={self.n_edges}"
        return parts + ")"


# ──────────────────────────────────────────────────────────────────────

def extract_circuit(
    attributions: Dict[int, torch.Tensor],
    k: int,
) -> Circuit:
    """
    Build a circuit from the top-k neurons by absolute attribution.

    Parameters
    ----------
    attributions : {layer: [d_ffn]}   (position-aggregated scores)
    k            : number of neurons to include

    Returns
    -------
    Circuit
    """
    # flatten all (layer, neuron) scores into one list
    entries: List[Tuple[float, int, int]] = []
    for layer_idx, scores in attributions.items():
        for neuron_idx in range(scores.shape[0]):
            entries.append((scores[neuron_idx].item(), layer_idx, neuron_idx))

    # sort descending by absolute score and take top-k
    entries.sort(key=lambda x: abs(x[0]), reverse=True)
    top = entries[:k]

    return Circuit(nodes={(layer, neuron) for _, layer, neuron in top})


def extract_circuit_with_edges(
    node_attributions: Dict[int, torch.Tensor],
    edge_scores: Dict[EdgeKey, float],
    k_nodes: int = 500,
    k_edges: int = 2000,
) -> Circuit:
    """
    Build a circuit with edge pruning (paper §5.4).

    1. Start with top *k_nodes* neurons by absolute node attribution.
    2. Keep top *k_edges* edges (by abs score) between those neurons.
    3. Prune neurons that have no remaining edges, except those in the
       first or last active layer.
    """
    # step 1: top nodes
    node_entries: List[Tuple[float, int, int]] = []
    for layer_idx, scores in node_attributions.items():
        for nidx in range(scores.shape[0]):
            node_entries.append((scores[nidx].abs().item(), layer_idx, nidx))
    node_entries.sort(reverse=True)
    candidate_nodes = {(l, n) for _, l, n in node_entries[:k_nodes]}

    # step 2: filter edges to candidate nodes, take top k_edges
    valid_edges = [
        (abs(score), score, edge)
        for edge, score in edge_scores.items()
        if edge[0] in candidate_nodes and edge[1] in candidate_nodes
    ]
    valid_edges.sort(reverse=True)
    top_edges = {edge for _, _, edge in valid_edges[:k_edges]}

    # step 3: prune isolated neurons (keep first/last layer)
    if candidate_nodes:
        all_layers = {l for l, _ in candidate_nodes}
        min_layer = min(all_layers)
        max_layer = max(all_layers)
    else:
        min_layer = max_layer = 0

    connected_nodes = set()
    for (src, tgt) in top_edges:
        connected_nodes.add(src)
        connected_nodes.add(tgt)

    # keep boundary-layer neurons even without edges
    final_nodes = set()
    for node in candidate_nodes:
        layer = node[0]
        if layer == min_layer or layer == max_layer or node in connected_nodes:
            final_nodes.add(node)

    return Circuit(nodes=final_nodes, edges=top_edges)


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_circuit(
    model,
    circuit: Circuit,
    mean_cache: MeanCache,
    examples,
    task,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute faithfulness and completeness of a circuit.

    faithfulness = E[m(C,x) - m(empty,x)] / E[m(M,x) - m(empty,x)]
    completeness = E[m(C_bar,x) - m(empty,x)] / E[m(M,x) - m(empty,x)]

    where
      m(M,x)     = metric on full model
      m(C,x)     = metric with complement ablated  (keep circuit)
      m(C_bar,x) = metric with circuit ablated      (keep complement)
      m(empty,x) = metric with everything ablated

    A perfect circuit has faithfulness ≈ 1 and completeness ≈ 0.
    """
    m_full_sum  = 0.0
    m_circ_sum  = 0.0
    m_comp_sum  = 0.0
    m_empty_sum = 0.0
    n = 0

    n_layers = model.config.num_hidden_layers

    # "all neurons" circuit for full ablation baseline
    all_neurons = set()
    for layer_idx in range(n_layers):
        for neuron_idx in range(model.config.intermediate_size):
            all_neurons.add((layer_idx, neuron_idx))
    all_circuit = Circuit(nodes=all_neurons)

    for ex in examples:
        ids = metric_input_ids(task, ex).to(device)
        metric_fn = task.make_metric_fn(ex)

        # full model (no ablation)
        with torch.no_grad():
            m_full = metric_fn(model(ids).logits).item()

        # circuit kept, complement ablated  (faithfulness)
        m_circ = mean_ablate_run(
            model, ids, metric_fn, mean_cache,
            circuit, ablate_complement=True, device=device,
        )

        # circuit ablated, complement kept  (completeness)
        m_comp = mean_ablate_run(
            model, ids, metric_fn, mean_cache,
            circuit, ablate_complement=False, device=device,
        )

        # everything ablated  (baseline)
        m_empty = mean_ablate_run(
            model, ids, metric_fn, mean_cache,
            all_circuit, ablate_complement=False, device=device,
        )

        m_full_sum  += m_full
        m_circ_sum  += m_circ
        m_comp_sum  += m_comp
        m_empty_sum += m_empty
        n += 1

    eps = 1e-8
    denom = (m_full_sum - m_empty_sum) / max(n, 1)

    faithfulness = ((m_circ_sum - m_empty_sum) / max(n, 1)) / (denom + eps)
    completeness = ((m_comp_sum - m_empty_sum) / max(n, 1)) / (denom + eps)

    return {
        "faithfulness": faithfulness,
        "completeness": completeness,
        "m_full":  m_full_sum  / max(n, 1),
        "m_circuit": m_circ_sum / max(n, 1),
        "m_complement": m_comp_sum / max(n, 1),
        "m_empty": m_empty_sum / max(n, 1),
        "circuit_size": circuit.size,
        "n_examples": n,
    }
