"""
RelP (Relevance Propagation) attribution for circuit tracing.

Implements the linearised backward pass from Table 2 of
"Language Model Circuits Are Sparse in the Neuron Basis" (Arora et al., 2026).

Linearisation rules
-------------------
* RMSNorm  :  x / freeze(sqrt(var + eps))   — freeze denominator
* SiLU     :  x * freeze(sigmoid(x))        — freeze sigmoid gate
* Gated MLP:  half-rule for the element-wise gate*up product
* Attention:  sum_k freeze(A_qk) * v_k      — freeze attn weights  (optional)

The forward pass is numerically identical to the original model;
only the backward pass differs, giving a single-pass attribution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Set, Tuple

from .tasks import metric_input_ids

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
except ImportError:
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )


# ──────────────────────────────────────────────────────────────────────
# Linearised primitives
# ──────────────────────────────────────────────────────────────────────


def _linearised_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU with frozen sigmoid:  x * detach(sigma(x))."""
    return x * torch.sigmoid(x).detach()


def _make_linearised_rmsnorm_forward(norm):
    """Return a patched forward that freezes the RMS denominator."""
    eps = norm.variance_epsilon
    weight = norm.weight

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        inv_rms = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps).detach()
        return (weight * (h * inv_rms)).to(in_dtype)

    return forward


def _make_linearised_mlp_forward(mlp):
    """
    Return a patched MLP forward with:
      1. linearised SiLU  (freeze the sigmoid)
      2. half-rule for  gate * up  multiplication
    """
    gate_proj = mlp.gate_proj
    up_proj = mlp.up_proj
    down_proj = mlp.down_proj

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        gate = _linearised_silu(gate_proj(hidden_states))  # linearised
        up = up_proj(hidden_states)
        # half-rule: each multiplicand receives half the gradient
        intermediate = 0.5 * (gate * up.detach() + gate.detach() * up)
        return down_proj(intermediate)

    return forward


def _make_linearised_attn_forward(attn):
    """
    Return a patched forward for HF Qwen2/Qwen3 attention that freezes
    attention weights.

    Per Table 2 / Appendix B of the paper:
      MHA_j(x) = W_out · Concat_h( freeze(A_j^h) · W_value^h · x )

    Gradients flow only through the value path; the QK softmax is detached.
    When the attention module has per-head q/k RMSNorms (Qwen3), those
    denominators are also frozen for consistency. Qwen2.5-Math follows the
    official HF Qwen2 path, which has no q_norm / k_norm modules.
    """
    q_proj = attn.q_proj
    k_proj = attn.k_proj
    v_proj = attn.v_proj
    o_proj = attn.o_proj
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    head_dim = attn.head_dim
    scaling = attn.scaling
    num_kv_groups = attn.num_key_value_groups

    # Qwen3 applies per-head RMSNorm to q/k before transpose; Qwen2 does not.
    if q_norm is not None and k_norm is not None:
        q_eps, q_weight = q_norm.variance_epsilon, q_norm.weight
        k_eps, k_weight = k_norm.variance_epsilon, k_norm.weight
    else:
        q_eps = q_weight = k_eps = k_weight = None

    def _frozen_rmsnorm(x, weight, eps):
        """RMSNorm with frozen denominator, works on arbitrary last dim."""
        in_dtype = x.dtype
        h = x.to(torch.float32)
        inv_rms = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps).detach()
        return (weight * (h * inv_rms)).to(in_dtype)

    def forward(
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        # This mirrors the official HF Qwen2/Qwen3 eager forward paths.
        q = q_proj(hidden_states).view(hidden_shape)
        k = k_proj(hidden_states).view(hidden_shape)
        v = v_proj(hidden_states).view(hidden_shape)

        if q_norm is not None and k_norm is not None:
            q = _frozen_rmsnorm(q, q_weight, q_eps)
            k = _frozen_rmsnorm(k, k_weight, k_eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE (linear in q, k)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expansion
        k = repeat_kv(k, num_kv_groups)
        v = repeat_kv(v, num_kv_groups)

        # attention weights — FROZEN (detached)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype).detach()

        # value path — gradients flow here only
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = o_proj(attn_output)

        return attn_output, None  # no attn_weights returned

    return forward


# ──────────────────────────────────────────────────────────────────────
# Core attributor
# ──────────────────────────────────────────────────────────────────────


class ReLPAttributor:
    """
    Compute RelP attribution scores for every MLP neuron.

    Usage
    -----
    >>> attr = ReLPAttributor(model, tokenizer)
    >>> scores = attr.attribute(input_ids, metric_fn)
    >>> # scores[layer] is  [batch, seq, d_ffn]
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_layers = model.config.num_hidden_layers
        self.d_ffn = model.config.intermediate_size

    # ---- context manager: patch / unpatch the model ----

    @contextmanager
    def _linearised_ctx(self):
        """Temporarily monkey-patch for a linearised backward.

        Note: the model must use ``attn_implementation='eager'``
        for the attention linearisation to be numerically exact.
        Pass ``attn_implementation='eager'`` when loading the model.
        """
        saved = {}
        layers = self.model.model.layers

        for i, layer in enumerate(layers):
            saved[f"mlp_{i}"] = layer.mlp.forward
            saved[f"ln1_{i}"] = layer.input_layernorm.forward
            saved[f"ln2_{i}"] = layer.post_attention_layernorm.forward
            saved[f"attn_{i}"] = layer.self_attn.forward

            layer.mlp.forward = _make_linearised_mlp_forward(layer.mlp)
            layer.input_layernorm.forward = _make_linearised_rmsnorm_forward(
                layer.input_layernorm
            )
            layer.post_attention_layernorm.forward = _make_linearised_rmsnorm_forward(
                layer.post_attention_layernorm
            )
            layer.self_attn.forward = _make_linearised_attn_forward(layer.self_attn)

        saved["final_norm"] = self.model.model.norm.forward
        self.model.model.norm.forward = _make_linearised_rmsnorm_forward(
            self.model.model.norm
        )

        try:
            yield
        finally:
            for i, layer in enumerate(layers):
                layer.mlp.forward = saved[f"mlp_{i}"]
                layer.input_layernorm.forward = saved[f"ln1_{i}"]
                layer.post_attention_layernorm.forward = saved[f"ln2_{i}"]
                layer.self_attn.forward = saved[f"attn_{i}"]
            self.model.model.norm.forward = saved["final_norm"]

    # ---- main entry point ----

    @torch.enable_grad()
    def attribute(
        self,
        input_ids: torch.Tensor,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute per-neuron attribution with a single linearised backward.

        Parameters
        ----------
        input_ids : [1, seq_len]
        metric_fn : logits -> scalar  (e.g. logit difference)
        layers    : subset of layers (default: all)

        Returns
        -------
        dict  {layer_idx: Tensor[batch, seq, d_ffn]}
            Attribution  = activation * gradient  for each MLP neuron.
        """
        if layers is None:
            layers = list(range(self.n_layers))

        input_ids = input_ids.to(self.device)
        activations: Dict[int, torch.Tensor] = {}
        hooks = []

        # hook: capture input to down_proj  (= MLP hidden activations)
        def _make_hook(idx):
            def hook_fn(module, args, output):
                act = args[0]  # [batch, seq, d_ffn]
                act.retain_grad()
                activations[idx] = act

            return hook_fn

        for i in layers:
            h = self.model.model.layers[i].mlp.down_proj.register_forward_hook(
                _make_hook(i)
            )
            hooks.append(h)

        try:
            self.model.zero_grad()
            with self._linearised_ctx():
                logits = self.model(input_ids).logits  # [1, seq, vocab]
                metric = metric_fn(logits)
                metric.backward()

            attributions: Dict[int, torch.Tensor] = {}
            for idx, act in activations.items():
                grad = act.grad if act.grad is not None else torch.zeros_like(act)
                attributions[idx] = (act * grad).detach()
        finally:
            for h in hooks:
                h.remove()

        return attributions

    # ---- convenience: aggregate over a dataset ----

    def attribute_dataset(
        self,
        examples,
        task,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Average per-neuron absolute attributions over a list of TaskExamples.

        Returns  {layer: [d_ffn]}  (summed over positions, averaged over examples).
        """
        agg: Dict[int, torch.Tensor] = {}
        n = 0

        for ex in examples:
            ids = metric_input_ids(task, ex).to(self.device)
            metric_fn = task.make_metric_fn(ex)
            scores = self.attribute(ids, metric_fn, layers=layers)

            for layer_idx, s in scores.items():
                # sum over batch & positions, take abs
                v = s.abs().sum(dim=(0, 1))  # [d_ffn]
                if layer_idx not in agg:
                    agg[layer_idx] = torch.zeros_like(v)
                agg[layer_idx] += v
            n += 1

        for k in agg:
            agg[k] /= max(n, 1)

        return agg

    # ---- edge attribution (paper §5.4, Eq 16-18) ----

    @torch.enable_grad()
    def attribute_edges(
        self,
        input_ids: torch.Tensor,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        source_neurons: List[Tuple[int, int]],
        target_neurons: List[Tuple[int, int]],
        node_scores: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """
        Compute edge attribution scores between neuron pairs.

        Uses ReLP with stop-gradients on intermediate MLPs to isolate
        the *direct* effect of source on target (paper Eq 16-18).

        Parameters
        ----------
        input_ids     : [1, seq_len]
        metric_fn     : logits -> scalar
        source_neurons: list of (layer, neuron_idx)
        target_neurons: list of (layer, neuron_idx)
        node_scores   : {layer: [d_ffn]} node-level ReLP scores for Eq 18
                        normalization. If None, only raw edge scores (Eq 16)
                        are returned.

        Returns
        -------
        dict  { ((src_layer, src_idx), (tgt_layer, tgt_idx)): float }
            Attribution flow scores (Eq 18) or raw edge scores (Eq 16).
        """
        input_ids = input_ids.to(self.device)

        # group neurons by layer
        src_by_layer: Dict[int, List[int]] = {}
        for l, n in source_neurons:
            src_by_layer.setdefault(l, []).append(n)
        tgt_by_layer: Dict[int, List[int]] = {}
        for l, n in target_neurons:
            tgt_by_layer.setdefault(l, []).append(n)

        # enumerate valid (src_layer, tgt_layer) pairs
        layer_pairs = []
        for sl in sorted(src_by_layer):
            for tl in sorted(tgt_by_layer):
                if sl < tl:
                    layer_pairs.append((sl, tl))

        edge_scores: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}

        for src_layer, tgt_layer in layer_pairs:
            src_indices = src_by_layer[src_layer]
            tgt_indices = tgt_by_layer[tgt_layer]

            # -- set up hooks --
            hooks = []
            captured: Dict[str, torch.Tensor] = {}

            # capture source activations
            def _src_hook(module, args, output):
                act = args[0]  # [batch, seq, d_ffn]
                act.retain_grad()
                captured["src"] = act

            hooks.append(
                self.model.model.layers[src_layer].mlp.down_proj.register_forward_hook(
                    _src_hook
                )
            )

            # capture target activations
            def _tgt_hook(module, args, output):
                act = args[0]
                act.retain_grad()
                captured["tgt"] = act

            hooks.append(
                self.model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(
                    _tgt_hook
                )
            )

            # stop-grad on intermediate MLP layers (direct effect)
            for mid in range(src_layer + 1, tgt_layer):

                def _stop_hook(module, args, output):
                    return output.detach()

                hooks.append(
                    self.model.model.layers[mid].mlp.register_forward_hook(_stop_hook)
                )

            try:
                self.model.zero_grad()
                with self._linearised_ctx():
                    logits = self.model(input_ids).logits
                    # we don't backward from metric — we backward from
                    # individual target neurons to get d(v_t)/d(v_s)

                src_act = captured["src"]  # [1, seq, d_ffn]
                tgt_act = captured["tgt"]  # [1, seq, d_ffn]

                for tgt_idx in tgt_indices:
                    # scalar: sum over batch & positions for this target neuron
                    tgt_scalar = tgt_act[:, :, tgt_idx].sum()
                    # gradient of target w.r.t. source activations
                    (grad,) = torch.autograd.grad(
                        tgt_scalar,
                        src_act,
                        retain_graph=True,
                    )
                    # grad is [1, seq, d_ffn] — d(v_t)/d(v_s) for all source neurons

                    for src_idx in src_indices:
                        # Eq 16: v_s(x) * d(v_t)/d(v_s), summed over positions
                        raw_score = (
                            (src_act[:, :, src_idx] * grad[:, :, src_idx]).sum().item()
                        )

                        # Eq 18: normalise by v_t(x) and multiply by node ReLP
                        if node_scores is not None:
                            v_t_val = tgt_act[:, :, tgt_idx].sum().item()
                            node_relp = node_scores.get(tgt_layer)
                            if node_relp is not None and abs(v_t_val) > 1e-10:
                                relp_vt = node_relp[tgt_idx].item()
                                flow = raw_score * relp_vt / v_t_val
                            else:
                                flow = raw_score
                        else:
                            flow = raw_score

                        key = ((src_layer, src_idx), (tgt_layer, tgt_idx))
                        edge_scores[key] = flow
            finally:
                for h in hooks:
                    h.remove()

        return edge_scores

    def attribute_edges_dataset(
        self,
        examples,
        task,
        node_attributions: Dict[int, torch.Tensor],
        k_nodes: int = 500,
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """
        Average edge attribution scores over examples.

        Selects the top *k_nodes* neurons from *node_attributions* first,
        then computes edges only between those neurons.

        Returns  { ((src_l, src_n), (tgt_l, tgt_n)): avg_abs_flow }
        """
        # select top neurons
        entries: List[Tuple[float, int, int]] = []
        for layer_idx, scores in node_attributions.items():
            for nidx in range(scores.shape[0]):
                entries.append((scores[nidx].abs().item(), layer_idx, nidx))
        entries.sort(reverse=True)
        top_neurons = [(l, n) for _, l, n in entries[:k_nodes]]

        # split into sources and targets (a neuron can be both)
        all_layers = sorted({l for l, _ in top_neurons})
        min_layer, max_layer = all_layers[0], all_layers[-1]
        sources = [(l, n) for l, n in top_neurons if l < max_layer]
        targets = [(l, n) for l, n in top_neurons if l > min_layer]

        agg: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
        n = 0

        for ex in examples:
            ids = task.tokenize(ex.input_text).to(self.device)
            metric_fn = task.make_metric_fn(ex)
            scores = self.attribute_edges(
                ids,
                metric_fn,
                sources,
                targets,
                node_scores=node_attributions,
            )
            for key, val in scores.items():
                agg[key] = agg.get(key, 0.0) + abs(val)
            n += 1

        for key in agg:
            agg[key] /= max(n, 1)

        return agg
