"""
Mean-ablation utilities for circuit evaluation.

Mean ablation replaces a neuron's activation with its mean over a
calibration dataset, effectively removing its contribution to the
model's computation on a specific input.

Uses pyvene's intervention primitives where possible, with direct
PyTorch hooks as the execution backend (since pyvene does not yet
have a built-in Qwen3 model type mapping).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from .circuit import Circuit


# ──────────────────────────────────────────────────────────────────────
# Mean activation cache
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MeanCache:
    """Per-layer mean MLP activations computed over a calibration set."""

    means: Dict[int, torch.Tensor]      # layer -> [d_ffn]

    @staticmethod
    def build(
        model,
        calibration_ids: list[torch.Tensor],
        device: str = "cpu",
        layers: Optional[list[int]] = None,
    ) -> "MeanCache":
        """
        Run *model* on *calibration_ids* and collect position-averaged
        MLP activations (inputs to ``down_proj``) at every layer.

        Parameters
        ----------
        calibration_ids : list of [1, seq_i] tensors
        """
        n_layers = model.config.num_hidden_layers
        if layers is None:
            layers = list(range(n_layers))

        sums:   Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}

        # register hooks
        hooks = []

        def _make_hook(idx):
            def hook_fn(module, args, output):
                act = args[0].detach()       # [1, seq, d_ffn]
                flat = act.reshape(-1, act.shape[-1])  # [seq, d_ffn]
                if idx not in sums:
                    sums[idx]   = torch.zeros(flat.shape[-1], device=device)
                    counts[idx] = 0
                sums[idx]   += flat.sum(dim=0).to(device)
                counts[idx] += flat.shape[0]
            return hook_fn

        for i in layers:
            h = model.model.layers[i].mlp.down_proj.register_forward_hook(
                _make_hook(i)
            )
            hooks.append(h)

        try:
            with torch.no_grad():
                for ids in calibration_ids:
                    model(ids.to(device))
        finally:
            for h in hooks:
                h.remove()

        means = {i: sums[i] / counts[i] for i in sums}
        return MeanCache(means=means)


# ──────────────────────────────────────────────────────────────────────
# Mean-ablation forward pass
# ──────────────────────────────────────────────────────────────────────

def mean_ablate_run(
    model,
    input_ids: torch.Tensor,
    metric_fn: Callable,
    mean_cache: MeanCache,
    circuit: "Circuit",
    ablate_complement: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run *model* on *input_ids* with mean-ablated MLP neurons.

    Parameters
    ----------
    ablate_complement : bool
        True  -> keep the circuit, ablate everything else  (faithfulness)
        False -> ablate the circuit, keep everything else  (completeness)

    Returns
    -------
    Scalar metric value.
    """
    n_layers = model.config.num_hidden_layers
    d_ffn    = model.config.intermediate_size

    # pre-compute per-layer masks:  True = neuron belongs to circuit
    layer_masks: Dict[int, torch.Tensor] = {}
    for layer_idx in range(n_layers):
        mask = torch.zeros(d_ffn, dtype=torch.bool, device=device)
        for l, n in circuit.nodes:
            if l == layer_idx:
                mask[n] = True
        if mask.any() or ablate_complement:
            layer_masks[layer_idx] = mask

    hooks = []

    def _make_pre_hook(layer_idx):
        mask = layer_masks.get(layer_idx)
        mean = mean_cache.means.get(layer_idx)
        if mask is None or mean is None:
            return None

        mean_dev = mean.to(device)

        def hook_fn(module, args):
            act = args[0]                    # [batch, seq, d_ffn]
            if ablate_complement:
                # keep circuit neurons, replace rest with means
                ablate_mask = ~mask           # True where we ablate
            else:
                # ablate circuit neurons, keep rest
                ablate_mask = mask            # True where we ablate

            modified = act.clone()
            mean_for_act = mean_dev.to(device=act.device, dtype=act.dtype)
            modified[:, :, ablate_mask] = mean_for_act[ablate_mask]
            return (modified,) + args[1:]

        return hook_fn

    for layer_idx in range(n_layers):
        hook_fn = _make_pre_hook(layer_idx)
        if hook_fn is not None:
            h = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                hook_fn
            )
            hooks.append(h)

    try:
        with torch.no_grad():
            logits = model(input_ids.to(device)).logits
            val = metric_fn(logits).item()
    finally:
        for h in hooks:
            h.remove()

    return val
