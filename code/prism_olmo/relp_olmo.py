"""OLMo-specific ReLP attribution helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch


def _linearised_silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x).detach()


def _linearised_rmsnorm_forward(norm):
    eps = getattr(norm, "eps", 1e-5)
    weight = getattr(norm, "weight", None)

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        inv_rms = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps).detach()
        out = h * inv_rms
        if weight is not None:
            out = weight * out
        return out.to(in_dtype)

    return forward


def _linearised_mlp_forward(mlp):
    gate_proj = mlp.gate_proj
    up_proj = mlp.up_proj
    down_proj = mlp.down_proj

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        gate = _linearised_silu(gate_proj(hidden_states))
        up = up_proj(hidden_states)
        intermediate = 0.5 * (gate * up.detach() + gate.detach() * up)
        return down_proj(intermediate)

    return forward


def olmo_decoder_root(model):
    return model.model


def olmo_layers(model):
    return olmo_decoder_root(model).layers


def olmo_num_layers(config) -> int:
    return int(getattr(config, "num_hidden_layers"))


def olmo_d_ffn(config) -> int:
    return int(getattr(config, "intermediate_size"))


class OLMoReLPAttributor:
    """ReLP over OLMo MLP neurons."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_layers = olmo_num_layers(model.config)
        self.d_ffn = olmo_d_ffn(model.config)

    @contextmanager
    def _linearised_ctx(self):
        saved = {}
        root = olmo_decoder_root(self.model)
        for i, layer in enumerate(olmo_layers(self.model)):
            saved[f"mlp_{i}"] = layer.mlp.forward
            layer.mlp.forward = _linearised_mlp_forward(layer.mlp)
            for name in ("input_layernorm", "post_attention_layernorm"):
                norm = getattr(layer, name, None)
                if norm is not None:
                    saved[f"{name}_{i}"] = norm.forward
                    norm.forward = _linearised_rmsnorm_forward(norm)
        norm = getattr(root, "norm", None)
        if norm is not None:
            saved["final_norm"] = norm.forward
            norm.forward = _linearised_rmsnorm_forward(norm)
        try:
            yield
        finally:
            for i, layer in enumerate(olmo_layers(self.model)):
                layer.mlp.forward = saved[f"mlp_{i}"]
                for name in ("input_layernorm", "post_attention_layernorm"):
                    norm = getattr(layer, name, None)
                    key = f"{name}_{i}"
                    if norm is not None and key in saved:
                        norm.forward = saved[key]
            if "final_norm" in saved:
                root.norm.forward = saved["final_norm"]

    @torch.enable_grad()
    def attribute(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, metric_fn: Callable[[torch.Tensor], torch.Tensor]):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        activations = {}
        hooks = []

        def make_hook(idx: int):
            def hook_fn(module, args, output):
                act = args[0]
                act.retain_grad()
                activations[idx] = act

            return hook_fn

        for i, layer in enumerate(olmo_layers(self.model)):
            hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(i)))
        try:
            self.model.zero_grad()
            with self._linearised_ctx():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                metric = metric_fn(logits)
                metric.backward()
            out = {}
            for idx, act in activations.items():
                grad = act.grad if act.grad is not None else torch.zeros_like(act)
                out[idx] = (act * grad).detach()
            return out
        finally:
            for hook in hooks:
                hook.remove()
