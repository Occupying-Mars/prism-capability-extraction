"""GLM/ChatGLM-specific ReLP attribution helpers.

Kept separate from ``relp.py`` so Qwen paths are not changed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch


def _linearised_silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x).detach()


def _linearised_rmsnorm_forward(norm):
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    weight = norm.weight

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        inv_rms = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps).detach()
        return (weight * (h * inv_rms)).to(in_dtype)

    return forward


def _linearised_glm_mlp_forward(mlp):
    dense_h_to_4h = mlp.dense_h_to_4h
    dense_4h_to_h = mlp.dense_4h_to_h

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        gate_raw, up = torch.chunk(dense_h_to_4h(hidden_states), 2, dim=-1)
        gate = _linearised_silu(gate_raw)
        intermediate = 0.5 * (gate * up.detach() + gate.detach() * up)
        return dense_4h_to_h(intermediate)

    return forward


def glm_decoder_root(model):
    return model.transformer.encoder


def glm_num_layers(config) -> int:
    return int(config.num_layers)


def glm_d_ffn(config) -> int:
    return int(config.ffn_hidden_size)


class GLMReLPAttributor:
    """ReLP over ChatGLM MLP neurons.

    This linearises RMSNorm and the GLM SwiGLU MLP path. Attention is left
    unpatched because ChatGLM's custom attention does not match the Qwen HF
    attention ABI used by ``relp.py``.
    """

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_layers = glm_num_layers(model.config)
        self.d_ffn = glm_d_ffn(model.config)

    @contextmanager
    def _linearised_ctx(self):
        root = glm_decoder_root(self.model)
        saved = {}
        for i, layer in enumerate(root.layers):
            saved[f"mlp_{i}"] = layer.mlp.forward
            saved[f"ln1_{i}"] = layer.input_layernorm.forward
            saved[f"ln2_{i}"] = layer.post_attention_layernorm.forward
            layer.mlp.forward = _linearised_glm_mlp_forward(layer.mlp)
            layer.input_layernorm.forward = _linearised_rmsnorm_forward(layer.input_layernorm)
            layer.post_attention_layernorm.forward = _linearised_rmsnorm_forward(layer.post_attention_layernorm)
        if getattr(root, "post_layer_norm", False):
            saved["final_norm"] = root.final_layernorm.forward
            root.final_layernorm.forward = _linearised_rmsnorm_forward(root.final_layernorm)
        try:
            yield
        finally:
            for i, layer in enumerate(root.layers):
                layer.mlp.forward = saved[f"mlp_{i}"]
                layer.input_layernorm.forward = saved[f"ln1_{i}"]
                layer.post_attention_layernorm.forward = saved[f"ln2_{i}"]
            if "final_norm" in saved:
                root.final_layernorm.forward = saved["final_norm"]

    @torch.enable_grad()
    def attribute(self, input_ids: torch.Tensor, metric_fn: Callable[[torch.Tensor], torch.Tensor]):
        input_ids = input_ids.to(self.device)
        activations = {}
        hooks = []

        def make_hook(idx: int):
            def hook_fn(module, args, output):
                act = args[0]
                act.retain_grad()
                activations[idx] = act

            return hook_fn

        for i, layer in enumerate(glm_decoder_root(self.model).layers):
            hooks.append(layer.mlp.dense_4h_to_h.register_forward_hook(make_hook(i)))
        try:
            self.model.zero_grad()
            with self._linearised_ctx():
                logits = self.model(input_ids, use_cache=False).logits
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
