"""Transformer->SSM hybrid for Qwen3 (arXiv:2602.11374, retrieval-aware distillation).

Keep the retained (Gather-and-Aggregate) attention heads; replace the rest with a
DiscreteMamba2 token mixer (MOHAWK / arXiv:2408.10189). Per layer, per-head outputs
are assembled (retained -> real attention, replaced -> SSM), passed through the
parameter-free Static-LN adapter, then the shared attention o_proj (paper A.2).

DiscreteMamba2 is vendored verbatim from goombalab/phi-mamba (materialize_mixer path
dropped; a forward_heads() returning gated per-head output before out_proj is added
so the SSM can share the attention o_proj).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pragma: no cover
    causal_conv1d_fn = None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


# --------------------------------------------------------------- DiscreteMamba2


class DiscreteMamba2(nn.Module):
    """Vendored from phi-mamba modules/mixers/discrete_mamba2.py (Mixer)."""

    def __init__(self, d_model, d_state=64, n_qk_heads=32, n_v_heads=32, d_conv=4,
                 expand=1, activation="identity", bias=False, conv_bias=True,
                 chunk_size=128, layer_idx=None, device=None, dtype=None, **kwargs):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.n_qk_heads = n_qk_heads
        self.n_v_heads = n_v_heads
        self.headdim = self.d_inner // n_v_heads
        assert self.d_inner % self.headdim == 0
        assert n_v_heads % n_qk_heads == 0
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.bias = bias

        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * n_qk_heads * d_state + n_v_heads,
            bias=bias, **factory)
        self.z_bias = nn.Parameter(torch.zeros(self.d_inner, device=device)) if not bias else 0
        conv_dim = self.d_inner + 2 * n_qk_heads * d_state
        self.conv_bias = conv_bias
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, bias=conv_bias, kernel_size=d_conv,
                                groups=conv_dim, padding=d_conv - 1, **factory)
        self.act = nn.Identity() if activation == "identity" else nn.SiLU()
        self.D = nn.Parameter(torch.ones(n_v_heads, device=device))
        self.D._optim = {"weight_decay": 0.0}
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory)

    def convolutional_forward(self, xBC, padded_len):
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish", "identity"]:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2))
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2), rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)
        return xBC

    def _ssm(self, u):
        """Shared trunk: returns gated per-head output y [B, L, n_v_heads, headdim]."""
        _, seqlen, _ = u.shape
        cs = self.chunk_size
        padded_len = (1 + (seqlen - 1) // cs) * cs
        u = F.pad(u, (0, 0, 0, padded_len - seqlen))
        xBC, z, A_log = torch.split(
            self.in_proj(u),
            [self.d_inner + 2 * self.n_qk_heads * self.d_state, self.d_inner, self.n_v_heads],
            dim=-1)
        xBC = self.convolutional_forward(xBC, padded_len)
        x, B, C = torch.split(
            xBC, [self.d_inner, self.n_qk_heads * self.d_state, self.n_qk_heads * self.d_state], dim=-1)
        x = rearrange(x, "b l (h n) -> b l h n", h=self.n_v_heads)
        B = rearrange(B, "b l (h n) -> b l h n", h=self.n_qk_heads)
        C = rearrange(C, "b l (h n) -> b l h n", h=self.n_qk_heads)
        y = mamba_chunk_scan_combined(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=A_log, dt_softplus=True,
            A=-torch.ones(self.n_v_heads, device=A_log.device),
            B=B, C=C, chunk_size=cs, return_final_states=False)
        y = y + torch.einsum("h,blhp->blhp", self.D, x)              # [B,L,h,p]
        zb = self.z_bias if isinstance(self.z_bias, torch.Tensor) else 0.0
        gate = F.silu(z + zb) if isinstance(zb, torch.Tensor) else F.silu(z)
        gate = rearrange(gate, "b l (h p) -> b l h p", h=self.n_v_heads)
        return (y * gate)[:, :seqlen]                                # [B,L,n_v_heads,headdim]

    def forward_heads(self, u):
        """Per-head SSM output (pre out_proj) for head-replacement in the hybrid."""
        return self._ssm(u)

    def forward(self, u):
        """Standalone: gated per-head output -> own out_proj (for sanity/pretrain)."""
        y = rearrange(self._ssm(u), "b l h p -> b l (h p)")
        return self.out_proj(y)


# ---------------------------------------------------- hybrid Qwen3 attention


def _static_ln(x, y, eps=1e-5):
    """Whiten x over last dim, recolor to y's per-token mean/std (paper A.2, param-free)."""
    mu_x, std_x = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
    mu_y, std_y = y.mean(-1, keepdim=True), y.std(-1, keepdim=True)
    return ((x - mu_x) / (std_x + eps)) * (std_y + eps) + mu_y


class HybridQwen3Attention(nn.Module):
    """Drops into a Qwen3 decoder layer in place of Qwen3Attention.

    Retained head positions use real attention (reusing the teacher q/k/v/o_proj +
    q_norm/k_norm); all other head positions use the DiscreteMamba2 per-head output.
    """

    def __init__(self, orig_attn, config, retained_heads: list[int], d_state=64):
        super().__init__()
        self.config = config
        self.layer_idx = getattr(orig_attn, "layer_idx", None)
        self.head_dim = orig_attn.head_dim
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.scaling = getattr(orig_attn, "scaling", self.head_dim ** -0.5)
        self.sliding_window = getattr(orig_attn, "sliding_window", None)

        # reuse teacher projections + QK norms (frozen anchors)
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj
        self.q_norm = orig_attn.q_norm
        self.k_norm = orig_attn.k_norm

        retain = torch.zeros(self.n_heads, dtype=torch.bool)
        for h in retained_heads:
            retain[h] = True
        self.register_buffer("retain_mask", retain, persistent=False)
        self.n_retained = int(retain.sum())

        self.ssm = DiscreteMamba2(
            d_model=self.n_heads * self.head_dim, d_state=d_state,
            n_qk_heads=self.n_kv, n_v_heads=self.n_heads, layer_idx=self.layer_idx)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

        B, L, _ = hidden_states.shape
        hd = self.head_dim
        # --- attention path (only needed if any head retained) ---
        attn_heads = None
        if self.n_retained > 0:
            q = self.q_norm(self.q_proj(hidden_states).view(B, L, self.n_heads, hd)).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(B, L, self.n_kv, hd)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, L, self.n_kv, hd).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, self.n_heads // self.n_kv)
            v = repeat_kv(v, self.n_heads // self.n_kv)
            mask = attention_mask
            if mask is not None and mask.dim() == 4:
                mask = mask[:, :, :, : k.shape[-2]]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=(mask is None), scale=self.scaling)
            attn_heads = out.transpose(1, 2)                        # [B,L,n_heads,hd]

        # --- SSM path (per-head replacement outputs) ---
        ssm_heads = self.ssm.forward_heads(hidden_states)          # [B,L,n_heads,hd]

        # --- assemble heads: retained from attention, rest from SSM ---
        if attn_heads is None:
            merged = ssm_heads
        else:
            m = self.retain_mask.view(1, 1, self.n_heads, 1)
            merged = torch.where(m, attn_heads.to(ssm_heads.dtype), ssm_heads)

        # --- Static-LN adapter: align retained-attention stats to SSM stats ---
        if 0 < self.n_retained < self.n_heads:
            ridx = self.retain_mask.nonzero(as_tuple=True)[0]
            sidx = (~self.retain_mask).nonzero(as_tuple=True)[0]
            att_c = merged[:, :, ridx, :].reshape(B, L, -1)
            ssm_c = merged[:, :, sidx, :].reshape(B, L, -1)
            aligned = _static_ln(att_c, ssm_c).view(B, L, self.n_retained, hd)
            merged = merged.clone()
            merged[:, :, ridx, :] = aligned.to(merged.dtype)

        merged = merged.reshape(B, L, self.n_heads * hd)
        return self.o_proj(merged), None


def build_hybrid(model, retained_by_layer: dict[int, list[int]], d_state=64):
    """Replace every decoder layer's self_attn with a HybridQwen3Attention in place."""
    cfg = model.config
    layers = model.model.layers if hasattr(model, "model") else model.layers
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    for i, layer in enumerate(layers):
        retained = retained_by_layer.get(i, [])
        hybrid = HybridQwen3Attention(layer.self_attn, cfg, retained, d_state=d_state)
        hybrid.to(device=dev, dtype=dt)   # moves retain_mask buffer + casts SSM to model dtype
        layer.self_attn = hybrid
    return model


def retained_by_layer_from_scores(scores_npz: Path, select: str, k: int, n_layers: int) -> dict[int, list[int]]:
    """Top-k retained heads by a chosen ablation-drop metric.

    select in {"total","aggregate","gather"}. gather is ~noise on BFCL (function
    selection is redundant), so "total" or "aggregate" are the meaningful choices.
    """
    import numpy as np
    d = np.load(scores_npz)
    key = {"total": "total_drop", "aggregate": "aggregate_drop", "gather": "gather_drop"}[select]
    mat = d[key]
    nh = mat.shape[1]
    order = np.argsort(mat.flatten())[::-1][:k]
    out: dict[int, list[int]] = {i: [] for i in range(n_layers)}
    for idx in order:
        out[int(idx) // nh].append(int(idx) % nh)
    return out


def freeze_except_ssm(model):
    """Only DiscreteMamba2 params train; teacher weights are frozen anchors."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for m in model.modules():
        if isinstance(m, DiscreteMamba2):
            for p in m.parameters():
                p.requires_grad_(True)
                n += p.numel()
    return n
