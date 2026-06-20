#!/usr/bin/env python3
"""Minimal Qwen3 substrate runner for exported physical MLP bundles."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional cuda runtime dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _triton_matmul_nt_kernel(
        x,
        w,
        out,
        m_size: tl.constexpr,
        k_size: tl.constexpr,
        n_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, k_size, BLOCK_K):
            k = k0 + offs_k
            x_tile = tl.load(
                x + offs_m[:, None] * k_size + k[None, :],
                mask=(offs_m[:, None] < m_size) & (k[None, :] < k_size),
                other=0.0,
            )
            w_tile = tl.load(
                w + offs_n[None, :] * k_size + k[:, None],
                mask=(offs_n[None, :] < n_size) & (k[:, None] < k_size),
                other=0.0,
            )
            acc += tl.dot(x_tile, w_tile)
        tl.store(
            out + offs_m[:, None] * n_size + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
        )

    @triton.jit
    def _triton_rms_inv_kernel(
        x,
        inv_rms,
        m_size: tl.constexpr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_h = tl.arange(0, BLOCK_H)
        vals = tl.load(
            x + pid_m * hidden_size + offs_h,
            mask=(pid_m < m_size) & (offs_h < hidden_size),
            other=0.0,
        ).to(tl.float32)
        sum_sq = tl.sum(vals * vals, axis=0)
        tl.store(inv_rms + pid_m, tl.rsqrt(sum_sq / hidden_size + eps), mask=pid_m < m_size)

    @triton.jit
    def _triton_normed_matmul_nt_kernel(
        x,
        norm_weight,
        inv_rms,
        w,
        out,
        m_size: tl.constexpr,
        k_size: tl.constexpr,
        n_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        inv = tl.load(inv_rms + offs_m, mask=offs_m < m_size, other=0.0)[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, k_size, BLOCK_K):
            k = k0 + offs_k
            x_tile = tl.load(
                x + offs_m[:, None] * k_size + k[None, :],
                mask=(offs_m[:, None] < m_size) & (k[None, :] < k_size),
                other=0.0,
            )
            norm = tl.load(norm_weight + k, mask=k < k_size, other=0.0)
            x_tile = (x_tile.to(tl.float32) * inv * norm[None, :].to(tl.float32)).to(x_tile.dtype)
            w_tile = tl.load(
                w + offs_n[None, :] * k_size + k[:, None],
                mask=(offs_n[None, :] < n_size) & (k[:, None] < k_size),
                other=0.0,
            )
            acc += tl.dot(x_tile, w_tile)
        tl.store(
            out + offs_m[:, None] * n_size + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
        )

    @triton.jit
    def _triton_silu_down_kernel(
        gate_up,
        down,
        out,
        residual,
        m_size: tl.constexpr,
        hidden_size: tl.constexpr,
        intermediate_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, intermediate_size, BLOCK_K):
            k = k0 + offs_k
            gate = tl.load(
                gate_up + offs_m[:, None] * (2 * intermediate_size) + k[None, :],
                mask=(offs_m[:, None] < m_size) & (k[None, :] < intermediate_size),
                other=0.0,
            )
            up = tl.load(
                gate_up + offs_m[:, None] * (2 * intermediate_size) + intermediate_size + k[None, :],
                mask=(offs_m[:, None] < m_size) & (k[None, :] < intermediate_size),
                other=0.0,
            )
            down_tile = tl.load(
                down + offs_n[None, :] * intermediate_size + k[:, None],
                mask=(offs_n[None, :] < hidden_size) & (k[:, None] < intermediate_size),
                other=0.0,
            )
            gate_f = gate.to(tl.float32)
            act = (gate_f * tl.sigmoid(gate_f) * up.to(tl.float32)).to(down_tile.dtype)
            acc += tl.dot(act, down_tile)
        store_value = acc
        if ADD_RESIDUAL:
            residual_tile = tl.load(
                residual + offs_m[:, None] * hidden_size + offs_n[None, :],
                mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < hidden_size),
                other=0.0,
            )
            store_value = acc.to(residual_tile.dtype) + residual_tile
        tl.store(
            out + offs_m[:, None] * hidden_size + offs_n[None, :],
            store_value,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < hidden_size),
        )


@dataclass
class QwenSubstrateConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float

    @classmethod
    def from_json(cls, path: Path) -> "QwenSubstrateConfig":
        raw = json.loads(path.read_text())
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        return cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            vocab_size=int(raw["vocab_size"]),
            rope_theta=float(rope.get("rope_theta") or raw.get("rope_theta") or 1000000.0),
        )


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    bsz, heads, seq_len, head_dim = x.shape
    return x[:, :, None, :, :].expand(bsz, heads, groups, seq_len, head_dim).reshape(
        bsz, heads * groups, seq_len, head_dim
    )


def round_up(value: int, multiple: int) -> int:
    if value == 0 or multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


class PackedGatedMLP(nn.Module):
    """Sparse Qwen gated MLP with gate/up packed into one projection.

    The exported substrate already stores physically sliced channels. This
    module makes the hot path closer to a native kernel shape:

        x -> packed gate/up matmul -> silu(gate) * up -> down matmul

    Optional padding is storage/kernel-layout only. The active gate/up rows stay
    contiguous and the down projection consumes only active channels, so padded
    zero channels cannot perturb greedy decode decisions.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, padding_multiple: int = 0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.requested_intermediate_size = int(intermediate_size)
        self.intermediate_size = round_up(self.requested_intermediate_size, padding_multiple)
        if self.intermediate_size == 0:
            self.register_buffer("_empty", torch.empty(0), persistent=False)
            return
        self.gate_up_proj = nn.Linear(hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.intermediate_size == 0:
            return torch.zeros((*x.shape[:-1], self.hidden_size), device=x.device, dtype=x.dtype)
        gate_up = self.gate_up_proj(x)
        kept = self.requested_intermediate_size
        gate = gate_up[..., :kept]
        up = gate_up[..., kept : 2 * kept]
        hidden = F.silu(gate) * up
        if kept == self.intermediate_size:
            return self.down_proj(hidden)
        return F.linear(hidden, self.down_proj.weight[:, :kept])

    @staticmethod
    def pack_gate_up_weights(
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        padded_intermediate_size: int,
    ) -> torch.Tensor:
        kept, hidden = gate_weight.shape
        if kept != up_weight.shape[0] or hidden != up_weight.shape[1]:
            raise ValueError(f"gate/up shape mismatch: {tuple(gate_weight.shape)} vs {tuple(up_weight.shape)}")
        if padded_intermediate_size < kept:
            raise ValueError(f"padded size {padded_intermediate_size} is smaller than kept size {kept}")
        out = gate_weight.new_zeros((2 * padded_intermediate_size, hidden))
        out[:kept].copy_(gate_weight)
        out[kept : 2 * kept].copy_(up_weight)
        return out

    @staticmethod
    def pack_down_weights(down_weight: torch.Tensor, padded_intermediate_size: int) -> torch.Tensor:
        hidden, kept = down_weight.shape
        if padded_intermediate_size < kept:
            raise ValueError(f"padded size {padded_intermediate_size} is smaller than kept size {kept}")
        out = down_weight.new_zeros((hidden, padded_intermediate_size))
        out[:, :kept].copy_(down_weight)
        return out


class FunctionalPackedGatedMLP(nn.Module):
    """Packed substrate MLP as a direct operator over raw weight parameters."""

    def __init__(self, hidden_size: int, intermediate_size: int, padding_multiple: int = 0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.requested_intermediate_size = int(intermediate_size)
        self.intermediate_size = round_up(self.requested_intermediate_size, padding_multiple)
        if self.intermediate_size == 0:
            self.register_buffer("_empty", torch.empty(0), persistent=False)
            return
        self.gate_up_weight = nn.Parameter(torch.empty(2 * self.intermediate_size, hidden_size))
        self.down_weight = nn.Parameter(torch.empty(hidden_size, self.intermediate_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.intermediate_size == 0:
            return torch.zeros((*x.shape[:-1], self.hidden_size), device=x.device, dtype=x.dtype)
        gate_up = F.linear(x, self.gate_up_weight)
        kept = self.requested_intermediate_size
        gate = gate_up[..., :kept]
        up = gate_up[..., kept : 2 * kept]
        hidden = F.silu(gate) * up
        return F.linear(hidden, self.down_weight[:, :kept])


class TritonSiluDownPackedGatedMLP(FunctionalPackedGatedMLP):
    """Packed gate/up matmul with a custom triton silu+down decode kernel."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        block_m: int = 8,
        block_n: int = 64,
        block_k: int = 64,
        num_warps: int = 4,
        torch_prefill: bool = True,
        custom_decode_small_width: int | None = None,
        custom_decode_large_width: int | None = None,
    ):
        if triton is None:
            raise RuntimeError("triton is required for mlp_impl='triton-silu-down'")
        super().__init__(hidden_size, intermediate_size, padding_multiple=0)
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.num_warps = int(num_warps)
        self.torch_prefill = bool(torch_prefill)
        self.custom_decode_small_width = custom_decode_small_width
        self.custom_decode_large_width = custom_decode_large_width

    def _use_triton_decode_for_width(self) -> bool:
        if self.custom_decode_small_width is None and self.custom_decode_large_width is None:
            return True
        kept = self.requested_intermediate_size
        if self.custom_decode_small_width is not None and kept <= self.custom_decode_small_width:
            return True
        if self.custom_decode_large_width is not None and kept >= self.custom_decode_large_width:
            return True
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.intermediate_size == 0:
            return torch.zeros((*x.shape[:-1], self.hidden_size), device=x.device, dtype=x.dtype)
        if x.device.type != "cuda":
            raise ValueError("triton-silu-down MLP requires cuda tensors")
        flat = x.reshape(-1, self.hidden_size)
        if self.torch_prefill and (flat.shape[0] > self.block_m or not self._use_triton_decode_for_width()):
            return super().forward(x)
        gate_up = F.linear(flat, self.gate_up_weight)
        out = torch.empty(flat.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        grid = (
            triton.cdiv(flat.shape[0], self.block_m),
            triton.cdiv(self.hidden_size, self.block_n),
        )
        _triton_silu_down_kernel[grid](
            gate_up,
            self.down_weight,
            out,
            out,
            flat.shape[0],
            self.hidden_size,
            self.intermediate_size,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            ADD_RESIDUAL=False,
            num_warps=self.num_warps,
        )
        return out.reshape(*x.shape[:-1], self.hidden_size)

    def forward_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.intermediate_size == 0:
            return residual
        flat = x.reshape(-1, self.hidden_size)
        if self.torch_prefill and (flat.shape[0] > self.block_m or not self._use_triton_decode_for_width()):
            return residual + super().forward(x)
        gate_up = F.linear(flat, self.gate_up_weight)
        out = torch.empty(flat.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        residual_flat = residual.reshape(-1, self.hidden_size)
        grid = (
            triton.cdiv(flat.shape[0], self.block_m),
            triton.cdiv(self.hidden_size, self.block_n),
        )
        _triton_silu_down_kernel[grid](
            gate_up,
            self.down_weight,
            out,
            residual_flat,
            flat.shape[0],
            self.hidden_size,
            self.intermediate_size,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            ADD_RESIDUAL=True,
            num_warps=self.num_warps,
        )
        return out.reshape(*x.shape[:-1], self.hidden_size)


class TritonFullPackedGatedMLP(nn.Module):
    """Experimental full custom triton MLP replacement."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        block_m: int = 8,
        block_n: int = 32,
        block_k: int = 64,
        prefill_block_m: int | None = None,
        prefill_block_n: int | None = None,
        prefill_block_k: int | None = None,
        num_warps: int = 4,
        torch_prefill: bool = False,
        decode_workspace_tokens: int = 0,
        custom_decode_small_width: int | None = None,
        custom_decode_large_width: int | None = None,
    ):
        super().__init__()
        if triton is None:
            raise RuntimeError("triton is required for mlp_impl='triton-full'")
        self.hidden_size = int(hidden_size)
        self.requested_intermediate_size = int(intermediate_size)
        self.intermediate_size = self.requested_intermediate_size
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.num_warps = int(num_warps)
        self.prefill_block_m = int(prefill_block_m) if prefill_block_m is not None else self.block_m
        self.prefill_block_n = int(prefill_block_n) if prefill_block_n is not None else self.block_n
        self.prefill_block_k = int(prefill_block_k) if prefill_block_k is not None else self.block_k
        self.torch_prefill = bool(torch_prefill)
        self.decode_workspace_tokens = int(decode_workspace_tokens)
        self.norm_block_h = 1 << (self.hidden_size - 1).bit_length()
        self._inv_rms_scratch: torch.Tensor | None = None
        self.custom_decode_small_width = custom_decode_small_width
        self.custom_decode_large_width = custom_decode_large_width
        if self.intermediate_size == 0:
            self.register_buffer("_empty", torch.empty(0), persistent=False)
            return
        self.gate_up_weight = nn.Parameter(torch.empty(2 * self.intermediate_size, hidden_size))
        self.down_weight = nn.Parameter(torch.empty(hidden_size, self.intermediate_size))
        self.prefill_gate_up_weight = None
        self.prefill_down_weight = None
        if self.decode_workspace_tokens:
            self.register_buffer(
                "gate_up_buffer",
                torch.empty(self.decode_workspace_tokens, 2 * self.intermediate_size),
                persistent=False,
            )
            self.register_buffer(
                "out_buffer",
                torch.empty(self.decode_workspace_tokens, self.hidden_size),
                persistent=False,
            )

    def _torch_forward(self, flat: torch.Tensor) -> torch.Tensor:
        kept = self.requested_intermediate_size
        if self.prefill_gate_up_weight is not None:
            gate_up = F.linear(flat, self.prefill_gate_up_weight)
            gate = gate_up[..., :kept]
            up = gate_up[..., kept : 2 * kept]
        else:
            gate = F.linear(flat, self.gate_up_weight[:kept])
            up = F.linear(flat, self.gate_up_weight[self.intermediate_size : self.intermediate_size + kept])
        hidden = F.silu(gate) * up
        down_weight = self.prefill_down_weight if self.prefill_down_weight is not None else self.down_weight[:, :kept]
        return F.linear(hidden, down_weight)

    def _torch_forward_post_norm(self, x: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
        dtype = x.dtype
        normed = x.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = norm_weight * normed.to(dtype)
        return self._torch_forward(normed.reshape(-1, self.hidden_size)).reshape(*x.shape[:-1], self.hidden_size)

    def _use_triton_decode_for_width(self) -> bool:
        if self.custom_decode_small_width is None and self.custom_decode_large_width is None:
            return True
        kept = self.requested_intermediate_size
        if self.custom_decode_small_width is not None and kept <= self.custom_decode_small_width:
            return True
        if self.custom_decode_large_width is not None and kept >= self.custom_decode_large_width:
            return True
        return False

    def _inv_rms_workspace(self, tokens: int, device: torch.device) -> torch.Tensor:
        if (
            self._inv_rms_scratch is None
            or self._inv_rms_scratch.device != device
            or self._inv_rms_scratch.shape[0] < tokens
        ):
            self._inv_rms_scratch = torch.empty(tokens, device=device, dtype=torch.float32)
        return self._inv_rms_scratch[:tokens]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.intermediate_size == 0:
            return torch.zeros((*x.shape[:-1], self.hidden_size), device=x.device, dtype=x.dtype)
        if x.device.type != "cuda":
            raise ValueError("triton-full MLP requires cuda tensors")
        flat = x.reshape(-1, self.hidden_size)
        if self.torch_prefill and (flat.shape[0] > self.block_m or not self._use_triton_decode_for_width()):
            return self._torch_forward(flat).reshape(*x.shape[:-1], self.hidden_size)
        if flat.shape[0] > self.block_m:
            block_m, block_n, block_k = self.prefill_block_m, self.prefill_block_n, self.prefill_block_k
        else:
            block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        if self.decode_workspace_tokens and flat.shape[0] <= self.decode_workspace_tokens:
            gate_up = self.gate_up_buffer[: flat.shape[0]]
            out = self.out_buffer[: flat.shape[0]]
        else:
            gate_up = torch.empty(flat.shape[0], 2 * self.intermediate_size, device=x.device, dtype=x.dtype)
            out = torch.empty(flat.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        grid_gate = (
            triton.cdiv(flat.shape[0], block_m),
            triton.cdiv(2 * self.intermediate_size, block_n),
        )
        _triton_matmul_nt_kernel[grid_gate](
            flat,
            self.gate_up_weight,
            gate_up,
            flat.shape[0],
            self.hidden_size,
            2 * self.intermediate_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=self.num_warps,
        )
        grid_down = (
            triton.cdiv(flat.shape[0], block_m),
            triton.cdiv(self.hidden_size, block_n),
        )
        _triton_silu_down_kernel[grid_down](
            gate_up,
            self.down_weight,
            out,
            out,
            flat.shape[0],
            self.hidden_size,
            self.intermediate_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            ADD_RESIDUAL=False,
            num_warps=self.num_warps,
        )
        return out.reshape(*x.shape[:-1], self.hidden_size)

    def forward_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.intermediate_size == 0:
            return residual
        if x.device.type != "cuda":
            raise ValueError("triton-full MLP requires cuda tensors")
        flat = x.reshape(-1, self.hidden_size)
        if self.torch_prefill and (flat.shape[0] > self.block_m or not self._use_triton_decode_for_width()):
            return residual + self._torch_forward(flat).reshape(*x.shape[:-1], self.hidden_size)
        if flat.shape[0] > self.block_m:
            block_m, block_n, block_k = self.prefill_block_m, self.prefill_block_n, self.prefill_block_k
        else:
            block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        if self.decode_workspace_tokens and flat.shape[0] <= self.decode_workspace_tokens:
            gate_up = self.gate_up_buffer[: flat.shape[0]]
            out = self.out_buffer[: flat.shape[0]]
        else:
            gate_up = torch.empty(flat.shape[0], 2 * self.intermediate_size, device=x.device, dtype=x.dtype)
            out = torch.empty(flat.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        grid_gate = (
            triton.cdiv(flat.shape[0], block_m),
            triton.cdiv(2 * self.intermediate_size, block_n),
        )
        _triton_matmul_nt_kernel[grid_gate](
            flat,
            self.gate_up_weight,
            gate_up,
            flat.shape[0],
            self.hidden_size,
            2 * self.intermediate_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=self.num_warps,
        )
        residual_flat = residual.reshape(-1, self.hidden_size)
        grid_down = (
            triton.cdiv(flat.shape[0], block_m),
            triton.cdiv(self.hidden_size, block_n),
        )
        _triton_silu_down_kernel[grid_down](
            gate_up,
            self.down_weight,
            out,
            residual_flat,
            flat.shape[0],
            self.hidden_size,
            self.intermediate_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            ADD_RESIDUAL=True,
            num_warps=self.num_warps,
        )
        return out.reshape(*x.shape[:-1], self.hidden_size)

    def forward_post_norm_residual(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if self.intermediate_size == 0:
            return residual
        if x.device.type != "cuda":
            raise ValueError("triton-full MLP requires cuda tensors")
        flat = x.reshape(-1, self.hidden_size)
        if self.torch_prefill and (flat.shape[0] > self.block_m or not self._use_triton_decode_for_width()):
            return residual + self._torch_forward_post_norm(x, norm_weight, eps)
        if flat.shape[0] > self.block_m:
            block_m, block_n, block_k = self.prefill_block_m, self.prefill_block_n, self.prefill_block_k
        else:
            block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        if self.decode_workspace_tokens and flat.shape[0] <= self.decode_workspace_tokens:
            gate_up = self.gate_up_buffer[: flat.shape[0]]
            out = self.out_buffer[: flat.shape[0]]
        else:
            gate_up = torch.empty(flat.shape[0], 2 * self.intermediate_size, device=x.device, dtype=x.dtype)
            out = torch.empty(flat.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        inv_rms = self._inv_rms_workspace(flat.shape[0], x.device)
        _triton_rms_inv_kernel[(flat.shape[0],)](
            flat,
            inv_rms,
            flat.shape[0],
            self.hidden_size,
            eps,
            BLOCK_H=self.norm_block_h,
            num_warps=8,
        )
        grid_gate = (
            triton.cdiv(flat.shape[0], block_m),
            triton.cdiv(2 * self.intermediate_size, block_n),
        )
        _triton_normed_matmul_nt_kernel[grid_gate](
            flat,
            norm_weight,
            inv_rms,
            self.gate_up_weight,
            gate_up,
            flat.shape[0],
            self.hidden_size,
            2 * self.intermediate_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=self.num_warps,
        )
        residual_flat = residual.reshape(-1, self.hidden_size)
        grid_down = (
            triton.cdiv(flat.shape[0], block_m),
            triton.cdiv(self.hidden_size, block_n),
        )
        _triton_silu_down_kernel[grid_down](
            gate_up,
            self.down_weight,
            out,
            residual_flat,
            flat.shape[0],
            self.hidden_size,
            self.intermediate_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            ADD_RESIDUAL=True,
            num_warps=self.num_warps,
        )
        return out.reshape(*x.shape[:-1], self.hidden_size)


class PaddedTritonFullPackedGatedMLP(TritonFullPackedGatedMLP):
    """Experimental full custom triton MLP with zero-padded active channels."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        padding_multiple: int = 128,
        block_m: int = 8,
        block_n: int = 16,
        block_k: int = 64,
        prefill_block_m: int | None = None,
        prefill_block_n: int | None = None,
        prefill_block_k: int | None = None,
        num_warps: int = 4,
        torch_prefill: bool = False,
        decode_workspace_tokens: int = 0,
        custom_decode_small_width: int | None = None,
        custom_decode_large_width: int | None = None,
    ):
        self.requested_intermediate_size = int(intermediate_size)
        self.padded_intermediate_size = round_up(self.requested_intermediate_size, int(padding_multiple))
        super().__init__(
            hidden_size,
            self.padded_intermediate_size,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            prefill_block_m=prefill_block_m,
            prefill_block_n=prefill_block_n,
            prefill_block_k=prefill_block_k,
            num_warps=num_warps,
            torch_prefill=torch_prefill,
            decode_workspace_tokens=decode_workspace_tokens,
            custom_decode_small_width=custom_decode_small_width,
            custom_decode_large_width=custom_decode_large_width,
        )
        self.requested_intermediate_size = int(intermediate_size)
        self.padded_intermediate_size = self.intermediate_size
        if self.torch_prefill and self.requested_intermediate_size != self.intermediate_size:
            self.prefill_gate_up_weight = nn.Parameter(
                torch.empty(2 * self.requested_intermediate_size, hidden_size)
            )
            self.prefill_down_weight = nn.Parameter(torch.empty(hidden_size, self.requested_intermediate_size))

    @staticmethod
    def pack_gate_up_weights(
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        padded_intermediate_size: int,
    ) -> torch.Tensor:
        kept, hidden = gate_weight.shape
        if kept != up_weight.shape[0] or hidden != up_weight.shape[1]:
            raise ValueError(f"gate/up shape mismatch: {tuple(gate_weight.shape)} vs {tuple(up_weight.shape)}")
        if padded_intermediate_size < kept:
            raise ValueError(f"padded size {padded_intermediate_size} is smaller than kept size {kept}")
        out = gate_weight.new_zeros((2 * padded_intermediate_size, hidden))
        out[:kept].copy_(gate_weight)
        out[padded_intermediate_size : padded_intermediate_size + kept].copy_(up_weight)
        return out


def _parse_triton_mlp_impl(
    mlp_impl: str,
) -> tuple[
    str,
    int | None,
    int,
    int,
    int,
    int | None,
    int | None,
    int | None,
    int,
    bool,
    int,
    int | None,
    int | None,
] | None:
    if mlp_impl == "triton-full":
        return ("full", None, 8, 32, 64, None, None, None, 4, False, 0, None, None)
    match = re.fullmatch(r"triton-full-m(\d+)n(\d+)k(\d+)", mlp_impl)
    if match:
        block_m, block_n, block_k = (int(item) for item in match.groups())
        return ("full", None, block_m, block_n, block_k, None, None, None, 4, False, 0, None, None)
    if mlp_impl == "triton-full-padded128":
        return ("padded", 128, 8, 16, 64, None, None, None, 4, False, 0, None, None)
    if mlp_impl == "triton-full-padded128-dyn":
        return ("padded", 128, 8, 16, 64, 64, 128, 64, 4, False, 0, None, None)
    if mlp_impl == "triton-full-padded128-decode":
        return ("padded", 128, 8, 16, 64, None, None, None, 4, True, 0, None, None)
    if mlp_impl == "triton-full-padded128-decode-buffered":
        return ("padded", 128, 8, 16, 64, None, None, None, 4, True, 8, None, None)
    if mlp_impl == "triton-full-padded128-decode-widthgated":
        return ("padded", 128, 8, 16, 64, None, None, None, 4, True, 0, 3400, 8500)
    if mlp_impl == "triton-full-padded128-decode-widthgated-split":
        return ("padded-split", 128, 8, 16, 64, None, None, None, 4, True, 0, 3400, 8500)
    if mlp_impl == "triton-silu-down-decode-widthgated-split":
        return ("silu-down-split", None, 8, 64, 64, None, None, None, 4, True, 0, 2500, None)
    match = re.fullmatch(r"triton-silu-down-decode-widthgated-s(\d+)l(\d+)-split", mlp_impl)
    if match:
        small_width, large_width = (int(item) for item in match.groups())
        return ("silu-down-split", None, 8, 64, 64, None, None, None, 4, True, 0, small_width, large_width)
    match = re.fullmatch(r"triton-silu-down-decode-widthgated-s(\d+)-split", mlp_impl)
    if match:
        (small_width,) = (int(item) for item in match.groups())
        return ("silu-down-split", None, 8, 64, 64, None, None, None, 4, True, 0, small_width, None)
    match = re.fullmatch(
        r"triton-silu-down-decode-m(\d+)n(\d+)k(\d+)w(\d+)-widthgated-s(\d+)l(\d+)-split",
        mlp_impl,
    )
    if match:
        block_m, block_n, block_k, num_warps, small_width, large_width = (int(item) for item in match.groups())
        return (
            "silu-down-split",
            None,
            block_m,
            block_n,
            block_k,
            None,
            None,
            None,
            num_warps,
            True,
            0,
            small_width,
            large_width,
        )
    match = re.fullmatch(r"triton-silu-down-decode-m(\d+)n(\d+)k(\d+)w(\d+)-widthgated-s(\d+)-split", mlp_impl)
    if match:
        block_m, block_n, block_k, num_warps, small_width = (int(item) for item in match.groups())
        return (
            "silu-down-split",
            None,
            block_m,
            block_n,
            block_k,
            None,
            None,
            None,
            num_warps,
            True,
            0,
            small_width,
            None,
        )
    match = re.fullmatch(r"triton-full-padded(\d+)-decode-widthgated-s(\d+)l(\d+)-split", mlp_impl)
    if match:
        padding_multiple, small_width, large_width = (int(item) for item in match.groups())
        return (
            "padded-split",
            padding_multiple,
            8,
            16,
            64,
            None,
            None,
            None,
            4,
            True,
            0,
            small_width,
            large_width,
        )
    match = re.fullmatch(
        r"triton-full-padded(\d+)-decode-m(\d+)n(\d+)k(\d+)w(\d+)-widthgated-s(\d+)l(\d+)-split",
        mlp_impl,
    )
    if match:
        padding_multiple, block_m, block_n, block_k, num_warps, small_width, large_width = (
            int(item) for item in match.groups()
        )
        return (
            "padded-split",
            padding_multiple,
            block_m,
            block_n,
            block_k,
            None,
            None,
            None,
            num_warps,
            True,
            0,
            small_width,
            large_width,
        )
    match = re.fullmatch(
        r"triton-full-padded(\d+)-decode-m(\d+)n(\d+)k(\d+)-widthgated-s(\d+)l(\d+)-split",
        mlp_impl,
    )
    if match:
        padding_multiple, block_m, block_n, block_k, small_width, large_width = (
            int(item) for item in match.groups()
        )
        return (
            "padded-split",
            padding_multiple,
            block_m,
            block_n,
            block_k,
            None,
            None,
            None,
            4,
            True,
            0,
            small_width,
            large_width,
        )
    match = re.fullmatch(r"triton-full-padded(\d+)-decode-m(\d+)n(\d+)k(\d+)w(\d+)(-buffered)?", mlp_impl)
    if match:
        padding_multiple, block_m, block_n, block_k, num_warps, buffered = match.groups()
        workspace_tokens = int(block_m) if buffered else 0
        return (
            "padded",
            int(padding_multiple),
            int(block_m),
            int(block_n),
            int(block_k),
            None,
            None,
            None,
            int(num_warps),
            True,
            workspace_tokens,
            None,
            None,
        )
    match = re.fullmatch(r"triton-full-padded(\d+)-decode-m(\d+)n(\d+)k(\d+)(-buffered)?", mlp_impl)
    if match:
        padding_multiple, block_m, block_n, block_k, buffered = match.groups()
        workspace_tokens = int(block_m) if buffered else 0
        return (
            "padded",
            int(padding_multiple),
            int(block_m),
            int(block_n),
            int(block_k),
            None,
            None,
            None,
            4,
            True,
            workspace_tokens,
            None,
            None,
        )
    match = re.fullmatch(r"triton-full-padded(\d+)-m(\d+)n(\d+)k(\d+)", mlp_impl)
    if match:
        padding_multiple, block_m, block_n, block_k = (int(item) for item in match.groups())
        return ("padded", padding_multiple, block_m, block_n, block_k, None, None, None, 4, False, 0, None, None)
    return None


class QwenLayer(nn.Module):
    def __init__(
        self,
        cfg: QwenSubstrateConfig,
        intermediate_size: int,
        mlp_padding_multiple: int = 0,
        mlp_impl: str = "packed-module",
    ):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if mlp_impl == "packed-module":
            self.mlp = PackedGatedMLP(cfg.hidden_size, intermediate_size, mlp_padding_multiple)
        elif mlp_impl == "packed-functional":
            self.mlp = FunctionalPackedGatedMLP(cfg.hidden_size, intermediate_size, mlp_padding_multiple)
        elif parsed := _parse_triton_mlp_impl(mlp_impl):
            (
                kind,
                padding_multiple,
                block_m,
                block_n,
                block_k,
                prefill_block_m,
                prefill_block_n,
                prefill_block_k,
                num_warps,
                torch_prefill,
                decode_workspace_tokens,
                custom_decode_small_width,
                custom_decode_large_width,
            ) = parsed
            if kind == "full":
                self.mlp = TritonFullPackedGatedMLP(
                    cfg.hidden_size,
                    intermediate_size,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    prefill_block_m=prefill_block_m,
                    prefill_block_n=prefill_block_n,
                    prefill_block_k=prefill_block_k,
                    num_warps=num_warps,
                    torch_prefill=torch_prefill,
                    decode_workspace_tokens=decode_workspace_tokens,
                    custom_decode_small_width=custom_decode_small_width,
                    custom_decode_large_width=custom_decode_large_width,
                )
            elif kind == "silu-down-split":
                if (
                    (custom_decode_small_width is not None and intermediate_size <= custom_decode_small_width)
                    or (custom_decode_large_width is not None and intermediate_size >= custom_decode_large_width)
                ):
                    self.mlp = TritonSiluDownPackedGatedMLP(
                        cfg.hidden_size,
                        intermediate_size,
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=num_warps,
                        torch_prefill=torch_prefill,
                    )
                else:
                    self.mlp = FunctionalPackedGatedMLP(cfg.hidden_size, intermediate_size, mlp_padding_multiple)
            elif kind == "padded":
                self.mlp = PaddedTritonFullPackedGatedMLP(
                    cfg.hidden_size,
                    intermediate_size,
                    padding_multiple=padding_multiple or 128,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    prefill_block_m=prefill_block_m,
                    prefill_block_n=prefill_block_n,
                    prefill_block_k=prefill_block_k,
                    num_warps=num_warps,
                    torch_prefill=torch_prefill,
                    decode_workspace_tokens=decode_workspace_tokens,
                    custom_decode_small_width=custom_decode_small_width,
                    custom_decode_large_width=custom_decode_large_width,
                )
            elif (
                (custom_decode_small_width is not None and intermediate_size <= custom_decode_small_width)
                or (custom_decode_large_width is not None and intermediate_size >= custom_decode_large_width)
            ):
                self.mlp = PaddedTritonFullPackedGatedMLP(
                    cfg.hidden_size,
                    intermediate_size,
                    padding_multiple=padding_multiple or 128,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    prefill_block_m=prefill_block_m,
                    prefill_block_n=prefill_block_n,
                    prefill_block_k=prefill_block_k,
                    num_warps=num_warps,
                    torch_prefill=torch_prefill,
                    decode_workspace_tokens=decode_workspace_tokens,
                )
            else:
                self.mlp = FunctionalPackedGatedMLP(cfg.hidden_size, intermediate_size, mlp_padding_multiple)
        else:
            raise ValueError(f"unknown mlp implementation: {mlp_impl}")

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        cache_position: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x = self.input_layernorm(x)
        bsz, q_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(bsz, q_len, self.cfg.num_attention_heads, self.cfg.head_dim)).transpose(
            1, 2
        )
        k = self.k_norm(self.k_proj(x).view(bsz, q_len, self.cfg.num_key_value_heads, self.cfg.head_dim)).transpose(
            1, 2
        )
        v = self.v_proj(x).view(bsz, q_len, self.cfg.num_key_value_heads, self.cfg.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if cache is not None:
            cache_k, cache_v = cache
            cache_k[:, :, cache_position : cache_position + q_len, :].copy_(k)
            cache_v[:, :, cache_position : cache_position + q_len, :].copy_(v)
            k = cache_k[:, :, : cache_position + q_len, :]
            v = cache_v[:, :, : cache_position + q_len, :]
            next_cache = cache
        else:
            next_cache = (k, v)
        k_full = repeat_kv(k, self.cfg.num_attention_heads // self.cfg.num_key_value_heads)
        v_full = repeat_kv(v, self.cfg.num_attention_heads // self.cfg.num_key_value_heads)
        attn = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attention_mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, self.cfg.hidden_size)
        x = residual + self.o_proj(attn)
        residual = x
        if hasattr(self.mlp, "forward_post_norm_residual"):
            x = self.mlp.forward_post_norm_residual(
                x,
                self.post_attention_layernorm.weight,
                self.cfg.rms_norm_eps,
                residual,
            )
        else:
            x = self.post_attention_layernorm(x)
            if hasattr(self.mlp, "forward_residual"):
                x = self.mlp.forward_residual(x, residual)
            else:
                x = residual + self.mlp(x)
        return x, next_cache


class PackedOutputProjection(nn.Module):
    """Output projection sliced to kept OV channels."""

    def __init__(
        self,
        in_features_full: int,
        out_features: int,
        keep_idx: torch.Tensor,
        *,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features_full = int(in_features_full)
        self.out_features = int(out_features)
        keep_idx = keep_idx.to(device=device, dtype=torch.long)
        self.register_buffer("keep_idx", keep_idx)
        self.weight = nn.Parameter(torch.empty((out_features, keep_idx.numel()), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None

    @classmethod
    def from_dense(cls, original: nn.Linear, keep: torch.Tensor) -> "PackedOutputProjection":
        keep_idx = torch.where(keep.to(device=original.weight.device))[0]
        packed = cls(
            original.in_features,
            original.out_features,
            keep_idx,
            bias=original.bias is not None,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        with torch.no_grad():
            packed.weight.copy_(original.weight.index_select(1, keep_idx))
            if original.bias is not None and packed.bias is not None:
                packed.bias.copy_(original.bias)
        return packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.keep_idx.numel() == 0:
            out_shape = (*x.shape[:-1], self.out_features)
            out = x.new_zeros(out_shape)
            if self.bias is not None:
                out = out + self.bias.to(device=x.device, dtype=x.dtype)
            return out
        return F.linear(x.index_select(-1, self.keep_idx), self.weight, self.bias)


def install_packed_ov_projections(model: "QwenSubstrateLM", keep: torch.Tensor) -> dict[str, int | float | str | dict[str, int]]:
    expected = (len(model.layers), model.cfg.num_attention_heads * model.cfg.head_dim)
    if tuple(keep.shape) != expected:
        raise ValueError(f"attention keep shape {tuple(keep.shape)} != expected {expected}")

    kept_per_layer: dict[str, int] = {}
    packed_layers = 0
    for layer_idx, layer in enumerate(model.layers):
        layer_keep = keep[layer_idx].to(dtype=torch.bool)
        kept = int(layer_keep.sum().item())
        kept_per_layer[str(layer_idx)] = kept
        if kept == expected[1]:
            continue
        layer.o_proj = PackedOutputProjection.from_dense(layer.o_proj, layer_keep)
        packed_layers += 1

    kept_total = int(sum(kept_per_layer.values()))
    total = int(keep.numel())
    return {
        "mode": "packed_ov_output_projection",
        "packed_layers": packed_layers,
        "kept_ov_channels": kept_total,
        "total_ov_channels": total,
        "ov_keep_fraction": kept_total / max(total, 1),
        "kept_per_layer": kept_per_layer,
    }


def build_sdpa_mask(
    attention_mask: torch.Tensor,
    q_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bsz, key_len = attention_mask.shape
    q_pos = past_len + torch.arange(q_len, device=device)
    k_pos = torch.arange(key_len, device=device)
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    valid = attention_mask[:, None, None, :].bool() & causal[None, None, :, :]
    mask = torch.zeros((bsz, 1, q_len, key_len), dtype=dtype, device=device)
    return mask.masked_fill(~valid, torch.finfo(dtype).min)


class QwenSubstrateLM(nn.Module):
    def __init__(
        self,
        cfg: QwenSubstrateConfig,
        intermediate_sizes: list[int],
        mlp_padding_multiple: int = 0,
        mlp_impl: str = "packed-module",
    ):
        super().__init__()
        self.cfg = cfg
        self.mlp_padding_multiple = int(mlp_padding_multiple)
        self.mlp_impl = mlp_impl
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [QwenLayer(cfg, size, self.mlp_padding_multiple, self.mlp_impl) for size in intermediate_sizes]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @classmethod
    def from_bundle(
        cls,
        bundle: Path,
        dtype: torch.dtype,
        device: str = "cuda",
        mlp_padding_multiple: int = 0,
        mlp_impl: str = "packed-module",
    ) -> "QwenSubstrateLM":
        if mlp_padding_multiple > 1:
            raise RuntimeError(
                "--mlp-padding-multiple is disabled: nonzero padding changed bf16 greedy outputs on the real "
                "BFCL substrate benchmark. Re-enable only with a gpu parity guard."
            )
        if mlp_impl not in {"packed-module", "packed-functional"} and _parse_triton_mlp_impl(mlp_impl) is None:
            raise ValueError(f"unknown mlp implementation: {mlp_impl}")
        cfg = QwenSubstrateConfig.from_json(bundle / "config.json")
        metadata = json.loads((bundle / "substrate_metadata.json").read_text())
        kept = metadata["isolation"]["kept_per_layer"]
        model = cls(cfg, [int(kept[str(i)]) for i in range(cfg.num_hidden_layers)], mlp_padding_multiple, mlp_impl)
        state = load_file(bundle / "model.safetensors", device="cpu")
        mapped: dict[str, torch.Tensor] = {
            "embed_tokens.weight": state["model.embed_tokens.weight"],
            "norm.weight": state["model.norm.weight"],
            "lm_head.weight": state["lm_head.weight"],
        }
        for i in range(cfg.num_hidden_layers):
            src = f"model.layers.{i}"
            dst = f"layers.{i}"
            mapped[f"{dst}.input_layernorm.weight"] = state[f"{src}.input_layernorm.weight"]
            mapped[f"{dst}.post_attention_layernorm.weight"] = state[f"{src}.post_attention_layernorm.weight"]
            mapped[f"{dst}.q_proj.weight"] = state[f"{src}.self_attn.q_proj.weight"]
            mapped[f"{dst}.k_proj.weight"] = state[f"{src}.self_attn.k_proj.weight"]
            mapped[f"{dst}.v_proj.weight"] = state[f"{src}.self_attn.v_proj.weight"]
            mapped[f"{dst}.o_proj.weight"] = state[f"{src}.self_attn.o_proj.weight"]
            mapped[f"{dst}.q_norm.weight"] = state[f"{src}.self_attn.q_norm.weight"]
            mapped[f"{dst}.k_norm.weight"] = state[f"{src}.self_attn.k_norm.weight"]
            mlp = model.layers[i].mlp
            if mlp.intermediate_size:
                gate_up_weight = PackedGatedMLP.pack_gate_up_weights(
                    state[f"{src}.mlp.gate_proj.weight"],
                    state[f"{src}.mlp.up_proj.weight"],
                    mlp.intermediate_size,
                )
                down_weight = PackedGatedMLP.pack_down_weights(
                    state[f"{src}.mlp.down_proj.weight"],
                    mlp.intermediate_size,
                )
                if isinstance(mlp, PackedGatedMLP):
                    mapped[f"{dst}.mlp.gate_up_proj.weight"] = gate_up_weight
                    mapped[f"{dst}.mlp.down_proj.weight"] = down_weight
                elif isinstance(mlp, PaddedTritonFullPackedGatedMLP):
                    mapped[f"{dst}.mlp.gate_up_weight"] = PaddedTritonFullPackedGatedMLP.pack_gate_up_weights(
                        state[f"{src}.mlp.gate_proj.weight"],
                        state[f"{src}.mlp.up_proj.weight"],
                        mlp.intermediate_size,
                    )
                    mapped[f"{dst}.mlp.down_weight"] = down_weight
                    if mlp.prefill_gate_up_weight is not None:
                        mapped[f"{dst}.mlp.prefill_gate_up_weight"] = PackedGatedMLP.pack_gate_up_weights(
                            state[f"{src}.mlp.gate_proj.weight"],
                            state[f"{src}.mlp.up_proj.weight"],
                            mlp.requested_intermediate_size,
                        )
                    if mlp.prefill_down_weight is not None:
                        mapped[f"{dst}.mlp.prefill_down_weight"] = state[f"{src}.mlp.down_proj.weight"]
                else:
                    mapped[f"{dst}.mlp.gate_up_weight"] = gate_up_weight
                    mapped[f"{dst}.mlp.down_weight"] = down_weight
        missing, unexpected = model.load_state_dict(mapped, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"state mismatch missing={missing} unexpected={unexpected}")
        return model.to(device=device, dtype=dtype).eval()

    def rotary(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv @ pos).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        cache_position: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(x, position_ids)
        attn_mask = build_sdpa_mask(
            attention_mask[:, : cache_position + input_ids.shape[-1]],
            q_len=input_ids.shape[-1],
            past_len=cache_position,
            dtype=x.dtype,
            device=x.device,
        )
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)  # type: ignore[list-item]
        next_cache = []
        for layer, cache in zip(self.layers, past_key_values):
            x, cache = layer(x, cos, sin, attn_mask, cache, cache_position)
            next_cache.append(cache)
        x = self.norm(x)
        return self.lm_head(x), next_cache

    def init_cache(self, batch_size: int, max_length: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        dtype = self.embed_tokens.weight.dtype
        return [
            (
                torch.empty(
                    batch_size,
                    self.cfg.num_key_value_heads,
                    max_length,
                    self.cfg.head_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.empty(
                    batch_size,
                    self.cfg.num_key_value_heads,
                    max_length,
                    self.cfg.head_dim,
                    device=device,
                    dtype=dtype,
                ),
            )
            for _ in self.layers
        ]


def manual_greedy_generate_custom(
    model: QwenSubstrateLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: int | list[int] | None,
    stop_token_ids: list[int] | None = None,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    generated = input_ids
    max_length = input_ids.shape[-1] + max_new_tokens
    eos_ids = None
    if eos_token_id is not None:
        eos_values = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        eos_ids = torch.tensor(eos_values, device=input_ids.device, dtype=input_ids.dtype)
    stop_ids = None
    if stop_token_ids:
        stop_ids = torch.tensor(stop_token_ids, device=input_ids.device, dtype=input_ids.dtype)
    unfinished = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
    past = model.init_cache(input_ids.shape[0], max_length=max_length, device=input_ids.device)
    full_attention_mask = torch.ones(
        input_ids.shape[0],
        max_length,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    full_attention_mask[:, : input_ids.shape[-1]].copy_(attention_mask)
    full_generated = torch.empty(input_ids.shape[0], max_length, device=input_ids.device, dtype=input_ids.dtype)
    full_generated[:, : input_ids.shape[-1]].copy_(input_ids)
    generated_len = input_ids.shape[-1]
    sequence_lengths = attention_mask.long().sum(dim=-1, keepdim=True)
    cache_position = 0
    next_ids = input_ids
    next_pos = position_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, past = model(next_ids, full_attention_mask, next_pos, past, cache_position=cache_position)
            cache_position += next_ids.shape[-1]
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            if eos_ids is not None:
                next_tokens = torch.where(unfinished, next_tokens, torch.full_like(next_tokens, pad_token_id))
                unfinished = unfinished & ~((next_tokens[:, None] == eos_ids[None, :]).any(dim=-1))
            full_generated[:, generated_len].copy_(next_tokens)
            generated_len += 1
            if stop_ids is not None and generated_len >= stop_ids.numel():
                matched_stop = (
                    full_generated[:, generated_len - stop_ids.numel() : generated_len] == stop_ids[None, :]
                ).all(dim=-1)
                unfinished = unfinished & ~matched_stop
            next_pos = sequence_lengths
            sequence_lengths = sequence_lengths + 1
            next_ids = next_tokens[:, None]
            if not bool(unfinished.any()):
                break
    return full_generated[:, :generated_len]


def maybe_compile_mlp(model: QwenSubstrateLM) -> None:
    raise RuntimeError(
        "--compile-mlp is disabled: torch.compile changed bf16 greedy outputs on the real BFCL substrate "
        "benchmark and was slower than eager. Re-enable only with a gpu parity guard."
    )
