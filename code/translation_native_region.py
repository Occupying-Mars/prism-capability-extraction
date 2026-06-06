#!/usr/bin/env python3
"""Native selected-MLP-region student helpers for EN->PT experiments.

This path starts from the live base selected-region circuit, not from a fresh
replacement writer. Non-selected MLP intermediate channels are mean-ablated at
the down_proj input. Selected channels stay live and receive trainable low-rank
slice edits to gate_proj rows, up_proj rows, and down_proj columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from translation_region_student import decoder_root


class LowRankLinearDelta(nn.Module):
    """Zero-initialized low-rank additive map."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x.to(torch.float32)))


class NativeRegionDeltaController(nn.Module):
    """Low-rank edits constrained to selected native MLP slices."""

    def __init__(self, *, mask: dict[int, torch.Tensor], means: dict[int, torch.Tensor],
                 hidden_size: int, rank: int):
        super().__init__()
        self.mask = {int(k): v.detach().cpu().bool() for k, v in mask.items()}
        self.means = {int(k): v.detach().cpu() for k, v in means.items()}
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.selected_indices: dict[int, torch.Tensor] = {}
        self.gate_deltas = nn.ModuleDict()
        self.up_deltas = nn.ModuleDict()
        self.down_deltas = nn.ModuleDict()

        for layer_idx, layer_mask in sorted(self.mask.items()):
            selected = torch.nonzero(layer_mask, as_tuple=False).flatten().cpu()
            self.selected_indices[layer_idx] = selected
            if selected.numel() == 0:
                continue
            key = str(layer_idx)
            width = int(selected.numel())
            self.gate_deltas[key] = LowRankLinearDelta(self.hidden_size, width, self.rank)
            self.up_deltas[key] = LowRankLinearDelta(self.hidden_size, width, self.rank)
            self.down_deltas[key] = LowRankLinearDelta(width, self.hidden_size, self.rank)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def install(self, model):
        hooks = []
        root = decoder_root(model)
        selected_act_cache: dict[int, torch.Tensor] = {}

        for layer_idx, layer in enumerate(root.layers):
            selected_cpu = self.selected_indices[layer_idx]
            if selected_cpu.numel() > 0:
                hooks.append(layer.mlp.gate_proj.register_forward_hook(
                    self._make_gate_or_up_hook(layer_idx, selected_cpu, self.gate_deltas)
                ))
                hooks.append(layer.mlp.up_proj.register_forward_hook(
                    self._make_gate_or_up_hook(layer_idx, selected_cpu, self.up_deltas)
                ))

            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
                self._make_down_pre_hook(layer_idx, selected_cpu, selected_act_cache)
            ))
            if selected_cpu.numel() > 0:
                hooks.append(layer.mlp.down_proj.register_forward_hook(
                    self._make_down_hook(layer_idx, selected_act_cache)
                ))
        return hooks

    def _make_gate_or_up_hook(self, layer_idx: int, selected_cpu: torch.Tensor,
                              modules: nn.ModuleDict):
        def hook_fn(module, hook_args, output):
            hidden = hook_args[0]
            selected = selected_cpu.to(output.device)
            delta = modules[str(layer_idx)](hidden).to(dtype=output.dtype)
            patched = output.clone()
            patched[..., selected] = patched[..., selected] + delta
            return patched
        return hook_fn

    def _make_down_pre_hook(self, layer_idx: int, selected_cpu: torch.Tensor,
                            selected_act_cache: dict[int, torch.Tensor]):
        def hook_fn(module, hook_args):
            act = hook_args[0]
            mean = self.means[layer_idx].to(device=act.device, dtype=act.dtype).view(1, 1, -1)
            keep = self.mask[layer_idx].to(device=act.device, dtype=act.dtype).view(1, 1, -1)
            patched = act * keep + mean * (1.0 - keep)
            if selected_cpu.numel() > 0:
                selected = selected_cpu.to(act.device)
                selected_act_cache[layer_idx] = act[..., selected]
            return (patched,) + hook_args[1:]
        return hook_fn

    def _make_down_hook(self, layer_idx: int, selected_act_cache: dict[int, torch.Tensor]):
        def hook_fn(module, hook_args, output):
            selected_act = selected_act_cache.pop(layer_idx, None)
            if selected_act is None:
                return output
            delta = self.down_deltas[str(layer_idx)](selected_act).to(dtype=output.dtype)
            return output + delta
        return hook_fn


def save_native_region_delta(out_dir: str | Path, controller: NativeRegionDeltaController,
                             *, mask_path: str, means_path: str,
                             extra: dict[str, Any]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(controller.state_dict(), out / "native_region_delta.pt")
    config = {
        "rank": controller.rank,
        "hidden_size": controller.hidden_size,
        "mask_path": mask_path,
        "means_path": means_path,
        "trainable_parameters": controller.trainable_parameter_count(),
        "parameterization": "selected native MLP gate/up/down low-rank slice edits",
        **extra,
    }
    with (out / "native_region_delta_config.json").open("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_native_region_delta(student_dir: str | Path, *, mask: dict[int, torch.Tensor],
                             means: dict[int, torch.Tensor], hidden_size: int,
                             map_location: str = "cpu") -> NativeRegionDeltaController:
    root = Path(student_dir)
    cfg = json.loads((root / "native_region_delta_config.json").read_text())
    controller = NativeRegionDeltaController(
        mask=mask,
        means=means,
        hidden_size=hidden_size,
        rank=int(cfg["rank"]),
    )
    state = torch.load(root / "native_region_delta.pt", map_location=map_location)
    controller.load_state_dict(state)
    return controller
