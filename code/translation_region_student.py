#!/usr/bin/env python3
"""Shared EN->PT region-student helpers.

The region student used by issue #31 is not a PEFT/LoRA delta. It is a set of
low-rank writers that overwrite selected MLP intermediate channels at the
`down_proj` input. Non-selected channels are mean-ablated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from translation_io import DEFAULT_PROMPT_STYLE, apply_chat


def decoder_root(model):
    cur = model
    for attr in ("base_model", "model", "language_model"):
        nxt = getattr(cur, attr, None)
        if nxt is not None and nxt is not cur and hasattr(nxt, "layers"):
            return nxt
        if nxt is not None and nxt is not cur:
            cur = nxt
    cur = model
    for _ in range(6):
        if hasattr(cur, "layers"):
            return cur
        if hasattr(cur, "model") and getattr(cur, "model") is not cur:
            cur = cur.model
            continue
        if hasattr(cur, "base_model") and getattr(cur, "base_model") is not cur:
            cur = cur.base_model
            continue
        break
    raise AttributeError("could not locate decoder .layers")


def text_config(model):
    return getattr(model.config, "text_config", model.config)


def load_mask_npz(path: str | Path, n_layers: int, d_ffn: int) -> dict[int, torch.Tensor]:
    arr = np.load(path)
    out: dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        key = f"layer_{layer}"
        if key in arr:
            out[layer] = torch.from_numpy(arr[key]).bool()
        else:
            out[layer] = torch.zeros(d_ffn, dtype=torch.bool)
    return out


def save_mask_npz(path: str | Path, mask: dict[int, torch.Tensor]) -> None:
    payload = {f"layer_{layer}": tensor.cpu().numpy().astype(bool)
               for layer, tensor in sorted(mask.items())}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def count_mask(mask: dict[int, torch.Tensor]) -> int:
    return sum(int(v.sum().item()) for v in mask.values())


def load_translation_rows(path: str | Path, *, target_field: str = "model_hyp",
                          max_rows: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            src = (raw.get("en") or raw.get("src") or "").strip()
            tgt = (raw.get(target_field) or raw.get("pt") or raw.get("tgt") or "").strip()
            if src and tgt:
                rows.append({"en": src, "target": tgt, "raw": raw})
                if max_rows is not None and len(rows) >= max_rows:
                    break
    return rows


def encode_supervised_row(row: dict[str, Any], tokenizer, *, target_language: str,
                          prompt_style: str = DEFAULT_PROMPT_STYLE,
                          max_seq_length: int = 1024,
                          kl_on: str = "answer") -> dict[str, Any] | None:
    prompt = apply_chat(
        tokenizer,
        row["en"],
        target_language=target_language,
        add_generation_prompt=True,
        prompt_style=prompt_style,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    if not target_ids:
        return None
    full_ids = prompt_ids + target_ids
    if len(full_ids) > max_seq_length:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    mask = [False] * len(full_ids)
    if kl_on == "prompt":
        lo, hi = 0, max(len(prompt_ids) - 1, 0)
    elif kl_on == "answer":
        lo, hi = max(len(prompt_ids) - 1, 0), len(full_ids) - 1
    elif kl_on == "all":
        lo, hi = 0, len(full_ids) - 1
    else:
        raise ValueError(f"unknown kl_on={kl_on!r}")
    for idx in range(lo, hi):
        mask[idx] = True
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "kl_logit_mask": torch.tensor(mask, dtype=torch.bool),
        "raw": row.get("raw", {}),
    }


class TranslationJsonlDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, *, target_language: str,
                 prompt_style: str, max_seq_length: int, kl_on: str = "answer"):
        self.rows: list[dict[str, Any]] = []
        dropped = 0
        for row in rows:
            enc = encode_supervised_row(
                row,
                tokenizer,
                target_language=target_language,
                prompt_style=prompt_style,
                max_seq_length=max_seq_length,
                kl_on=kl_on,
            )
            if enc is None:
                dropped += 1
                continue
            self.rows.append(enc)
        self.dropped = dropped

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate_translation(rows: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(int(r["input_ids"].shape[0]) for r in rows)
    batch: dict[str, Any] = {}
    for key, pad_val, dtype in (
        ("input_ids", pad_id, torch.long),
        ("labels", -100, torch.long),
        ("attention_mask", 0, torch.long),
    ):
        value = torch.full((len(rows), max_len), pad_val, dtype=dtype)
        for idx, row in enumerate(rows):
            item = row[key]
            value[idx, : item.shape[0]] = item
        batch[key] = value
    kl_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for idx, row in enumerate(rows):
        item = row["kl_logit_mask"]
        kl_mask[idx, : item.shape[0]] = item
    batch["kl_logit_mask"] = kl_mask
    return batch


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def answer_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            mask: torch.Tensor, *, temperature: float) -> torch.Tensor:
    logit_mask = mask[:, :-1]
    if int(logit_mask.sum().item()) == 0:
        return student_logits.new_zeros(())
    s = student_logits[:, :-1, :][logit_mask] / temperature
    t = teacher_logits[:, :-1, :][logit_mask] / temperature
    return F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def token_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    if int(valid.sum().item()) == 0:
        return logits.new_zeros(()), logits.new_zeros(())
    nll = F.cross_entropy(
        shifted_logits[valid],
        shifted_labels[valid],
        reduction="sum",
    )
    return nll, valid.sum()


def build_mean_cache(model, examples: list[dict[str, Any]], *, n_calib: int,
                     mean_on: str, device: str, dtype: torch.dtype,
                     n_layers: int, d_ffn: int) -> dict[int, torch.Tensor]:
    sums = {i: torch.zeros(d_ffn, device=device, dtype=torch.float32)
            for i in range(n_layers)}
    counts = {i: 0 for i in range(n_layers)}
    hooks = []
    root = decoder_root(model)

    def make_hook(layer_idx: int):
        def hook_fn(module, hook_args, output):
            act = hook_args[0].detach().to(torch.float32)
            flat = act.reshape(-1, act.shape[-1])
            sums[layer_idx] += flat.sum(dim=0)
            counts[layer_idx] += flat.shape[0]
        return hook_fn

    for i, layer in enumerate(root.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            for row in examples[:n_calib]:
                ids = row["prompt_ids"] if mean_on == "prompt" else row["input_ids"]
                model(ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return {
        i: (sums[i] / max(counts[i], 1)).to(dtype=dtype).detach().cpu()
        for i in range(n_layers)
    }


def install_mean_ablation_hooks(model, mask: dict[int, torch.Tensor],
                                means: dict[int, torch.Tensor], *,
                                device: str, dtype: torch.dtype):
    hooks = []
    root = decoder_root(model)
    for layer_idx, layer in enumerate(root.layers):
        keep_bool = mask[layer_idx].to(device)
        keep = keep_bool.to(dtype=dtype).view(1, 1, -1)
        mean = means[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)

        def make_hook(keep, mean):
            def hook_fn(module, hook_args):
                act = hook_args[0]
                patched = act * keep + mean * (1.0 - keep)
                return (patched,) + hook_args[1:]
            return hook_fn

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(keep, mean)))
    return hooks


class LowRankRegionWriter(nn.Module):
    def __init__(self, hidden_size: int, out_features: int, rank: int,
                 init_bias: torch.Tensor | None = None):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=True)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        if init_bias is not None and init_bias.numel() == out_features:
            with torch.no_grad():
                self.up.bias.copy_(init_bias.to(dtype=self.up.bias.dtype))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_fp32 = hidden.to(dtype=torch.float32)
        return self.up(self.down(hidden_fp32))


class RegionStudentController(nn.Module):
    """Low-rank writer set that overwrites selected MLP channels."""

    def __init__(self, *, mask: dict[int, torch.Tensor], means: dict[int, torch.Tensor],
                 hidden_size: int, rank: int, detach_writer_input: bool = True):
        super().__init__()
        self.mask = {int(k): v.detach().cpu().bool() for k, v in mask.items()}
        self.means = {int(k): v.detach().cpu() for k, v in means.items()}
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.detach_writer_input = bool(detach_writer_input)
        self.writers = nn.ModuleDict()
        self.selected_indices: dict[int, torch.Tensor] = {}
        for layer_idx, layer_mask in sorted(self.mask.items()):
            idx = torch.nonzero(layer_mask, as_tuple=False).flatten().cpu()
            self.selected_indices[layer_idx] = idx
            if idx.numel() == 0:
                continue
            init_bias = self.means[layer_idx][idx].to(torch.float32)
            self.writers[str(layer_idx)] = LowRankRegionWriter(
                self.hidden_size,
                int(idx.numel()),
                self.rank,
                init_bias=init_bias,
            )

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def install(self, model):
        hooks = []
        hidden_cache: dict[int, torch.Tensor] = {}
        root = decoder_root(model)

        for layer_idx, layer in enumerate(root.layers):
            def make_mlp_pre(idx: int):
                def hook_fn(module, hook_args):
                    hidden = hook_args[0]
                    hidden_cache[idx] = hidden.detach() if self.detach_writer_input else hidden
                return hook_fn

            def make_down_pre(idx: int):
                def hook_fn(module, hook_args):
                    act = hook_args[0]
                    mean = self.means[idx].to(device=act.device, dtype=act.dtype).view(1, 1, -1)
                    patched = mean.expand_as(act).clone()
                    selected = self.selected_indices[idx].to(act.device)
                    if selected.numel() > 0:
                        hidden = hidden_cache.get(idx)
                        if hidden is None:
                            raise RuntimeError(f"missing cached MLP input for layer {idx}")
                        writer = self.writers[str(idx)]
                        values = writer(hidden).to(dtype=act.dtype)
                        patched[..., selected] = values
                    return (patched,) + hook_args[1:]
                return hook_fn

            hooks.append(layer.mlp.register_forward_pre_hook(make_mlp_pre(layer_idx)))
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_down_pre(layer_idx)))
        return hooks


def save_region_student(out_dir: str | Path, controller: RegionStudentController,
                        *, mask_path: str, means_path: str, extra: dict[str, Any]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(controller.state_dict(), out / "region_student.pt")
    config = {
        "rank": controller.rank,
        "hidden_size": controller.hidden_size,
        "detach_writer_input": controller.detach_writer_input,
        "mask_path": mask_path,
        "means_path": means_path,
        "trainable_parameters": controller.trainable_parameter_count(),
        **extra,
    }
    with (out / "region_student_config.json").open("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_region_student(student_dir: str | Path, *, mask: dict[int, torch.Tensor],
                        means: dict[int, torch.Tensor], hidden_size: int,
                        map_location: str = "cpu") -> RegionStudentController:
    root = Path(student_dir)
    cfg = json.load(open(root / "region_student_config.json"))
    controller = RegionStudentController(
        mask=mask,
        means=means,
        hidden_size=hidden_size,
        rank=int(cfg["rank"]),
        detach_writer_input=bool(cfg.get("detach_writer_input", True)),
    )
    state = torch.load(root / "region_student.pt", map_location=map_location)
    controller.load_state_dict(state)
    return controller
