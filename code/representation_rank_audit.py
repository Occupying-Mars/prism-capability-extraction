#!/usr/bin/env python3
"""Audit activation spectra for 2-digit addition representations."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from full_answer_group_search import load_mlp_final_npz
from merge_lora_kl_sweep import apply_lora_scale, capture_lora_scaling
from union_attribution import pick_device


POSITIONS = ("hundreds", "tens", "ones")


def parse_variant(text: str) -> dict[str, Any]:
    parts = text.split(":", 3)
    if len(parts) == 2:
        name, model = parts
        return {"name": name, "model": model, "adapter": None, "scale": None}
    if len(parts) == 4:
        name, model, adapter, scale = parts
        return {"name": name, "model": model, "adapter": adapter, "scale": float(scale)}
    raise ValueError("--variant must be NAME:MODEL or NAME:MODEL:ADAPTER:SCALE")


def parse_mask(text: str) -> tuple[str, Path]:
    if ":" not in text:
        raise ValueError("--mask must be NAME:PATH")
    name, path = text.split(":", 1)
    return name, Path(path)


def carry_regime(a: int, b: int) -> str:
    ones_carry = (a % 10) + (b % 10) >= 10
    overflow = a + b >= 100
    if overflow:
        return "overflow"
    if ones_carry:
        return "ones_carry_no_overflow"
    return "no_carry_no_overflow"


def position_digit_index(answer: str, position: str) -> int | None:
    if position == "ones":
        return len(answer) - 1
    if position == "tens" and len(answer) >= 2:
        return len(answer) - 2
    if position == "hundreds" and len(answer) >= 3:
        return len(answer) - 3
    return None


def build_rows(tokenizer, *, min_value: int, max_value: int, max_examples_per_regime: int, seed: int) -> list[dict[str, Any]]:
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in range(min_value, max_value + 1):
        for b in range(min_value, max_value + 1):
            answer = str(a + b)
            prompt = f"{a} + {b} ="
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
            answer_ids = full_ids[len(prompt_ids) :]
            if len(answer_ids) != len(answer):
                continue
            decoded_digits = [tokenizer.decode([tok]).strip() for tok in answer_ids]
            if decoded_digits != list(answer):
                continue
            row = {
                "a": a,
                "b": b,
                "prompt": prompt,
                "answer": answer,
                "input_ids": full_ids,
                "prompt_len": len(prompt_ids),
                "regime": carry_regime(a, b),
                "positions": {},
            }
            for position in POSITIONS:
                digit_idx = position_digit_index(answer, position)
                if digit_idx is None:
                    continue
                # The hidden state at this index predicts the target digit.
                row["positions"][position] = len(prompt_ids) + digit_idx - 1
            by_regime[row["regime"]].append(row)

    rng = random.Random(seed)
    rows = []
    for regime in sorted(by_regime):
        vals = by_regime[regime]
        rng.shuffle(vals)
        rows.extend(vals[:max_examples_per_regime])
    rows.sort(key=lambda row: (row["regime"], row["a"], row["b"]))
    return rows


def batch_rows(rows: list[dict[str, Any]], pad_id: int, batch_size: int):
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for idx, row in enumerate(batch):
            ids = torch.tensor(row["input_ids"], dtype=torch.long)
            input_ids[idx, : len(ids)] = ids
            attention_mask[idx, : len(ids)] = 1
        yield batch, input_ids, attention_mask


def load_model(variant: dict[str, Any], *, dtype: torch.dtype, device: str):
    base = AutoModelForCausalLM.from_pretrained(
        variant["model"],
        dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    if not variant.get("adapter"):
        return base
    model = PeftModel.from_pretrained(base, variant["adapter"]).to(device).eval()
    captured = capture_lora_scaling(model)
    apply_lora_scale(captured, float(variant["scale"]))
    return model


def decoder_layers(model):
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("base_model.model.model.layers", lambda m: m.base_model.model.model.layers),
    ]
    for _name, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None:
            return layers
    raise AttributeError("could not locate decoder layers on model")


def key_for(layer: int, position: str, regime: str) -> str:
    return f"layer_{layer:02d}/{position}/{regime}"


def collect_variant(
    variant: dict[str, Any],
    rows: list[dict[str, Any]],
    tokenizer,
    masks: dict[str, dict[int, torch.Tensor]],
    *,
    dtype: torch.dtype,
    device: str,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, dict[str, float]]]]:
    print(f"[variant] {variant['name']}", flush=True)
    model = load_model(variant, dtype=dtype, device=device)
    n_layers = model.config.num_hidden_layers
    mlp_inputs: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer: int):
        def hook(_module, hook_args):
            mlp_inputs[layer] = hook_args[0].detach()

        return hook

    for layer_idx, layer in enumerate(decoder_layers(model)):
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_idx)))

    hidden_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    energy_sums: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"selected_sq": 0.0, "total_sq": 0.0, "n": 0})
    )
    try:
        for batch_meta, input_ids, attention_mask in batch_rows(rows, tokenizer.pad_token_id, batch_size):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            mlp_inputs.clear()
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states[1:]
            for layer_idx in range(n_layers):
                hs = hidden_states[layer_idx]
                mlp = mlp_inputs[layer_idx]
                for row_idx, row in enumerate(batch_meta):
                    for position, logit_idx in row["positions"].items():
                        key = key_for(layer_idx, position, row["regime"])
                        hidden_chunks[key].append(hs[row_idx, logit_idx].detach().float().cpu())
                        mlp_vec = mlp[row_idx, logit_idx].detach().float()
                        total_sq = float((mlp_vec * mlp_vec).sum().cpu())
                        if total_sq <= 0.0:
                            continue
                        for mask_name, by_layer in masks.items():
                            mask = by_layer[layer_idx].to(device)
                            selected_sq = float((mlp_vec[mask] * mlp_vec[mask]).sum().cpu())
                            entry = energy_sums[key][mask_name]
                            entry["selected_sq"] += selected_sq
                            entry["total_sq"] += total_sq
                            entry["n"] += 1
            print(f"  processed={len(batch_meta)}", flush=True)
    finally:
        for handle in hooks:
            handle.remove()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    matrices = {key: torch.stack(vals, dim=0) for key, vals in hidden_chunks.items() if vals}
    energy = {}
    for key, by_mask in energy_sums.items():
        energy[key] = {}
        for mask_name, entry in by_mask.items():
            frac = entry["selected_sq"] / max(entry["total_sq"], 1e-12)
            energy[key][mask_name] = {**entry, "fraction": frac}
    return matrices, energy


def spectral_summary(x: torch.Tensor, top_ks: list[int]) -> dict[str, Any]:
    x = x.float()
    xc = x - x.mean(dim=0, keepdim=True)
    if xc.shape[0] < 2 or float((xc * xc).sum()) <= 0.0:
        return {"n": int(x.shape[0]), "effective_rank_entropy": 0.0, "effective_rank_pr": 0.0, "topk_explained": {}}
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    eig = s.square()
    probs = eig / eig.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    out = {
        "n": int(x.shape[0]),
        "effective_rank_entropy": float(torch.exp(entropy).item()),
        "effective_rank_pr": float((1.0 / probs.square().sum().clamp_min(1e-12)).item()),
        "topk_explained": {},
        "top10_basis": vh[: min(10, vh.shape[0])].cpu(),
    }
    for k in top_ks:
        out["topk_explained"][str(k)] = float(probs[: min(k, probs.numel())].sum().item())
    return out


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    xy = torch.linalg.matrix_norm(x.T @ y).square()
    xx = torch.linalg.matrix_norm(x.T @ x)
    yy = torch.linalg.matrix_norm(y.T @ y)
    denom = (xx * yy).clamp_min(1e-12)
    return float((xy / denom).item())


def subspace_overlap(bx: torch.Tensor, by: torch.Tensor) -> float:
    k = min(bx.shape[0], by.shape[0])
    if k == 0:
        return 0.0
    return float(torch.linalg.matrix_norm(bx[:k] @ by[:k].T).square().item() / k)


def summarize_layers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant_layer: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_variant_layer[row["variant"]][row["layer"]].append(row["effective_rank_pr"])
    out = {}
    for variant, by_layer in by_variant_layer.items():
        out[variant] = {
            str(layer): sum(vals) / max(len(vals), 1)
            for layer, vals in sorted(by_layer.items())
        }
    return out


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    def fmt(value: float) -> str:
        return f"{value:.3f}"

    lines = [
        "# Issue 11 Representation-Rank Audit Summary",
        "",
        "This audit compares teacher-forced answer-position representations on the same sampled 2-digit examples.",
        "",
        "## Variants",
        "",
        "| variant | model | adapter | scale |",
        "|---|---|---|---:|",
    ]
    for variant in payload["variants"]:
        lines.append(
            f"| {variant['name']} | `{variant['model']}` | "
            f"{variant.get('adapter') or 'none'} | {variant.get('scale') if variant.get('scale') is not None else 'n/a'} |"
        )
    lines.extend([
        "",
        "## Mean Effective Rank By Layer",
        "",
        "Participation-ratio effective rank averaged over answer positions and carry regimes.",
        "",
        "| layer | " + " | ".join(v["name"] for v in payload["variants"]) + " |",
        "|---:|" + "---:|" * len(payload["variants"]),
    ])
    layer_summary = payload["layer_summary"]
    all_layers = sorted({int(layer) for by_layer in layer_summary.values() for layer in by_layer})
    for layer in all_layers:
        vals = []
        for variant in payload["variants"]:
            vals.append(fmt(layer_summary.get(variant["name"], {}).get(str(layer), float("nan"))))
        lines.append(f"| {layer} | " + " | ".join(vals) + " |")

    if payload["pairwise"]:
        lines.extend(["", "## Mean Pairwise Similarity", "", "| pair | mean CKA | mean top-10 overlap |", "|---|---:|---:|"])
        by_pair: dict[str, list[dict[str, float]]] = defaultdict(list)
        for row in payload["pairwise"]:
            by_pair[row["pair"]].append(row)
        for pair, vals in sorted(by_pair.items()):
            mean_cka = sum(v["cka"] for v in vals) / len(vals)
            mean_overlap = sum(v["top10_subspace_overlap"] for v in vals) / len(vals)
            lines.append(f"| {pair} | {fmt(mean_cka)} | {fmt(mean_overlap)} |")

    if payload["mask_energy"]:
        lines.extend(["", "## Mean MLP Activation Energy In Masks", "", "| variant | mask | mean fraction |", "|---|---|---:|"])
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in payload["mask_energy"]:
            grouped[(row["variant"], row["mask"])].append(row["fraction"])
        for (variant, mask), vals in sorted(grouped.items()):
            lines.append(f"| {variant} | {mask} | {fmt(sum(vals) / len(vals))} |")

    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, help="NAME:MODEL or NAME:MODEL:ADAPTER:SCALE")
    parser.add_argument("--mask", action="append", default=[], help="NAME:PATH to mlp_final NPZ")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min", type=int, default=10)
    parser.add_argument("--max", type=int, default=99)
    parser.add_argument("--max-examples-per-regime", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-ks", default="1,5,10,20,50")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [parse_variant(item) for item in args.variant]
    device = pick_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(variants[0]["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = build_rows(
        tokenizer,
        min_value=args.min,
        max_value=args.max,
        max_examples_per_regime=args.max_examples_per_regime,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError("no audit rows built")

    temp = AutoModelForCausalLM.from_pretrained(variants[0]["model"], dtype=dtype, attn_implementation="eager")
    n_layers = temp.config.num_hidden_layers
    d_ffn = temp.config.intermediate_size
    del temp
    masks = {
        name: load_mlp_final_npz(path, n_layers=n_layers, d_ffn=d_ffn, device=device)
        for name, path in (parse_mask(item) for item in args.mask)
    }

    started = time.time()
    all_mats = {}
    all_energy = {}
    for variant in variants:
        mats, energy = collect_variant(
            variant,
            rows,
            tokenizer,
            masks,
            dtype=dtype,
            device=device,
            batch_size=args.batch_size,
        )
        all_mats[variant["name"]] = mats
        all_energy[variant["name"]] = energy

    spectral_rows = []
    bases: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for variant in variants:
        name = variant["name"]
        for key, matrix in sorted(all_mats[name].items()):
            layer_text, position, regime = key.split("/")
            layer = int(layer_text.split("_")[1])
            spec = spectral_summary(matrix, top_ks)
            basis = spec.pop("top10_basis")
            bases[name][key] = basis
            spectral_rows.append(
                {
                    "variant": name,
                    "layer": layer,
                    "position": position,
                    "regime": regime,
                    **spec,
                }
            )

    pairwise_rows = []
    for idx, left in enumerate(variants):
        for right in variants[idx + 1 :]:
            lname = left["name"]
            rname = right["name"]
            shared = sorted(set(all_mats[lname]) & set(all_mats[rname]))
            for key in shared:
                layer_text, position, regime = key.split("/")
                layer = int(layer_text.split("_")[1])
                pairwise_rows.append(
                    {
                        "pair": f"{lname}__{rname}",
                        "layer": layer,
                        "position": position,
                        "regime": regime,
                        "cka": linear_cka(all_mats[lname][key], all_mats[rname][key]),
                        "top10_subspace_overlap": subspace_overlap(bases[lname][key], bases[rname][key]),
                    }
                )

    energy_rows = []
    for variant_name, by_key in all_energy.items():
        for key, by_mask in by_key.items():
            layer_text, position, regime = key.split("/")
            layer = int(layer_text.split("_")[1])
            for mask_name, entry in by_mask.items():
                energy_rows.append(
                    {
                        "variant": variant_name,
                        "layer": layer,
                        "position": position,
                        "regime": regime,
                        "mask": mask_name,
                        "fraction": entry["fraction"],
                        "n": entry["n"],
                    }
                )

    payload = {
        "args": vars(args),
        "variants": variants,
        "n_rows": len(rows),
        "rows_by_regime": {regime: sum(1 for row in rows if row["regime"] == regime) for regime in sorted({row["regime"] for row in rows})},
        "spectra": spectral_rows,
        "pairwise": pairwise_rows,
        "mask_energy": energy_rows,
        "layer_summary": summarize_layers(spectral_rows),
        "elapsed_s": time.time() - started,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    write_summary_md(out_dir / "summary.md", payload)
    print(f"[done] {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
