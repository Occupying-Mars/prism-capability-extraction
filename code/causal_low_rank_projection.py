#!/usr/bin/env python3
"""Evaluate causal low-rank MLP projections for issue #12."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_route_conditioned_full_answer import route_fields
from position_interface_decomposition import resolve_batch_size, split_pairs
from union_attribution import pick_device


ROUTE_KEYS = ["source_position", "carry_signature", "n_carries", "leading_carry"]


def parse_layers(text: str, n_layers: int) -> list[int]:
    if text == "all":
        return list(range(n_layers))
    out: set[int] = set()
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            start, end = [int(v) for v in raw.split(":", 1)]
            out.update(range(start, end + 1))
        else:
            out.add(int(raw))
    layers = sorted(layer for layer in out if 0 <= layer < n_layers)
    if not layers:
        raise ValueError(f"no valid layers from {text!r}")
    return layers


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def build_records(tokenizer, pairs: list[dict], source_position: str) -> tuple[list[dict], int]:
    records = []
    skipped = 0
    for pair in pairs:
        ex = pair["target"]
        cf = pair["cf"]
        ans = str(ex["answer"])
        prompt = ex["prompt"] + " "
        cf_prompt = cf["prompt"] + " "
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        cf_prompt_ids = tokenizer(cf_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if int(prompt_ids.shape[0]) != int(cf_prompt_ids.shape[0]):
            skipped += 1
            continue
        full_ids = tokenizer(prompt + ans, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        answer_start = int(prompt_ids.shape[0])
        gold_ids = full_ids[answer_start:]
        gold_text = [tokenizer.decode([tid]).strip() for tid in gold_ids.tolist()]
        if len(gold_text) != len(ans) or any(len(tok) != 1 or not tok.isdigit() for tok in gold_text):
            skipped += 1
            continue
        records.append(
            {
                "input_ids": full_ids,
                "cf_prompt_ids": cf_prompt_ids,
                "answer_start": answer_start,
                "answer_token_ids": gold_ids.clone().long(),
                "routes": route_fields(ex, source_position),
            }
        )
    return records, skipped


def load_records(out_dir: Path, tokenizer, positions: list[str], n_prune_test: int, max_records: int | None):
    records = []
    skipped_by_position = {}
    n_pairs_by_position = {}
    for position in positions:
        payload = json.load(open(out_dir / f"{position}_pairs.json"))
        _attr_pairs, test_pairs = split_pairs(payload["pairs"], n_prune_test)
        pos_records, skipped = build_records(tokenizer, test_pairs, position)
        records.extend(pos_records)
        skipped_by_position[position] = skipped
        n_pairs_by_position[position] = len(test_pairs)
    if max_records is not None:
        records = records[:max_records]
    return records, n_pairs_by_position, skipped_by_position


def grouped_indices(records: list[dict]) -> list[list[int]]:
    idx = sorted(
        range(len(records)),
        key=lambda i: (
            int(records[i]["input_ids"].shape[0]),
            int(records[i]["cf_prompt_ids"].shape[0]),
            int(records[i]["answer_token_ids"].shape[0]),
        ),
    )
    groups = []
    start = 0
    while start < len(idx):
        end = start
        key = (
            int(records[idx[start]]["input_ids"].shape[0]),
            int(records[idx[start]]["cf_prompt_ids"].shape[0]),
            int(records[idx[start]]["answer_token_ids"].shape[0]),
        )
        while end < len(idx):
            probe = (
                int(records[idx[end]]["input_ids"].shape[0]),
                int(records[idx[end]]["cf_prompt_ids"].shape[0]),
                int(records[idx[end]]["answer_token_ids"].shape[0]),
            )
            if probe != key:
                break
            end += 1
        groups.append(idx[start:end])
        start = end
    return groups


def cache_mlp_inputs(model, input_ids: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    cache: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]

        def make_hook(li):
            def hook(_module, hook_args):
                cache[li] = hook_args[0].detach().clone()

            return hook

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_idx)))
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return cache


def collect_deltas(
    model,
    records: list[dict],
    layers: list[int],
    *,
    device: str,
    batch_size: int,
    max_samples_per_layer: int,
) -> dict[int, torch.Tensor]:
    samples: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    counts = {layer: 0 for layer in layers}
    for group in grouped_indices(records):
        for offset in range(0, len(group), batch_size):
            chunk = group[offset : offset + batch_size]
            clean_ids = torch.stack([records[i]["input_ids"] for i in chunk]).to(device)
            cf_ids = torch.stack([records[i]["cf_prompt_ids"] for i in chunk]).to(device)
            answer_start = int(records[chunk[0]]["answer_start"])
            cf_cache = cache_mlp_inputs(model, cf_ids, layers)
            clean_cache = cache_mlp_inputs(model, clean_ids, layers)
            for layer in layers:
                if counts[layer] >= max_samples_per_layer:
                    continue
                delta = clean_cache[layer][:, :answer_start, :] - cf_cache[layer][:, :answer_start, :]
                flat = delta.reshape(-1, delta.shape[-1]).detach().float().cpu()
                need = max_samples_per_layer - counts[layer]
                if flat.shape[0] > need:
                    flat = flat[:need]
                samples[layer].append(flat)
                counts[layer] += int(flat.shape[0])
        if all(counts[layer] >= max_samples_per_layer for layer in layers):
            break
    return {layer: torch.cat(parts, dim=0) for layer, parts in samples.items() if parts}


def fit_bases(samples: dict[int, torch.Tensor], *, max_rank: int, device: str, seed: int):
    torch.manual_seed(seed)
    bases: dict[int, torch.Tensor] = {}
    random_bases: dict[int, torch.Tensor] = {}
    spectra: dict[str, Any] = {}
    for layer, x_cpu in samples.items():
        q = min(max_rank, x_cpu.shape[0] - 1, x_cpu.shape[1])
        if q <= 0:
            continue
        x = x_cpu.to(device)
        centered = x - x.mean(dim=0, keepdim=True)
        total_var = float(centered.pow(2).sum().detach().cpu())
        _u, s, v = torch.pca_lowrank(x, q=q, center=True, niter=2)
        basis = v[:, :q].T.contiguous().detach().cpu()
        bases[layer] = basis
        rand = torch.randn(x_cpu.shape[1], q, device=device)
        rand_q, _r = torch.linalg.qr(rand, mode="reduced")
        random_bases[layer] = rand_q[:, :q].T.contiguous().detach().cpu()
        eig = (s[:q].detach().cpu() ** 2).tolist()
        spectra[str(layer)] = {
            "n_samples": int(x_cpu.shape[0]),
            "rank_available": int(q),
            "total_variance": total_var,
            "explained_variance_by_component": eig,
            "cumulative_explained": [float(sum(eig[: i + 1]) / max(total_var, 1e-12)) for i in range(len(eig))],
        }
    return bases, random_bases, spectra


def project_tensor(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    work = x.reshape(-1, x.shape[-1]).float()
    basis_f = basis.float()
    out = (work @ basis_f.T) @ basis_f
    return out.reshape(orig_shape).to(dtype=x.dtype)


def finish_counts(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {"correct": vals["correct"], "total": vals["total"], "accuracy": vals["correct"] / max(vals["total"], 1)}
        for key, vals in sorted(counts.items())
    }


def evaluate_projection(
    model,
    records: list[dict],
    layers: list[int],
    bases: dict[int, torch.Tensor],
    *,
    rank: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    dtype = next(model.parameters()).dtype
    basis_by_layer = {layer: bases[layer][:rank].to(device=device, dtype=dtype) for layer in layers if layer in bases}
    correct = 0
    total = 0
    by_route = {key: defaultdict(lambda: {"correct": 0, "total": 0}) for key in ROUTE_KEYS}
    for group in grouped_indices(records):
        for offset in range(0, len(group), batch_size):
            chunk = group[offset : offset + batch_size]
            clean_ids = torch.stack([records[i]["input_ids"] for i in chunk]).to(device)
            cf_ids = torch.stack([records[i]["cf_prompt_ids"] for i in chunk]).to(device)
            gold = torch.stack([records[i]["answer_token_ids"] for i in chunk]).to(device)
            answer_start = int(records[chunk[0]]["answer_start"])
            answer_len = int(gold.shape[1])
            cf_cache = cache_mlp_inputs(model, cf_ids, list(basis_by_layer))
            handles = []
            for layer_idx, layer_basis in basis_by_layer.items():
                layer = model.model.layers[layer_idx]
                neg_mlp = cf_cache[layer_idx]
                prompt_len = int(neg_mlp.shape[1])

                def make_patch(basis, neg_act, pl=prompt_len):
                    def hook(_module, hook_args):
                        x = hook_args[0]
                        out = x.clone()
                        delta = x[:, :pl, :] - neg_act[:, :pl, :]
                        out[:, :pl, :] = neg_act[:, :pl, :] + project_tensor(delta, basis)
                        if x.shape[1] > pl:
                            out[:, pl:, :] = project_tensor(x[:, pl:, :], basis)
                        return (out,) + hook_args[1:]

                    return hook

                handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_patch(layer_basis, neg_mlp)))
            try:
                with torch.no_grad():
                    logits = model(clean_ids).logits
                pos = torch.arange(answer_start - 1, answer_start - 1 + answer_len, device=device)
                preds = logits[:, pos, :].argmax(dim=-1)
                matches = (preds == gold).all(dim=1)
                correct += int(matches.sum().item())
                total += int(clean_ids.shape[0])
                for batch_idx, rec_idx in enumerate(chunk):
                    ok = int(matches[batch_idx].item())
                    for route_key in ROUTE_KEYS:
                        value = records[rec_idx]["routes"].get(route_key, "missing")
                        by_route[route_key][value]["correct"] += ok
                        by_route[route_key][value]["total"] += 1
            finally:
                for handle in handles:
                    handle.remove()
    return {
        "rank": rank,
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "by_route": {key: finish_counts(vals) for key, vals in by_route.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-pair-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--n-prune-test", type=int, default=500)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--positions", nargs="+", default=["hundreds", "tens", "ones"])
    parser.add_argument("--layers", default="all")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--max-samples-per-layer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records, n_pairs_by_position, skipped_by_position = load_records(
        Path(args.position_pair_dir), tokenizer, args.positions, args.n_prune_test, args.max_records
    )
    if not records:
        raise RuntimeError("no projection records")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager").to(device).eval()
    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    ranks = sorted({rank for rank in args.ranks if rank > 0})
    max_rank = max(ranks)
    result: dict[str, Any] = {
        "model": args.model,
        "variant": args.variant or args.model,
        "position_pair_dir": args.position_pair_dir,
        "positions": args.positions,
        "n_prune_test": args.n_prune_test,
        "max_records": args.max_records,
        "n_records": len(records),
        "n_pairs_by_position": n_pairs_by_position,
        "skipped_by_position": skipped_by_position,
        "layers": layers,
        "ranks": ranks,
        "max_samples_per_layer": args.max_samples_per_layer,
        "topk": {},
        "random": {},
    }
    write_json(out_path, result)

    samples = collect_deltas(
        model,
        records,
        layers,
        device=device,
        batch_size=batch_size,
        max_samples_per_layer=args.max_samples_per_layer,
    )
    bases, random_bases, spectra = fit_bases(samples, max_rank=max_rank, device=device, seed=args.seed)
    result["spectra"] = spectra
    write_json(out_path, result)

    for rank in ranks:
        print(f"[projection] variant={result['variant']} rank={rank} topk", flush=True)
        result["topk"][str(rank)] = evaluate_projection(
            model, records, layers, bases, rank=rank, device=device, batch_size=batch_size
        )
        write_json(out_path, result)
        print(f"[projection] variant={result['variant']} rank={rank} random", flush=True)
        result["random"][str(rank)] = evaluate_projection(
            model, records, layers, random_bases, rank=rank, device=device, batch_size=batch_size
        )
        write_json(out_path, result)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
