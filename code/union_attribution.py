#!/usr/bin/env python3
"""
Per-example ReLP attribution over a dataset of correct 2-digit sums.
Takes the UNION (not average, not top-k) of every neuron / attention head
that gets positive attribution on at least one example.

Reports:
  - total MLP neurons vs. activated (union) count
  - total attention heads vs. activated (union) count

Usage:
    uv run python union_attribution.py
    uv run python union_attribution.py --max-examples 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.circuit_tracing.batching import grouped_batches
from src.circuit_tracing.profiling import add_profile_args, make_profiler
from src.circuit_tracing.relp import ReLPAttributor


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def attribute_mlp_and_heads(attributor, model, input_ids, metric_fn, num_heads: int):
    """
    Single linearised backward pass. Capture:
      * MLP neurons: down_proj input  -> [d_ffn] score per layer
      * Attn heads : o_proj input     -> [num_heads] score per layer

    Returns  (mlp_attr, attn_attr)
      mlp_attr[layer]  : Tensor[d_ffn]
      attn_attr[layer] : Tensor[num_heads]
    """
    mlp_acts: dict[int, torch.Tensor] = {}
    attn_acts: dict[int, torch.Tensor] = {}
    hooks = []

    def _mk_mlp_hook(idx):
        def hook(module, args, output):
            act = args[0]  # [b, s, d_ffn]
            act.retain_grad()
            mlp_acts[idx] = act

        return hook

    def _mk_attn_hook(idx):
        def hook(module, args, output):
            act = args[0]  # [b, s, num_heads * head_dim]
            act.retain_grad()
            attn_acts[idx] = act

        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(_mk_mlp_hook(i)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(_mk_attn_hook(i)))

    try:
        model.zero_grad()
        with attributor._linearised_ctx():
            logits = model(input_ids).logits
            metric = metric_fn(logits)
            metric.backward()

        mlp_attr: dict[int, torch.Tensor] = {}
        for l, act in mlp_acts.items():
            grad = act.grad if act.grad is not None else torch.zeros_like(act)
            # sum over batch & seq positions -> [d_ffn]
            mlp_attr[l] = (act * grad).detach().sum(dim=(0, 1)).float().cpu()

        attn_attr: dict[int, torch.Tensor] = {}
        for l, act in attn_acts.items():
            grad = act.grad if act.grad is not None else torch.zeros_like(act)
            attr = (act * grad).detach()  # [b, s, H*d_h]
            b, s, d = attr.shape
            # per head: sum along head_dim, then sum over batch & seq
            per_head = attr.view(b, s, num_heads, -1).sum(dim=-1)  # [b,s,H]
            attn_attr[l] = per_head.sum(dim=(0, 1)).float().cpu()
    finally:
        for h in hooks:
            h.remove()

    return mlp_attr, attn_attr


def attribute_mlp_and_heads_batched(
    attributor,
    model,
    input_ids,
    target_ids,
    cf_ids,
    num_heads: int,
    *,
    to_cpu: bool = True,
):
    """
    Batched version of `attribute_mlp_and_heads` for examples with identical
    prompt token length.

    Returns batched attribution tensors:
      mlp_attr[layer]  : Tensor[batch, d_ffn]
      attn_attr[layer] : Tensor[batch, num_heads]
    """
    mlp_acts: dict[int, torch.Tensor] = {}
    attn_acts: dict[int, torch.Tensor] = {}
    hooks = []

    def _mk_mlp_hook(idx):
        def hook(module, args, output):
            act = args[0]
            act.retain_grad()
            mlp_acts[idx] = act

        return hook

    def _mk_attn_hook(idx):
        def hook(module, args, output):
            act = args[0]
            act.retain_grad()
            attn_acts[idx] = act

        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(_mk_mlp_hook(i)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(_mk_attn_hook(i)))

    try:
        model.zero_grad()
        with attributor._linearised_ctx():
            logits = model(input_ids).logits
            batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            metric = (
                logits[batch_idx, -1, target_ids] - logits[batch_idx, -1, cf_ids]
            ).sum()
            metric.backward()

        mlp_attr: dict[int, torch.Tensor] = {}
        for l, act in mlp_acts.items():
            grad = act.grad if act.grad is not None else torch.zeros_like(act)
            tensor = (act * grad).detach().sum(dim=1).float()
            mlp_attr[l] = tensor.cpu() if to_cpu else tensor

        attn_attr: dict[int, torch.Tensor] = {}
        for l, act in attn_acts.items():
            grad = act.grad if act.grad is not None else torch.zeros_like(act)
            attr = (act * grad).detach()
            b, s, d = attr.shape
            per_head = attr.view(b, s, num_heads, -1).sum(dim=-1)
            tensor = per_head.sum(dim=1).float()
            attn_attr[l] = tensor.cpu() if to_cpu else tensor
    finally:
        for h in hooks:
            h.remove()

    return mlp_attr, attn_attr


def attribute_mlp_and_heads_full_answer_batched(
    attributor,
    model,
    input_ids,
    answer_token_ids,
    cf_token_ids,
    answer_start: int,
    num_heads: int,
    *,
    to_cpu: bool = True,
):
    """
    Teacher-forced full-answer attribution.

    input_ids        : [B, L] where L = prompt_len + A (prompt + gold answer)
    answer_token_ids : [B, A] gold answer tokens
    cf_token_ids     : [B, A] per-position close-digit competitor tokens
    answer_start     : int -- position index in input_ids of the first gold token.
                       logits[:, answer_start - 1 + t] predicts answer_token_ids[:, t].

    Metric: sum over batch and answer positions of
            [log_softmax(logits)[gold_t] - log_softmax(logits)[cf_t]]

    This is the "strict" generalisation of the single-token logit-delta used
    by attribute_mlp_and_heads_batched: it scores the whole teacher-forced
    answer instead of only the leading digit.
    """
    mlp_acts: dict[int, torch.Tensor] = {}
    attn_acts: dict[int, torch.Tensor] = {}
    hooks = []

    def _mk_mlp_hook(idx):
        def hook(module, args, output):
            act = args[0]
            act.retain_grad()
            mlp_acts[idx] = act
        return hook

    def _mk_attn_hook(idx):
        def hook(module, args, output):
            act = args[0]
            act.retain_grad()
            attn_acts[idx] = act
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(_mk_mlp_hook(i)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(_mk_attn_hook(i)))

    try:
        model.zero_grad()
        with attributor._linearised_ctx():
            logits = model(input_ids).logits  # [B, L, V]
            B, L, V = logits.shape
            A = answer_token_ids.shape[1]
            # positions that predict gold[0..A-1]
            pos = torch.arange(answer_start - 1, answer_start - 1 + A, device=logits.device)
            # [B, A, V]
            logp = torch.log_softmax(logits[:, pos, :], dim=-1)
            batch_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, A)
            pos_idx = torch.arange(A, device=logits.device).unsqueeze(0).expand(B, A)
            gold_lp = logp[batch_idx, pos_idx, answer_token_ids]  # [B, A]
            cf_lp = logp[batch_idx, pos_idx, cf_token_ids]        # [B, A]
            metric = (gold_lp - cf_lp).sum()
            metric.backward()

        mlp_attr: dict[int, torch.Tensor] = {}
        for l, act in mlp_acts.items():
            grad = act.grad if act.grad is not None else torch.zeros_like(act)
            tensor = (act * grad).detach().sum(dim=1).float()
            mlp_attr[l] = tensor.cpu() if to_cpu else tensor

        attn_attr: dict[int, torch.Tensor] = {}
        for l, act in attn_acts.items():
            grad = act.grad if act.grad is not None else torch.zeros_like(act)
            attr = (act * grad).detach()
            b, s, d = attr.shape
            per_head = attr.view(b, s, num_heads, -1).sum(dim=-1)
            tensor = per_head.sum(dim=1).float()
            attn_attr[l] = tensor.cpu() if to_cpu else tensor
    finally:
        for h in hooks:
            h.remove()

    return mlp_attr, attn_attr


def build_attr_records(tokenizer, examples):
    digit_id_cache: dict[str, int] = {}
    records = []
    for ex in examples:
        prompt = ex["prompt"]
        ans = str(ex["answer"])
        ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        tgt = ans[0]
        cf = str((int(tgt) + 1) % 10) if tgt != "9" else "8"
        if tgt not in digit_id_cache:
            digit_id_cache[tgt] = tokenizer.encode(tgt, add_special_tokens=False)[0]
        if cf not in digit_id_cache:
            digit_id_cache[cf] = tokenizer.encode(cf, add_special_tokens=False)[0]
        records.append(
            {
                "input_ids": ids,
                "target_id": digit_id_cache[tgt],
                "cf_id": digit_id_cache[cf],
            }
        )
    return records


def _init_union_accumulators(n_layers: int, d_ffn: int, num_heads: int, device: str = "cpu"):
    return {
        mode: {l: torch.zeros(d_ffn, dtype=torch.bool, device=device) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }, {
        mode: {l: torch.zeros(num_heads, dtype=torch.bool, device=device) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }


def _tensor_union_size(layer_map: dict[int, torch.Tensor]) -> int:
    return int(sum(int(v.sum().item()) for v in layer_map.values()))


def _batched_selection_counts(
    attr: dict[int, torch.Tensor], mode: str
) -> dict[int, torch.Tensor]:
    """
    Given {layer: [batch, dim]} batched attributions, return per-dimension counts
    over the batch for the requested selection rule.
    """
    return _batched_selection_counts_multi(attr, [mode])[mode]


def _batched_selection_counts_multi(
    attr: dict[int, torch.Tensor], modes: list[str]
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Same as `_batched_selection_counts` but computes counts for many modes at
    once, sharing the stacked tensor and `|attr|` across modes. Cuts GPU
    kernel launch count and redundant work when TRACKED_MODES is large.
    """
    layers = sorted(attr)
    if not layers:
        return {m: {} for m in modes}

    # Assume every layer has the same feature dim (true for our MLP down_proj
    # inputs and attention head counts within a given model).
    dim = int(attr[layers[0]].shape[1])
    stack = torch.stack([attr[l] for l in layers], dim=0)  # [L, B, D]
    abs_stack = stack.abs()
    # Per-example global max across all layers (shape [B]).
    global_max_per_ex = abs_stack.amax(dim=(0, 2))

    # For topk modes, we re-flatten per-example across layer dimension once.
    flat_abs = None  # [B, L*D] lazily

    out: dict[str, dict[int, torch.Tensor]] = {}
    for mode in modes:
        if mode == "positive":
            counts = (stack > 0).sum(dim=1).to(torch.int32)  # [L, D]
            out[mode] = {l: counts[i] for i, l in enumerate(layers)}
            continue

        if mode.startswith("rel_"):
            frac = float(mode.split("_", 1)[1])
            thresh = (frac * global_max_per_ex).view(1, -1, 1)  # [1, B, 1]
            counts = (abs_stack > thresh).sum(dim=1).to(torch.int32)  # [L, D]
            out[mode] = {l: counts[i] for i, l in enumerate(layers)}
            continue

        if mode.startswith("topk_"):
            if flat_abs is None:
                L, B, D = abs_stack.shape
                # [B, L*D] so topk per example selects across all (layer, dim).
                flat_abs = abs_stack.permute(1, 0, 2).reshape(B, L * D)
            k = int(mode.split("_", 1)[1])
            total_dim = int(flat_abs.shape[1])
            k = min(k, total_dim)
            device = flat_abs.device
            if k <= 0:
                zero = torch.zeros(dim, dtype=torch.int32, device=device)
                out[mode] = {l: zero for l in layers}
                continue
            idx = torch.topk(flat_abs, k=k, dim=1).indices.reshape(-1)
            counts_flat = torch.bincount(idx, minlength=total_dim).to(torch.int32)
            # [L*D] -> [L, D]
            counts_l = counts_flat.view(len(layers), dim)
            out[mode] = {l: counts_l[i] for i, l in enumerate(layers)}
            continue

        raise ValueError(f"Unknown selection mode: {mode}")

    return out


def _update_union_accumulator(acc, counts) -> None:
    for l, c in counts.items():
        acc[l] |= c.bool()


def _layer_union_pairs(layer_map: dict[int, torch.Tensor]):
    pairs = []
    for l, mask in layer_map.items():
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        pairs.extend((l, i) for i in idx)
    return pairs


def _select_indices(
    attr: dict[int, torch.Tensor], mode: str
) -> dict[int, torch.Tensor]:
    """
    Given {layer: [dim]} attributions for one example, return
    {layer: LongTensor of activated indices} according to mode.

    Modes
    -----
    positive         : attr > 0                 (per-layer)
    rel_<frac>       : |attr| > frac * max(|attr|) across ALL layers
    topk_<K>         : top-K by |attr| across ALL layers (global)
    """
    out: dict[int, torch.Tensor] = {}
    if mode == "positive":
        for l, v in attr.items():
            out[l] = (v > 0).nonzero(as_tuple=True)[0]
        return out

    if mode.startswith("rel_"):
        frac = float(mode.split("_", 1)[1])
        global_max = max((v.abs().max().item() for v in attr.values()), default=0.0)
        thresh = frac * global_max
        if thresh <= 0:
            for l in attr:
                out[l] = torch.empty(0, dtype=torch.long)
            return out
        for l, v in attr.items():
            out[l] = (v.abs() > thresh).nonzero(as_tuple=True)[0]
        return out

    if mode.startswith("topk_"):
        k = int(mode.split("_", 1)[1])
        # flatten across layers: (layer, idx, |attr|)
        flat_vals: list[torch.Tensor] = []
        layers: list[int] = []
        offsets: list[int] = []
        off = 0
        for l, v in attr.items():
            flat_vals.append(v.abs())
            layers.append(l)
            offsets.append(off)
            off += v.numel()
        cat = torch.cat(flat_vals)
        k = min(k, cat.numel())
        if k <= 0:
            for l in attr:
                out[l] = torch.empty(0, dtype=torch.long)
            return out
        _, flat_idx = torch.topk(cat, k)
        # map back to (layer, idx)
        picked: dict[int, list[int]] = {l: [] for l in attr}
        # build a per-layer size list for search
        sizes = [v.numel() for v in flat_vals]
        layer_of = []
        for l, sz in zip(layers, sizes):
            layer_of.extend([l] * sz)
        # per-layer offsets
        starts = {}
        acc = 0
        for l, sz in zip(layers, sizes):
            starts[l] = acc
            acc += sz
        for fi in flat_idx.tolist():
            l = layer_of[fi]
            picked[l].append(fi - starts[l])
        for l in attr:
            out[l] = torch.tensor(picked[l], dtype=torch.long)
        return out

    raise ValueError(mode)


# Thresholds tracked concurrently — one union per threshold.
TRACKED_MODES = [
    "positive",  # anything contributing positively
    "rel_0.001",  # |attr| > 0.1% of global max per example
    "rel_0.01",  # |attr| > 1%
    "rel_0.05",  # |attr| > 5%
    "topk_100",  # top-100 globally per example
    "topk_500",  # top-500 globally per example
    "topk_2000",  # top-2000 globally per example
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dataset", default="data/correct_addition_dataset.json")
    ap.add_argument("--output", default="results/union_attribution.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    ap.add_argument("--checkpoint-every", type=int, default=200)
    # bs=256 is the throughput default for 2-digit attribution. bf16 + cuBLAS
    # reduction ordering is batch-size sensitive, so exact reproducibility of
    # bs=64 runs (e.g. perf_v3 numbers) requires passing --batch-size 64.
    # Threshold-level union fractions drift a few percent between the two.
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    add_profile_args(ap)
    args = ap.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    device = pick_device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    print(f"Device: {device}   dtype: {args.dtype}")
    profiler = make_profiler(args, device)

    print(f"Loading {args.model} ...")
    with profiler.phase("model_load", model=args.model):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )

    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    total_mlp = n_layers * d_ffn
    total_heads = n_layers * num_heads

    print(f"  layers={n_layers}  d_ffn={d_ffn}  num_heads={num_heads}")
    print(f"  total MLP neurons = {total_mlp:,}")
    print(f"  total attention heads = {total_heads:,}")

    with open(args.dataset) as f:
        ds = json.load(f)
    examples = ds["examples"]
    if args.max_examples:
        examples = examples[: args.max_examples]
    examples = examples[args.shard_index :: args.num_shards]
    with profiler.phase("build_attr_records", n_examples=len(examples)):
        records = build_attr_records(tokenizer, examples)
    print(
        f"  examples: {len(records):,}  batch_size={args.batch_size}  "
        f"shard={args.shard_index + 1}/{args.num_shards}"
    )

    attributor = ReLPAttributor(model, tokenizer, device=device)

    mlp_unions, head_unions = _init_union_accumulators(n_layers, d_ffn, num_heads, device=device)

    t0 = time.time()
    processed = 0
    for batch in grouped_batches(
        records,
        key_fn=lambda rec: int(rec["input_ids"].shape[0]),
        batch_size=args.batch_size,
    ):
        with profiler.phase(
            "attr_batch_prepare",
            batch_size=len(batch),
            seq_len=int(batch[0]["input_ids"].shape[0]),
        ):
            batch_ids = torch.stack([rec["input_ids"] for rec in batch]).to(device)
            target_ids = torch.tensor([rec["target_id"] for rec in batch], device=device)
            cf_ids = torch.tensor([rec["cf_id"] for rec in batch], device=device)
        try:
            with profiler.phase(
                "attr_batch_forward_backward",
                batch_size=len(batch),
                seq_len=int(batch_ids.shape[1]),
            ):
                mlp_batch_attr, attn_batch_attr = attribute_mlp_and_heads_batched(
                    attributor,
                    model,
                    batch_ids,
                    target_ids,
                    cf_ids,
                    num_heads,
                    to_cpu=False,
                )
                batch_examples = None
        except Exception as e:
            print(f"  [!] batch at {processed} failed: {e}; falling back to singles")
            batch_examples = []
            for rec in batch:
                ids = rec["input_ids"].unsqueeze(0).to(device)

                def metric_fn(logits, _t=rec["target_id"], _c=rec["cf_id"]):
                    return logits[0, -1, _t] - logits[0, -1, _c]

                try:
                    with profiler.phase(
                        "attr_single_forward_backward",
                        seq_len=int(ids.shape[1]),
                    ):
                        batch_examples.append(
                            attribute_mlp_and_heads(
                                attributor,
                                model,
                                ids,
                                metric_fn,
                                num_heads,
                            )
                        )
                except Exception as single_e:
                    print(f"  [!] example at {processed} failed: {single_e}")
                    batch_examples.append((None, None))

        with profiler.phase(
            "attr_batch_reduce",
            batch_size=len(batch) if batch_examples is None else len(batch_examples),
        ):
            if batch_examples is None:
                processed += len(batch)
                mlp_by_mode = _batched_selection_counts_multi(mlp_batch_attr, TRACKED_MODES)
                head_by_mode = _batched_selection_counts_multi(attn_batch_attr, TRACKED_MODES)
                for mode in TRACKED_MODES:
                    _update_union_accumulator(mlp_unions[mode], mlp_by_mode[mode])
                    _update_union_accumulator(head_unions[mode], head_by_mode[mode])
            else:
                for mlp_attr, attn_attr in batch_examples:
                    if mlp_attr is None or attn_attr is None:
                        continue
                    processed += 1
                    for mode in TRACKED_MODES:
                        mlp_pick = _select_indices(mlp_attr, mode)
                        for l, idx in mlp_pick.items():
                            mlp_unions[mode][l][idx.to(mlp_unions[mode][l].device)] = True
                        head_pick = _select_indices(attn_attr, mode)
                        for l, idx in head_pick.items():
                            head_unions[mode][l][idx.to(head_unions[mode][l].device)] = True

        profiler.step(processed=processed)

        if processed % 25 == 0:
            dt = time.time() - t0
            rate = processed / dt
            eta = (len(records) - processed) / rate
            sizes = " ".join(
                f"{m}={_tensor_union_size(mlp_unions[m])}/{total_mlp}({_tensor_union_size(mlp_unions[m]) / total_mlp:.0%})"
                for m in ("rel_0.01", "topk_500")
            )
            print(
                f"  {processed:>5d}/{len(records)}  {sizes}  "
                f"{rate:.2f} ex/s  eta {eta / 60:.1f} min"
            )

        if args.checkpoint_every and processed % args.checkpoint_every == 0:
            with profiler.phase("checkpoint_save", processed=processed):
                _save(
                    args,
                    ds,
                    records,
                    processed,
                    mlp_unions,
                    head_unions,
                    n_layers,
                    total_mlp,
                    total_heads,
                )

    with profiler.phase("final_save", processed=processed):
        _save(
            args,
            ds,
            records,
            processed,
            mlp_unions,
            head_unions,
            n_layers,
            total_mlp,
            total_heads,
        )

    print(f"\n{'=' * 72}")
    print(
        f"  {'Threshold':<14} {'MLP (union / total)':<28} {'Heads (union / total)':<22}"
    )
    print(f"  {'-' * 14} {'-' * 28} {'-' * 22}")
    for m in TRACKED_MODES:
        mn, hn = _tensor_union_size(mlp_unions[m]), _tensor_union_size(head_unions[m])
        print(
            f"  {m:<14} "
            f"{mn:>7,d} / {total_mlp:<7,d} ({mn / total_mlp:>6.2%})   "
            f"{hn:>3d} / {total_heads:<3d} ({hn / total_heads:>6.2%})"
        )
    print(f"{'=' * 72}")
    print(f"  Results: {args.output}")
    profiler.close()


def _save(
    args,
    ds,
    examples,
    n_done,
    mlp_unions,
    head_unions,
    n_layers,
    total_mlp,
    total_heads,
):
    try:
        import numpy as np

        npz_path = Path(args.output).with_suffix(".full.npz")
        to_save = {}
        for mode, layer_map in mlp_unions.items():
            mpairs = _layer_union_pairs(layer_map)
            if mpairs:
                to_save[f"mlp_{mode}"] = np.array(mpairs, dtype=np.int32)
            hpairs = _layer_union_pairs(head_unions[mode])
            if hpairs:
                to_save[f"heads_{mode}"] = np.array(hpairs, dtype=np.int32)
        if to_save:
            np.savez_compressed(npz_path, **to_save)
    except Exception as e:
        print(f"  [!] union npz sidecar save failed: {e}")

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "n_examples_processed": n_done,
        "n_examples_total": len(examples),
        "total_mlp_neurons": total_mlp,
        "total_attention_heads": total_heads,
        "by_threshold": {},
    }
    for mode, layer_map in mlp_unions.items():
        head_map = head_unions[mode]
        mn = _tensor_union_size(layer_map)
        hn = _tensor_union_size(head_map)
        out["by_threshold"][mode] = {
            "mlp_neurons_activated_union": mn,
            "mlp_fraction": mn / total_mlp,
            "attention_heads_activated_union": hn,
            "attention_head_fraction": hn / total_heads,
            "mlp_per_layer": {
                str(l): int(layer_map[l].sum().item()) for l in range(n_layers)
            },
            "heads_per_layer": {
                str(l): int(head_map[l].sum().item()) for l in range(n_layers)
            },
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
