#!/usr/bin/env python3
"""
carry_into_thousands primitive-subtask tracer for 3-digit addition.

Traces the sub-circuit that handles the hundreds->thousands carry on
Qwen/Qwen2.5-Math-1.5B. That carry is the dominant failure mode on the
3-digit exhaustive benchmark (88.45% accuracy when the result stays 3
digits, 53.01% when it gains a fourth digit).

Design
------
Slices from the canonical 3-digit correct set:

- pos         : correct, `a+b >= 1000` (carry into thousands fires)
- neg_paired  : correct, `a+b <  1000`, token-length-matched and
                lower-digit-carry-pattern-matched to pos (so only the
                hundreds-position decision differs).

Attribution metric (stricter than the single-token logit delta used by
the 2-digit pipeline): teacher-forced, full-answer log-softmax delta

    sum_t [ log p(gold_t | prompt + gold_<t) - log p(cf_t | ...) ]

where `cf_t` is the close-digit competitor for each answer token (same
scheme as `build_attr_records`: d -> (d+1) mod 10, 9 -> 8).

Pruning check: mask every MLP neuron and attention head that is not in
the pos slice's `rel_0.01` union, then measure teacher-forced full-answer
accuracy on a held-out pos test set.

Artifacts under `--out-dir`:
  pos_union.json, pos_union.full.npz
  neg_union.json, neg_union.full.npz
  differential.json      (pos_only = pos & ~neg per mode)
  pruning_check.json     (held-out pos accuracy under pos-union mask)
  frequency.json         (per-slice firing frequency; optional)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.circuit_tracing.batching import grouped_batches
from src.circuit_tracing.profiling import add_profile_args, make_profiler
from src.circuit_tracing.relp import ReLPAttributor
from union_attribution import (
    _batched_selection_counts_multi,
    _init_union_accumulators,
    _layer_union_pairs,
    _tensor_union_size,
    _update_union_accumulator,
    TRACKED_MODES,
    attribute_mlp_and_heads_full_answer_batched,
    pick_device,
)


def lower_carries(a: int, b: int) -> tuple[int, int]:
    ones = (a % 10) + (b % 10)
    c1 = 1 if ones >= 10 else 0
    tens = ((a // 10) % 10) + ((b // 10) % 10) + c1
    c2 = 1 if tens >= 10 else 0
    return (c1, c2)


def _close_digit(c: str) -> str:
    return str((int(c) + 1) % 10) if c != "9" else "8"


def build_records(tokenizer, examples):
    digit_ids = {d: tokenizer.encode(d, add_special_tokens=False)[0] for d in "0123456789"}
    records = []
    # Qwen's natural generation after "... =" starts with a space token before
    # the first digit. Insert that space in the teacher-forced input so
    # argmax at position ans_start - 1 predicts gold[0].
    for ex in examples:
        prompt = ex["prompt"]
        ans = str(ex["answer"])
        prompt_spaced = prompt + " "
        prompt_ids = tokenizer(prompt_spaced, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_ids = tokenizer(prompt_spaced + ans, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ans_start = int(prompt_ids.shape[0])
        gold_ids = full_ids[ans_start:]
        cf_list = []
        for tid in gold_ids.tolist():
            tok_text = tokenizer.decode([tid]).strip()
            if len(tok_text) == 1 and tok_text.isdigit():
                cf_list.append(digit_ids[_close_digit(tok_text)])
            else:
                # non-digit or multi-char piece: no meaningful competitor, use the
                # same id so its logit-delta contribution is zero.
                cf_list.append(tid)
        records.append(
            {
                "input_ids": full_ids,
                "answer_start": ans_start,
                "answer_token_ids": gold_ids.clone().long(),
                "cf_token_ids": torch.tensor(cf_list, dtype=torch.long),
                "a": ex["a"],
                "b": ex["b"],
                "answer": ex["answer"],
            }
        )
    return records


def build_slices(canonical_path: str, *, n_per_slice: int, seed: int):
    """Paired slices: pos[i] and neg[i] share the same lower-carry pattern.

    The pairing matters for the CF-patching pruning check — neg[i]'s activations
    supply the "no-carry but matched-lower-carries" replacement for pos[i]'s
    pruned components.
    """
    ds = json.load(open(canonical_path))
    buckets = defaultdict(lambda: ([], []))  # lc -> (pos_list, neg_list)
    for ex in ds["examples"]:
        cit = (ex["a"] + ex["b"]) >= 1000
        lc = lower_carries(ex["a"], ex["b"])
        bpos, bneg = buckets[lc]
        (bpos if cit else bneg).append(ex)
    rng = random.Random(seed)
    usable = [lc for lc in sorted(buckets) if buckets[lc][0] and buckets[lc][1]]
    if not usable:
        raise RuntimeError("No lower-carry patterns have both pos and neg examples")
    per_lc = max(1, n_per_slice // len(usable))
    pairs: list[tuple[dict, dict]] = []
    for lc in usable:
        bpos, bneg = buckets[lc]
        p = list(bpos)
        n = list(bneg)
        rng.shuffle(p)
        rng.shuffle(n)
        k = min(per_lc, len(p), len(n))
        for i in range(k):
            pairs.append((p[i], n[i]))
    rng.shuffle(pairs)
    pairs = pairs[:n_per_slice]
    pos = [p for p, _ in pairs]
    neg = [n for _, n in pairs]
    return pos, neg


def _group_key(r):
    return (int(r["input_ids"].shape[0]), int(r["answer_token_ids"].shape[0]))


def _init_freq(n_layers, d, device):
    return {m: {l: torch.zeros(d, dtype=torch.int32, device=device) for l in range(n_layers)} for m in TRACKED_MODES}


def run_attribution(
    records,
    model,
    attributor,
    *,
    device,
    n_layers,
    d_ffn,
    num_heads,
    batch_size,
    profiler,
    label,
):
    mlp_u, head_u = _init_union_accumulators(n_layers, d_ffn, num_heads, device=device)
    mlp_f = _init_freq(n_layers, d_ffn, device)
    head_f = _init_freq(n_layers, num_heads, device)
    processed = 0
    t0 = time.time()
    for batch in grouped_batches(records, key_fn=_group_key, batch_size=batch_size):
        ids = torch.stack([r["input_ids"] for r in batch]).to(device)
        ans = torch.stack([r["answer_token_ids"] for r in batch]).to(device)
        cf = torch.stack([r["cf_token_ids"] for r in batch]).to(device)
        ans_start = int(batch[0]["answer_start"])
        with profiler.phase(f"{label}_fwd_bwd", batch_size=len(batch), seq_len=int(ids.shape[1]), ans=int(ans.shape[1])):
            mlp_a, attn_a = attribute_mlp_and_heads_full_answer_batched(
                attributor,
                model,
                ids,
                ans,
                cf,
                ans_start,
                num_heads,
                to_cpu=False,
            )
        with profiler.phase(f"{label}_reduce", batch_size=len(batch)):
            mlp_by = _batched_selection_counts_multi(mlp_a, TRACKED_MODES)
            head_by = _batched_selection_counts_multi(attn_a, TRACKED_MODES)
            for m in TRACKED_MODES:
                _update_union_accumulator(mlp_u[m], mlp_by[m])
                _update_union_accumulator(head_u[m], head_by[m])
                for l, c in mlp_by[m].items():
                    mlp_f[m][l] += c
                for l, c in head_by[m].items():
                    head_f[m][l] += c
        processed += len(batch)
        profiler.step(processed=processed)
    dt = time.time() - t0
    return {
        "mlp_u": mlp_u,
        "head_u": head_u,
        "mlp_f": mlp_f,
        "head_f": head_f,
        "n": processed,
        "sec": dt,
    }


def save_union(out_dir: Path, side: str, res, n_layers: int, d_ffn: int, num_heads: int):
    import numpy as np

    summary = {
        "slice": side,
        "n_examples": res["n"],
        "runtime_sec": res["sec"],
        "by_threshold": {},
    }
    for mode in TRACKED_MODES:
        mu = _tensor_union_size(res["mlp_u"][mode])
        hu = _tensor_union_size(res["head_u"][mode])
        summary["by_threshold"][mode] = {
            "mlp_union": mu,
            "mlp_fraction": mu / (n_layers * d_ffn),
            "heads_union": hu,
            "heads_fraction": hu / (n_layers * num_heads),
        }
    with open(out_dir / f"{side}_union.json", "w") as f:
        json.dump(summary, f, indent=2)
    npz = {}
    for mode in TRACKED_MODES:
        mp = _layer_union_pairs(res["mlp_u"][mode])
        hp = _layer_union_pairs(res["head_u"][mode])
        if mp:
            npz[f"mlp_{mode}"] = np.array(mp, dtype=np.int32)
        if hp:
            npz[f"heads_{mode}"] = np.array(hp, dtype=np.int32)
    if npz:
        np.savez_compressed(out_dir / f"{side}_union.full.npz", **npz)


def compute_differential(pos_res, neg_res, n_layers: int):
    diff = {"by_threshold": {}}
    pos_only_mlp = {m: {} for m in TRACKED_MODES}
    pos_only_head = {m: {} for m in TRACKED_MODES}
    for mode in TRACKED_MODES:
        for l in range(n_layers):
            pos_only_mlp[mode][l] = pos_res["mlp_u"][mode][l] & ~neg_res["mlp_u"][mode][l]
            pos_only_head[mode][l] = pos_res["head_u"][mode][l] & ~neg_res["head_u"][mode][l]
        pos_size = _tensor_union_size(pos_res["mlp_u"][mode])
        diff_mlp = _tensor_union_size(pos_only_mlp[mode])
        pos_head = _tensor_union_size(pos_res["head_u"][mode])
        diff_head = _tensor_union_size(pos_only_head[mode])
        diff["by_threshold"][mode] = {
            "pos_only_mlp": diff_mlp,
            "pos_only_heads": diff_head,
            "pos_only_mlp_fraction_of_pos": diff_mlp / max(pos_size, 1),
            "pos_only_heads_fraction_of_pos": diff_head / max(pos_head, 1),
            "pos_mlp": pos_size,
            "pos_heads": pos_head,
        }
    return diff, pos_only_mlp, pos_only_head


def _install_prune_hooks_zero(model, mlp_keep, head_keep, *, num_heads, dtype, device):
    hooks = []
    for i, layer in enumerate(model.model.layers):
        mlp_mask = mlp_keep[i].to(device=device, dtype=dtype)
        head_mask = head_keep[i].to(device=device, dtype=dtype)

        def _mlp_pre(mask):
            def hook(module, args):
                x = args[0]
                return (x * mask.view(1, 1, -1),) + args[1:]
            return hook

        def _attn_pre(mask, H=num_heads):
            def hook(module, args):
                x = args[0]
                b, s, d = x.shape
                per_head = x.view(b, s, H, -1)
                y = per_head * mask.view(1, 1, -1, 1)
                return (y.reshape(b, s, d),) + args[1:]
            return hook

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mlp_pre(mlp_mask)))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_attn_pre(head_mask)))
    return hooks


def build_prompt_records(tokenizer, examples):
    """Build prompt-only (prompt + " ") records for leading-digit probes.

    For 3-digit operands these are uniformly 10 tokens. The trailing space
    is the position at which the model predicts the leading answer digit
    (e.g. '1' for pos, '9'/'8'/.. for neg).
    """
    digit_ids = {d: tokenizer.encode(d, add_special_tokens=False)[0] for d in "0123456789"}
    records = []
    for ex in examples:
        text = ex["prompt"] + " "
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ans = str(ex["answer"])
        lead = digit_ids[ans[0]]
        records.append(
            {
                "input_ids": ids,
                "lead_gold": lead,
                "a": ex["a"], "b": ex["b"], "answer": ex["answer"],
            }
        )
    return records


def run_pruning_check_zero(
    model, tokenizer, examples, *,
    mlp_keep, head_keep, device, num_heads, batch_size, label,
):
    """Zero-ablation pruning; leading-digit argmax accuracy at last prompt position."""
    records = build_prompt_records(tokenizer, examples)
    dtype = next(model.parameters()).dtype
    hooks = _install_prune_hooks_zero(
        model, mlp_keep, head_keep, num_heads=num_heads, dtype=dtype, device=device
    )
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                ids = torch.stack([r["input_ids"] for r in batch]).to(device)
                lead = torch.tensor([r["lead_gold"] for r in batch], device=device)
                logits = model(ids).logits[:, -1, :]
                preds = logits.argmax(dim=-1)
                correct += int((preds == lead).sum().item())
                total += ids.shape[0]
    finally:
        for h in hooks:
            h.remove()
    return {"label": label, "n": total, "correct": correct, "accuracy": correct / max(total, 1)}


def run_fullanswer_check_cf_patch(
    model, tokenizer, target_examples, cf_examples, *,
    mlp_keep, head_keep, device, num_heads, batch_size, label,
    prompt_len: int = 10,
):
    """Full-answer teacher-forced accuracy with CF-patch restricted to the
    shared prompt positions.

    target_examples are teacher-forced as (prompt + " " + gold).
    cf_examples supply the CF activations at prompt positions [0, prompt_len).
    Positions >= prompt_len (the gold answer tokens) use target's own
    activations — the mask still fires there, but there is no CF to patch
    with (cf's sequence ends at prompt_len).

    An example counts as correct iff argmax equals gold at EVERY answer
    position (i.e. the decoded answer string matches the gold digits).

    This probes the actually-interesting question: with the carry mask
    applied, can the pruned network still do the rest of the addition
    (ones, tens, higher digits) under CF-patch?
    """
    assert len(target_examples) == len(cf_examples)
    target_records = build_records(tokenizer, target_examples)  # prompt + " " + gold
    cf_records = build_prompt_records(tokenizer, cf_examples)   # prompt + " "
    dtype = next(model.parameters()).dtype
    correct = 0
    total = 0
    # Group by (target_len, answer_len) to batch cleanly.
    idx = sorted(range(len(target_records)), key=lambda i: (int(target_records[i]["input_ids"].shape[0]), int(target_records[i]["answer_token_ids"].shape[0])))
    group = 0
    while group < len(idx):
        # scan a contiguous same-shape group
        end = group
        key = (int(target_records[idx[group]]["input_ids"].shape[0]), int(target_records[idx[group]]["answer_token_ids"].shape[0]))
        while end < len(idx) and (int(target_records[idx[end]]["input_ids"].shape[0]), int(target_records[idx[end]]["answer_token_ids"].shape[0])) == key:
            end += 1
        sub = idx[group:end]
        group = end
        for i in range(0, len(sub), batch_size):
            chunk = sub[i : i + batch_size]
            t_ids = torch.stack([target_records[k]["input_ids"] for k in chunk]).to(device)
            c_ids = torch.stack([cf_records[k]["input_ids"] for k in chunk]).to(device)
            gold = torch.stack([target_records[k]["answer_token_ids"] for k in chunk]).to(device)
            ans_start = int(target_records[chunk[0]]["answer_start"])
            A = int(gold.shape[1])
            B = t_ids.shape[0]

            cache_mlp: dict[int, torch.Tensor] = {}
            cache_attn: dict[int, torch.Tensor] = {}

            def _mk_mlp_cache(li):
                def h(mod, args):
                    cache_mlp[li] = args[0].detach().clone()
                return h

            def _mk_attn_cache(li):
                def h(mod, args):
                    cache_attn[li] = args[0].detach().clone()
                return h

            cache_hooks = []
            for li, layer in enumerate(model.model.layers):
                cache_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mk_mlp_cache(li)))
                cache_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_mk_attn_cache(li)))
            try:
                with torch.no_grad():
                    model(c_ids)
            finally:
                for h in cache_hooks:
                    h.remove()

            patch_hooks = []
            for li, layer in enumerate(model.model.layers):
                mlp_mask = mlp_keep[li].to(device=device, dtype=dtype)
                head_mask = head_keep[li].to(device=device, dtype=dtype)
                neg_mlp = cache_mlp[li]  # [B, prompt_len, d_ffn]
                neg_attn = cache_attn[li]  # [B, prompt_len, H*d_h]
                PL = min(prompt_len, int(neg_mlp.shape[1]))

                def _mlp_patch(mask, neg_act, pl=PL):
                    def h(mod, args):
                        x = args[0]
                        out = x * mask.view(1, 1, -1)
                        keep_pref = mask.view(1, 1, -1)
                        # patch prompt positions only
                        x_prompt = x[:, :pl, :]
                        patched = x_prompt * keep_pref + neg_act[:, :pl, :] * (1.0 - keep_pref)
                        out = x.clone()
                        out[:, :pl, :] = patched
                        out[:, pl:, :] = x[:, pl:, :] * mask.view(1, 1, -1)
                        return (out,) + args[1:]
                    return h

                def _attn_patch(mask, neg_act, H=num_heads, pl=PL):
                    def h(mod, args):
                        x = args[0]
                        b, s, d = x.shape
                        per = x.view(b, s, H, -1)
                        neg_per = neg_act.view(b, int(neg_act.shape[1]), H, -1)
                        km = mask.view(1, 1, -1, 1)
                        out = per.clone()
                        out[:, :pl, :, :] = per[:, :pl, :, :] * km + neg_per[:, :pl, :, :] * (1.0 - km)
                        out[:, pl:, :, :] = per[:, pl:, :, :] * km
                        return (out.reshape(b, s, d),) + args[1:]
                    return h

                patch_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mlp_patch(mlp_mask, neg_mlp)))
                patch_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_attn_patch(head_mask, neg_attn)))

            try:
                with torch.no_grad():
                    logits = model(t_ids).logits
                preds = logits[:, ans_start - 1 : ans_start - 1 + A, :].argmax(dim=-1)
                match = (preds == gold).all(dim=1)
                correct += int(match.sum().item())
                total += B
            finally:
                for h in patch_hooks:
                    h.remove()
                cache_mlp.clear()
                cache_attn.clear()

    return {"label": label, "n": total, "correct": correct, "accuracy": correct / max(total, 1)}


def run_pruning_check_cf_patch(
    model, tokenizer, pos_examples, neg_examples, *,
    mlp_keep, head_keep, device, num_heads, batch_size, label,
):
    """Activation patching from paired neg; leading-digit argmax accuracy on pos.

    For every layer l, every position s, the components of the MLP down_proj
    input (d_ffn) and o_proj input (per-head) that are NOT in the keep mask
    are replaced with their values from neg's forward pass at the same
    (layer, position). Components that ARE kept stay at their pos values.

    If the kept set is truly the causal carry circuit, pos accuracy should
    stay high. If the kept set is merely backbone plumbing, pos accuracy
    collapses because the actual carry computation now runs off neg's
    residual stream.
    """
    assert len(pos_examples) == len(neg_examples)
    pos_records = build_prompt_records(tokenizer, pos_examples)
    neg_records = build_prompt_records(tokenizer, neg_examples)
    dtype = next(model.parameters()).dtype
    n_layers = model.config.num_hidden_layers
    correct = 0
    total = 0
    for i in range(0, len(pos_records), batch_size):
        pos_b = pos_records[i : i + batch_size]
        neg_b = neg_records[i : i + batch_size]
        pos_ids = torch.stack([r["input_ids"] for r in pos_b]).to(device)
        neg_ids = torch.stack([r["input_ids"] for r in neg_b]).to(device)
        lead = torch.tensor([r["lead_gold"] for r in pos_b], device=device)
        B = pos_ids.shape[0]

        cache_mlp: dict[int, torch.Tensor] = {}
        cache_attn: dict[int, torch.Tensor] = {}

        def _mk_mlp_cache(li):
            def h(mod, args):
                cache_mlp[li] = args[0].detach().clone()
            return h

        def _mk_attn_cache(li):
            def h(mod, args):
                cache_attn[li] = args[0].detach().clone()
            return h

        cache_hooks = []
        for li, layer in enumerate(model.model.layers):
            cache_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mk_mlp_cache(li)))
            cache_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_mk_attn_cache(li)))
        try:
            with torch.no_grad():
                model(neg_ids)
        finally:
            for h in cache_hooks:
                h.remove()

        patch_hooks = []
        for li, layer in enumerate(model.model.layers):
            mlp_mask = mlp_keep[li].to(device=device, dtype=dtype)
            head_mask = head_keep[li].to(device=device, dtype=dtype)
            neg_mlp = cache_mlp[li]
            neg_attn = cache_attn[li]

            def _mlp_patch(mask, neg_act):
                def h(mod, args):
                    x = args[0]
                    keep = mask.view(1, 1, -1)
                    return (x * keep + neg_act * (1.0 - keep),) + args[1:]
                return h

            def _attn_patch(mask, neg_act, H=num_heads):
                def h(mod, args):
                    x = args[0]
                    b, s, d = x.shape
                    per = x.view(b, s, H, -1)
                    neg_per = neg_act.view(b, s, H, -1)
                    km = mask.view(1, 1, -1, 1)
                    y = per * km + neg_per * (1.0 - km)
                    return (y.reshape(b, s, d),) + args[1:]
                return h

            patch_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mlp_patch(mlp_mask, neg_mlp)))
            patch_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_attn_patch(head_mask, neg_attn)))

        try:
            with torch.no_grad():
                logits = model(pos_ids).logits[:, -1, :]
            preds = logits.argmax(dim=-1)
            correct += int((preds == lead).sum().item())
            total += B
        finally:
            for h in patch_hooks:
                h.remove()
            # release cache
            cache_mlp.clear()
            cache_attn.clear()

    return {"label": label, "n": total, "correct": correct, "accuracy": correct / max(total, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument(
        "--dataset",
        default="data/qwen25_math_1p5b_3digit_correct_addition_v1.json",
    )
    ap.add_argument(
        "--out-dir",
        default="results/qwen25_math_1p5b_3digit_carry_into_thousands_v1",
    )
    ap.add_argument("--n-per-slice", type=int, default=2000)
    ap.add_argument("--n-prune-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    add_profile_args(ap)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiler = make_profiler(args, device)

    print(f"Device: {device}   dtype: {args.dtype}")
    print(f"[slicing] {args.dataset}")
    pos_pool, neg_pool = build_slices(
        args.dataset, n_per_slice=args.n_per_slice, seed=args.seed
    )
    print(f"  pos={len(pos_pool)}  neg_paired={len(neg_pool)}")

    n_test = min(args.n_prune_test, len(pos_pool) // 4)
    pos_attr, pos_test = pos_pool[:-n_test], pos_pool[-n_test:]
    neg_attr, neg_test = neg_pool[:-n_test], neg_pool[-n_test:]
    print(f"  attribution: pos={len(pos_attr)}  neg={len(neg_attr)}")
    print(f"  held-out test: pos={len(pos_test)}  neg={len(neg_test)}")

    print(f"[model] {args.model}")
    with profiler.phase("model_load", model=args.model):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model, dtype=dtype, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )
    for p in model.parameters():
        p.requires_grad_(True)

    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    print(f"  layers={n_layers}  d_ffn={d_ffn}  heads={num_heads}")

    attributor = ReLPAttributor(model, tokenizer, device=device)

    print("[records]")
    pos_records = build_records(tokenizer, pos_attr)
    neg_records = build_records(tokenizer, neg_attr)
    print(f"  pos_records={len(pos_records)}  neg_records={len(neg_records)}")

    print("[attribution] pos")
    pos_res = run_attribution(
        pos_records, model, attributor,
        device=device, n_layers=n_layers, d_ffn=d_ffn, num_heads=num_heads,
        batch_size=args.batch_size, profiler=profiler, label="pos",
    )
    print(f"  pos done: n={pos_res['n']} in {pos_res['sec']:.1f}s ({pos_res['n']/max(pos_res['sec'],1e-6):.1f} ex/s)")
    save_union(out_dir, "pos", pos_res, n_layers, d_ffn, num_heads)

    print("[attribution] neg_paired")
    neg_res = run_attribution(
        neg_records, model, attributor,
        device=device, n_layers=n_layers, d_ffn=d_ffn, num_heads=num_heads,
        batch_size=args.batch_size, profiler=profiler, label="neg",
    )
    print(f"  neg done: n={neg_res['n']} in {neg_res['sec']:.1f}s ({neg_res['n']/max(neg_res['sec'],1e-6):.1f} ex/s)")
    save_union(out_dir, "neg", neg_res, n_layers, d_ffn, num_heads)

    print("[differential] pos_only = pos & ~neg (per mode)")
    diff, pos_only_mlp, pos_only_head = compute_differential(pos_res, neg_res, n_layers)
    with open(out_dir / "differential.json", "w") as f:
        json.dump(diff, f, indent=2)
    for m in TRACKED_MODES:
        d = diff["by_threshold"][m]
        print(
            f"  {m:>12}  pos_mlp={d['pos_mlp']:>7d}  pos_only_mlp={d['pos_only_mlp']:>7d} ({d['pos_only_mlp_fraction_of_pos']:.2%})  "
            f"pos_heads={d['pos_heads']:>3d}  pos_only_heads={d['pos_only_heads']:>3d} ({d['pos_only_heads_fraction_of_pos']:.2%})"
        )

    print("[pruning check] mask outside pos union at multiple thresholds (zero + CF-patch)")
    full_mlp = {l: torch.ones(d_ffn, dtype=torch.bool, device=device) for l in range(n_layers)}
    full_head = {l: torch.ones(num_heads, dtype=torch.bool, device=device) for l in range(n_layers)}
    # Leading-digit baseline (no mask)
    baseline_pos = run_pruning_check_zero(
        model, tokenizer, pos_test,
        mlp_keep=full_mlp, head_keep=full_head,
        device=device, num_heads=num_heads, batch_size=args.batch_size,
        label="pos_no_mask_zero",
    )
    baseline_neg = run_pruning_check_zero(
        model, tokenizer, neg_test,
        mlp_keep=full_mlp, head_keep=full_head,
        device=device, num_heads=num_heads, batch_size=args.batch_size,
        label="neg_no_mask_zero",
    )
    print(f"  baseline (leading-digit argmax): pos={baseline_pos['accuracy']:.2%}  neg={baseline_neg['accuracy']:.2%}")

    # Full-answer baselines: pos (4-digit) and neg (3-digit) teacher-forced
    # accuracy with NO mask applied — how many held-out examples does the
    # unmasked model get every answer digit right on?
    def _eval_full_answer_no_mask(examples, label):
        records = build_records(tokenizer, examples)
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(records), args.batch_size):
                batch = records[i : i + args.batch_size]
                ids = torch.stack([r["input_ids"] for r in batch]).to(device)
                gold = torch.stack([r["answer_token_ids"] for r in batch]).to(device)
                ans_start = int(batch[0]["answer_start"])
                A = int(gold.shape[1])
                logits = model(ids).logits
                preds = logits[:, ans_start - 1 : ans_start - 1 + A, :].argmax(dim=-1)
                match = (preds == gold).all(dim=1)
                correct += int(match.sum().item())
                total += int(match.shape[0])
        return {"label": label, "n": total, "correct": correct, "accuracy": correct / max(total, 1)}

    baseline_pos_full = _eval_full_answer_no_mask(pos_test, "pos_full_no_mask")
    baseline_neg_full = _eval_full_answer_no_mask(neg_test, "neg_full_no_mask")
    print(
        f"  baseline (full-answer teacher-forced): pos={baseline_pos_full['accuracy']:.2%}  neg={baseline_neg_full['accuracy']:.2%}"
    )

    prune_results = {
        "leading_digit_probe": {
            "probe": "leading_digit_argmax_at_last_prompt_position",
            "baseline": {"pos": baseline_pos, "neg": baseline_neg},
            "by_threshold": {},
        },
        "full_answer_probe": {
            "probe": "teacher_forced_all_answer_positions_argmax (CF-patch at prompt positions only)",
            "baseline": {"pos": baseline_pos_full, "neg": baseline_neg_full},
            "by_threshold": {},
        },
    }
    for mode in TRACKED_MODES:
        mlp_keep = pos_res["mlp_u"][mode]
        head_keep = pos_res["head_u"][mode]
        mlp_kept = _tensor_union_size(mlp_keep)
        head_kept = _tensor_union_size(head_keep)
        shared = {
            "mlp_kept": mlp_kept,
            "mlp_keep_fraction": mlp_kept / (n_layers * d_ffn),
            "heads_kept": head_kept,
            "heads_keep_fraction": head_kept / (n_layers * num_heads),
        }

        # Leading-digit probe
        pp_zero = run_pruning_check_zero(
            model, tokenizer, pos_test,
            mlp_keep=mlp_keep, head_keep=head_keep,
            device=device, num_heads=num_heads, batch_size=args.batch_size,
            label=f"lead_pos_{mode}_zero",
        )
        pp_cf = run_pruning_check_cf_patch(
            model, tokenizer, pos_test, neg_test,
            mlp_keep=mlp_keep, head_keep=head_keep,
            device=device, num_heads=num_heads, batch_size=args.batch_size,
            label=f"lead_pos_{mode}_cfpatch",
        )
        prune_results["leading_digit_probe"]["by_threshold"][mode] = {
            **shared,
            "pos_acc_zero_ablation": pp_zero["accuracy"],
            "pos_acc_cf_patch": pp_cf["accuracy"],
            "n_test": pp_zero["n"],
        }

        # Full-answer probe — the real "can pruned network solve 3-digit addition?" check
        full_pos = run_fullanswer_check_cf_patch(
            model, tokenizer, pos_test, neg_test,
            mlp_keep=mlp_keep, head_keep=head_keep,
            device=device, num_heads=num_heads, batch_size=args.batch_size,
            label=f"full_pos_{mode}",
        )
        full_neg = run_fullanswer_check_cf_patch(
            model, tokenizer, neg_test, pos_test,
            mlp_keep=mlp_keep, head_keep=head_keep,
            device=device, num_heads=num_heads, batch_size=args.batch_size,
            label=f"full_neg_{mode}",
        )
        prune_results["full_answer_probe"]["by_threshold"][mode] = {
            **shared,
            "pos_full_acc_cf_patch": full_pos["accuracy"],
            "neg_full_acc_cf_patch": full_neg["accuracy"],
            "n_test_pos": full_pos["n"],
            "n_test_neg": full_neg["n"],
        }
        print(
            f"  {mode:>12}  mlp_keep={mlp_kept:>7d}({mlp_kept/(n_layers*d_ffn):.1%})  "
            f"heads_keep={head_kept:>3d}({head_kept/(n_layers*num_heads):.1%})  "
            f"LEAD zero={pp_zero['accuracy']:.2%} cf={pp_cf['accuracy']:.2%}  "
            f"FULL pos_cf={full_pos['accuracy']:.2%} neg_cf={full_neg['accuracy']:.2%}"
        )

    with open(out_dir / "pruning_check.json", "w") as f:
        json.dump(prune_results, f, indent=2)

    # frequency dump (fired-ever counts per mode)
    freq = {"by_threshold": {}}
    for mode in TRACKED_MODES:
        pos_mlp_flat = torch.cat([pos_res["mlp_f"][mode][l] for l in range(n_layers)])
        pos_head_flat = torch.cat([pos_res["head_f"][mode][l] for l in range(n_layers)])
        neg_mlp_flat = torch.cat([neg_res["mlp_f"][mode][l] for l in range(n_layers)])
        neg_head_flat = torch.cat([neg_res["head_f"][mode][l] for l in range(n_layers)])
        freq["by_threshold"][mode] = {
            "pos_mlp_n_fired_ever": int((pos_mlp_flat > 0).sum().item()),
            "pos_heads_n_fired_ever": int((pos_head_flat > 0).sum().item()),
            "neg_mlp_n_fired_ever": int((neg_mlp_flat > 0).sum().item()),
            "neg_heads_n_fired_ever": int((neg_head_flat > 0).sum().item()),
        }
    with open(out_dir / "frequency.json", "w") as f:
        json.dump(freq, f, indent=2)

    profiler.close()
    print(f"[done] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
