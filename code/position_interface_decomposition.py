#!/usr/bin/env python3
"""Position-interface decomposition scaffold for 3-digit addition.

Builds tightly matched correct-vs-correct contrast sets for answer positions
(`overflow`, `hundreds`, `tens`, `ones`), traces position-specific unions with
teacher-forced attribution, runs local causal pruning checks with matched
prompt-level CF patching, and exports composed MLP masks across positions.

This is the first concrete implementation of the repo's shift from tracing
whole-task circuits toward tracing decision interfaces.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from carry_into_thousands import _install_prune_hooks_zero, run_attribution, save_union
from src.circuit_tracing.profiling import add_profile_args, make_profiler
from src.circuit_tracing.relp import ReLPAttributor
from union_attribution import (
    TRACKED_MODES,
    _layer_union_pairs,
    _tensor_union_size,
    pick_device,
)


POSITION_OFFSETS = {
    "overflow": 4,
    "hundreds": 3,
    "tens": 2,
    "ones": 1,
}


def resolve_batch_size(batch_size_arg: str, device: str) -> int:
    if batch_size_arg != "auto":
        val = int(batch_size_arg)
        if val <= 0:
            raise ValueError("--batch-size must be positive or 'auto'")
        return val
    if device != "cuda" or not torch.cuda.is_available():
        return 32
    idx = 0
    if ":" in device:
        idx = int(device.split(":", 1)[1])
    total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    if total_gb >= 70:
        return 384
    if total_gb >= 44:
        return 256
    if total_gb >= 28:
        return 192
    if total_gb >= 18:
        return 128
    if total_gb >= 12:
        return 64
    return 32


def carry_positions(a: int, b: int) -> tuple[int, ...]:
    carries = []
    carry = 0
    pos = 1
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + carry
        carry_out = 1 if s >= 10 else 0
        if carry_out:
            carries.append(pos)
        a //= 10
        b //= 10
        carry = carry_out
        pos += 1
    if carry:
        carries.append(pos)
    return tuple(carries)


def carry_signature_label(a: int, b: int) -> str:
    sig = carry_positions(a, b)
    return "none" if not sig else "carry_" + "_".join(str(v) for v in sig)


def normalize_carry_signature(value: str) -> str:
    value = value.strip()
    if value == "none":
        return value
    if value.startswith("carry_"):
        return value
    raise ValueError(f"invalid carry signature {value!r}; expected 'none' or 'carry_<positions>'")


def filter_pairs_by_target_carry(pairs: list[dict], signatures: set[str] | None) -> list[dict]:
    if not signatures:
        return pairs
    return [
        pair
        for pair in pairs
        if carry_signature_label(int(pair["target"]["a"]), int(pair["target"]["b"])) in signatures
    ]


def _position_index(answer: str, position: str) -> int | None:
    off = POSITION_OFFSETS[position]
    if len(answer) < off:
        return None
    return len(answer) - off


def _answer_without_position(answer: str, position: str) -> str | None:
    idx = _position_index(answer, position)
    if idx is None:
        return None
    chars = list(answer)
    chars[idx] = "*"
    return "".join(chars)


def _target_digit(answer: str, position: str) -> str | None:
    idx = _position_index(answer, position)
    if idx is None:
        return None
    return answer[idx]


def _serializable_example(ex: dict) -> dict:
    return {
        "a": ex["a"],
        "b": ex["b"],
        "answer": ex["answer"],
        "prompt": ex["prompt"],
        "predicted": ex.get("predicted"),
    }


def load_enriched_examples(dataset_path: str, tokenizer) -> list[dict]:
    ds = json.load(open(dataset_path))
    out = []
    for ex in ds["examples"]:
        ans = str(ex["answer"])
        prompt_token_len = len(
            tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        )
        sig = carry_positions(ex["a"], ex["b"])
        out.append(
            {
                **ex,
                "answer_str": ans,
                "result_len": len(ans),
                "prompt_token_len": prompt_token_len,
                "n_carries": len(sig),
                "carry_signature": list(sig),
            }
        )
    return out


def build_position_pairs(
    examples: list[dict],
    *,
    position: str,
    n_per_position: int,
    seed: int,
    match_on_carries: bool,
    two_digit_leading_carry: bool = False,
) -> tuple[list[dict], dict]:
    if two_digit_leading_carry and position == "hundreds":
        return build_two_digit_leading_carry_pairs(
            examples,
            n_per_position=n_per_position,
            seed=seed,
            match_on_carries=match_on_carries,
        )
    if position == "overflow":
        return build_overflow_pairs(
            examples,
            n_per_position=n_per_position,
            seed=seed,
            match_on_carries=match_on_carries,
        )

    rng = random.Random(seed)
    buckets: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    usable = 0
    for ex in examples:
        answer = ex["answer_str"]
        target_digit = _target_digit(answer, position)
        answer_wo = _answer_without_position(answer, position)
        if target_digit is None or answer_wo is None:
            continue
        key = (ex["result_len"], ex["prompt_token_len"], answer_wo)
        if match_on_carries:
            key += (ex["n_carries"], tuple(ex["carry_signature"]))
        buckets[key][target_digit].append(ex)
        usable += 1

    pairs = []
    multi_digit_buckets = 0
    for key, by_digit in buckets.items():
        live = {d: list(v) for d, v in by_digit.items() if v}
        if len(live) < 2:
            continue
        multi_digit_buckets += 1
        answer_wo = key[2]
        for vals in live.values():
            rng.shuffle(vals)
        for digit, vals in live.items():
            others = [rec for od, recs in live.items() if od != digit for rec in recs]
            if not others:
                continue
            for rec in vals:
                cf = others[rng.randrange(len(others))]
                pairs.append(
                    {
                        "position": position,
                        "target": _serializable_example(rec),
                        "cf": _serializable_example(cf),
                        "target_digit": digit,
                        "cf_digit": _target_digit(cf["answer_str"], position),
                        "match": {
                            "result_len": rec["result_len"],
                            "prompt_token_len": rec["prompt_token_len"],
                            "answer_without_position": answer_wo,
                            "n_carries": rec["n_carries"],
                            "carry_signature": rec["carry_signature"],
                        },
                    }
                )

    rng.shuffle(pairs)
    if n_per_position is not None:
        pairs = pairs[:n_per_position]

    stats = {
        "position": position,
        "n_examples_considered": usable,
        "n_match_buckets": len(buckets),
        "n_multi_digit_buckets": multi_digit_buckets,
        "n_pairs": len(pairs),
        "match_on_carries": match_on_carries,
    }
    return pairs, stats


def build_two_digit_leading_carry_pairs(
    examples: list[dict],
    *,
    n_per_position: int,
    seed: int,
    match_on_carries: bool,
) -> tuple[list[dict], dict]:
    """Pair 2-digit overflow answers `1XY` against no-overflow answers `XY`.

    For 2-digit addition, the leading carry is the hundreds answer position, not
    the 3-digit task's thousands/overflow position. The attribution target is the
    first token of the overflow answer, with the no-overflow first suffix digit as
    counterfactual token.
    """
    rng = random.Random(seed)
    buckets: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    usable = 0
    for ex in examples:
        answer = ex["answer_str"]
        if len(answer) == 3:
            suffix = answer[1:]
            side = "overflow"
        elif len(answer) == 2:
            suffix = answer
            side = "no_overflow"
        else:
            continue
        key = (ex["prompt_token_len"], suffix)
        if match_on_carries:
            key += (_lower_carry_signature(ex),)
        buckets[key][side].append(ex)
        usable += 1

    pairs = []
    contrast_buckets = 0
    for key, by_side in buckets.items():
        overflow = list(by_side.get("overflow", []))
        no_overflow = list(by_side.get("no_overflow", []))
        if not overflow or not no_overflow:
            continue
        contrast_buckets += 1
        rng.shuffle(overflow)
        rng.shuffle(no_overflow)
        for rec in overflow:
            cf = no_overflow[rng.randrange(len(no_overflow))]
            pairs.append(
                {
                    "position": "hundreds",
                    "target": _serializable_example(rec),
                    "cf": _serializable_example(cf),
                    "target_digit": str(rec["answer_str"])[0],
                    "cf_digit": str(cf["answer_str"])[0],
                    "match": {
                        "prompt_token_len": rec["prompt_token_len"],
                        "answer_suffix": str(rec["answer_str"])[1:],
                        "n_carries": rec["n_carries"],
                        "carry_signature": rec["carry_signature"],
                    },
                }
            )

    rng.shuffle(pairs)
    if n_per_position is not None:
        pairs = pairs[:n_per_position]

    stats = {
        "position": "hundreds",
        "kind": "two_digit_leading_carry",
        "n_examples_considered": usable,
        "n_match_buckets": len(buckets),
        "n_multi_digit_buckets": contrast_buckets,
        "n_pairs": len(pairs),
        "match_on_carries": match_on_carries,
    }
    return pairs, stats


def _lower_carry_signature(ex: dict) -> tuple[int, ...]:
    return tuple(p for p in ex["carry_signature"] if p < 3)


def build_overflow_pairs(
    examples: list[dict],
    *,
    n_per_position: int,
    seed: int,
    match_on_carries: bool,
) -> tuple[list[dict], dict]:
    """Pair overflow answers `1XYZ` against no-overflow answers `XYZ`.

    The ordinary digit-position matcher cannot work for overflow because the
    leading overflow digit is always `1` in 3-digit addition. The meaningful
    contrast is whether the first generated answer token should be `1` or the
    first digit of the no-overflow suffix.
    """
    rng = random.Random(seed)
    buckets: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    usable = 0
    for ex in examples:
        answer = ex["answer_str"]
        if len(answer) == 4:
            suffix = answer[1:]
            side = "overflow"
        elif len(answer) == 3:
            suffix = answer
            side = "no_overflow"
        else:
            continue
        key = (ex["prompt_token_len"], suffix)
        if match_on_carries:
            key += (_lower_carry_signature(ex),)
        buckets[key][side].append(ex)
        usable += 1

    pairs = []
    multi_digit_buckets = 0
    for key, sides in buckets.items():
        overflow = list(sides.get("overflow", []))
        no_overflow = list(sides.get("no_overflow", []))
        if not overflow or not no_overflow:
            continue
        multi_digit_buckets += 1
        rng.shuffle(overflow)
        rng.shuffle(no_overflow)
        suffix = key[1]
        answer_wo = f"*{suffix}"
        for rec in overflow:
            cf = no_overflow[rng.randrange(len(no_overflow))]
            pairs.append(
                {
                    "position": "overflow",
                    "target": _serializable_example(rec),
                    "cf": _serializable_example(cf),
                    "target_digit": rec["answer_str"][0],
                    "cf_digit": cf["answer_str"][0],
                    "match": {
                        "result_len": rec["result_len"],
                        "cf_result_len": cf["result_len"],
                        "prompt_token_len": rec["prompt_token_len"],
                        "answer_without_position": answer_wo,
                        "overflow_suffix": suffix,
                        "lower_carry_signature": list(_lower_carry_signature(rec)),
                    },
                }
            )

    rng.shuffle(pairs)
    if n_per_position is not None:
        pairs = pairs[:n_per_position]

    stats = {
        "position": "overflow",
        "n_examples_considered": usable,
        "n_match_buckets": len(buckets),
        "n_multi_digit_buckets": multi_digit_buckets,
        "n_pairs": len(pairs),
        "match_on_carries": match_on_carries,
    }
    return pairs, stats


def split_pairs(pairs: list[dict], n_prune_test: int) -> tuple[list[dict], list[dict]]:
    if not pairs:
        return [], []
    n_test = min(n_prune_test, max(len(pairs) // 4, 1))
    if len(pairs) <= n_test:
        return pairs, []
    return pairs[:-n_test], pairs[-n_test:]


def build_position_attr_records(tokenizer, pairs: list[dict], position: str):
    digit_ids = {d: tokenizer.encode(d, add_special_tokens=False)[0] for d in "0123456789"}
    records = []
    skipped = 0
    for pair in pairs:
        ex = pair["target"]
        ans = str(ex["answer"])
        target_idx = _position_index(ans, position)
        if target_idx is None:
            skipped += 1
            continue
        prompt_spaced = ex["prompt"] + " "
        prompt_ids = tokenizer(
            prompt_spaced, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        full_ids = tokenizer(
            prompt_spaced + ans, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        ans_start = int(prompt_ids.shape[0])
        gold_ids = full_ids[ans_start:]
        gold_text = [tokenizer.decode([tid]).strip() for tid in gold_ids.tolist()]
        if len(gold_text) != len(ans) or any(len(tok) != 1 or not tok.isdigit() for tok in gold_text):
            skipped += 1
            continue
        cf_digit = pair["cf_digit"]
        if cf_digit is None:
            skipped += 1
            continue
        cf_ids = gold_ids.clone().long()
        cf_ids[target_idx] = digit_ids[cf_digit]
        records.append(
            {
                "input_ids": full_ids,
                "answer_start": ans_start,
                "answer_token_ids": gold_ids.clone().long(),
                "cf_token_ids": cf_ids,
                "target_index": target_idx,
                "target_digit": pair["target_digit"],
                "cf_digit": cf_digit,
            }
        )
    return records, skipped


def build_position_eval_records(tokenizer, pairs: list[dict], position: str):
    records = []
    skipped = 0
    for pair in pairs:
        ex = pair["target"]
        cf = pair["cf"]
        ans = str(ex["answer"])
        target_idx = _position_index(ans, position)
        if target_idx is None:
            skipped += 1
            continue
        prompt_spaced = ex["prompt"] + " "
        full_ids = tokenizer(
            prompt_spaced + ans, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        prompt_ids = tokenizer(
            prompt_spaced, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        cf_prompt_ids = tokenizer(
            cf["prompt"] + " ", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        ans_start = int(prompt_ids.shape[0])
        gold_ids = full_ids[ans_start:]
        gold_text = [tokenizer.decode([tid]).strip() for tid in gold_ids.tolist()]
        if len(gold_text) != len(ans) or any(len(tok) != 1 or not tok.isdigit() for tok in gold_text):
            skipped += 1
            continue
        if int(prompt_ids.shape[0]) != int(cf_prompt_ids.shape[0]):
            skipped += 1
            continue
        records.append(
            {
                "input_ids": full_ids,
                "cf_prompt_ids": cf_prompt_ids,
                "answer_start": ans_start,
                "answer_len": int(gold_ids.shape[0]),
                "target_index": target_idx,
                "target_token_id": int(gold_ids[target_idx].item()),
            }
        )
    return records, skipped


def run_position_check_zero(
    model,
    records,
    *,
    mlp_keep,
    head_keep,
    device,
    num_heads,
    batch_size,
    label,
): 
    dtype = next(model.parameters()).dtype
    hooks = _install_prune_hooks_zero(
        model, mlp_keep, head_keep, num_heads=num_heads, dtype=dtype, device=device
    )
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            idx = sorted(
                range(len(records)),
                key=lambda i: (
                    int(records[i]["input_ids"].shape[0]),
                    int(records[i]["target_index"]),
                ),
            )
            group = 0
            while group < len(idx):
                end = group
                key = (
                    int(records[idx[group]]["input_ids"].shape[0]),
                    int(records[idx[group]]["target_index"]),
                )
                while end < len(idx):
                    rec = records[idx[end]]
                    probe = (
                        int(rec["input_ids"].shape[0]),
                        int(rec["target_index"]),
                    )
                    if probe != key:
                        break
                    end += 1
                group_idx = idx[group:end]
                group = end
                for i in range(0, len(group_idx), batch_size):
                    chunk = group_idx[i : i + batch_size]
                    ids = torch.stack([records[k]["input_ids"] for k in chunk]).to(device)
                    target_ids = torch.tensor(
                        [records[k]["target_token_id"] for k in chunk], device=device
                    )
                    answer_start = int(records[chunk[0]]["answer_start"])
                    target_index = int(records[chunk[0]]["target_index"])
                    pos = answer_start - 1 + target_index
                    logits = model(ids).logits[:, pos, :]
                    preds = logits.argmax(dim=-1)
                    correct += int((preds == target_ids).sum().item())
                    total += ids.shape[0]
    finally:
        for h in hooks:
            h.remove()
    return {
        "label": label,
        "n": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
    }


def run_position_check_cf_patch(
    model,
    records,
    *,
    mlp_keep,
    head_keep,
    device,
    num_heads,
    batch_size,
    label,
):
    dtype = next(model.parameters()).dtype
    correct = 0
    total = 0
    idx = sorted(
        range(len(records)),
        key=lambda i: (
            int(records[i]["input_ids"].shape[0]),
            int(records[i]["cf_prompt_ids"].shape[0]),
            int(records[i]["target_index"]),
        ),
    )
    group = 0
    while group < len(idx):
        end = group
        key = (
            int(records[idx[group]]["input_ids"].shape[0]),
            int(records[idx[group]]["cf_prompt_ids"].shape[0]),
            int(records[idx[group]]["target_index"]),
        )
        while end < len(idx):
            rec = records[idx[end]]
            probe = (
                int(rec["input_ids"].shape[0]),
                int(rec["cf_prompt_ids"].shape[0]),
                int(rec["target_index"]),
            )
            if probe != key:
                break
            end += 1
        group_idx = idx[group:end]
        group = end
        for j in range(0, len(group_idx), batch_size):
            chunk = group_idx[j : j + batch_size]
            t_ids = torch.stack([records[k]["input_ids"] for k in chunk]).to(device)
            c_ids = torch.stack([records[k]["cf_prompt_ids"] for k in chunk]).to(device)
            target_ids = torch.tensor([records[k]["target_token_id"] for k in chunk], device=device)
            answer_start = int(records[chunk[0]]["answer_start"])
            target_index = int(records[chunk[0]]["target_index"])

            cache_mlp: dict[int, torch.Tensor] = {}
            cache_attn: dict[int, torch.Tensor] = {}

            def _mk_mlp_cache(li):
                def hook(mod, args):
                    cache_mlp[li] = args[0].detach().clone()

                return hook

            def _mk_attn_cache(li):
                def hook(mod, args):
                    cache_attn[li] = args[0].detach().clone()

                return hook

            cache_hooks = []
            for li, layer in enumerate(model.model.layers):
                cache_hooks.append(
                    layer.mlp.down_proj.register_forward_pre_hook(_mk_mlp_cache(li))
                )
                cache_hooks.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(_mk_attn_cache(li))
                )
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
                neg_mlp = cache_mlp[li]
                neg_attn = cache_attn[li]
                prompt_len = int(neg_mlp.shape[1])

                def _mlp_patch(mask, neg_act, pl=prompt_len):
                    def hook(mod, args):
                        x = args[0]
                        keep = mask.view(1, 1, -1)
                        out = x.clone()
                        out[:, :pl, :] = x[:, :pl, :] * keep + neg_act[:, :pl, :] * (1.0 - keep)
                        out[:, pl:, :] = x[:, pl:, :] * keep
                        return (out,) + args[1:]

                    return hook

                def _attn_patch(mask, neg_act, H=num_heads, pl=prompt_len):
                    def hook(mod, args):
                        x = args[0]
                        b, s, d = x.shape
                        per = x.view(b, s, H, -1)
                        neg_per = neg_act.view(b, int(neg_act.shape[1]), H, -1)
                        keep = mask.view(1, 1, -1, 1)
                        out = per.clone()
                        out[:, :pl, :, :] = per[:, :pl, :, :] * keep + neg_per[:, :pl, :, :] * (1.0 - keep)
                        out[:, pl:, :, :] = per[:, pl:, :, :] * keep
                        return (out.reshape(b, s, d),) + args[1:]

                    return hook

                patch_hooks.append(
                    layer.mlp.down_proj.register_forward_pre_hook(
                        _mlp_patch(mlp_mask, neg_mlp)
                    )
                )
                patch_hooks.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        _attn_patch(head_mask, neg_attn)
                    )
                )

            try:
                with torch.no_grad():
                    pos = answer_start - 1 + target_index
                    logits = model(t_ids).logits[:, pos, :]
                preds = logits.argmax(dim=-1)
                correct += int((preds == target_ids).sum().item())
                total += int(t_ids.shape[0])
            finally:
                for h in patch_hooks:
                    h.remove()
                cache_mlp.clear()
                cache_attn.clear()

    return {
        "label": label,
        "n": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
    }


def save_composed_union(
    out_dir: Path,
    composed_mlp: dict[str, dict[int, torch.Tensor]],
    composed_heads: dict[str, dict[int, torch.Tensor]],
    *,
    n_layers: int,
    d_ffn: int,
    num_heads: int,
    positions: list[str],
):
    out = {
        "positions": positions,
        "total_mlp_neurons": n_layers * d_ffn,
        "total_attention_heads": n_layers * num_heads,
        "by_threshold": {},
    }
    try:
        import numpy as np

        npz = {}
        for mode in TRACKED_MODES:
            mpairs = _layer_union_pairs(composed_mlp[mode])
            hpairs = _layer_union_pairs(composed_heads[mode])
            if mpairs:
                npz[f"mlp_{mode}"] = np.array(mpairs, dtype=np.int32)
            if hpairs:
                npz[f"heads_{mode}"] = np.array(hpairs, dtype=np.int32)
        if npz:
            np.savez_compressed(out_dir / "composed_union.full.npz", **npz)
    except Exception as e:
        print(f"  [!] composed union npz save failed: {e}")

    for mode in TRACKED_MODES:
        mn = _tensor_union_size(composed_mlp[mode])
        hn = _tensor_union_size(composed_heads[mode])
        out["by_threshold"][mode] = {
            "mlp_union": mn,
            "mlp_fraction": mn / max(n_layers * d_ffn, 1),
            "heads_union": hn,
            "heads_fraction": hn / max(n_layers * num_heads, 1),
        }
    with open(out_dir / "composed_union.json", "w") as f:
        json.dump(out, f, indent=2)


def load_union_npz(npz_path: Path, *, n_layers: int, d_ffn: int):
    import numpy as np

    out = {
        mode: {l: torch.zeros(d_ffn, dtype=torch.bool) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }
    if not npz_path.exists():
        return out
    data = np.load(npz_path)
    for mode in TRACKED_MODES:
        key = f"mlp_{mode}"
        if key not in data:
            continue
        for layer, idx in data[key].tolist():
            out[mode][int(layer)][int(idx)] = True
    return out


def build_worker_cmd(args, *, positions: list[str], out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        args.dataset,
        "--out-dir",
        str(out_dir),
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--n-per-position",
        str(args.n_per_position),
        "--n-prune-test",
        str(args.n_prune_test),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--positions",
        *positions,
    ]
    if args.max_examples is not None:
        cmd.extend(["--max-examples", str(args.max_examples)])
    if args.match_on_carries:
        cmd.append("--match-on-carries")
    if args.two_digit_leading_carry:
        cmd.append("--two-digit-leading-carry")
    if args.filter_result_len:
        cmd.extend(["--filter-result-len", args.filter_result_len])
    if args.target_carry_signature:
        cmd.extend(["--target-carry-signature", *args.target_carry_signature])
    if args.build_only:
        cmd.append("--build-only")
    return cmd


def merge_worker_outputs(args, worker_specs: list[dict]) -> None:
    out_dir = Path(args.out_dir)
    first_summary = None
    for spec in worker_specs:
        summary_path = spec["out_dir"] / "summary.json"
        if summary_path.exists():
            first_summary = json.load(open(summary_path))
            break
    if first_summary is None:
        raise RuntimeError("No worker summaries found to merge")

    merged_summary = {
        "dataset": args.dataset,
        "model": args.model,
        "positions": args.positions,
        "n_examples_dataset": first_summary.get("n_examples_dataset", 0),
        "max_examples": args.max_examples,
        "n_per_position": args.n_per_position,
        "n_prune_test": args.n_prune_test,
        "match_on_carries": args.match_on_carries,
        "two_digit_leading_carry": args.two_digit_leading_carry,
        "filter_result_len": args.filter_result_len,
        "target_carry_signature": args.target_carry_signature,
        "multi_gpu": True,
        "worker_assignments": [
            {
                "cuda_visible_devices": spec["gpu"],
                "positions": spec["positions"],
                "out_dir": str(spec["out_dir"]),
            }
            for spec in worker_specs
        ],
        "per_position": {},
    }

    if args.build_only:
        for spec in worker_specs:
            worker_summary = json.load(open(spec["out_dir"] / "summary.json"))
            merged_summary["per_position"].update(worker_summary.get("per_position", {}))
            for pos in worker_summary.get("positions", []):
                src = spec["out_dir"] / f"{pos}_pairs.json"
                if src.exists():
                    shutil.copy2(src, out_dir / src.name)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(merged_summary, f, indent=2)
        return

    n_layers = int(first_summary["n_layers"])
    d_ffn = int(first_summary["d_ffn"])
    num_heads = int(first_summary["num_heads"])
    merged_summary["n_layers"] = n_layers
    merged_summary["d_ffn"] = d_ffn
    merged_summary["num_heads"] = num_heads

    composed_mlp = {
        mode: {l: torch.zeros(d_ffn, dtype=torch.bool) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }
    full_head_by_mode = {
        mode: {l: torch.ones(num_heads, dtype=torch.bool) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }

    for spec in worker_specs:
        worker_summary = json.load(open(spec["out_dir"] / "summary.json"))
        merged_summary["per_position"].update(worker_summary.get("per_position", {}))
        for pos in worker_summary.get("positions", []):
            for suffix in ["pairs.json", "union.json", "union.full.npz", "pruning_check.json"]:
                src = spec["out_dir"] / f"{pos}_{suffix}"
                if src.exists():
                    shutil.copy2(src, out_dir / src.name)
            mlp_by_mode = load_union_npz(
                spec["out_dir"] / f"{pos}_union.full.npz",
                n_layers=n_layers,
                d_ffn=d_ffn,
            )
            for mode in TRACKED_MODES:
                for layer in range(n_layers):
                    composed_mlp[mode][layer] |= mlp_by_mode[mode][layer]

    save_composed_union(
        out_dir,
        composed_mlp,
        full_head_by_mode,
        n_layers=n_layers,
        d_ffn=d_ffn,
        num_heads=num_heads,
        positions=args.positions,
    )
    merged_summary["composed_union"] = {
        mode: {
            "mlp_union": _tensor_union_size(composed_mlp[mode]),
            "mlp_fraction": _tensor_union_size(composed_mlp[mode]) / max(n_layers * d_ffn, 1),
        }
        for mode in TRACKED_MODES
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(merged_summary, f, indent=2)


def run_multi_gpu(args) -> None:
    if args.device not in (None, "cuda"):
        raise ValueError("--multi-gpu currently supports only --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("--multi-gpu requested but CUDA is not available")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError("--multi-gpu requested but fewer than 2 GPUs are visible")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers_root = out_dir / "_workers"
    workers_root.mkdir(parents=True, exist_ok=True)

    n_workers = min(gpu_count, len(args.positions))
    position_groups = [[] for _ in range(n_workers)]
    for i, pos in enumerate(args.positions):
        position_groups[i % n_workers].append(pos)

    procs = []
    worker_specs = []
    for gpu_idx, positions in enumerate(position_groups):
        worker_out = workers_root / f"gpu{gpu_idx}"
        worker_out.mkdir(parents=True, exist_ok=True)
        cmd = build_worker_cmd(args, positions=positions, out_dir=worker_out)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((gpu_idx, positions, proc))
        worker_specs.append({"gpu": gpu_idx, "positions": positions, "out_dir": worker_out})

    failed = []
    for gpu_idx, positions, proc in procs:
        output, _ = proc.communicate()
        print(f"[worker gpu={gpu_idx} positions={','.join(positions)}]\n{output}", flush=True)
        if proc.returncode != 0:
            failed.append((gpu_idx, positions, proc.returncode))
    if failed:
        raise RuntimeError(f"multi-gpu workers failed: {failed}")

    merge_worker_outputs(args, worker_specs)
    print(f"[done] merged multi-gpu artifacts -> {args.out_dir}")


def run_single(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dataset] {args.dataset}")
    print("[pairing] loading tokenizer for prompt-length matching")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    examples = load_enriched_examples(args.dataset, tokenizer)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    filter_result_len = None
    if args.filter_result_len:
        filter_result_len = {int(x) for x in args.filter_result_len.split(",") if x.strip()}
        examples = [ex for ex in examples if ex["result_len"] in filter_result_len]
    print(f"  enriched examples={len(examples):,}")

    pair_bank = {}
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "positions": args.positions,
        "n_examples_dataset": len(examples),
        "max_examples": args.max_examples,
        "n_per_position": args.n_per_position,
        "n_prune_test": args.n_prune_test,
        "match_on_carries": args.match_on_carries,
        "filter_result_len": sorted(filter_result_len) if filter_result_len else None,
        "target_carry_signature": args.target_carry_signature,
        "per_position": {},
    }

    target_carry_signatures = None
    if args.target_carry_signature:
        target_carry_signatures = {normalize_carry_signature(v) for v in args.target_carry_signature}

    for pos in args.positions:
        pairs, stats = build_position_pairs(
            examples,
            position=pos,
            n_per_position=args.n_per_position,
            seed=args.seed,
            match_on_carries=args.match_on_carries,
            two_digit_leading_carry=args.two_digit_leading_carry,
        )
        unfiltered_pairs = len(pairs)
        pairs = filter_pairs_by_target_carry(pairs, target_carry_signatures)
        stats["n_pairs_before_target_carry_filter"] = unfiltered_pairs
        stats["target_carry_signature"] = sorted(target_carry_signatures) if target_carry_signatures else None
        pair_bank[pos] = pairs
        with open(out_dir / f"{pos}_pairs.json", "w") as f:
            json.dump({"position": pos, "stats": stats, "pairs": pairs}, f, indent=2)
        summary["per_position"][pos] = {
            **stats,
            "attr_pairs": 0,
            "test_pairs": 0,
            "attr_records": 0,
            "attr_records_skipped": 0,
            "eval_records": 0,
            "eval_records_skipped": 0,
        }
        print(f"  {pos:>9}  pairs={len(pairs):>5d}  buckets={stats['n_multi_digit_buckets']:>5d}")

    if args.build_only:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] build-only artifacts -> {out_dir}")
        return

    profiler = make_profiler(args, pick_device(args.device))
    device = pick_device(args.device)
    batch_size = resolve_batch_size(str(args.batch_size), device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    print(f"[model] {args.model}")
    print(f"  device={device} dtype={args.dtype} batch_size={batch_size}")
    with profiler.phase("model_load", model=args.model):
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
    summary["n_layers"] = n_layers
    summary["d_ffn"] = d_ffn
    summary["num_heads"] = num_heads
    print(f"  layers={n_layers} d_ffn={d_ffn} heads={num_heads}")

    attributor = ReLPAttributor(model, tokenizer, device=device)
    full_head_by_mode = {
        mode: {l: torch.ones(num_heads, dtype=torch.bool, device=device) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }
    composed_mlp = {
        mode: {l: torch.zeros(d_ffn, dtype=torch.bool, device=device) for l in range(n_layers)}
        for mode in TRACKED_MODES
    }

    for pos in args.positions:
        pairs = pair_bank[pos]
        attr_pairs, test_pairs = split_pairs(pairs, args.n_prune_test)
        summary["per_position"][pos]["attr_pairs"] = len(attr_pairs)
        summary["per_position"][pos]["test_pairs"] = len(test_pairs)
        print(f"[{pos}] attr_pairs={len(attr_pairs)} test_pairs={len(test_pairs)}")

        attr_records, attr_skipped = build_position_attr_records(tokenizer, attr_pairs, pos)
        summary["per_position"][pos]["attr_records"] = len(attr_records)
        summary["per_position"][pos]["attr_records_skipped"] = attr_skipped
        if not attr_records:
            print(f"  [!] no valid attribution records for {pos}; skipping")
            continue

        res = run_attribution(
            attr_records,
            model,
            attributor,
            device=device,
            n_layers=n_layers,
            d_ffn=d_ffn,
            num_heads=num_heads,
            batch_size=batch_size,
            profiler=profiler,
            label=pos,
        )
        save_union(out_dir, pos, res, n_layers, d_ffn, num_heads)

        for mode in TRACKED_MODES:
            for layer in range(n_layers):
                composed_mlp[mode][layer] |= res["mlp_u"][mode][layer]

        eval_records, eval_skipped = build_position_eval_records(tokenizer, test_pairs, pos)
        summary["per_position"][pos]["eval_records"] = len(eval_records)
        summary["per_position"][pos]["eval_records_skipped"] = eval_skipped
        pruning = {
            "probe": f"teacher_forced_argmax_at_{pos}_answer_position",
            "baseline": {},
            "by_threshold": {},
        }
        if eval_records:
            full_mlp = {l: torch.ones(d_ffn, dtype=torch.bool, device=device) for l in range(n_layers)}
            full_head = {l: torch.ones(num_heads, dtype=torch.bool, device=device) for l in range(n_layers)}
            baseline = run_position_check_cf_patch(
                model,
                eval_records,
                mlp_keep=full_mlp,
                head_keep=full_head,
                device=device,
                num_heads=num_heads,
                batch_size=batch_size,
                label=f"{pos}_baseline_cf",
            )
            pruning["baseline"] = baseline
            for mode in TRACKED_MODES:
                mlp_keep = res["mlp_u"][mode]
                head_keep = full_head_by_mode[mode]
                zero = run_position_check_zero(
                    model,
                    eval_records,
                    mlp_keep=mlp_keep,
                    head_keep=head_keep,
                    device=device,
                    num_heads=num_heads,
                    batch_size=batch_size,
                    label=f"{pos}_{mode}_zero",
                )
                cf = run_position_check_cf_patch(
                    model,
                    eval_records,
                    mlp_keep=mlp_keep,
                    head_keep=head_keep,
                    device=device,
                    num_heads=num_heads,
                    batch_size=batch_size,
                    label=f"{pos}_{mode}_cf",
                )
                mlp_kept = _tensor_union_size(mlp_keep)
                pruning["by_threshold"][mode] = {
                    "mlp_kept": mlp_kept,
                    "mlp_keep_fraction": mlp_kept / max(n_layers * d_ffn, 1),
                    "heads_kept": n_layers * num_heads,
                    "heads_keep_fraction": 1.0,
                    "acc_zero_ablation": zero["accuracy"],
                    "acc_cf_patch": cf["accuracy"],
                    "n_test": cf["n"],
                }
                print(
                    f"  {pos:>9} {mode:>12}  mlp={mlp_kept:>7d} "
                    f"zero={zero['accuracy']:.2%} cf={cf['accuracy']:.2%}"
                )
        with open(out_dir / f"{pos}_pruning_check.json", "w") as f:
            json.dump(pruning, f, indent=2)

    save_composed_union(
        out_dir,
        composed_mlp,
        full_head_by_mode,
        n_layers=n_layers,
        d_ffn=d_ffn,
        num_heads=num_heads,
        positions=args.positions,
    )
    summary["composed_union"] = {
        mode: {
            "mlp_union": _tensor_union_size(composed_mlp[mode]),
            "mlp_fraction": _tensor_union_size(composed_mlp[mode]) / max(n_layers * d_ffn, 1),
        }
        for mode in TRACKED_MODES
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    profiler.close()
    print(f"[done] artifacts -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="data/qwen25_math_1p5b_3digit_correct_addition_v1.json",
    )
    ap.add_argument(
        "--positions",
        nargs="*",
        default=["overflow", "hundreds", "tens", "ones"],
        choices=sorted(POSITION_OFFSETS),
    )
    ap.add_argument(
        "--out-dir",
        default="results/qwen25_math_1p5b_3digit_position_interface_v1",
    )
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--n-per-position", type=int, default=2000)
    ap.add_argument("--n-prune-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match-on-carries", action="store_true")
    ap.add_argument(
        "--two-digit-leading-carry",
        action="store_true",
        help="Treat the hundreds position as the 2-digit overflow/leading-carry token.",
    )
    ap.add_argument(
        "--filter-result-len",
        default=None,
        help="Comma-separated answer lengths to keep before building position pairs.",
    )
    ap.add_argument(
        "--target-carry-signature",
        nargs="+",
        default=None,
        help="Keep only pairs whose target example has one of these carry signatures, e.g. none carry_1 carry_2_3.",
    )
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    ap.add_argument("--batch-size", default="auto")
    add_profile_args(ap)
    args = ap.parse_args()
    if args.multi_gpu and not args.worker:
        run_multi_gpu(args)
        return
    run_single(args)


if __name__ == "__main__":
    main()
