#!/usr/bin/env python3
"""Gather-and-Aggregate (G&A) attention-head substrate for BFCL function calling.

Standalone implementation of arXiv:2602.11374 ("Retrieval-Aware Distillation for
Transformer-SSM Hybrids") applied to Qwen3-8B single-call function calling.

The paper's claim: a *specific* capability (retrieval) is carried by a tiny set of
whole attention heads -- "Gather-and-Aggregate" heads -- that you find by
capability-targeted head ablation, not by magnitude/uniform pruning. ~2% of heads
recover >95% of teacher performance.

We map BFCL single-call onto that motif:
  * gather    = pick the right tool          -> the function-name tokens of the gold call
  * aggregate = copy argument values in       -> the argument-value tokens of the gold call

Pipeline (all self-contained, no repo deps):
  1. mean-cache : mean o_proj-input activation per (layer, head) over the pairs
  2. attribute  : per-head act*grad importance, split into gather / aggregate scores
  3. ladder     : keep top-k heads, mean-ablate the rest, measure teacher-forced
                  recovery vs the full model across a k-ladder (2/5/10/25/50/100%)

Data contract (jsonl, one row per pair). Either:
  {"messages": [...chat...], "tools": [...]}   # last message = gold assistant tool call
or:
  {"prompt": "<already chat-templated text>", "target": "<tool_call>{...}</tool_call>"}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse the exact BFCL prompt/target construction AND scoring from #3 so recovery
# numbers are byte-identical and directly comparable. Pure helpers, not the atlas harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bfcl_direct_qwen3 import (  # noqa: E402
    canonical,
    encode_prompt,
    format_tool_call_target,
    messages_for_generation,
    normalized_prediction_ok,
    parse_tool_calls,
    prediction_ok,
    read_records,
    tool_call_options,
)

# Full unmasked BFCL single-call anchor (normalized exact, n=1007) from #3.
FULL_BFCL_ANCHOR = 664


# --------------------------------------------------------------------------- data


_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]*)"')
_ARGS_KEY_RE = re.compile(r'"arguments"\s*:\s*')


def _arguments_block_span(target: str) -> tuple[int, int] | None:
    """Char span of the {...} object following "arguments": via balanced-brace scan."""
    m = _ARGS_KEY_RE.search(target)
    if not m or m.end() >= len(target) or target[m.end()] != "{":
        return None
    depth, in_str, esc = 0, False, False
    for j in range(m.end(), len(target)):
        c = target[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return m.end(), j + 1
    return None


def value_char_spans(target: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Char spans of gather (function name) and aggregate (argument values) in target."""
    arg_span = _arguments_block_span(target)

    # gather = top-level "name" values (outside the arguments object)
    gather: list[tuple[int, int]] = []
    for m in _NAME_RE.finditer(target):
        if arg_span and arg_span[0] <= m.start(1) < arg_span[1]:
            continue
        gather.append((m.start(1), m.end(1)))

    # aggregate = scalar values copied into the arguments object
    aggregate: list[tuple[int, int]] = []
    if arg_span:
        s, e = arg_span
        block = target[s:e]
        try:
            args = json.loads(block)
        except json.JSONDecodeError:
            args = None
        if args is not None:
            cursor = 0
            for val in _iter_scalar_strings(args):
                start = block.find(val, cursor)
                if start >= 0:
                    aggregate.append((s + start, s + start + len(val)))
                    cursor = start + len(val)
    return gather, aggregate


def _iter_scalar_strings(obj: Any):
    if isinstance(obj, str):
        if obj:
            yield obj
    elif isinstance(obj, bool):
        yield "true" if obj else "false"
    elif isinstance(obj, (int, float)):
        yield str(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_scalar_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_scalar_strings(v)


def token_type_masks(
    tokenizer, target: str, n_target_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bool masks over target tokens: (gather_tokens, aggregate_tokens)."""
    enc = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"][:n_target_tokens]
    gather_spans, agg_spans = value_char_spans(target)

    def build(spans: list[tuple[int, int]]) -> torch.Tensor:
        mask = torch.zeros(n_target_tokens, dtype=torch.bool)
        for i, (a, b) in enumerate(offsets):
            if a == b:  # empty token
                continue
            for (s, e) in spans:
                if a < e and b > s:  # token overlaps a value span
                    mask[i] = True
                    break
        return mask

    return build(gather_spans), build(agg_spans)


# --------------------------------------------------------------------------- model


def torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def load_model(args) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation=getattr(args, "attn_impl", None) or "eager",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        if args.merge_adapter:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def decoder_layers(model):
    cur = model
    for _ in range(8):
        if hasattr(cur, "layers"):
            return list(cur.layers)
        cur = getattr(cur, "model", None) or getattr(cur, "base_model", None)
    raise AttributeError("could not find decoder .layers")


def attn_shape(model) -> tuple[int, int, int, int]:
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(cfg.num_attention_heads)
    hidden = int(cfg.hidden_size)
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
    return n_layers, n_heads, hidden, head_dim


def encode_pair(tokenizer, model, row) -> tuple[torch.Tensor, int, torch.Tensor, str]:
    prompt = encode_prompt(tokenizer, row, enable_thinking=False)
    prompt_ids = prompt["input_ids"]
    target_text = format_tool_call_target(row)
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([prompt_ids, target_ids], dim=1).to(model.device)
    return input_ids, prompt_ids.shape[1], target_ids, target_text


def gold_logprob(logits, prompt_len, target_ids, token_mask=None) -> torch.Tensor:
    """Summed teacher-forced log p(gold) over target positions (optionally a subset)."""
    n = target_ids.shape[1]
    pos = torch.arange(prompt_len - 1, prompt_len - 1 + n, device=logits.device)
    logp = torch.log_softmax(logits[:, pos, :], dim=-1)
    gold = target_ids.to(logits.device)[0].view(1, -1, 1)
    per_tok = logp.gather(2, gold).squeeze(0).squeeze(-1)  # (n,)
    if token_mask is not None:
        per_tok = per_tok[token_mask.to(per_tok.device)]
    return per_tok.sum()


def gold_token_accuracy(logits, prompt_len, target_ids, token_mask=None) -> tuple[int, int]:
    n = target_ids.shape[1]
    pos = torch.arange(prompt_len - 1, prompt_len - 1 + n, device=logits.device)
    pred = logits[:, pos, :].argmax(-1).squeeze(0)  # (n,)
    gold = target_ids.to(logits.device)[0]
    correct = pred.eq(gold)
    if token_mask is not None:
        correct = correct[token_mask.to(correct.device)]
    return int(correct.sum().item()), int(correct.numel())


# ------------------------------------------------------------------- mean caching


def build_mean_cache(model, tokenizer, rows) -> torch.Tensor:
    """Mean o_proj-input activation per (layer, hidden-channel), averaged over tokens."""
    n_layers, _, hidden, _ = attn_shape(model)
    sums = torch.zeros(n_layers, hidden, dtype=torch.float64)
    counts = torch.zeros(n_layers, dtype=torch.float64)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def mk(idx):
        def hook(_m, inp):
            captured[idx] = inp[0].detach()
        return hook

    for i, layer in enumerate(decoder_layers(model)):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(mk(i)))
    try:
        with torch.no_grad():
            for row in rows:
                captured.clear()
                input_ids, *_ = encode_pair(tokenizer, model, row)
                model(input_ids=input_ids)
                for idx, act in captured.items():
                    sums[idx] += act.double().sum(dim=(0, 1)).cpu()
                    counts[idx] += act.shape[0] * act.shape[1]
    finally:
        for h in hooks:
            h.remove()
    return (sums / counts.clamp_min(1).unsqueeze(1)).float()


# --------------------------------------------------------------------- attribution


def attribute_heads(model, tokenizer, rows, log_every=10) -> dict[str, np.ndarray]:
    """Per-head act*grad importance, split into gather / aggregate / total."""
    n_layers, n_heads, hidden, head_dim = attn_shape(model)
    scores = {
        k: torch.zeros(n_layers, n_heads, dtype=torch.float32)
        for k in ("gather", "aggregate", "total")
    }
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def mk(idx):
        def hook(_m, inp):
            act = inp[0]
            if not act.requires_grad:
                act = act.detach().requires_grad_(True)
                act.retain_grad()
                captured[idx] = act
                return (act,) + inp[1:]
            act.retain_grad()
            captured[idx] = act
        return hook

    for i, layer in enumerate(decoder_layers(model)):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(mk(i)))

    started = time.time()
    try:
        for step, row in enumerate(rows, 1):
            input_ids, prompt_len, target_ids, target = encode_pair(tokenizer, model, row)
            gather_tok, agg_tok = token_type_masks(tokenizer, target, target_ids.shape[1])
            for name, tok_mask in (("gather", gather_tok), ("aggregate", agg_tok), ("total", None)):
                if tok_mask is not None and tok_mask.sum() == 0:
                    continue
                captured.clear()
                model.zero_grad(set_to_none=True)
                logits = model(input_ids=input_ids).logits
                metric = gold_logprob(logits, prompt_len, target_ids, tok_mask)
                metric.backward()
                for idx, act in captured.items():
                    grad = act.grad if act.grad is not None else torch.zeros_like(act)
                    attr = (act * grad).detach().abs().float()
                    head_attr = attr.view(attr.shape[0], attr.shape[1], n_heads, head_dim)
                    scores[name][idx] += head_attr.sum(dim=(0, 1, 3)).cpu()
                del logits, metric
            if step % log_every == 0 or step == len(rows):
                print(f"attributed {step}/{len(rows)} elapsed_s={time.time()-started:.1f}", flush=True)
            if torch.cuda.is_available() and step % 25 == 0:
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    denom = max(len(rows), 1)
    return {k: (v / denom).numpy() for k, v in scores.items()}


# ------------------------------------------------------------- keep-only recovery


def rank_heads(head_scores: np.ndarray) -> list[tuple[int, int]]:
    """Flat (layer, head) ranking, most important first."""
    n_layers, n_heads = head_scores.shape
    order = np.argsort(head_scores.flatten())[::-1]
    return [(int(i) // n_heads, int(i) % n_heads) for i in order]


def install_keep_hooks(model, keep_heads: set[tuple[int, int]], mean_cache: torch.Tensor):
    """Keep listed heads live; mean-ablate every other head's o_proj-input block."""
    _, n_heads, hidden, head_dim = attn_shape(model)
    hooks = []

    def mk(idx):
        m = torch.zeros(hidden, dtype=torch.bool)
        for h in range(n_heads):
            if (idx, h) in keep_heads:
                m[h * head_dim : (h + 1) * head_dim] = True
        keep = m.to(model.device)
        mean = mean_cache[idx].to(model.device)

        def hook(_mod, inp):
            act = inp[0]
            return (torch.where(keep, act, mean.to(act.dtype)),) + inp[1:]
        return hook

    for i, layer in enumerate(decoder_layers(model)):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(mk(i)))
    return hooks


def gold_names(row) -> set[str]:
    names = set()
    for opt in tool_call_options(row):
        for call in opt if isinstance(opt, list) else [opt]:
            c = canonical(call)
            if isinstance(c, dict) and "name" in c:
                names.add(str(c["name"]))
    return names


def pred_name(calls) -> str | None:
    for call in calls:
        c = canonical(call)
        if isinstance(c, dict) and "name" in c:
            return str(c["name"])
    return None


def classify_failure(calls, row) -> str:
    """wrong_function vs arg_value vs parse_fail — mirrors the #3 failure buckets."""
    name = pred_name(calls)
    if name is None:
        return "parse_fail"
    if name not in gold_names(row):
        return "wrong_function"
    return "arg_value"  # right tool, wrong/misaggregated arguments


def eval_bfcl_generate(
    model, tokenizer, rows, *, keep_heads, mean_cache, batch_size, max_new_tokens
) -> dict[str, Any]:
    """Real BFCL eval under a keep-only head mask: generate -> normalized exact match."""
    tokenizer.padding_side = "left"
    hooks = install_keep_hooks(model, keep_heads, mean_cache)
    norm_ok = raw_ok = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    fails: Counter = Counter()
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            enc_items = [
                tokenizer.apply_chat_template(
                    messages_for_generation(r, bfcl_canonicalization_prompt=False),
                    tools=r.get("tools") or None,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    enable_thinking=False,
                )
                for r in batch
            ]
            enc = tokenizer.pad(enc_items, padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            plen = enc["input_ids"].shape[1]
            for r, seq in zip(batch, out):
                text = tokenizer.decode(seq[plen:], skip_special_tokens=True)
                calls = parse_tool_calls(text)
                n_ok = normalized_prediction_ok(calls, r)
                norm_ok += int(n_ok)
                raw_ok += int(prediction_ok(calls, r))
                cat = r.get("category", "?")
                by_cat[cat][0] += int(n_ok)
                by_cat[cat][1] += 1
                if not n_ok:
                    fails[classify_failure(calls, r)] += 1
    finally:
        for h in hooks:
            h.remove()
    n = len(rows)
    return {
        "normalized_exact": norm_ok,
        "raw_exact": raw_ok,
        "n": n,
        "normalized_acc": norm_ok / max(n, 1),
        "per_category": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(by_cat.items())},
        "failure_buckets": dict(fails),
    }


# ---------------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tokenizer")
    ap.add_argument("--adapter")
    ap.add_argument("--merge-adapter", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--pairs", type=Path, required=True, help="jsonl of BFCL pairs")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--attr-limit", type=int, help="cap pairs used for attribution/mean")
    ap.add_argument(
        "--keep-fracs",
        default="0.02,0.05,0.10,0.25,0.50,1.0",
        help="head-keep fractions for the recovery ladder (paper headline: 0.02)",
    )
    ap.add_argument("--select", choices=["total", "gather", "aggregate", "ga_union"], default="ga_union",
                    help="which typed ranking selects the kept heads")
    ap.add_argument("--load-attribution", type=Path,
                    help="reuse an existing gaa_head_scores.npz instead of recomputing")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--full-anchor", type=int, default=FULL_BFCL_ANCHOR,
                    help="reference full-model normalized-exact count (default 664)")
    args = ap.parse_args()

    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    attr_rows = rows[: args.attr_limit] if args.attr_limit else rows

    model, tokenizer = load_model(args)
    n_layers, n_heads, _, _ = attn_shape(model)
    total_heads = n_layers * n_heads
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model} layers={n_layers} heads/layer={n_heads} total_heads={total_heads}", flush=True)

    print("== mean cache ==", flush=True)
    mean_cache = build_mean_cache(model, tokenizer, attr_rows)

    if args.load_attribution and args.load_attribution.exists():
        print(f"== load attribution {args.load_attribution} ==", flush=True)
        d = np.load(args.load_attribution)
        scores = {k: d[f"{k}_scores"] for k in ("gather", "aggregate", "total")}
    else:
        print("== attribute (gather/aggregate/total) ==", flush=True)
        scores = attribute_heads(model, tokenizer, attr_rows)
    np.savez_compressed(
        args.output_dir / "gaa_head_scores.npz",
        head_scores=scores["total"],  # eval-ladder compatible key
        gather_scores=scores["gather"],
        aggregate_scores=scores["aggregate"],
        total_scores=scores["total"],
        n_layers=n_layers,
        n_heads=n_heads,
    )

    def selected_ranking() -> list[tuple[int, int]]:
        if args.select == "ga_union":
            # normalize each typed map, take the max -> a head that is critical for
            # EITHER gathering the function or aggregating an argument ranks high
            def norm(x):
                x = scores[x]
                return x / (x.max() + 1e-9)
            combined = np.maximum(norm("gather"), norm("aggregate"))
            return rank_heads(combined)
        return rank_heads(scores[args.select])

    ranking = selected_ranking()

    def bfcl_eval(keep_heads):
        return eval_bfcl_generate(
            model, tokenizer, rows, keep_heads=keep_heads, mean_cache=mean_cache,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        )

    print("== BFCL keep-only recovery ladder (normalized exact) ==", flush=True)
    fracs = sorted({float(x) for x in args.keep_fracs.split(",")} | {1.0})
    ladder = []
    full_norm = None
    for frac in fracs:
        k = total_heads if frac >= 1.0 else max(1, round(frac * total_heads))
        keep = set(ranking[:k])
        t0 = time.time()
        res = bfcl_eval(keep)
        if frac >= 1.0:
            full_norm = res["normalized_exact"]
        rec = {
            "keep_frac": frac,
            "kept_heads": k,
            "kept_pct": 100 * k / total_heads,
            **res,
            "recovery_vs_anchor": res["normalized_exact"] / max(args.full_anchor, 1),
            "elapsed_s": round(time.time() - t0, 1),
        }
        ladder.append(rec)
        fb = res["failure_buckets"]
        print(
            f"keep {rec['kept_pct']:5.1f}% ({k:4d}h): "
            f"norm_exact={res['normalized_exact']:4d}/{res['n']} "
            f"({100*res['normalized_acc']:5.1f}%)  recovery={rec['recovery_vs_anchor']:.3f}  "
            f"wrong_fn={fb.get('wrong_function',0)} arg_val={fb.get('arg_value',0)} "
            f"parse_fail={fb.get('parse_fail',0)}",
            flush=True,
        )

    for rec in ladder:
        rec["recovery_vs_full"] = rec["normalized_exact"] / max(full_norm or args.full_anchor, 1)

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "select": args.select,
        "pairs": len(rows),
        "attr_pairs": len(attr_rows),
        "total_heads": total_heads,
        "full_anchor_reference": args.full_anchor,
        "full_measured_normalized_exact": full_norm,
        "ladder": ladder,
        "top_heads": [{"layer": l, "head": h} for l, h in ranking[:32]],
    }
    (args.output_dir / "gaa_recovery_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
