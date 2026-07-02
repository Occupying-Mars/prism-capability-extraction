#!/usr/bin/env python3
"""Phase 1 (paper-exact Step 1): rank Qwen3 attention heads by ZERO-ABLATION importance.

arXiv:2602.11374 §4.1: "systematically ablate each attention head individually
(masking its output to zero) and measure the resulting drop in accuracy ... rank all
heads by their retrieval importance score = the magnitude of the performance drop."

Our probe is the BFCL gold tool-call continuation (teacher-forced), typed into:
  gather    = function-name tokens   (pick the right tool)
  aggregate = argument-value tokens  (copy values in)
So each head gets a (gather_drop, aggregate_drop, total_drop). The retained-head set
for the hybrid = top-k by the chosen typed drop.

Output: gaa_ablation_scores.npz + gaa_ablation_ranking.json (kept-head lists at several k).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaa_heads_qwen3 import (  # noqa: E402
    attn_shape,
    decoder_layers,
    encode_pair,
    load_model,
    token_type_masks,
)
from bfcl_direct_qwen3 import read_records  # noqa: E402


def prep_items(model, tokenizer, rows):
    """Per-pair: cpu input_ids, prompt_len, gold ids, gather/aggregate token masks."""
    items = []
    for row in rows:
        input_ids, prompt_len, target_ids, target_text = encode_pair(tokenizer, model, row)
        tlen = target_ids.shape[1]
        gmask, amask = token_type_masks(tokenizer, target_text, tlen)
        items.append({
            "ids": input_ids[0].cpu(),
            "plen": prompt_len,
            "tlen": tlen,
            "gold": target_ids[0].cpu(),
            "gmask": gmask,
            "amask": amask,
        })
    return items


def make_batches(items, pad_id, batch_size):
    """Right-pad batches (real tokens start at 0 -> correct RoPE; no left-pad shift)."""
    batches = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        Lmax = max(len(it["ids"]) for it in chunk)
        B = len(chunk)
        input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
        attn = torch.zeros((B, Lmax), dtype=torch.long)
        for i, it in enumerate(chunk):
            L = len(it["ids"])
            input_ids[i, :L] = it["ids"]
            attn[i, :L] = 1
        batches.append((chunk, input_ids, attn))
    return batches


def typed_logprob(hidden, lm_head, chunk):
    """Per-example gold logprob over gather/aggregate/all target tokens.

    Applies lm_head ONLY at the target positions (not all positions x full vocab),
    which is ~40 tokens/example vs hundreds -> far cheaper than materializing logits.
    """
    g = torch.zeros(len(chunk)); a = torch.zeros(len(chunk)); t = torch.zeros(len(chunk))
    for i, it in enumerate(chunk):
        plen, tlen = it["plen"], it["tlen"]
        h = hidden[i, plen - 1: plen - 1 + tlen, :]               # [tlen, D] predicts target
        logp = torch.log_softmax(lm_head(h).float(), dim=-1)      # [tlen, V]
        lp = logp.gather(1, it["gold"].to(logp.device).view(-1, 1)).squeeze(1)
        t[i] = lp.sum()
        gm, am = it["gmask"], it["amask"]
        g[i] = lp[gm].sum() if gm.any() else 0.0
        a[i] = lp[am].sum() if am.any() else 0.0
    return g, a, t


def head_zero_hook(layer_idx, head, head_dim, device):
    lo, hi = head * head_dim, (head + 1) * head_dim

    def hook(_m, inp):
        act = inp[0].clone()
        act[..., lo:hi] = 0.0                                     # mask this head's output to zero
        return (act,) + inp[1:]
    return hook


def measure(model, batches, hook_factory=None):
    """Sum typed logprob over all examples, optionally with one head ablated."""
    handles = []
    if hook_factory is not None:
        for i, layer in enumerate(decoder_layers(model)):
            h = hook_factory(i, layer)
            if h is not None:
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(h))
    backbone = model.model if hasattr(model, "model") else model
    g = a = t = 0.0
    try:
        with torch.no_grad():
            for chunk, input_ids, attn in batches:
                hidden = backbone(input_ids=input_ids.to(model.device),
                                  attention_mask=attn.to(model.device)).last_hidden_state
                gg, aa, tt = typed_logprob(hidden, model.lm_head, chunk)
                g += float(gg.sum()); a += float(aa.sum()); t += float(tt.sum())
    finally:
        for h in handles:
            h.remove()
    return g, a, t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tokenizer")
    ap.add_argument("--adapter")
    ap.add_argument("--merge-adapter", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ablation-pairs", type=int, default=96, help="subset used to score heads")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--attn-impl", default="sdpa", help="sdpa (fast) fine here; hook is impl-agnostic")
    ap.add_argument("--keep-ks", default="10,20,30,40,58,115", help="kept-head counts to emit")
    args = ap.parse_args()

    rows = read_records(args.pairs)[: args.ablation_pairs]
    model, tokenizer = load_model(args)
    tokenizer.padding_side = "right"
    n_layers, n_heads, hidden, head_dim = attn_shape(model)
    total = n_layers * n_heads
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} heads={total} ablation_pairs={len(rows)}", flush=True)

    items = prep_items(model, tokenizer, rows)
    batches = make_batches(items, tokenizer.pad_token_id, args.batch_size)

    base_g, base_a, base_t = measure(model, batches)
    print(f"baseline logprob: gather={base_g:.1f} aggregate={base_a:.1f} total={base_t:.1f}", flush=True)

    gather_drop = np.zeros((n_layers, n_heads), dtype=np.float32)
    agg_drop = np.zeros((n_layers, n_heads), dtype=np.float32)
    total_drop = np.zeros((n_layers, n_heads), dtype=np.float32)

    started = time.time()
    for L in range(n_layers):
        for H in range(n_heads):
            g, a, t = measure(model, batches,
                              hook_factory=lambda i, layer, _L=L, _H=H:
                              head_zero_hook(_L, _H, head_dim, model.device) if i == _L else None)
            # drop = baseline - ablated (higher = more critical)
            gather_drop[L, H] = base_g - g
            agg_drop[L, H] = base_a - a
            total_drop[L, H] = base_t - t
        done = (L + 1) * n_heads
        print(f"ablated {done}/{total} heads (layer {L}) elapsed_s={time.time()-started:.0f}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez_compressed(
        args.output_dir / "gaa_ablation_scores.npz",
        gather_drop=gather_drop, aggregate_drop=agg_drop, total_drop=total_drop,
        head_scores=total_drop, n_layers=n_layers, n_heads=n_heads,
    )

    def rank(mat):
        order = np.argsort(mat.flatten())[::-1]
        return [(int(i) // n_heads, int(i) % n_heads, float(mat.flatten()[i])) for i in order]

    rankings = {"total": rank(total_drop), "gather": rank(gather_drop), "aggregate": rank(agg_drop)}
    ks = [int(x) for x in args.keep_ks.split(",")]
    kept = {}
    for k in ks:
        # union of top heads by gather and by aggregate (G&A = either role matters)
        gset = [(l, h) for l, h, _ in rankings["gather"][:k]]
        aset = [(l, h) for l, h, _ in rankings["aggregate"][:k]]
        union = sorted(set(gset) | set(aset))
        kept[str(k)] = {"gather_top": gset, "aggregate_top": aset, "ga_union": union,
                        "union_size": len(union)}

    summary = {
        "model": args.model, "adapter": args.adapter, "total_heads": total,
        "ablation_pairs": len(rows),
        "baseline": {"gather": base_g, "aggregate": base_a, "total": base_t},
        "top32_total": [{"layer": l, "head": h, "drop": s} for l, h, s in rankings["total"][:32]],
        "top32_gather": [{"layer": l, "head": h, "drop": s} for l, h, s in rankings["gather"][:32]],
        "top32_aggregate": [{"layer": l, "head": h, "drop": s} for l, h, s in rankings["aggregate"][:32]],
        "kept_by_k": kept,
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.output_dir / "gaa_ablation_ranking.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("baseline", "top32_aggregate")}, indent=2))
    print("wrote", args.output_dir / "gaa_ablation_ranking.json", flush=True)


if __name__ == "__main__":
    main()
