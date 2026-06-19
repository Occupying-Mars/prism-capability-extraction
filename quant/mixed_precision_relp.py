#!/usr/bin/env python3
"""ReLP-driven mixed-precision quantization (the eval-aware lever that *reduces*
quant error instead of training it away).

Recovery training failed for this substrate (it drifts the tree-search-tuned
b007 adapter). The other eval-aware lever: spend bits where the task lives.
We rank the 36 decoder layers by ReLP MLP saliency (the b007 attribution =
free task-saliency oracle) and keep the **top-K most salient layers' MLP at
bf16**, NF4-4bit the rest. Attention stays bf16 here so we isolate the MLP
precision/score tradeoff. No training -> no drift. Sweep K to map score vs bits.

K=0  -> all-MLP NF4 (lower anchor)   K=36 -> all bf16 (= 599 substrate ceiling)

Usage (pod, .venv):
  python mixed_precision_relp.py --hi-layers-list 0,6,12,24 --eval-after
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bfcl = _load("bfcl_direct_qwen3", SCRIPTS / "bfcl_direct_qwen3.py")
qs = _load("quantize_substrate", HERE / "quantize_substrate.py")

ART = Path("/workspace/qwen-quant/artifacts/bfcl")
DEF_ADAPTER = ART / "issue6_tree_search_v1/run/branches/b007/unmasked_r32/adapter"
DEF_RELP = ART / "issue6_tree_search_v1/run/branches/b007/relp_full_collimated.npz"
DEF_MASK = (
    ART
    / "issue12_recursive_coactivation_mace_v1/runs/issue12_recursive_coactivation_mace"
    / "mace90_v13_java500_shrink_pressure_rebuild_tf4576/candidate_masks"
    / "category_repair_java_r500_protect_tail_b140875_p10000.npz"
)
DEF_PAIRS = ART / "issue12_recursive_coactivation_mace_v1/data/bfcl_single_call/pairs.jsonl"
DEF_TOPK = 140875


def salient_layer_order(relp_path):
    import numpy as np

    scores = np.load(relp_path)["mlp_scores"]  # (n_layers, d_ffn)
    per_layer = np.abs(scores).sum(axis=1)
    return [int(i) for i in np.argsort(-per_layer)], scores.shape[0]


def build_mixed(model_name, dtype_str, hi_layers, attn_bf16=True):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # keep top-K salient layers' MLP (and all attention) in bf16; NF4 the rest.
    skip = [f"layers.{i}.mlp" for i in hi_layers]
    if attn_bf16:
        skip.append("self_attn")
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, dtype_str),
        llm_int8_skip_modules=skip or None,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=qcfg, device_map="auto", attn_implementation="eager"
    )


def eff_mlp_bits(k, n_layers):
    return round((k * 16 + (n_layers - k) * 4) / n_layers, 2)


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--relp", type=Path, default=DEF_RELP)
    ap.add_argument("--adapter", type=Path, default=DEF_ADAPTER)
    ap.add_argument("--mask", type=Path, default=DEF_MASK)
    ap.add_argument("--topk", type=int, default=DEF_TOPK)
    ap.add_argument("--pairs", type=Path, default=DEF_PAIRS)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--hi-layers-list", default="0,6,12,24", help="comma list of K (top-K salient MLP layers kept bf16)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-after", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    order, n_layers = salient_layer_order(args.relp)
    ks = [int(x) for x in args.hi_layers_list.split(",") if x.strip() != ""]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected = bfcl.load_topk_mask(args.mask, args.topk)

    results = []
    for k in ks:
        hi = order[:k]
        print(f"\n==== K={k} hi-bf16 layers={sorted(hi)} | eff MLP bits={eff_mlp_bits(k, n_layers)} ====", flush=True)
        run = None
        if args.wandb:
            key = os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key")
            try:
                import wandb
                if key:
                    wandb.login(key=key)
                run = wandb.init(
                    entity=os.environ.get("WANDB_ENTITY") or "krishnapg2315",
                    project=os.environ.get("WANDB_PROJECT", "prism-bfcl"),
                    group=os.environ.get("WANDB_GROUP", "qwen-substrate-quant"),
                    name=f"mixprec-relp-K{k}", job_type="mixed-precision",
                    config={"hi_layers": k, "eff_mlp_bits": eff_mlp_bits(k, n_layers), "topk": args.topk},
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] disabled ({e})", flush=True)

        base = build_mixed(args.model, args.dtype, hi)
        model = PeftModel.from_pretrained(base, str(args.adapter))
        model.eval()
        hooks = bfcl.install_mlp_keep_hooks(model, selected)
        eargs = SimpleNamespace(method=f"mixprec-K{k}", target="mlp-mixed", topk=args.topk,
                                pairs=args.pairs, limit=args.limit, batch_size=args.batch_size,
                                max_new_tokens=args.max_new_tokens)
        ev = qs.evaluate(model, tokenizer, eargs)
        ev.update({"hi_layers": k, "hi_layer_ids": sorted(hi), "eff_mlp_bits": eff_mlp_bits(k, n_layers)})
        results.append(ev)
        print(json.dumps({k2: ev[k2] for k2 in ("hi_layers", "eff_mlp_bits", "normalized_exact_correct", "recovery_vs_full_anchor")}), flush=True)
        if run is not None:
            run.summary.update(ev)
            run.log({k2: v for k2, v in ev.items() if isinstance(v, (int, float))})
            run.finish()
        for h in hooks:
            h.remove()
        del model, base
        torch.cuda.empty_cache()

    summary = {"sweep": results, "n_layers": n_layers, "salient_order": order}
    print("\n=== SWEEP ===")
    for r in results:
        print(f"  K={r['hi_layers']:>2}  bits={r['eff_mlp_bits']:>5}  score={r['normalized_exact_correct']}  recovery={r.get('recovery_vs_full_anchor')}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
