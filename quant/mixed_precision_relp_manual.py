#!/usr/bin/env python3
"""ReLP-driven mixed-precision quant — CORRECTED (manual bnb.Linear4bit path).

The original `mixed_precision_relp.py` selected hi-bf16 layers via
`llm_int8_skip_modules`, which is SILENTLY IGNORED on this stack
(transformers 5.12 / bnb 0.49) -> every K collapsed to the same all-NF4 config
(the invalidated sweep). This version reuses the verified per-module replacement
from `manual_quant.py`: NF4 every attn+mlp Linear EXCEPT the top-K ReLP-salient
layers' MLP (and all attention), which stay bf16. Dtypes are asserted per run.

Rank = b007 ReLP MLP saliency (free task oracle). Keep top-K salient layers'
MLP in bf16, NF4 the rest. Sweep K to map score vs effective MLP bits.

K=0  -> all-MLP NF4, attn bf16 (lower point)   K=n_layers -> all bf16 (= ceiling)

Usage (pod, .venv):
  python mixed_precision_relp_manual.py --hi-layers-list 0,6,12,18,36 --eval-after
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
mq = _load("manual_quant", HERE / "manual_quant.py")
mp = _load("mixed_precision_relp", HERE / "mixed_precision_relp.py")

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


def build_mixed_manual(model_name, dtype, lo_layers):
    """bf16 base; manual NF4 on the lo (non-salient) layers' MLP only. Attention
    and the hi-salient layers' MLP stay bf16. Returns (model, n_quantized)."""
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cuda", attn_implementation="eager"
    )
    # ".{i}.mlp" delimiter is collision-safe ("layers.1.mlp" not in "layers.12.mlp").
    include = [f".{i}.mlp" for i in lo_layers]
    n_quant = mq.manual_nf4(model, include, dtype) if include else 0
    return model, n_quant


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
    ap.add_argument("--hi-layers-list", default="0,6,12,18,36")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-after", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    order, n_layers = mp.salient_layer_order(args.relp)
    ks = [min(int(x), n_layers) for x in args.hi_layers_list.split(",") if x.strip() != ""]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected = bfcl.load_topk_mask(args.mask, args.topk)

    results = []
    for k in ks:
        hi = sorted(order[:k])
        lo = sorted(order[k:])
        eff_bits = mp.eff_mlp_bits(k, n_layers)
        print(f"\n==== K={k} hi-bf16-MLP={hi} | NF4-MLP layers={len(lo)} | eff MLP bits={eff_bits} ====", flush=True)

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
                    name=f"mixprec-manual-K{k}", job_type="mixed-precision-manual",
                    config={"hi_layers": k, "eff_mlp_bits": eff_bits, "topk": args.topk},
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] disabled ({e})", flush=True)

        torch.cuda.reset_peak_memory_stats()
        base, n_quant = build_mixed_manual(args.model, dtype, lo)
        torch.cuda.synchronize()
        weight_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

        # ASSERT targeting actually took effect (the whole point of the rewrite).
        def _kind(layer_idx):
            return type(base.model.layers[layer_idx].mlp.down_proj).__name__
        hi_kind = _kind(hi[0]) if hi else None
        lo_kind = _kind(lo[0]) if lo else None
        print(f"[dtype-check] hi-layer{hi[0] if hi else '-'} mlp.down={hi_kind} | "
              f"lo-layer{lo[0] if lo else '-'} mlp.down={lo_kind} | quant_linears={n_quant} | {weight_gb} GB", flush=True)
        if hi and lo:
            assert hi_kind == "Linear" and lo_kind == "Linear4bit", \
                f"targeting FAILED: hi={hi_kind} lo={lo_kind}"

        model = PeftModel.from_pretrained(base, str(args.adapter))
        model.eval()
        hooks = bfcl.install_mlp_keep_hooks(model, selected)

        ev = {}
        if args.eval_after:
            eargs = SimpleNamespace(method=f"mixprec-manual-K{k}", target="mlp-mixed", topk=args.topk,
                                    pairs=args.pairs, limit=args.limit, batch_size=args.batch_size,
                                    max_new_tokens=args.max_new_tokens)
            ev = qs.evaluate(model, tokenizer, eargs)
        ev.update({"hi_layers": k, "hi_layer_ids": hi, "eff_mlp_bits": eff_bits,
                   "weight_footprint_gb": weight_gb, "quantized_linears": n_quant})
        results.append(ev)
        print("[RESULT] " + json.dumps({kk: ev.get(kk) for kk in
              ("hi_layers", "eff_mlp_bits", "weight_footprint_gb",
               "normalized_exact_correct", "recovery_vs_full_anchor")}), flush=True)
        if run is not None:
            run.summary.update(ev)
            run.log({kk: v for kk, v in ev.items() if isinstance(v, (int, float))})
            run.finish()

        for h in hooks:
            h.remove()
        del model, base
        torch.cuda.empty_cache()

    summary = {"sweep": results, "n_layers": n_layers, "salient_order": order}
    print("\n=== SWEEP ===")
    for r in results:
        print(f"  K={r['hi_layers']:>2}  bits={r['eff_mlp_bits']:>5}  "
              f"score={r.get('normalized_exact_correct')}  recovery={r.get('recovery_vs_full_anchor')}  "
              f"{r.get('weight_footprint_gb')}GB")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))
    print("MIXPREC_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
