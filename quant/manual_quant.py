#!/usr/bin/env python3
"""Manual per-module NF4 quantization — bypasses transformers' broken
`llm_int8_skip_modules` so we can ACTUALLY target attention-only / mlp-only.

transformers 5.12 / bnb 0.49 silently ignore the skip list (verified: every
module stays Linear4bit). Here we load bf16 and hand-replace exactly the target
`nn.Linear`s with `bnb.nn.Linear4bit` (NF4 + double-quant), leaving the rest in
bf16. Reports the real GPU weight footprint per config (the cost-savings number).

Substrate (b007 adapter + MACE-90 mask) is applied in every run, as before.

Usage (pod, .venv):
  python manual_quant.py --target attn --eval --report reports/attn_only_nf4.json
  python manual_quant.py --target both --measure-only   # just the memory number
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
DEF_MASK = (
    ART
    / "issue12_recursive_coactivation_mace_v1/runs/issue12_recursive_coactivation_mace"
    / "mace90_v13_java500_shrink_pressure_rebuild_tf4576/candidate_masks"
    / "category_repair_java_r500_protect_tail_b140875_p10000.npz"
)
DEF_PAIRS = ART / "issue12_recursive_coactivation_mace_v1/data/bfcl_single_call/pairs.jsonl"
DEF_TOPK = 140875

# substrings selecting which decoder projections to NF4-quantize
TARGET_SUBSTR = {
    "attn": ["self_attn"],
    "mlp": ["mlp"],
    "both": ["self_attn", "mlp"],
    "none": [],
}


def manual_nf4(model, include, compute_dtype):
    """Replace target nn.Linear with bnb.nn.Linear4bit (NF4). Returns count."""
    import bitsandbytes as bnb
    import torch.nn as nn

    names = [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and any(s in n for s in include)
    ]
    for name in names:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[-1]
        lin = getattr(parent, child)
        new = bnb.nn.Linear4bit(
            lin.in_features, lin.out_features, bias=lin.bias is not None,
            compute_dtype=compute_dtype, quant_type="nf4", compress_statistics=True,
        )
        new.weight = bnb.nn.Params4bit(
            lin.weight.data.to("cpu"), requires_grad=False, quant_type="nf4", compress_statistics=True
        )
        if lin.bias is not None:
            new.bias = nn.Parameter(lin.bias.data.clone())
        new = new.to(lin.weight.device)  # .to(cuda) triggers the NF4 quantization
        setattr(parent, child, new)
        del lin
    return len(names)


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="attn", choices=list(TARGET_SUBSTR))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", type=Path, default=DEF_ADAPTER)
    ap.add_argument("--mask", type=Path, default=DEF_MASK)
    ap.add_argument("--topk", type=int, default=DEF_TOPK)
    ap.add_argument("--pairs", type=Path, default=DEF_PAIRS)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    torch.cuda.reset_peak_memory_stats()
    print(f"[load] bf16 base, then manual NF4 on target={args.target}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda", attn_implementation="eager"
    )
    n_quant = manual_nf4(model, TARGET_SUBSTR[args.target], dtype)
    torch.cuda.synchronize()
    weight_gb = round(torch.cuda.memory_allocated() / 1e9, 2)
    # verify it actually applied
    import torch.nn as nn
    dp = type(model.model.layers[5].mlp.down_proj).__name__
    qp = type(model.model.layers[5].self_attn.q_proj).__name__
    print(f"[quant] target={args.target} quantized_linears={n_quant} | "
          f"weight_footprint={weight_gb} GB | layer5: mlp.down={dp} attn.q={qp}", flush=True)

    summary = {"target": args.target, "quantized_linears": n_quant, "weight_footprint_gb": weight_gb,
               "layer5_mlp_down": dp, "layer5_attn_q": qp}

    if not args.measure_only:
        if not args.no_adapter:
            model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        selected = bfcl.load_topk_mask(args.mask, args.topk)
        hooks = bfcl.install_mlp_keep_hooks(model, selected)
        print(f"[mask] kept={sum(len(v) for v in selected.values())}", flush=True)

        run = None
        if args.wandb and args.eval:
            key = os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key")
            try:
                import wandb
                if key:
                    wandb.login(key=key)
                run = wandb.init(entity=os.environ.get("WANDB_ENTITY") or "krishnapg2315",
                                 project=os.environ.get("WANDB_PROJECT", "prism-bfcl"),
                                 group=os.environ.get("WANDB_GROUP", "qwen-substrate-quant"),
                                 name=f"manual-nf4-{args.target}", job_type="manual-quant",
                                 config={"target": args.target, "weight_gb": weight_gb})
            except Exception as e:
                print(f"[wandb] disabled ({e})", flush=True)

        if args.eval:
            eargs = SimpleNamespace(method=f"nf4-{args.target}", target=args.target, topk=args.topk,
                                    pairs=args.pairs, limit=args.limit, batch_size=args.batch_size,
                                    max_new_tokens=args.max_new_tokens)
            ev = qs.evaluate(model, tokenizer, eargs)
            summary.update(ev)
            print(json.dumps(ev, indent=2), flush=True)
            if run is not None:
                run.summary.update(summary)
                run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        for h in hooks:
            h.remove()
        if run is not None:
            run.finish()

    print("[SUMMARY] " + json.dumps(summary), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
