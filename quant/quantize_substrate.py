#!/usr/bin/env python3
"""Quantize the best-substrate Qwen (b007 + issue12 MACE-90 mask) — basic SOTA path.

Substrate (issue #3/#4 context):
    Qwen/Qwen3-8B
    + b007 r32 rsLoRA adapter  (epsilon_repair, issue #6 tree search)
    + issue #12 v13 MACE-90 kept-channel MLP mask
      (category_repair_java_r500_protect_tail_b140875_p10000.npz, topk=140875,
       ~31.85% of 442,368 MLP channels, score 600/664 = 90.4% recovery)

The intervention path is *identical* to tokenbender's `bfcl_direct_qwen3.py
eval-mask` (we import its helpers, not reimplement): keep-only hook on each
`mlp.down_proj` input, b007 adapter on top. The only addition here is a
weight-quantization stage on the base model.

Quant backends (survey issue #4 shortlist):
    nf4    bitsandbytes NF4 4-bit + double-quant  (QLoRA-style; LoRA stays bf16)  [default]
    int8   bitsandbytes LLM.int8()  (W8A8 outlier-aware)
    int4wo torchao Int4WeightOnly  (PyTorch-native, Marlin-friendly)
    int8wo torchao Int8WeightOnly
    none   bf16 baseline (sanity)

Order is QLoRA-correct: quantize the BASE, then attach the LoRA, then hooks.

Usage (on the pod, inside .venv):
    python quantize_substrate.py --method nf4 --limit 64 --eval
    python quantize_substrate.py --method int4wo --eval            # full 1007
    python quantize_substrate.py --method nf4 --save out/qwen8b-b007-mace90-nf4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Reuse tokenbender's canonical helpers verbatim (mask + intervention + scoring).
_spec = importlib.util.spec_from_file_location("bfcl_direct_qwen3", SCRIPTS / "bfcl_direct_qwen3.py")
bfcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bfcl)  # type: ignore[union-attr]

# Default artifact layout produced by download_artifacts.py --mode full.
ART = Path("/workspace/qwen-quant/artifacts/bfcl")
DEF_ADAPTER = ART / "issue6_tree_search_v1/run/branches/b007/unmasked_r32/adapter"
DEF_MASK = (
    ART
    / "issue12_recursive_coactivation_mace_v1/runs/issue12_recursive_coactivation_mace"
    / "mace90_v13_java500_shrink_pressure_rebuild_tf4576/candidate_masks"
    / "category_repair_java_r500_protect_tail_b140875_p10000.npz"
)
DEF_PAIRS = ART / "issue12_recursive_coactivation_mace_v1/data/bfcl_single_call/pairs.jsonl"
DEF_TOPK = 140875  # v13 MACE-90 kept-channel budget


# Which decoder submodules each --target touches. We stage quant attention-first
# (issue #4): quantize self_attn projections, leave the MLP substrate in bf16,
# then quantize MLP as a later stage.
TARGET_MODULES = {
    "attn": ["self_attn"],
    "mlp": ["mlp"],
    "both": ["self_attn", "mlp"],
}


def _fqn_in_target(fqn: str, target: str) -> bool:
    return any(tok in fqn for tok in TARGET_MODULES[target])


_GPTQ_KEEPALIVE = []  # holds GPTQModel wrappers so they aren't GC'd (not nn submodules)


def build_quantized_base(method: str, model_name: str, dtype_str: str, target: str, gptq_path=None):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, dtype_str)
    common = dict(attn_implementation="eager")
    # bitsandbytes can only *exclude* modules from quant -> skip the complement.
    skip = [] if target == "both" else (["mlp"] if target == "attn" else ["self_attn"])

    if method == "gptq":
        # full-model 4-bit GPTQ checkpoint from gptq_quantize.py (calibrated,
        # leak-gated). Force the Triton backend: Marlin needs nvcc/CUDA_HOME to
        # JIT-build a C++ kernel (absent here) and lacks sm_120 support; Triton
        # compiles at runtime, no CUDA toolkit needed -> works on Blackwell.
        if not gptq_path:
            raise ValueError("--method gptq requires --gptq-path")
        from gptqmodel import BACKEND, GPTQModel

        gm = GPTQModel.load(str(gptq_path), backend=BACKEND.TRITON, device_map="auto")
        inner = getattr(gm, "model", gm)
        _GPTQ_KEEPALIVE.append(gm)  # keep wrapper alive WITHOUT making it a submodule
        return inner
    if method in ("nf4", "int8"):
        from transformers import BitsAndBytesConfig

        if method == "nf4":
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
                llm_int8_skip_modules=skip or None,
            )
        else:
            qcfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip or None)
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=qcfg, device_map="auto", **common
        )
    if method in ("int4wo", "int8wo", "none"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cuda", **common
        )
        if method != "none":
            from torchao.quantization import (
                Int4WeightOnlyConfig,
                Int8WeightOnlyConfig,
                quantize_,
            )

            cfg = Int4WeightOnlyConfig() if method == "int4wo" else Int8WeightOnlyConfig()
            # Quantize only the target projections; leave embeddings / lm_head / complement.
            quantize_(
                model,
                cfg,
                filter_fn=lambda m, fqn: m.__class__.__name__ == "Linear"
                and _fqn_in_target(fqn, target),
            )
        return model
    raise ValueError(f"unknown method: {method}")


def load_substrate(args):
    import torch
    from transformers import AutoTokenizer

    print(f"[load] base={args.model} method={args.method} target={args.target} dtype={args.dtype}", flush=True)
    t0 = time.time()
    model = build_quantized_base(args.method, args.model, args.dtype, args.target, getattr(args, "gptq_path", None))

    if args.adapter:
        from peft import PeftModel

        print(f"[load] adapter={args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    hooks = []
    if args.topk and args.mask:
        selected = bfcl.load_topk_mask(args.mask, args.topk)
        kept = sum(len(v) for v in selected.values())
        print(f"[mask] topk={args.topk} kept_channels={kept} layers={len(selected)}", flush=True)
        hooks = bfcl.install_mlp_keep_hooks(model, selected)

    # rough footprint
    try:
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[load] done in {time.time()-t0:.1f}s, peak {mem:.2f} GB", flush=True)
    except Exception:
        pass
    return model, tokenizer, hooks


def evaluate(model, tokenizer, args) -> dict:
    import torch

    rows = bfcl.read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    out_rows = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        enc_items = [
            tokenizer.apply_chat_template(
                bfcl.messages_for_generation(row, bfcl_canonicalization_prompt=True),
                tools=row.get("tools") or None,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        enc = tokenizer.pad(enc_items, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        plen = enc["input_ids"].shape[-1]
        for row, seq in zip(batch, output):
            text = tokenizer.decode(seq[plen:], skip_special_tokens=True)
            pred = bfcl.parse_tool_calls(text)
            out_rows.append(
                {
                    "raw_correct": bfcl.prediction_ok(pred, row),
                    "normalized_correct": bfcl.normalized_prediction_ok(pred, row),
                }
            )
        print(f"[eval] {len(out_rows)}/{len(rows)}", flush=True)

    judged = len(out_rows)
    norm = sum(int(r["normalized_correct"]) for r in out_rows)
    raw = sum(int(r["raw_correct"]) for r in out_rows)
    full_set = judged == 1007  # recovery vs the 664 anchor only meaningful on full eval
    return {
        "method": args.method,
        "examples": judged,
        "target": args.target,
        "normalized_exact_correct": norm,
        "normalized_exact_accuracy": norm / judged if judged else None,
        "raw_exact_correct": raw,
        "raw_exact_accuracy": raw / judged if judged else None,
        "recovery_vs_full_anchor": (norm / 664) if full_set else None,
        "full_anchor": 664,
        "full_set": full_set,
        "topk": args.topk,
        "elapsed_s": round(time.time() - t0, 1),
    }


def init_wandb(args):
    """Start a wandb run from .env keys; returns the run or None on failure/disabled."""
    if not args.wandb:
        return None
    key = os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key")
    try:
        import wandb

        if key:
            wandb.login(key=key)
        run = wandb.init(
            # the API key's default entity is a team without write access;
            # log to the personal entity explicitly.
            entity=os.environ.get("WANDB_ENTITY") or "krishnapg2315",
            project=os.environ.get("WANDB_PROJECT", "prism-bfcl"),
            group=os.environ.get("WANDB_GROUP", "qwen-substrate-quant"),
            name=f"quant-{args.target}-{args.method}" + (f"-limit{args.limit}" if args.limit else "-full"),
            job_type="quantize-eval",
            config={
                "method": args.method,
                "target": args.target,
                "model": args.model,
                "adapter": str(args.adapter) if args.adapter else None,
                "mask": str(args.mask) if args.topk else None,
                "topk": args.topk,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "limit": args.limit or 1007,
                "substrate": "qwen3-8b+b007+issue12_v13_mace90",
            },
        )
        print(f"[wandb] logging to {run.url}", flush=True)
        return run
    except Exception as e:  # never let logging break the eval
        print(f"[wandb] disabled ({e})", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="nf4", choices=["nf4", "int8", "int4wo", "int8wo", "gptq", "none"])
    ap.add_argument("--target", default="attn", choices=["attn", "mlp", "both"],
                    help="which projections to quantize (attention-first; MLP later). N/A for gptq (full-model)")
    ap.add_argument("--gptq-path", type=Path, help="path to a GPTQ checkpoint (--method gptq)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", type=Path, default=DEF_ADAPTER)
    ap.add_argument("--mask", type=Path, default=DEF_MASK)
    ap.add_argument("--topk", type=int, default=DEF_TOPK)
    ap.add_argument("--pairs", type=Path, default=DEF_PAIRS)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 = full 1007")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                    help="log run to wandb (keys from .env); --no-wandb to disable")
    ap.add_argument("--save", type=Path, help="save quantized model + tokenizer here")
    ap.add_argument("--report", type=Path, help="write eval summary json here")
    args = ap.parse_args()
    if args.no_adapter:
        args.adapter = None
    if args.no_mask:
        args.topk = 0

    run = init_wandb(args) if args.eval else None
    model, tokenizer, hooks = load_substrate(args)
    try:
        if args.eval:
            summary = evaluate(model, tokenizer, args)
            print(json.dumps(summary, indent=2))
            if run is not None:
                run.summary.update(summary)
                run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(summary, indent=2))
    finally:
        for h in hooks:
            h.remove()
        if run is not None:
            run.finish()

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        print(f"[save] -> {args.save}", flush=True)
        model.save_pretrained(str(args.save))
        tokenizer.save_pretrained(str(args.save))


if __name__ == "__main__":
    main()
