#!/usr/bin/env python3
"""Bake the substrate into a dense bf16 checkpoint (no adapter, no runtime hook).

vLLM (our serving leg for Blackwell-prebuilt W4A16 Marlin kernels) will NOT run
the MACE keep-only forward hook, and we want the adapter folded in. So we bake
the full substrate into plain weights:

  1. merge the b007 rsLoRA adapter into Qwen3-8B  (merge_and_unload)
  2. bake the MACE-90 mask: the keep-only hook zeros every NON-kept input channel
     of mlp.down_proj. y = W @ x with x[drop]=0  ==  y = (W[:,drop]=0) @ x. So we
     zero the DROPPED input columns of each down_proj.weight — exactly equivalent.

Result: a standard Qwen3ForCausalLM that *is* the 599-substrate, quantizable by
AutoRound/GPTQ and servable by vLLM with no hooks. Sanity-gate it against 599
(via bfcl_direct_qwen3 eval, no --topk no --adapter) BEFORE quantizing.

Usage (pod, .venv):
  python bake_dense.py --out out/qwen3-8b-b007-mace90-dense
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
spec = importlib.util.spec_from_file_location("bfcl_direct_qwen3", SCRIPTS / "bfcl_direct_qwen3.py")
bfcl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfcl)

ART = Path("/workspace/qwen-quant/artifacts/bfcl")
DEF_ADAPTER = ART / "issue6_tree_search_v1/run/branches/b007/unmasked_r32/adapter"
DEF_MASK = (
    ART
    / "issue12_recursive_coactivation_mace_v1/runs/issue12_recursive_coactivation_mace"
    / "mace90_v13_java500_shrink_pressure_rebuild_tf4576/candidate_masks"
    / "category_repair_java_r500_protect_tail_b140875_p10000.npz"
)
DEF_TOPK = 140875


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", type=Path, default=DEF_ADAPTER)
    ap.add_argument("--mask", type=Path, default=DEF_MASK)
    ap.add_argument("--topk", type=int, default=DEF_TOPK)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    print(f"[bake] load {args.model} bf16", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="cpu"
    )
    print(f"[bake] merge b007 adapter {args.adapter}", flush=True)
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()

    # bake MACE-90 keep-only mask into down_proj input columns
    selected = bfcl.load_topk_mask(args.mask, args.topk)  # {layer: set(kept channels)}
    layers = bfcl.decoder_layers(merged)
    d_ffn = int(merged.config.intermediate_size)
    total_zeroed = 0
    for li, layer in enumerate(layers):
        keep = selected.get(li, set())
        keep_idx = torch.tensor(sorted(keep), dtype=torch.long)
        drop_mask = torch.ones(d_ffn, dtype=torch.bool)
        if keep_idx.numel():
            drop_mask[keep_idx] = False
        dp = layer.mlp.down_proj.weight.data  # (hidden, d_ffn)
        dp[:, drop_mask] = 0
        total_zeroed += int(drop_mask.sum().item())
    kept_total = sum(len(v) for v in selected.values())
    print(f"[bake] kept {kept_total} channels, zeroed {total_zeroed} down_proj cols across {len(layers)} layers", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(str(args.out))
    print(f"[bake] saved dense substrate -> {args.out}", flush=True)
    print("[bake] SANITY GATE: eval this dir with bfcl_direct_qwen3 eval-mask (NO --topk, NO --adapter); must reproduce ~599/1007", flush=True)


if __name__ == "__main__":
    main()
