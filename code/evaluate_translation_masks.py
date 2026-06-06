#!/usr/bin/env python3
"""Evaluate FLORES devtest English -> Hindi translation under MLP-neuron
mask + mean ablation. For each (name, mask.npz) the kept neurons pass
through unchanged; the rest are replaced by the calibration-set mean.

This is the autoregressive analogue of evaluate_accuracy() in
src/circuit_tracing/accuracy_eval.py, but for a chat-template translation
task and chrF++/BLEU on FLORES devtest.

Usage
-----
    uv run python evaluate_translation_masks.py \
        --model tencent/HY-MT1.5-1.8B \
        --mask k1000:results/attribution/base/relp_k1000.full.npz \
        --mask k5000:results/attribution/base/relp_k5000.full.npz \
        --out results/attribution/base/eval_masks.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.circuit_tracing.ablation import MeanCache
from translation_io import (
    DEFAULT_PROMPT_STYLE,
    DEFAULT_TARGET_LANGUAGE,
    Pair,
    apply_chat,
    generate_translations,
    load_flores_devtest_any,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--mask", action="append", default=[], metavar="NAME:PATH",
                   help="Repeat for each mask. PATH is the .full.npz produced "
                        "by attribute_translation.py")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE)
    p.add_argument("--prompt-style", default=DEFAULT_PROMPT_STYLE,
                   choices=["hy_mt", "sarvam"])
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-lang", default="hin_Deva")
    p.add_argument("--input-jsonl", default=None,
                   help="if set, load EN/PT pairs from this jsonl (fields 'en','pt') "
                        "instead of FLORES devtest")
    p.add_argument("--n-calib", type=int, default=64,
                   help="prompts used to build the mean activation cache")
    p.add_argument("--max-examples", type=int, default=None,
                   help="optional cap on FLORES devtest size")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--include-no-mask", action="store_true",
                   help="also run an unpatched baseline alongside masks")
    p.add_argument("--dump-hyps-dir", default=None,
                   help="if set, write {block_name}.jsonl with per-row hyps "
                        "(fields: en, pt, model_hyp, category, tag, mask_name) "
                        "for downstream XCOMET-XXL / GEMBA-MQM scoring.")
    p.add_argument("--dump-category", default="flores_devtest")
    p.add_argument("--dump-tag", default="heldout")
    return p.parse_args()


def load_mask_npz(path: Path, n_layers: int, d_ffn: int) -> Dict[int, torch.Tensor]:
    arr = np.load(path)
    out: Dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        key = f"layer_{layer}"
        if key in arr:
            out[layer] = torch.from_numpy(arr[key]).bool()
        else:
            out[layer] = torch.zeros(d_ffn, dtype=torch.bool)
    return out


def parse_mask_specs(values: List[str]) -> List[Tuple[str, Path]]:
    specs = []
    for v in values:
        if ":" not in v:
            raise ValueError(f"--mask must be NAME:PATH, got {v!r}")
        name, raw = v.split(":", 1)
        specs.append((name, Path(raw)))
    return specs


def install_mask_hooks(model, keep_per_layer, base_per_layer):
    hooks = []
    root = model.model
    if not hasattr(root, "layers") and hasattr(root, "language_model"):
        root = root.language_model
    for layer_idx, layer in enumerate(root.layers):
        keep = keep_per_layer[layer_idx]
        base = base_per_layer[layer_idx]

        def make(keep, base):
            def hook_fn(module, args):
                act = args[0]
                modified = act * keep + base * (1.0 - keep)
                return (modified,) + args[1:]
            return hook_fn

        h = layer.mlp.down_proj.register_forward_pre_hook(make(keep, base))
        hooks.append(h)
    return hooks


def evaluate_with_mask(model, tokenizer, sources, refs, *,
                        n_layers, d_ffn, mask, mean_cache,
                        device, dtype, batch_size, max_new_tokens,
                        target_language, prompt_style=DEFAULT_PROMPT_STYLE):
    keep_per_layer = {}
    for layer_idx in range(n_layers):
        keep = torch.zeros(d_ffn, device=device, dtype=dtype)
        if mask is not None:
            mask_layer = mask[layer_idx].to(device)
            keep[mask_layer] = 1.0
        keep_per_layer[layer_idx] = keep.view(1, 1, -1)

    base_per_layer = {}
    for layer_idx in range(n_layers):
        mu = mean_cache.means.get(layer_idx)
        if mu is None:
            base_per_layer[layer_idx] = torch.zeros(1, 1, d_ffn,
                                                     device=device, dtype=dtype)
        else:
            base_per_layer[layer_idx] = mu.to(device=device, dtype=dtype).view(1, 1, -1)

    if mask is not None:
        hooks = install_mask_hooks(model, keep_per_layer, base_per_layer)
    else:
        hooks = []
    try:
        hyps = generate_translations(
            model, tokenizer, sources,
            target_language=target_language,
            prompt_style=prompt_style,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            do_sample=False, device=device,
        )
    finally:
        for h in hooks:
            h.remove()

    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return {"chrFpp": chrfpp, "chrF": chrf, "BLEU": bleu, "n": len(hyps)}, hyps


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[load] tokenizer + model from {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = text_cfg.num_hidden_layers
    d_ffn = text_cfg.intermediate_size

    if args.input_jsonl:
        print(f"[load] jsonl {args.input_jsonl}", flush=True)
        pairs = []
        with open(args.input_jsonl) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                pairs.append(Pair(src=row["en"], tgt=row.get("pt", "")))
                if args.max_examples and len(pairs) >= args.max_examples:
                    break
    else:
        print(f"[load] FLORES devtest ({args.src_lang} -> {args.tgt_lang})", flush=True)
        pairs = load_flores_devtest_any(src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                                          max_examples=args.max_examples)
    sources = [p.src for p in pairs]
    refs = [p.tgt for p in pairs]

    print(f"[mean] building mean cache from {args.n_calib} prompts", flush=True)
    calib_ids = []
    for src in sources[:args.n_calib]:
        prompt = apply_chat(tokenizer, src, target_language=args.target_language,
                              add_generation_prompt=True,
                              prompt_style=args.prompt_style)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        calib_ids.append(ids)
    mean_cache = MeanCache.build(model, calib_ids, device=args.device)
    mean_cache = MeanCache(means={
        k: v.to(dtype=dtype) for k, v in mean_cache.means.items()
    })

    dump_dir = Path(args.dump_hyps_dir) if args.dump_hyps_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    def dump_block(block_name: str, hyps: List[str]) -> None:
        if dump_dir is None:
            return
        path = dump_dir / f"{block_name}.jsonl"
        with open(path, "w") as fh:
            for i, (src, ref, hyp) in enumerate(zip(sources, refs, hyps)):
                fh.write(json.dumps({
                    "id": i, "en": src, "pt": ref, "model_hyp": hyp,
                    "category": args.dump_category, "tag": args.dump_tag,
                    "mask_name": block_name,
                }, ensure_ascii=False) + "\n")
        print(f"     dumped {len(hyps)} hyps -> {path}", flush=True)

    results = {}
    if args.include_no_mask:
        print(f"[eval] no-mask (sanity)", flush=True)
        scores, hyps = evaluate_with_mask(
            model, tokenizer, sources, refs,
            n_layers=n_layers, d_ffn=d_ffn,
            mask=None, mean_cache=mean_cache,
            device=args.device, dtype=dtype,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            target_language=args.target_language,
            prompt_style=args.prompt_style,
        )
        print(f"     no_mask: {scores}", flush=True)
        results["no_mask"] = {"scores": scores, "kept": -1}
        dump_block("no_mask", hyps)

    for name, mask_path in parse_mask_specs(args.mask):
        print(f"[eval] mask={name} from {mask_path}", flush=True)
        mask = load_mask_npz(mask_path, n_layers, d_ffn)
        kept = sum(int(m.sum().item()) for m in mask.values())
        t0 = time.time()
        scores, hyps = evaluate_with_mask(
            model, tokenizer, sources, refs,
            n_layers=n_layers, d_ffn=d_ffn,
            mask=mask, mean_cache=mean_cache,
            device=args.device, dtype=dtype,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            target_language=args.target_language,
            prompt_style=args.prompt_style,
        )
        elapsed = time.time() - t0
        print(f"     {name}: kept={kept} {scores} ({elapsed:.1f}s)", flush=True)
        results[name] = {"scores": scores, "kept": kept,
                         "mask_path": str(mask_path), "elapsed_s": elapsed}
        dump_block(name, hyps)

    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_layers": n_layers, "d_ffn": d_ffn,
            "n_examples": len(pairs),
            "n_calib": args.n_calib,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
