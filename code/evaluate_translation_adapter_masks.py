#!/usr/bin/env python3
"""Evaluate HY-MT EN->PT translation masks with an optional PEFT adapter.

This is the issue #28 reproduction evaluator. It mirrors
evaluate_translation_masks.py, but loads a base model plus a saved LoRA adapter
and merges the adapter in memory before generation. That preserves the original
LoRA rescue inference contract without writing full merged model checkpoints to
disk.
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
from peft import PeftModel
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
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default=None,
                   help="Optional PEFT adapter dir. If set, it is merged in memory.")
    p.add_argument("--mask", action="append", default=[], metavar="NAME:PATH")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE)
    p.add_argument("--prompt-style", default=DEFAULT_PROMPT_STYLE,
                   choices=["hy_mt", "sarvam"])
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-lang", default="por_Latn")
    p.add_argument("--input-jsonl", default=None)
    p.add_argument("--n-calib", type=int, default=64)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--include-no-mask", action="store_true")
    p.add_argument("--dump-hyps-dir", default=None)
    p.add_argument("--dump-category", default="flores_devtest")
    p.add_argument("--dump-tag", default="heldout")
    return p.parse_args()


def text_config(model):
    return getattr(model.config, "text_config", model.config)


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
    for value in values:
        if ":" not in value:
            raise ValueError(f"--mask must be NAME:PATH, got {value!r}")
        name, raw = value.split(":", 1)
        specs.append((name, Path(raw)))
    return specs


def decoder_root(model):
    cur = model
    for _ in range(8):
        if hasattr(cur, "layers"):
            return cur
        for attr in ("model", "language_model", "base_model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    raise AttributeError("could not locate decoder .layers")


def install_mask_hooks(model, keep_per_layer, base_per_layer):
    hooks = []
    root = decoder_root(model)
    for layer_idx, layer in enumerate(root.layers):
        keep = keep_per_layer[layer_idx]
        base = base_per_layer[layer_idx]

        def make(keep, base):
            def hook_fn(module, args):
                act = args[0]
                modified = act * keep + base * (1.0 - keep)
                return (modified,) + args[1:]
            return hook_fn

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make(keep, base)))
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
            base_per_layer[layer_idx] = torch.zeros(1, 1, d_ffn, device=device, dtype=dtype)
        else:
            base_per_layer[layer_idx] = mu.to(device=device, dtype=dtype).view(1, 1, -1)

    hooks = install_mask_hooks(model, keep_per_layer, base_per_layer) if mask is not None else []
    try:
        hyps = generate_translations(
            model, tokenizer, sources,
            target_language=target_language,
            prompt_style=prompt_style,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=device,
        )
    finally:
        for hook in hooks:
            hook.remove()

    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return {"chrFpp": chrfpp, "chrF": chrf, "BLEU": bleu, "n": len(hyps)}, hyps


def load_pairs(args: argparse.Namespace) -> list[Pair]:
    if args.input_jsonl:
        pairs = []
        with open(args.input_jsonl) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                pairs.append(Pair(src=row["en"], tgt=row.get("pt", "")))
                if args.max_examples and len(pairs) >= args.max_examples:
                    break
        return pairs
    return load_flores_devtest_any(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_examples=args.max_examples,
    )


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[load] tokenizer from {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"[load] base model {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    if args.adapter:
        print(f"[load] adapter {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        print("[merge] merging adapter in memory", flush=True)
        model = model.merge_and_unload().to(args.device).eval()

    cfg = text_config(model)
    n_layers = int(cfg.num_hidden_layers)
    d_ffn = int(cfg.intermediate_size)
    pairs = load_pairs(args)
    sources = [p.src for p in pairs]
    refs = [p.tgt for p in pairs]
    print(f"[data] eval_rows={len(pairs)}", flush=True)

    print(f"[mean] building mean cache from {args.n_calib} prompts", flush=True)
    calib_ids = []
    for src in sources[: args.n_calib]:
        prompt = apply_chat(
            tokenizer, src,
            target_language=args.target_language,
            add_generation_prompt=True,
            prompt_style=args.prompt_style,
        )
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        calib_ids.append(ids)
    mean_cache = MeanCache.build(model, calib_ids, device=args.device)
    mean_cache = MeanCache(means={k: v.to(dtype=dtype) for k, v in mean_cache.means.items()})

    dump_dir = Path(args.dump_hyps_dir) if args.dump_hyps_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    def dump_block(block_name: str, hyps: List[str]) -> None:
        if dump_dir is None:
            return
        path = dump_dir / f"{block_name}.jsonl"
        with path.open("w") as fh:
            for idx, (src, ref, hyp) in enumerate(zip(sources, refs, hyps)):
                fh.write(json.dumps({
                    "id": idx,
                    "en": src,
                    "pt": ref,
                    "model_hyp": hyp,
                    "category": args.dump_category,
                    "tag": args.dump_tag,
                    "mask_name": block_name,
                    "adapter": args.adapter,
                }, ensure_ascii=False) + "\n")
        print(f"     dumped {len(hyps)} hyps -> {path}", flush=True)

    results = {}
    if args.include_no_mask:
        print("[eval] no-mask", flush=True)
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
        results[name] = {
            "scores": scores,
            "kept": kept,
            "mask_path": str(mask_path),
            "elapsed_s": elapsed,
        }
        dump_block(name, hyps)

    with out_path.open("w") as f:
        json.dump({
            "base_model": args.base_model,
            "adapter": args.adapter,
            "n_layers": n_layers,
            "d_ffn": d_ffn,
            "n_examples": len(pairs),
            "n_calib": args.n_calib,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
