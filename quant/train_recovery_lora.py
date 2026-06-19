#!/usr/bin/env python3
"""Recover quant error with a LoRA, trained ONLY on the leak-gated b007 mix.

After 4-bit quantizing the base (GPTQ or NF4), the BFCL score drops. We continue
the b007 rsLoRA adapter on the quantized base to absorb the quant error — QLoRA-
style recovery — then eval on the held-out 1007.

Integrity (the point the whole thing turns on):
  * TRAIN/calibration data = b007 train_mixed.jsonl, leak-audited vs the eval
    (leak_audit.py + tokenbender mixed_overlap_audit.json). Never the eval rows.
  * EVAL = held-out 1007 BFCL pairs, untouched.
  * The training intervention matches the EVAL contract exactly: zero-ablation
    keep-only hooks on mlp.down_proj input (tokenbender's eval-mask path), so the
    LoRA recovers the substrate we actually score.

Usage (pod, .venv):
  python train_recovery_lora.py --base-method gptq --gptq-path out/qwen3-8b-gptq4 \
      --train train_data/train_mixed.jsonl --max-steps 200 --eval-after
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
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


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def encode_row(row, tokenizer, max_seq_len):
    """Chat prompt + gold tool_call; loss only on the target tokens."""
    target_text = (row.get("target_text") or "").strip()
    if not target_text:
        return None
    prompt_ids = tokenizer.apply_chat_template(
        row["messages"], tools=row.get("tools") or None,
        add_generation_prompt=True, tokenize=True, enable_thinking=False,
    )
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    ids = prompt_ids + target_ids
    if not target_ids or len(ids) > max_seq_len:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    return ids, labels


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-method", default="gptq", choices=["gptq", "nf4", "nf4-attn"])
    ap.add_argument("--gptq-path", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", type=Path, default=DEF_ADAPTER)
    ap.add_argument("--mask", type=Path, default=DEF_MASK)
    ap.add_argument("--topk", type=int, default=DEF_TOPK)
    ap.add_argument("--train", type=Path, required=True, help="leak-gated train mix (NOT eval)")
    ap.add_argument("--pairs", type=Path, default=DEF_PAIRS)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-after", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
                name=f"recovery-{args.base_method}",
                job_type="lora-recovery",
                config={k: str(v) for k, v in vars(args).items()},
            )
            print(f"[wandb] {run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] disabled ({e})", flush=True)

    # ---- base (quantized) ----
    method = "gptq" if args.base_method == "gptq" else "nf4"
    target = "attn" if args.base_method == "nf4-attn" else "both"
    print(f"[base] {args.base_method} method={method} target={target}", flush=True)
    base = qs.build_quantized_base(method, args.model, args.dtype, target, args.gptq_path)

    # ---- b007 adapter as trainable ----
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] trainable params: {trainable/1e6:.2f}M", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- mask hooks: zero-ablation keep-only, identical to the eval contract ----
    selected = bfcl.load_topk_mask(args.mask, args.topk)
    hooks = bfcl.install_mlp_keep_hooks(model, selected)
    print(f"[mask] topk={args.topk} kept={sum(len(v) for v in selected.values())}", flush=True)

    # ---- training data (leak-gated) ----
    rows = [r for r in read_jsonl(args.train)]
    random.shuffle(rows)
    rows = rows[: args.max_rows]
    encoded = [e for r in rows if (e := encode_row(r, tokenizer, args.max_seq_len))]
    print(f"[data] usable train rows: {len(encoded)} (leak-gated mix)", flush=True)
    dev = model.device

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    t0 = time.time()
    step = 0
    micro = 0
    opt.zero_grad(set_to_none=True)
    logs = []
    idx = 0
    while step < args.max_steps:
        ids, labels = encoded[idx % len(encoded)]
        idx += 1
        input_ids = torch.tensor([ids], device=dev)
        label_t = torch.tensor([labels], device=dev)
        out = model(input_ids=input_ids, labels=label_t, use_cache=False)
        (out.loss / args.grad_accum).backward()
        micro += 1
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 10 == 0 or step == args.max_steps:
                rec = {"step": step, "loss": float(out.loss.detach().cpu()), "elapsed_s": round(time.time() - t0, 1)}
                logs.append(rec)
                print(json.dumps(rec), flush=True)
                if run is not None:
                    run.log(rec)

    model.eval()
    train_secs = round(time.time() - t0, 1)
    print(f"[train] done in {train_secs}s over {step} steps", flush=True)

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.save))
        tokenizer.save_pretrained(str(args.save))
        print(f"[save] {args.save}", flush=True)

    summary = {"base_method": args.base_method, "train_steps": step, "train_secs": train_secs}
    if args.eval_after:
        eargs = SimpleNamespace(
            method=f"recovery-{args.base_method}", target=target, topk=args.topk,
            pairs=args.pairs, limit=args.eval_limit, batch_size=args.eval_batch_size,
            max_new_tokens=512,
        )
        ev = qs.evaluate(model, tokenizer, eargs)
        summary.update(ev)
        print(json.dumps(ev, indent=2), flush=True)
        if run is not None:
            run.summary.update(ev)
            run.log({k: v for k, v in ev.items() if isinstance(v, (int, float))})

    for h in hooks:
        h.remove()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
