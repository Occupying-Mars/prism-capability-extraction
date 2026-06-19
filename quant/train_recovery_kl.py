#!/usr/bin/env python3
"""Quant-error recovery by DISTILLATION from the bf16 teacher (the real fix).

Plain-CE recovery (train_recovery_lora.py) regressed: it re-fits train-mix labels
and drifts the already-optimal b007 adapter off the substrate. The principled
"eval-aware" recovery is to make the quantized model behave like the bf16 model
it was quantized from:

    teacher = bf16 Qwen3-8B + b007 + MACE mask        (the 599 substrate, frozen)
    student = NF4  Qwen3-8B + b007(trainable) + mask
    loss    = KL(teacher || student) on the answer tokens + small CE on gold

KL pulls the student toward *what bf16 outputs* (undo the rounding), not toward
train labels (which caused the drift). Trained ONLY on the leak-gated b007 mix;
eval is the untouched held-out 1007.

Usage (pod, .venv):
  python train_recovery_kl.py --train train_data/train_mixed.jsonl \
      --max-steps 150 --eval-after
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


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def encode_row(row, tokenizer, max_seq_len):
    """Return (input_ids, labels, kl_positions) for one example."""
    target_text = (row.get("target_text") or "").strip()
    if not target_text:
        return None
    enc = tokenizer.apply_chat_template(
        row["messages"], tools=row.get("tools") or None,
        add_generation_prompt=True, tokenize=True, return_dict=True, enable_thinking=False,
    )
    prompt_ids = list(enc["input_ids"])
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    ids = prompt_ids + target_ids
    if not target_ids or len(ids) > max_seq_len:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    # positions whose NEXT token is a target token -> where teacher/student must agree
    kl_positions = list(range(max(len(prompt_ids) - 1, 0), len(ids) - 1))
    return ids, labels, kl_positions


def main():
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", type=Path, default=DEF_ADAPTER)
    ap.add_argument("--mask", type=Path, default=DEF_MASK)
    ap.add_argument("--topk", type=int, default=DEF_TOPK)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, default=DEF_PAIRS)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--ce-beta", type=float, default=0.1)
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
                name="recovery-kl-nf4", job_type="lora-recovery-kl",
                config={k: str(v) for k, v in vars(args).items()},
            )
            print(f"[wandb] {run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] disabled ({e})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected = bfcl.load_topk_mask(args.mask, args.topk)

    # ---- teacher: bf16 + b007 + mask, frozen ----
    print("[teacher] loading bf16 + b007 + mask (frozen)", flush=True)
    tbase = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="cuda", attn_implementation="eager"
    )
    teacher = PeftModel.from_pretrained(tbase, str(args.adapter))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_hooks = bfcl.install_mlp_keep_hooks(teacher, selected)

    # ---- student: NF4 + b007(trainable) + mask ----
    print("[student] loading NF4 + b007 (trainable) + mask", flush=True)
    sbase = qs.build_quantized_base("nf4", args.model, args.dtype, "both")
    student = PeftModel.from_pretrained(sbase, str(args.adapter), is_trainable=True)
    student.config.use_cache = False
    if hasattr(student, "enable_input_require_grads"):
        student.enable_input_require_grads()
    s_hooks = bfcl.install_mlp_keep_hooks(student, selected)
    sdev = student.device
    tdev = teacher.device
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[lora] trainable: {trainable/1e6:.1f}M | kept {sum(len(v) for v in selected.values())}", flush=True)

    rows = list(read_jsonl(args.train))
    random.shuffle(rows)
    rows = rows[: args.max_rows]
    data = [e for r in rows if (e := encode_row(r, tokenizer, args.max_seq_len))]
    print(f"[data] usable: {len(data)} (leak-gated)", flush=True)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    student.train()
    T = args.kl_temp
    t0 = time.time()
    step = micro = idx = 0
    opt.zero_grad(set_to_none=True)
    while step < args.max_steps:
        ids, labels, kl_pos = data[idx % len(data)]
        idx += 1
        inp_s = torch.tensor([ids], device=sdev)
        inp_t = torch.tensor([ids], device=tdev)
        with torch.inference_mode():
            t_logits = teacher(input_ids=inp_t, use_cache=False).logits[0, kl_pos, :].to(sdev).float()
        s_logits = student(input_ids=inp_s, use_cache=False).logits[0, kl_pos, :].float()
        kl = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        ce = F.cross_entropy(
            s_logits, torch.tensor([labels[p + 1] for p in kl_pos], device=sdev), ignore_index=-100
        )
        loss = kl + args.ce_beta * ce
        (loss / args.grad_accum).backward()
        micro += 1
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 10 == 0 or step == args.max_steps:
                rec = {"step": step, "kl": float(kl.detach().cpu()), "ce": float(ce.detach().cpu()),
                       "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                if run is not None:
                    run.log(rec)

    student.eval()
    train_secs = round(time.time() - t0, 1)
    print(f"[train] done {step} steps in {train_secs}s", flush=True)
    for h in t_hooks:
        h.remove()

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(str(args.save))
        tokenizer.save_pretrained(str(args.save))

    summary = {"base_method": "nf4-kl-distill", "train_steps": step, "train_secs": train_secs,
               "kl_temp": T, "ce_beta": args.ce_beta, "lr": args.lr}
    if args.eval_after:
        eargs = SimpleNamespace(method="recovery-kl-nf4", target="both", topk=args.topk,
                                pairs=args.pairs, limit=args.eval_limit,
                                batch_size=args.eval_batch_size, max_new_tokens=512)
        ev = qs.evaluate(student, tokenizer, eargs)
        summary.update(ev)
        print(json.dumps(ev, indent=2), flush=True)
        if run is not None:
            run.summary.update(ev)
            run.log({k: v for k, v in ev.items() if isinstance(v, (int, float))})
    for h in s_hooks:
        h.remove()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
