"""MOHAWK 2-stage distillation of the Qwen3 Transformer->SSM hybrid (arXiv:2602.11374).

Teacher = frozen Qwen3-8B. Student = hybrid (retained G&A heads + DiscreteMamba2).
Only the SSM params train; teacher weights are frozen anchors, so MOHAWK's Matrix
Orientation stage is skipped (paper §5.1).

Stage A  Hidden-State Alignment: MSE between student and teacher per-layer hidden
         states (teacher detached). Peak LR 1e-4.
Stage B  Knowledge Distillation: KL(student || teacher) on logits. Peak LR 1e-5.

Data: BFCL strict-10k mix (our eval is BFCL), 1-2 epochs. Prompt+gold tokenized and
packed to --seq-len. Simplification vs paper: full-model hidden alignment rather than
per-layer teacher-forced decoupling (student shares all non-SSM weights with teacher).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# transformers 5.x dropped symbols mamba_ssm's __init__ expects; shim before mamba import
import transformers.generation as _G  # noqa: E402
for _n in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
    if not hasattr(_G, _n):
        setattr(_G, _n, getattr(_G, "GenerateDecoderOnlyOutput", object))

from prism_qwen.hybrid_mamba import (  # noqa: E402
    build_hybrid, freeze_except_ssm, retained_by_layer_from_scores,
)
from bfcl_direct_qwen3 import encode_prompt, format_tool_call_target, read_records  # noqa: E402


def load(model_path, dtype, device="cuda"):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device, attn_implementation="eager")


def build_packed(tokenizer, rows, seq_len, device):
    """Concatenate prompt+gold token streams and slice into [n, seq_len] blocks."""
    from transformers import AutoTokenizer  # noqa
    stream: list[int] = []
    eos = tokenizer.eos_token_id
    for row in rows:
        p = encode_prompt(tokenizer, row, enable_thinking=False)["input_ids"][0].tolist()
        t = tokenizer(format_tool_call_target(row), add_special_tokens=False)["input_ids"]
        stream.extend(p + t + [eos])
    n = len(stream) // seq_len
    ids = torch.tensor(stream[: n * seq_len], dtype=torch.long).view(n, seq_len)
    print(f"packed {len(rows)} rows -> {n} blocks of {seq_len} ({len(stream)} tokens)", flush=True)
    return ids


def hidden_align_loss(student, teacher, batch):
    with torch.no_grad():
        t = teacher(batch, output_hidden_states=True).hidden_states
    s = student(batch, output_hidden_states=True).hidden_states
    loss = 0.0
    for si, ti in zip(s[1:], t[1:]):                       # skip embedding layer
        loss = loss + F.mse_loss(si.float(), ti.float().detach())
    return loss / (len(s) - 1)


def kd_loss(student, teacher, batch, temp=1.0):
    with torch.no_grad():
        tl = teacher(batch).logits
    sl = student(batch).logits
    s_logp = F.log_softmax(sl.float() / temp, dim=-1)
    t_p = F.softmax(tl.float() / temp, dim=-1)
    return F.kl_div(s_logp, t_p, reduction="batchmean") * (temp * temp)


def run_stage(name, student, teacher, blocks, loss_fn, *, lr, steps, bs, accum, log_every):
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    student.train()
    dev = next(student.parameters()).device
    n = blocks.shape[0]
    started = time.time()
    step = 0
    order = torch.randperm(n)
    ptr = 0
    while step < steps:
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(accum):
            if ptr + bs > n:
                order = torch.randperm(n); ptr = 0
            idx = order[ptr: ptr + bs]; ptr += bs
            batch = blocks[idx].to(dev)
            loss = loss_fn(student, teacher, batch) / accum
            loss.backward()
            acc_loss += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % log_every == 0 or step == steps:
            print(f"[{name}] step {step}/{steps} loss={acc_loss:.5f} "
                  f"elapsed_s={time.time()-started:.0f}", flush=True)
    return {"stage": name, "final_loss": acc_loss, "steps": steps, "lr": lr}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/root/models/Qwen3-8B")
    ap.add_argument("--tokenizer")
    ap.add_argument("--scores", type=Path, required=True, help="gaa_ablation_scores.npz")
    ap.add_argument("--select", default="total", choices=["total", "aggregate", "gather"])
    ap.add_argument("--keep-k", type=int, default=40, help="number of retained heads (top-k)")
    ap.add_argument("--d-state", type=int, default=64)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--rows", type=int, help="cap training rows")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--align-steps", type=int, default=400)
    ap.add_argument("--kd-steps", type=int, default=200)
    ap.add_argument("--align-lr", type=float, default=1e-4)
    ap.add_argument("--kd-lr", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--grad-checkpointing", action="store_true")
    args = ap.parse_args()

    dtype = torch.bfloat16
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading teacher + student (2x Qwen3-8B)...", flush=True)
    teacher = load(args.model, dtype); teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = load(args.model, dtype)

    n_layers = student.config.num_hidden_layers
    retained = retained_by_layer_from_scores(args.scores, args.select, args.keep_k, n_layers)
    n_ret = sum(len(v) for v in retained.values())
    build_hybrid(student, retained, d_state=args.d_state)
    n_train = freeze_except_ssm(student)
    if args.grad_checkpointing:
        student.gradient_checkpointing_enable()
    print(f"retained heads={n_ret}/{n_layers*student.config.num_attention_heads} "
          f"ssm_trainable={n_train/1e6:.1f}M d_state={args.d_state}", flush=True)

    rows = read_records(args.train_jsonl)
    if args.rows:
        rows = rows[: args.rows]
    blocks = build_packed(tokenizer, rows, args.seq_len, student.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = []
    log.append(run_stage("align", student, teacher, blocks, hidden_align_loss,
                         lr=args.align_lr, steps=args.align_steps, bs=args.batch_size,
                         accum=args.accum, log_every=args.log_every))
    log.append(run_stage("kd", student, teacher, blocks, kd_loss,
                         lr=args.kd_lr, steps=args.kd_steps, bs=args.batch_size,
                         accum=args.accum, log_every=args.log_every))

    ssm_state = {k: v.cpu() for k, v in student.state_dict().items() if ".ssm." in k}
    torch.save({"ssm_state": ssm_state, "retained": retained, "d_state": args.d_state,
                "keep_k": args.keep_k}, args.output_dir / "hybrid_ssm.pt")
    (args.output_dir / "train_log.json").write_text(json.dumps(
        {"model": args.model, "retained_heads": n_ret, "ssm_trainable_M": n_train / 1e6,
         "d_state": args.d_state, "seq_len": args.seq_len, "blocks": int(blocks.shape[0]),
         "stages": log}, indent=2) + "\n")
    print("saved hybrid_ssm.pt + train_log.json", flush=True)


if __name__ == "__main__":
    main()
