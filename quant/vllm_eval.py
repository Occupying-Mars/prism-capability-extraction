#!/usr/bin/env python3
"""Eval a quantized checkpoint on held-out BFCL 1007 via vLLM (Blackwell-prebuilt
W4A16 Marlin kernels — the serving leg that runs on sm_120 where the local
GPTQModel/torchao kernels don't).

The substrate is already baked into the dense->quantized weights, so we eval the
model directly (no adapter, no mask hook). Prompts + scoring are tokenbender's
eval path verbatim (chat template + tools + bfcl-canonicalization prompt, greedy,
normalized-exact), so the number is apples-to-apples with the transformers runs.

Usage (pod, .venv-vllm with vllm installed):
  python vllm_eval.py --model out/qwen3-8b-ar-w4 --report reports/ar_w4_vllm.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
spec = importlib.util.spec_from_file_location("bfcl_direct_qwen3", SCRIPTS / "bfcl_direct_qwen3.py")
bfcl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfcl)

ART = Path("/workspace/qwen-quant/artifacts/bfcl")
DEF_PAIRS = ART / "issue12_recursive_coactivation_mace_v1/data/bfcl_single_call/pairs.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", type=Path, default=DEF_PAIRS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--quantization", default=None, help="force vLLM quant method, e.g. gptq_marlin, compressed-tensors")
    ap.add_argument("--enforce-eager", action="store_true", help="disable cudagraph (needed on Blackwell sm_120; leave OFF on A100 for real throughput)")
    ap.add_argument("--strip-think", action="store_true", help="strip the empty <think></think> block transformers 4.57 injects (match the substrate's eval format)")
    ap.add_argument("--smoke", type=int, default=0, help="generate N and print (serving sanity)")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = bfcl.read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    prompts = [
        tok.apply_chat_template(
            bfcl.messages_for_generation(r, bfcl_canonicalization_prompt=True),
            tools=r.get("tools") or None,
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        for r in rows
    ]

    # enforce_eager only needed on Blackwell sm_120 (no nvcc for flashinfer/inductor JIT);
    # on A100 leave it OFF so cudagraph is on and throughput numbers are real.
    llm_kwargs = dict(model=args.model, dtype="bfloat16", enforce_eager=args.enforce_eager,
                      gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    if args.smoke:
        outs = llm.generate(prompts[: args.smoke], sp)
        for o in outs:
            print("--- SMOKE ---\n" + o.outputs[0].text[:300], flush=True)
        return

    import time

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    judged = norm = raw = out_tokens = 0
    for row, out in zip(rows, outs):
        text = out.outputs[0].text
        out_tokens += len(out.outputs[0].token_ids)
        pred = bfcl.parse_tool_calls(text)
        judged += 1
        norm += int(bfcl.normalized_prediction_ok(pred, row))
        raw += int(bfcl.prediction_ok(pred, row))
    summary = {
        "model": args.model,
        "serving": "vllm",
        "quantization": args.quantization or "none",
        "examples": judged,
        "normalized_exact_correct": norm,
        "normalized_exact_accuracy": norm / judged if judged else None,
        "raw_exact_correct": raw,
        "recovery_vs_full_anchor": (norm / 664) if judged == 1007 else None,
        "full_anchor": 664,
        # throughput (whole batch through vLLM, same prompt set across models)
        "elapsed_s": round(elapsed, 1),
        "output_tokens": out_tokens,
        "requests_per_s": round(judged / elapsed, 2) if elapsed else None,
        "output_tokens_per_s": round(out_tokens / elapsed, 1) if elapsed else None,
    }
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
