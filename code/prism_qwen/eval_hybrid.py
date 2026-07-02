"""Evaluate a distilled Qwen3->SSM hybrid on BFCL single-call (normalized exact).

Builds the hybrid from a saved hybrid_ssm.pt (retained heads + trained SSM weights),
generates tool calls, scores normalized-exact vs the 664 anchor, and reports the
wrong_function / arg_value / parse_fail buckets. Generation runs with use_cache=False
(the hybrid attention/SSM has no incremental-decode cache path).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import transformers.generation as _G  # noqa: E402
for _n in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
    if not hasattr(_G, _n):
        setattr(_G, _n, getattr(_G, "GenerateDecoderOnlyOutput", object))

from prism_qwen.hybrid_mamba import build_hybrid  # noqa: E402
from bfcl_direct_qwen3 import (  # noqa: E402
    messages_for_generation, normalized_prediction_ok, parse_tool_calls,
    prediction_ok, read_records,
)
from gaa_heads_qwen3 import classify_failure  # noqa: E402

FULL_BFCL_ANCHOR = 664


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/root/models/Qwen3-8B")
    ap.add_argument("--tokenizer")
    ap.add_argument("--hybrid-ckpt", type=Path, required=True, help="hybrid_ssm.pt")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--full-anchor", type=int, default=FULL_BFCL_ANCHOR)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ckpt = torch.load(args.hybrid_ckpt, map_location="cpu")
    retained = {int(k): v for k, v in ckpt["retained"].items()}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    build_hybrid(model, retained, d_state=ckpt["d_state"])
    missing, unexpected = model.load_state_dict(ckpt["ssm_state"], strict=False)
    loaded = [k for k in ckpt["ssm_state"] if ".ssm." in k]
    print(f"loaded {len(loaded)} ssm tensors; unexpected={len(unexpected)}", flush=True)
    model.config.use_cache = False
    model.eval()

    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    norm_ok = raw_ok = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    fails: Counter = Counter()
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start: start + args.batch_size]
        enc_items = [
            tok.apply_chat_template(
                messages_for_generation(r, bfcl_canonicalization_prompt=False),
                tools=r.get("tools") or None, add_generation_prompt=True,
                tokenize=True, return_dict=True, enable_thinking=False)
            for r in batch
        ]
        enc = tok.pad(enc_items, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, use_cache=False, pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for r, seq in zip(batch, out):
            text = tok.decode(seq[plen:], skip_special_tokens=True)
            calls = parse_tool_calls(text)
            n_ok = normalized_prediction_ok(calls, r)
            norm_ok += int(n_ok); raw_ok += int(prediction_ok(calls, r))
            c = r.get("category", "?"); by_cat[c][0] += int(n_ok); by_cat[c][1] += 1
            if not n_ok:
                fails[classify_failure(calls, r)] += 1
        done = min(start + args.batch_size, len(rows))
        print(f"eval {done}/{len(rows)} norm_exact={norm_ok} elapsed_s={time.time()-t0:.0f}", flush=True)

    summary = {
        "model": args.model, "hybrid_ckpt": str(args.hybrid_ckpt),
        "retained_heads": sum(len(v) for v in retained.values()),
        "n": len(rows), "normalized_exact": norm_ok, "raw_exact": raw_ok,
        "normalized_acc": norm_ok / max(len(rows), 1),
        "recovery_vs_anchor": norm_ok / max(args.full_anchor, 1),
        "per_category": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(by_cat.items())},
        "failure_buckets": dict(fails),
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("normalized_exact", "recovery_vs_anchor", "failure_buckets")}, indent=2))


if __name__ == "__main__":
    main()
