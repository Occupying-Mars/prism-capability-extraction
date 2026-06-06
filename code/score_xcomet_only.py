#!/usr/bin/env python3
"""Phase 2 of post-LoRA eval: load COMET/XCOMET alone and score the hyps
jsonl produced by gen_translations_only.py. Writes a per-row results
json with seg scores + per-(category, tag) pass-rate summary.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyps-jsonl", required=True,
                   help="jsonl with {en, pt, model_hyp, category, tag} rows")
    p.add_argument("--out", required=True)
    p.add_argument("--comet-model", default="Unbabel/XCOMET-XXL",
                   help="COMET model id to load from Hugging Face. Defaults "
                        "to XCOMET-XXL; use Unbabel/wmt22-comet-da when "
                        "XCOMET gated access is unavailable.")
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--xcomet-batch", type=int, default=4)
    p.add_argument("--xcomet-chunk", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(l) for l in open(args.hyps_jsonl)]
    print(f"[load] {len(rows)} rows from {args.hyps_jsonl}")
    cnt = Counter(r.get("category", "?") for r in rows)
    print(f"[load] by category: {dict(cnt)}")

    print(f"[comet] load {args.comet_model}")
    import os
    from comet import load_from_checkpoint
    ckpt_env = os.environ.get("XCOMET_CKPT_PATH")
    if ckpt_env:
        ckpt = ckpt_env
    else:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(repo_id=args.comet_model)
        ckpt = os.path.join(snap, "checkpoints", "model.ckpt")
    xc = load_from_checkpoint(ckpt)
    data = [{"src": r["en"], "mt": r["model_hyp"], "ref": r["pt"]} for r in rows]
    chunk = args.xcomet_chunk
    seg_scores: list[float] = []
    print(f"[comet] scoring {len(data)} segs  (chunk={chunk}, batch={args.xcomet_batch})")
    t0 = time.time()
    for ci in range(0, len(data), chunk):
        sub = data[ci: ci + chunk]
        preds = xc.predict(sub, batch_size=args.xcomet_batch, gpus=1, progress_bar=False)
        seg_scores.extend(map(float, preds["scores"]))
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(f"  [chunk] {min(ci + chunk, len(data))}/{len(data)} done  "
              f"({elapsed:.0f}s, mean so far {sum(seg_scores)/len(seg_scores):.4f})",
              flush=True)
    sys_score = sum(seg_scores) / len(seg_scores)
    print(f"[comet] system={sys_score:.4f}")

    for r, s in zip(rows, seg_scores):
        r["comet_score"] = s

    by_cat: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_cat[(r.get("category", "?"), r.get("tag", "?"))].append(r["comet_score"])

    print(f"\n[summary] system {args.comet_model} = {sys_score:.4f}  threshold = {args.threshold}")
    print(f"\n{'category':<24s} {'tag':<14s} {'n':>5s} {'mean':>7s} {'≥thr':>6s}  {'%':>6s}")
    print("-" * 70)
    rows_summary = []
    for (cat, tag), ss in sorted(by_cat.items()):
        n = len(ss)
        mean = sum(ss) / n
        passed = sum(1 for s in ss if s >= args.threshold)
        pct = 100 * passed / n
        rows_summary.append((cat, tag, n, mean, passed, pct))
        print(f"{cat:<24s} {tag:<14s} {n:>5d} {mean:>7.4f} {passed:>6d}  {pct:>5.1f}%")

    out = {
        "hyps_jsonl": args.hyps_jsonl,
        "comet_model": args.comet_model,
        "n": len(rows),
        "threshold": args.threshold,
        "system_score": sys_score,
        "system_xcomet_xxl": sys_score if args.comet_model == "Unbabel/XCOMET-XXL" else None,
        "by_category_tag": [
            {"category": c, "tag": t, "n": n, "mean": mean, "passed": p, "pct": pct}
            for (c, t, n, mean, p, pct) in rows_summary
        ],
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
