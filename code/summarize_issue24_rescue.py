#!/usr/bin/env python3
"""Summarize issue #24 EN->PT low-rank rescue runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--k", type=int, default=120000)
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_eval(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def score_block(payload: dict, block: str) -> dict | None:
    result = payload.get("results", {}).get(block)
    if not result:
        return None
    scores = result.get("scores", {})
    return {
        "chrFpp": scores.get("chrFpp"),
        "chrF": scores.get("chrF"),
        "BLEU": scores.get("BLEU"),
        "n": scores.get("n"),
    }


def load_metric_score(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = load_eval(path)
    for key in ("system_score", "system_xcomet_xxl", "system_mqm"):
        value = payload.get(key)
        if value is not None:
            return value
    return None


def metric_paths(root: Path, rank: str, k: int) -> dict[str, Path]:
    if rank == "base":
        stem_no_mask = "base_no_mask"
        stem_masked = f"base_fixed_k{k}"
    elif rank == "r32_anchor":
        stem_no_mask = "r32_anchor_no_mask"
        stem_masked = f"r32_anchor_fixed_k{k}"
    else:
        stem_no_mask = f"r{rank}_no_mask"
        stem_masked = f"r{rank}_fixed_k{k}"
    return {
        "no_mask": root / "comet" / f"{stem_no_mask}.json",
        "masked": root / "comet" / f"{stem_masked}.json",
        "xcomet_no_mask": root / "xcomet" / f"{stem_no_mask}.json",
        "xcomet_masked": root / "xcomet" / f"{stem_masked}.json",
    }


def attach_external_metrics(root: Path, row: dict, k: int) -> None:
    rank = str(row["rank"])
    paths = metric_paths(root, rank, k)
    comet_no_mask = load_metric_score(paths["no_mask"])
    comet_masked = load_metric_score(paths["masked"])
    xcomet_no_mask = load_metric_score(paths["xcomet_no_mask"])
    xcomet_masked = load_metric_score(paths["xcomet_masked"])
    if comet_no_mask is not None or comet_masked is not None:
        row["comet"] = {"no_mask": comet_no_mask, "masked": comet_masked}
    if xcomet_no_mask is not None or xcomet_masked is not None:
        row["xcomet"] = {"no_mask": xcomet_no_mask, "masked": xcomet_masked}


def row_order(row: dict) -> tuple[int, int]:
    rank = str(row["rank"])
    if rank == "base":
        return (-1, 0)
    if rank == "r32_anchor":
        return (32, 1)
    if rank.isdigit():
        return (int(rank), 0)
    return (10_000, 0)


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    k = args.k
    rows = []

    base_path = root / "eval" / f"base_k{k}_masks.json"
    if base_path.exists():
        payload = load_eval(base_path)
        rows.append({
            "rank": "base",
            "trainable_params": 0,
            "no_mask": score_block(payload, "no_mask"),
            "masked": score_block(payload, f"fixed_k{k}"),
        })

    anchor_dir = root / f"k{k}" / "r32_anchor"
    anchor_eval = root / "eval" / f"r32_anchor_k{k}_masks.json"
    if anchor_dir.exists() and anchor_eval.exists():
        payload = load_eval(anchor_eval)
        train_summary = anchor_dir / "train_summary.json"
        final_train = None
        if train_summary.exists():
            summary = load_eval(train_summary)
            logs = summary.get("logs") or []
            final_train = logs[-1] if logs else None
        rows.append({
            "rank": "r32_anchor",
            "trainable_params": "rank 32, alpha 64",
            "final_train": final_train,
            "no_mask": score_block(payload, "no_mask"),
            "masked": score_block(payload, f"fixed_k{k}"),
        })

    for run_dir in sorted((root / f"k{k}").glob("r*")):
        rank_label = run_dir.name.removeprefix("r")
        if not rank_label.isdigit():
            continue
        eval_path = root / "eval" / f"r{rank_label}_k{k}_masks.json"
        if not eval_path.exists():
            continue
        train_summary = run_dir / "train_summary.json"
        trainable = None
        final_train = None
        if train_summary.exists():
            summary = load_eval(train_summary)
            logs = summary.get("logs") or []
            final_train = logs[-1] if logs else None
            # Exact trainable params are printed, not serialized, by PEFT.
            r = int(rank_label)
            trainable = f"rank {r}, alpha {2 * r}"
        payload = load_eval(eval_path)
        rows.append({
            "rank": rank_label,
            "trainable_params": trainable,
            "final_train": final_train,
            "no_mask": score_block(payload, "no_mask"),
            "masked": score_block(payload, f"fixed_k{k}"),
        })

    for row in rows:
        attach_external_metrics(root, row, k)

    base_masked = next((r["masked"] for r in rows if r["rank"] == "base"), None)
    base_chrfpp = base_masked.get("chrFpp") if base_masked else None
    for row in rows:
        masked = row.get("masked") or {}
        if base_chrfpp is not None and masked.get("chrFpp") is not None:
            row["gain_chrFpp_over_base_mask"] = masked["chrFpp"] - base_chrfpp

    rows.sort(key=row_order)

    out = {
        "run_root": str(root),
        "k": k,
        "metric_note": "XCOMET/GEMBA unavailable in this run; chrF++/chrF/BLEU are cheap generation metrics.",
        "rows": rows,
    }
    out_path = Path(args.out) if args.out else root / "summary_issue24.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
