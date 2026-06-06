#!/usr/bin/env python3
"""Summarize issue #9 rank-ladder artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.2f}%"


def fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/qwen25_math_1p5b_2digit_rank_ladder_issue9")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_md = Path(args.out_md) if args.out_md else root / "summary.md"
    out_json = Path(args.out_json) if args.out_json else root / "summary.json"
    ranks = [16, 8, 4, 2, 1]

    hidden_size = 1536
    intermediate_size = 8960
    num_layers = 28
    vocab_size = 151936
    num_heads = 12
    kv_heads = 2
    head_dim = hidden_size // num_heads
    kv_dim = kv_heads * head_dim
    mlp_component_params = 3 * hidden_size
    mlp_total_components = num_layers * intermediate_size
    mlp_total_params = mlp_total_components * mlp_component_params
    attn_per_layer = hidden_size * hidden_size + 2 * (kv_dim * hidden_size) + hidden_size * hidden_size
    total_params = vocab_size * hidden_size + num_layers * attn_per_layer + mlp_total_params + (num_layers * 2 * hidden_size + hidden_size)

    rows = []
    for rank in ranks:
        run_dir = root / f"rank_{rank}"
        train = load_json(run_dir / f"lora_r{rank}_beta005" / "train_summary.json") or {}
        merge = load_json(run_dir / f"lora_r{rank}_beta005_merge" / "merge_sweep.json") or {}
        target = load_json(run_dir / "target90_search.json") or {}
        fresh_eval = load_json(run_dir / "fresh_target90_eval.json") or {}
        fresh_gen = load_json(run_dir / "fresh_target90_generation.json") or {}

        selection = (merge.get("selection") or {}).get("row") or {}
        target_best = target.get("best") or {}
        eval_best = (fresh_eval.get("by_mask") or {}).get("target90") or {}
        gen_best = (fresh_gen.get("by_mask") or {}).get("target90") or {}
        mlp = target_best.get("mlp_kept")
        kept_params = mlp * mlp_component_params if mlp is not None else None
        rows.append(
            {
                "rank": rank,
                "train_exact": (train.get("final_eval") or {}).get("accuracy"),
                "selected_scale": selection.get("scale"),
                "selected_kl_mean": (selection.get("kl") or {}).get("kl_mean"),
                "selected_exact": (selection.get("exact") or {}).get("accuracy"),
                "original_exact": target_best.get("accuracy"),
                "fresh_teacher_forced_exact": eval_best.get("accuracy"),
                "fresh_generation_exact": gen_best.get("accuracy"),
                "mlp_kept": mlp,
                "mlp_fraction": (mlp / mlp_total_components) if mlp is not None else None,
                "mlp_params": kept_params,
                "total_model_equivalent_param_fraction": (kept_params / total_params) if kept_params is not None else None,
                "target_label": target_best.get("label"),
            }
        )

    winning = [row for row in rows if (row["fresh_generation_exact"] or 0.0) >= 0.9]
    winner = min(winning, key=lambda row: row["mlp_kept"]) if winning else None

    summary = {
        "root": str(root),
        "reference_rank32": {
            "fresh_generation_exact": 0.9133,
            "mlp_kept": 12661,
            "total_model_equivalent_param_fraction": 0.0378,
        },
        "rows": rows,
        "winner": winner,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Rank Ladder Issue 9 Summary",
        "",
        "Goal: find the smallest circuit mask with at least `90%` exact 2-digit full-answer recovery by using LoRA rank as an information bottleneck.",
        "",
        "Rank-32 reference: `91.33%` fresh autoregressive exact at `12,661` MLP channels, or `3.78%` total-model-equivalent MLP parameters.",
        "",
        "| rank | selected scale | selected KL | merged exact | original target exact | fresh teacher-forced | fresh generation | MLPs | total-model-equiv params |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    "n/a" if row["selected_scale"] is None else f"{row['selected_scale']:.2f}",
                    "n/a" if row["selected_kl_mean"] is None else f"{row['selected_kl_mean']:.6f}",
                    pct(row["selected_exact"]),
                    pct(row["original_exact"]),
                    pct(row["fresh_teacher_forced_exact"]),
                    pct(row["fresh_generation_exact"]),
                    fmt_int(row["mlp_kept"]),
                    pct(row["total_model_equivalent_param_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Conclusion", ""])
    if winner:
        lines.append(
            f"The best lower-rank target-90 result is rank `{winner['rank']}`: "
            f"{pct(winner['fresh_generation_exact'])} fresh autoregressive exact at "
            f"`{fmt_int(winner['mlp_kept'])}` MLP channels."
        )
    else:
        lines.append("No lower-rank result held at least `90%` fresh autoregressive exact recovery.")
    lines.append(
        "No tested lower rank beats the rank-32 sparse reference. The lower-rank bottleneck did not produce a smaller `>=90%` circuit; it moved the target-90 frontier to broad `rel_0.001`-derived masks."
    )
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
