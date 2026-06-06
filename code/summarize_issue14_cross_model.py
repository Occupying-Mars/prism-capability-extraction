#!/usr/bin/env python3
"""Summarize issue #14 cross-model sparse-frontier replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HIDDEN_SIZE = 1536
TOTAL_MODEL_EQUIV_PARAMS = 1_548_000_000
TOTAL_MLP_CHANNELS = 28 * 8960


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def selected_weight_fraction(mlp: int | None) -> float | None:
    if mlp is None:
        return None
    return (int(mlp) * 3 * HIDDEN_SIZE) / TOTAL_MODEL_EQUIV_PARAMS


def mask_row(run_dir: Path, variant: str, mask: str) -> dict[str, Any]:
    fresh = load_json(run_dir / variant / "fresh_eval.json")
    gen = load_json(run_dir / f"{variant}_fresh_generation.json")
    tf_row = (fresh.get("by_mask") or {}).get(mask) or {}
    gen_row = (gen.get("by_mask") or {}).get(mask) or {}
    mlp = tf_row.get("mlp_kept") or gen_row.get("mlp_kept")
    return {
        "mask": mask,
        "mlp_kept": mlp,
        "mlp_channel_fraction": None if mlp is None else int(mlp) / TOTAL_MLP_CHANNELS,
        "selected_weight_fraction": selected_weight_fraction(mlp),
        "teacher_forced_accuracy": tf_row.get("accuracy"),
        "generation_accuracy": gen_row.get("accuracy"),
    }


def summarize_variant(run_dir: Path, variant: str) -> dict[str, Any]:
    vdir = run_dir / variant
    train = load_json(vdir / "lora" / "train_summary.json")
    merge = load_json(vdir / "merge" / "merge_sweep.json")
    data = load_json(run_dir / "data" / f"{run_dir.name.split('_2digit_')[0]}_{variant}_correct_addition.json")
    rate = load_json(vdir / "rate_distortion.json")
    target = load_json(vdir / "target90_search.json")
    masks = [mask_row(run_dir, variant, "topk_500")]
    if target.get("best"):
        masks.append(mask_row(run_dir, variant, "target90"))
    return {
        "variant": variant,
        "train_final_accuracy": (train.get("final_eval") or {}).get("accuracy"),
        "merge_selection": merge.get("selection") or {},
        "merged_exact_accuracy": (((merge.get("selection") or {}).get("row") or {}).get("exact") or {}).get("accuracy"),
        "merged_kl_mean": ((((merge.get("selection") or {}).get("row") or {}).get("kl") or {}).get("kl_mean")),
        "correct_dataset": {
            "n_tested": data.get("n_tested"),
            "n_correct": data.get("n_correct"),
            "accuracy": None
            if data.get("n_tested") in (None, 0)
            else data.get("n_correct", 0) / max(int(data.get("n_tested", 0)), 1),
        },
        "rate_distortion": {
            name: {
                "accuracy": row.get("accuracy"),
                "mlp_kept": row.get("mlp_kept"),
                "mlp_channel_fraction": row.get("mlp_keep_fraction"),
                "selected_weight_fraction": selected_weight_fraction(row.get("mlp_kept")),
            }
            for name, row in (rate.get("by_mask") or {}).items()
        },
        "target90_search": {
            "best": target.get("best"),
            "best_mask_npz": target.get("best_mask_npz"),
        },
        "fresh_masks": masks,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Issue 14 Cross-Model Replication Summary", ""]
    base = summary.get("base_accuracy") or {}
    lines.extend(
        [
            f"Model: `{summary['model']}`",
            "",
            f"Base 2-digit exact: `{pct(base.get('accuracy'))}` (`{fmt_int(base.get('n_correct'))}/{fmt_int(base.get('n_tested'))}`).",
            "",
            "## Fresh Recovery",
            "",
            "| variant | mask | MLPs | MLP-channel % | selected-weight % | teacher-forced | generation |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant, data in summary["variants"].items():
        for row in data.get("fresh_masks") or []:
            lines.append(
                f"| `{variant}` | `{row['mask']}` | {fmt_int(row.get('mlp_kept'))} | "
                f"{pct(row.get('mlp_channel_fraction'))} | {pct(row.get('selected_weight_fraction'))} | "
                f"{pct(row.get('teacher_forced_accuracy'))} | {pct(row.get('generation_accuracy'))} |"
            )
    lines.extend(["", "## Merge Selection", "", "| variant | merged exact | selected KL | rule |", "|---|---:|---:|---|"])
    for variant, data in summary["variants"].items():
        sel = data.get("merge_selection") or {}
        lines.append(
            f"| `{variant}` | {pct(data.get('merged_exact_accuracy'))} | "
            f"{data.get('merged_kl_mean', 'n/a')} | `{sel.get('rule', 'n/a')}` |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base = load_json(run_dir / "data" / f"{args.model_key}_base_correct_addition.json")
    base_accuracy = {
        "n_tested": base.get("n_tested"),
        "n_correct": base.get("n_correct"),
        "accuracy": None
        if base.get("n_tested") in (None, 0)
        else base.get("n_correct", 0) / max(int(base.get("n_tested", 0)), 1),
    }
    summary = {
        "run_dir": str(run_dir),
        "model": args.model,
        "model_key": args.model_key,
        "base_accuracy": base_accuracy,
        "variants": {
            "rank32_kl": summarize_variant(run_dir, "rank32_kl"),
            "rank32_nokl": summarize_variant(run_dir, "rank32_nokl"),
        },
    }
    out_json = Path(args.out) if args.out else run_dir / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_json.with_suffix(".md"), summary)
    print(f"[done] wrote {out_json} and {out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
