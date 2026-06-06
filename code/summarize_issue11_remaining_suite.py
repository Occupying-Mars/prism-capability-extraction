#!/usr/bin/env python3
"""Summarize the remaining issue #11 experiment suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def rows_from_rate(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, row in (payload.get("by_mask") or {}).items():
        rows.append(
            {
                "mask": name,
                "accuracy": row.get("accuracy"),
                "mlp_kept": row.get("mlp_kept"),
                "by_source_position": row.get("by_source_position") or {},
            }
        )
    return sorted(rows, key=lambda row: (row.get("mlp_kept") or 10**18, row["mask"]))


def smallest_at(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    winners = [row for row in rows if (row.get("accuracy") or 0.0) >= target]
    if not winners:
        return None
    return min(winners, key=lambda row: (row.get("mlp_kept") or 10**18, -(row.get("accuracy") or 0.0)))


def best_route(payload: dict[str, Any], route_key: str) -> dict[str, Any] | None:
    routes = payload.get("route_conditioned") or {}
    row = routes.get(route_key)
    if not row:
        return None
    return {
        "route_key": route_key,
        "accuracy": row.get("accuracy"),
        "correct": row.get("correct"),
        "total": row.get("total"),
        "selected_by_value": row.get("selected_by_value") or {},
    }


def route_breakdown(payload: dict[str, Any], mask_name: str, route_key: str) -> dict[str, Any]:
    mask = (payload.get("by_mask") or {}).get(mask_name) or {}
    return (mask.get("by_route") or {}).get(route_key) or {}


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Issue 11 Remaining Suite Summary", ""]
    lines.append("## Rate-Distortion")
    for variant in ["rank32_kl", "rank32_nokl"]:
        lines.extend(["", f"### {variant}", "", "| mask | MLPs | teacher-forced exact |", "|---|---:|---:|"])
        for row in summary["rate"].get(variant, {}).get("rows", []):
            lines.append(f"| {row['mask']} | {fmt_int(row.get('mlp_kept'))} | {pct(row.get('accuracy'))} |")
        winner = summary["rate"].get(variant, {}).get("smallest_ge_90")
        if winner:
            lines.append("")
            lines.append(
                f"Smallest found >=90%: `{winner['mask']}` at `{fmt_int(winner.get('mlp_kept'))}` MLPs "
                f"with `{pct(winner.get('accuracy'))}` teacher-forced exact."
            )

    lines.extend(["", "## Carry-Regime Routing", "", "| variant | route key | oracle routed exact |", "|---|---|---:|"])
    for variant, by_route in summary["routing"].items():
        for route_key, row in by_route.items():
            lines.append(f"| {variant} | {route_key} | {pct(row.get('accuracy'))} |")

    lines.extend(["", "## Sparse-Mask Carry Breakdown", ""])
    for variant, breakdown in summary["sparse_carry_breakdown"].items():
        lines.extend([f"### {variant}", "", "| carry signature | exact | n |", "|---|---:|---:|"])
        for name, row in sorted(breakdown.items()):
            lines.append(f"| {name} | {pct(row.get('accuracy'))} | {fmt_int(row.get('total'))} |")
        lines.append("")

    lines.extend(["## Denoising / Self-Consistency", ""])
    denoise = summary.get("denoise") or {}
    if denoise:
        initial = denoise.get("initial") or {}
        final = denoise.get("final") or {}
        gen = denoise.get("generation") or {}
        lines.extend(
            [
                "| mask | MLPs | teacher-forced exact | generation exact |",
                "|---|---:|---:|---:|",
                f"| initial sparse | {fmt_int(initial.get('mlp_kept'))} | {pct(initial.get('accuracy'))} | n/a |",
                f"| compressed sparse | {fmt_int(final.get('mlp_kept'))} | {pct(final.get('accuracy'))} | {pct(gen.get('accuracy'))} |",
                "",
            ]
        )
        lines.append(f"Dropped layers: `{denoise.get('dropped_layers', [])}`")
    variants = summary.get("mask_variant_controls") or []
    if variants:
        lines.extend(["", "### Random/Dropout Controls", "", "| mask | MLPs | teacher-forced exact |", "|---|---:|---:|"])
        for row in variants:
            lines.append(f"| {row['mask']} | {fmt_int(row.get('mlp_kept'))} | {pct(row.get('accuracy'))} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rate_kl = rows_from_rate(load_json(run_dir / "rate_distortion_rank32_kl.json"))
    rate_nokl = rows_from_rate(load_json(run_dir / "rate_distortion_rank32_nokl.json"))
    route_kl = load_json(run_dir / "carry_route_rank32_kl.json")
    route_nokl = load_json(run_dir / "carry_route_rank32_nokl.json")
    variants = rows_from_rate(load_json(run_dir / "kl_sparse_variant_controls.json"))
    compress = load_json(run_dir / "kl_sparse_min90_compress.json")
    gen = ((load_json(run_dir / "kl_sparse_min90_generation.json").get("by_mask") or {}).get("compressed_min90") or {})

    summary = {
        "run_dir": str(run_dir),
        "rate": {
            "rank32_kl": {"rows": rate_kl, "smallest_ge_90": smallest_at(rate_kl, 0.9)},
            "rank32_nokl": {"rows": rate_nokl, "smallest_ge_90": smallest_at(rate_nokl, 0.9)},
        },
        "routing": {
            "rank32_kl": {
                key: best_route(route_kl, key)
                for key in ["carry_signature", "n_carries", "leading_carry", "source_x_carry_signature"]
                if best_route(route_kl, key)
            },
            "rank32_nokl": {
                key: best_route(route_nokl, key)
                for key in ["carry_signature", "n_carries", "leading_carry", "source_x_carry_signature"]
                if best_route(route_nokl, key)
            },
        },
        "sparse_carry_breakdown": {
            "rank32_kl_topk500": route_breakdown(route_kl, "topk_500", "carry_signature"),
            "rank32_nokl_topk500": route_breakdown(route_nokl, "topk_500", "carry_signature"),
        },
        "denoise": {
            "initial": compress.get("initial"),
            "final": compress.get("final"),
            "dropped_layers": compress.get("dropped_layers"),
            "generation": gen,
        },
        "mask_variant_controls": variants,
    }

    out_json = Path(args.out) if args.out else run_dir / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_json.with_suffix(".md"), summary)
    print(f"[done] wrote {out_json} and {out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
