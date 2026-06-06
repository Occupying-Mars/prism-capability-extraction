"""
Procedural dataset generator for 2-digit addition circuit tracing.

Generates equations  a + b = c  (10 ≤ a,b ≤ 49, 20 ≤ c ≤ 99) organised
by **digit-position properties**.

Hierarchy
---------
property  = "digit D appears somewhere in the 6 digits of the equation"
  └─ subproperty = "digit D is at position P"
       P ∈ {a_tens, a_ones, b_tens, b_ones, sum_tens, sum_ones}

Extra behavioural properties
  └─ carry / no_carry  (ones column produces a carry or not)

Each example is paired with a **minimal counterfactual** that flips the
target property while keeping the other operand unchanged.  This gives
a strong logit-difference signal for circuit tracing.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

POSITIONS = ("a_tens", "a_ones", "b_tens", "b_ones", "sum_tens", "sum_ones")


def _digit_at(a: int, b: int, s: int, pos: str) -> int:
    return {
        "a_tens":   a // 10,
        "a_ones":   a % 10,
        "b_tens":   b // 10,
        "b_ones":   b % 10,
        "sum_tens": s // 10,
        "sum_ones": s % 10,
    }[pos]


def _valid(a: int, b: int) -> bool:
    s = a + b
    return 10 <= a <= 49 and 10 <= b <= 49 and 20 <= s <= 99


def _find_counterfactual(
    a: int, b: int, s: int, pos: str, digit: int,
) -> Optional[Tuple[int, int, int]]:
    """
    Minimal edit: change one operand so the digit at *pos* differs.

    Returns (cf_a, cf_b, cf_sum) or None.
    """
    if pos == "a_tens":
        for delta in (10, -10, 20, -20, 30, -30):
            ca = a + delta
            if 10 <= ca <= 49 and ca // 10 != digit:
                cs = ca + b
                if 20 <= cs <= 99 and cs // 10 != s // 10:
                    return ca, b, cs

    elif pos == "a_ones":
        base = (a // 10) * 10
        for d in range(10):
            if d != digit:
                ca = base + d
                if 10 <= ca <= 49:
                    cs = ca + b
                    if 20 <= cs <= 99 and cs != s:
                        return ca, b, cs

    elif pos == "b_tens":
        for delta in (10, -10, 20, -20, 30, -30):
            cb = b + delta
            if 10 <= cb <= 49 and cb // 10 != digit:
                cs = a + cb
                if 20 <= cs <= 99 and cs // 10 != s // 10:
                    return a, cb, cs

    elif pos == "b_ones":
        base = (b // 10) * 10
        for d in range(10):
            if d != digit:
                cb = base + d
                if 10 <= cb <= 49:
                    cs = a + cb
                    if 20 <= cs <= 99 and cs != s:
                        return a, cb, cs

    elif pos == "sum_tens":
        for delta in (10, -10, 20, -20, 30, -30):
            cb = b + delta
            if 10 <= cb <= 49:
                cs = a + cb
                if 20 <= cs <= 99 and cs // 10 != digit:
                    return a, cb, cs

    elif pos == "sum_ones":
        for delta in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
            cb = b + delta
            if 10 <= cb <= 49:
                cs = a + cb
                if 20 <= cs <= 99 and cs % 10 != digit:
                    return a, cb, cs

    return None


# ──────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────

def generate_dataset(
    a_range: Tuple[int, int] = (10, 49),
    b_range: Tuple[int, int] = (10, 49),
    max_sum: int = 99,
    max_per_subprop: int = 80,
    seed: int = 42,
) -> dict:
    """
    Generate the full structured dataset.

    Returns a nested dict ready to be saved as JSON.
    """
    rng = random.Random(seed)

    # ── enumerate all valid equations ──────────────────────────────
    equations: List[Tuple[int, int, int]] = []
    for a in range(a_range[0], a_range[1] + 1):
        for b in range(b_range[0], b_range[1] + 1):
            s = a + b
            if s <= max_sum:
                equations.append((a, b, s))

    # ── index by (position, digit) ────────────────────────────────
    pos_digit_idx: Dict[Tuple[str, int], List[Tuple[int, int, int]]] = defaultdict(list)
    for a, b, s in equations:
        for pos in POSITIONS:
            d = _digit_at(a, b, s, pos)
            pos_digit_idx[(pos, d)].append((a, b, s))

    # ── build properties ──────────────────────────────────────────
    dataset: dict = {
        "metadata": {
            "total_equations": len(equations),
            "a_range": list(a_range),
            "b_range": list(b_range),
            "max_sum": max_sum,
            "seed": seed,
        },
        "properties": {},
    }

    total_examples = 0

    for digit in range(10):
        dkey = f"digit_{digit}"
        subprops: dict = {}

        for pos in POSITIONS:
            pool = list(pos_digit_idx[(pos, digit)])
            if not pool:
                continue

            rng.shuffle(pool)
            used_pairs: set = set()
            paired: List[dict] = []

            for a, b, s in pool:
                if (a, b) in used_pairs:
                    continue

                cf = _find_counterfactual(a, b, s, pos, digit)
                if cf is None:
                    continue
                cf_a, cf_b, cf_s = cf
                if (cf_a, cf_b) in used_pairs:
                    continue

                # decide metric position (which answer digit changes)
                if s // 10 != cf_s // 10:
                    metric_pos = "tens"
                    target_d = str(s // 10)
                    cf_d = str(cf_s // 10)
                    input_text    = f"{a} + {b} ="
                    cf_input_text = f"{cf_a} + {cf_b} ="
                elif s % 10 != cf_s % 10:
                    metric_pos = "ones"
                    target_d = str(s % 10)
                    cf_d = str(cf_s % 10)
                    # include tens digit in prompt so model predicts ones
                    input_text    = f"{a} + {b} = {s // 10}"
                    cf_input_text = f"{cf_a} + {cf_b} = {cf_s // 10}"
                else:
                    continue  # sums identical — skip

                if target_d == cf_d:
                    continue

                used_pairs.add((a, b))
                used_pairs.add((cf_a, cf_b))

                paired.append({
                    "input_text":    input_text,
                    "answer":        str(s),
                    "a": a,  "b": b,  "sum": s,
                    "cf_input_text": cf_input_text,
                    "cf_answer":     str(cf_s),
                    "cf_a": cf_a,  "cf_b": cf_b,  "cf_sum": cf_s,
                    "target_digit":  target_d,
                    "cf_digit":      cf_d,
                    "metric_position": metric_pos,
                })

                if len(paired) >= max_per_subprop:
                    break

            if paired:
                subprops[pos] = {
                    "description": f"digit {digit} at {pos}",
                    "n_examples":  len(paired),
                    "examples":    paired,
                }
                total_examples += len(paired)

        if subprops:
            dataset["properties"][dkey] = {
                "description": f"Digit {digit} appears in the equation",
                "subproperties": subprops,
            }

    # ── carry / no-carry property ─────────────────────────────────
    carry_examples: List[dict] = []
    no_carry_examples: List[dict] = []
    used_carry: set = set()

    rng.shuffle(equations)
    for a, b, s in equations:
        if (a, b) in used_carry:
            continue
        has_carry = (a % 10) + (b % 10) >= 10

        # counterfactual: flip carry by changing b's ones digit
        target_list = carry_examples if has_carry else no_carry_examples
        if len(target_list) >= max_per_subprop:
            continue

        # find cf that flips carry
        cf = None
        base_b = (b // 10) * 10
        for d in range(10):
            cb = base_b + d
            if not _valid(a, cb):
                continue
            cf_carry = (a % 10) + d >= 10
            if cf_carry != has_carry:
                cf = (a, cb, a + cb)
                break
        if cf is None:
            continue
        cf_a, cf_b, cf_s = cf
        if (cf_a, cf_b) in used_carry:
            continue

        # metric: tens digit should differ (carry shifts it)
        if s // 10 != cf_s // 10:
            metric_pos = "tens"
            target_d = str(s // 10)
            cf_d = str(cf_s // 10)
            input_text    = f"{a} + {b} ="
            cf_input_text = f"{cf_a} + {cf_b} ="
        elif s % 10 != cf_s % 10:
            metric_pos = "ones"
            target_d = str(s % 10)
            cf_d = str(cf_s % 10)
            input_text    = f"{a} + {b} = {s // 10}"
            cf_input_text = f"{cf_a} + {cf_b} = {cf_s // 10}"
        else:
            continue

        if target_d == cf_d:
            continue

        used_carry.add((a, b))
        used_carry.add((cf_a, cf_b))

        entry = {
            "input_text": input_text,
            "answer": str(s),
            "a": a, "b": b, "sum": s,
            "cf_input_text": cf_input_text,
            "cf_answer": str(cf_s),
            "cf_a": cf_a, "cf_b": cf_b, "cf_sum": cf_s,
            "target_digit": target_d,
            "cf_digit": cf_d,
            "metric_position": metric_pos,
            "has_carry": has_carry,
        }
        target_list.append(entry)

    carry_prop: dict = {}
    if carry_examples:
        carry_prop["carry"] = {
            "description": "Ones column produces a carry (a%10 + b%10 >= 10)",
            "n_examples": len(carry_examples),
            "examples": carry_examples,
        }
        total_examples += len(carry_examples)
    if no_carry_examples:
        carry_prop["no_carry"] = {
            "description": "Ones column does NOT produce a carry",
            "n_examples": len(no_carry_examples),
            "examples": no_carry_examples,
        }
        total_examples += len(no_carry_examples)

    if carry_prop:
        dataset["properties"]["carry_behavior"] = {
            "description": "Whether ones-column addition produces a carry",
            "subproperties": carry_prop,
        }

    dataset["metadata"]["total_paired_examples"] = total_examples

    return dataset


# ──────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────

def save_dataset(dataset: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    n = dataset["metadata"]["total_paired_examples"]
    props = len(dataset["properties"])
    print(f"Saved {n} examples across {props} properties -> {path}")


def load_dataset(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate addition circuit-tracing dataset")
    p.add_argument("--output", default="data/addition_dataset.json")
    p.add_argument("--max-per-subprop", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = generate_dataset(max_per_subprop=args.max_per_subprop, seed=args.seed)
    save_dataset(ds, args.output)

    # summary
    print("\nDataset summary:")
    for pname, pdata in ds["properties"].items():
        subs = pdata["subproperties"]
        counts = {k: v["n_examples"] for k, v in subs.items()}
        print(f"  {pname}: {counts}")
