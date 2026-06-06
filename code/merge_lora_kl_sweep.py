#!/usr/bin/env python3
"""Select and merge a LoRA adapter scale by KL/accuracy frontier."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.circuit_tracing.batching import greedy_generate_batch
from train_lora_2digit_kl import OBJECTIVES, AdditionDataset, build_examples, collate_rows, move_batch
from union_attribution import pick_device


def parse_scales(text: str) -> list[float]:
    out: list[float] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            parts = [float(x) for x in raw.split(":")]
            if len(parts) != 3:
                raise ValueError(f"invalid scale range {raw!r}; expected start:stop:step")
            start, stop, step = parts
            val = start
            while val <= stop + 1e-9:
                out.append(round(val, 10))
                val += step
        else:
            out.append(float(raw))
    return sorted(set(out))


def capture_lora_scaling(model) -> list[tuple[Any, dict[str, float]]]:
    captured = []
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and scaling:
            captured.append((module, {name: float(value) for name, value in scaling.items()}))
    if not captured:
        raise RuntimeError("no LoRA scaling dictionaries found")
    return captured


def apply_lora_scale(captured: list[tuple[Any, dict[str, float]]], scale: float) -> None:
    for module, base_scaling in captured:
        for name, value in base_scaling.items():
            module.scaling[name] = value * scale


def exact_accuracy(
    model,
    tokenizer,
    *,
    min_value: int,
    max_value: int,
    objective: str,
    seed: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    examples = build_examples(
        min_value=min_value,
        max_value=max_value,
        include_compact=True,
        include_spaced=False,
        objective=objective,
        seed=seed,
    )
    prompts = [ex.prompt for ex in examples]
    answers = [ex.answer for ex in examples]
    preds = greedy_generate_batch(model, tokenizer, prompts, device, max_tokens=6, batch_size=batch_size)
    correct = sum(int(pred == ans) for pred, ans in zip(preds, answers))
    overlong_digit_predictions = sum(
        int(pred != ans and pred.startswith(ans) and pred.isdigit())
        for pred, ans in zip(preds, answers)
    )
    wrong = [
        {"prompt": prompt, "answer": ans, "pred": pred}
        for prompt, ans, pred in zip(prompts, answers, preds)
        if pred != ans
    ]
    return {
        "n": len(answers),
        "correct": correct,
        "accuracy": correct / max(len(answers), 1),
        "overlong_digit_predictions": overlong_digit_predictions,
        "overlong_digit_fraction": overlong_digit_predictions / max(len(answers), 1),
        "wrong_sample": wrong[:50],
    }


def mean_kl_for_loader(model, loader, *, device: str, temperature: float) -> dict[str, float]:
    total_kl = 0.0
    total_positions = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.no_grad(), model.disable_adapter():
            base_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
        with torch.no_grad():
            adapted_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits

        mask = batch["kl_logit_mask"][:, :-1]
        if int(mask.sum().item()) == 0:
            continue
        base = base_logits[:, :-1, :][mask] / temperature
        adapted = adapted_logits[:, :-1, :][mask] / temperature
        kl = F.kl_div(
            F.log_softmax(adapted, dim=-1),
            F.softmax(base, dim=-1),
            reduction="none",
        ).sum(dim=-1) * (temperature**2)
        total_kl += float(kl.sum().cpu())
        total_positions += int(kl.shape[0])

    return {
        "kl_sum": total_kl,
        "kl_positions": total_positions,
        "kl_mean": total_kl / max(total_positions, 1),
    }


def choose_scale(rows: list[dict[str, Any]], min_accuracy: float) -> dict[str, Any]:
    eligible = [row for row in rows if row["exact"]["accuracy"] >= min_accuracy]
    if eligible:
        chosen = min(eligible, key=lambda row: (row["kl"]["kl_mean"], row["scale"]))
        return {"rule": "lowest_kl_at_or_above_min_accuracy", "row": chosen}
    chosen = max(rows, key=lambda row: (row["exact"]["accuracy"], -row["kl"]["kl_mean"]))
    return {"rule": "max_accuracy_no_scale_met_threshold", "row": chosen}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--merged-output-dir", default=None)
    parser.add_argument("--min", type=int, default=10)
    parser.add_argument("--max", type=int, default=99)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kl-batch-size", type=int, default=16)
    parser.add_argument("--kl-examples", type=int, default=2048)
    parser.add_argument("--objective", default="addition", choices=OBJECTIVES)
    parser.add_argument("--kl-on", default="all", choices=["none", "prompt", "answer", "all"])
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--scales", default="0.0:1.5:0.1")
    parser.add_argument("--min-accuracy", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-save-merged", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_output_dir = Path(args.merged_output_dir) if args.merged_output_dir else out_dir / "merged_model"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    scales = parse_scales(args.scales)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kl_examples = build_examples(
        min_value=args.min,
        max_value=args.max,
        include_compact=True,
        include_spaced=True,
        objective=args.objective,
        seed=args.seed,
    )
    random.shuffle(kl_examples)
    kl_examples = kl_examples[: min(args.kl_examples, len(kl_examples))]
    kl_dataset = AdditionDataset(kl_examples, tokenizer, kl_on=args.kl_on)
    kl_loader = DataLoader(
        kl_dataset,
        batch_size=args.kl_batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_rows(rows, tokenizer.pad_token_id),
    )

    print(f"[model] {args.model} adapter={args.adapter_dir} device={device}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    model = PeftModel.from_pretrained(base, args.adapter_dir).to(device).eval()
    captured = capture_lora_scaling(model)

    summary: dict[str, Any] = {
        "args": vars(args),
        "device": device,
        "dtype": args.dtype,
        "scales": scales,
        "frontier": [],
        "started_at": time.time(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(summary, f, indent=2)

    for scale in scales:
        apply_lora_scale(captured, scale)
        kl = mean_kl_for_loader(model, kl_loader, device=device, temperature=args.kl_temperature)
        exact = exact_accuracy(
            model,
            tokenizer,
            min_value=args.min,
            max_value=args.max,
            objective=args.objective,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
        )
        row = {"scale": scale, "kl": kl, "exact": exact}
        if args.objective != "addition":
            row["arithmetic_exact"] = exact_accuracy(
                model,
                tokenizer,
                min_value=args.min,
                max_value=args.max,
                objective="addition",
                seed=args.seed,
                device=device,
                batch_size=args.batch_size,
            )
        summary["frontier"].append(row)
        with open(out_dir / "merge_sweep.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"scale={scale:.4g} kl={kl['kl_mean']:.6g} "
            f"acc={exact['accuracy']:.2%} correct={exact['correct']}/{exact['n']}",
            flush=True,
        )

    selection = choose_scale(summary["frontier"], args.min_accuracy)
    selected_scale = float(selection["row"]["scale"])
    summary["selection"] = selection
    summary["elapsed_s"] = time.time() - summary["started_at"]

    if not args.skip_save_merged:
        apply_lora_scale(captured, selected_scale)
        try:
            merged = model.merge_and_unload(safe_merge=True)
        except TypeError:
            merged = model.merge_and_unload()
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        summary["merged_output_dir"] = str(merged_output_dir)
    else:
        summary["merged_output_dir"] = None

    with open(out_dir / "merge_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[done] selected scale={selected_scale} rule={selection['rule']} "
        f"merged={summary['merged_output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
