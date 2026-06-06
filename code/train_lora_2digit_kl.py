#!/usr/bin/env python3
"""Train a KL-regularized LoRA adapter for 2-digit addition.

This is intentionally small and explicit instead of using Trainer: we need a
reference-model KL term with the LoRA adapter disabled on the same base model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.circuit_tracing.batching import greedy_generate_batch


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class AdditionExample:
    a: int
    b: int
    answer: str
    prompt: str
    train_prompt: str
    variant: str
    objective: str
    carry_signature: str


OBJECTIVES = ["addition", "format_only", "digit_reversal", "random_labels", "copy", "kl_only"]
PROMPT_TEMPLATE_SETS = ["legacy", "canonical", "diverse"]


def carry_positions(a: int, b: int) -> tuple[int, ...]:
    carries = []
    carry = 0
    pos = 1
    while a > 0 or b > 0:
        total = (a % 10) + (b % 10) + carry
        carry = 1 if total >= 10 else 0
        if carry:
            carries.append(pos)
        a //= 10
        b //= 10
        pos += 1
    if carry:
        carries.append(pos)
    return tuple(carries)


def carry_signature_label(a: int, b: int) -> str:
    sig = carry_positions(a, b)
    return "none" if not sig else "carry_" + "_".join(str(pos) for pos in sig)


def train_prompt_variants(
    a: int,
    b: int,
    *,
    canonical_prompt: str,
    include_compact: bool,
    include_spaced: bool,
    prompt_template_set: str,
) -> list[tuple[str, str]]:
    if prompt_template_set == "legacy":
        variants = []
        if include_compact:
            variants.append(("compact", canonical_prompt))
        if include_spaced:
            variants.append(("spaced", canonical_prompt + " "))
        return variants
    if prompt_template_set == "canonical":
        return [("canonical", canonical_prompt), ("canonical_spaced", canonical_prompt + " ")]
    if prompt_template_set == "diverse":
        return [
            ("canonical", canonical_prompt),
            ("canonical_spaced", canonical_prompt + " "),
            ("compact", f"{a}+{b}="),
            ("compact_spaced", f"{a}+{b}= "),
            ("words", f"{a} plus {b} equals "),
            ("question", f"What is {a} + {b}? "),
            ("sum", f"Sum: {a} + {b} = "),
        ]
    raise ValueError(f"unknown prompt_template_set={prompt_template_set!r}")


def objective_answer(a: int, b: int, objective: str, rng: random.Random) -> str:
    correct = str(a + b)
    if objective in {"addition", "kl_only"}:
        return correct
    if objective == "format_only":
        return "0" * len(correct)
    if objective == "digit_reversal":
        return correct[::-1]
    if objective == "random_labels":
        lo = 10 if len(correct) > 1 else 0
        hi = 10 ** len(correct) - 1
        return str(rng.randint(lo, hi))
    if objective == "copy":
        return str(a)
    raise ValueError(f"unknown objective={objective!r}")


class AdditionDataset(Dataset):
    def __init__(self, examples: list[AdditionExample], tokenizer, *, kl_on: str, append_eos: bool = False):
        self.rows = [encode_example(ex, tokenizer, kl_on=kl_on, append_eos=append_eos) for ex in examples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def build_examples(
    *,
    min_value: int,
    max_value: int,
    include_compact: bool,
    include_spaced: bool,
    objective: str = "addition",
    seed: int = 42,
    prompt_template_set: str = "legacy",
    oversample_carry_signatures: set[str] | None = None,
    carry_oversample_multiplier: int = 1,
    hard_pairs: set[tuple[int, int]] | None = None,
    hard_pair_multiplier: int = 1,
) -> list[AdditionExample]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective={objective!r}")
    if prompt_template_set not in PROMPT_TEMPLATE_SETS:
        raise ValueError(f"unknown prompt_template_set={prompt_template_set!r}")
    examples: list[AdditionExample] = []
    label_rng = random.Random(seed + 17_291)
    for a in range(min_value, max_value + 1):
        for b in range(min_value, max_value + 1):
            prompt = f"{a} + {b} ="
            answer = objective_answer(a, b, objective, label_rng)
            carry_signature = carry_signature_label(a, b)
            repeat = 1
            if oversample_carry_signatures and carry_signature in oversample_carry_signatures:
                repeat = max(repeat, carry_oversample_multiplier)
            if hard_pairs and (a, b) in hard_pairs:
                repeat = max(repeat, hard_pair_multiplier)
            variants = train_prompt_variants(
                a,
                b,
                canonical_prompt=prompt,
                include_compact=include_compact,
                include_spaced=include_spaced,
                prompt_template_set=prompt_template_set,
            )
            for _ in range(repeat):
                for variant, train_prompt in variants:
                    examples.append(
                        AdditionExample(
                            a=a,
                            b=b,
                            answer=answer,
                            prompt=prompt,
                            train_prompt=train_prompt,
                            variant=variant,
                            objective=objective,
                            carry_signature=carry_signature,
                        )
                    )
    return examples


def load_hard_pairs(path: str | None) -> set[tuple[int, int]] | None:
    if not path:
        return None
    with open(path) as f:
        payload = json.load(f)
    pairs = set()
    for row in payload.get("examples", []):
        pairs.add((int(row["a"]), int(row["b"])))
    return pairs


def normalize_signature_set(values: list[str]) -> set[str]:
    out = set()
    for value in values:
        value = value.strip()
        if not value:
            continue
        if value != "none" and not value.startswith("carry_"):
            raise ValueError(f"invalid carry signature {value!r}")
        out.add(value)
    return out


def encode_example(ex: AdditionExample, tokenizer, *, kl_on: str, append_eos: bool = False) -> dict[str, Any]:
    prompt_ids = tokenizer(ex.train_prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(ex.train_prompt + ex.answer, add_special_tokens=False)["input_ids"]
    if append_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("--append-eos requested but tokenizer has no eos_token_id")
        full_ids = full_ids + [int(tokenizer.eos_token_id)]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(f"answer did not add tokens for {ex}")

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    logit_mask = [False] * len(full_ids)
    if kl_on == "prompt":
        for idx in range(max(len(prompt_ids) - 1, 0)):
            logit_mask[idx] = True
    elif kl_on == "answer":
        for idx in range(max(len(prompt_ids) - 1, 0), len(full_ids) - 1):
            logit_mask[idx] = True
    elif kl_on == "all":
        for idx in range(len(full_ids) - 1):
            logit_mask[idx] = True
    elif kl_on == "none":
        pass
    else:
        raise ValueError(f"unknown kl_on={kl_on!r}")

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "kl_logit_mask": torch.tensor(logit_mask, dtype=torch.bool),
        "meta": asdict(ex),
    }


def collate_rows(rows: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(int(row["input_ids"].shape[0]) for row in rows)
    batch: dict[str, Any] = {"meta": [row["meta"] for row in rows]}
    for key, pad_value, dtype in [
        ("input_ids", pad_id, torch.long),
        ("labels", -100, torch.long),
        ("attention_mask", 0, torch.long),
    ]:
        values = torch.full((len(rows), max_len), pad_value, dtype=dtype)
        for idx, row in enumerate(rows):
            item = row[key]
            values[idx, : item.shape[0]] = item
        batch[key] = values

    kl_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for idx, row in enumerate(rows):
        item = row["kl_logit_mask"]
        kl_mask[idx, : item.shape[0]] = item
    batch["kl_logit_mask"] = kl_mask
    return batch


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def answer_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def kl_loss(
    adapted_logits: torch.Tensor,
    base_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    logit_mask = mask[:, :-1]
    if int(logit_mask.sum().item()) == 0:
        return adapted_logits.new_zeros(())
    adapted = adapted_logits[:, :-1, :][logit_mask] / temperature
    base = base_logits[:, :-1, :][logit_mask] / temperature
    return F.kl_div(
        F.log_softmax(adapted, dim=-1),
        F.softmax(base, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def exact_accuracy(model, tokenizer, examples: list[AdditionExample], *, device: str, batch_size: int) -> dict[str, Any]:
    compact = [ex for ex in examples if ex.variant == "compact"]
    prompts = [ex.prompt for ex in compact]
    answers = [ex.answer for ex in compact]
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
        "wrong_sample": wrong[:25],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min", type=int, default=10)
    parser.add_argument("--max", type=int, default=99)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--objective", default="addition", choices=OBJECTIVES)
    parser.add_argument("--ce-beta", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--kl-on", default="prompt", choices=["none", "prompt", "answer", "all"])
    parser.add_argument("--include-compact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-spaced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-template-set", default="legacy", choices=PROMPT_TEMPLATE_SETS)
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--oversample-carry-signatures", nargs="*", default=[])
    parser.add_argument("--carry-oversample-multiplier", type=int, default=1)
    parser.add_argument("--hard-pairs-json", default=None)
    parser.add_argument("--hard-pair-multiplier", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    hard_pairs = load_hard_pairs(args.hard_pairs_json)
    oversample_carry_signatures = normalize_signature_set(args.oversample_carry_signatures)
    examples = build_examples(
        min_value=args.min,
        max_value=args.max,
        include_compact=args.include_compact,
        include_spaced=args.include_spaced,
        objective=args.objective,
        seed=args.seed,
        prompt_template_set=args.prompt_template_set,
        oversample_carry_signatures=oversample_carry_signatures,
        carry_oversample_multiplier=args.carry_oversample_multiplier,
        hard_pairs=hard_pairs,
        hard_pair_multiplier=args.hard_pair_multiplier,
    )
    random.shuffle(examples)
    dataset = AdditionDataset(examples, tokenizer, kl_on=args.kl_on, append_eos=args.append_eos)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda rows: collate_rows(rows, tokenizer.pad_token_id),
    )

    print(f"[model] {args.model} device={device} dtype={args.dtype}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    base.config.use_cache = False
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        use_rslora=args.use_rslora,
        bias="none",
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()
    model.train()

    total_batches = math.ceil(len(loader) * args.epochs)
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps * args.grad_accum)
    total_steps = math.ceil(total_batches / args.grad_accum)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    summary: dict[str, Any] = {
        "args": vars(args),
        "device": device,
        "dtype": args.dtype,
        "n_examples": len(examples),
        "n_unique_pairs": (args.max - args.min + 1) ** 2,
        "adapter_dir": str(adapter_dir),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "logs": [],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(summary, f, indent=2)

    compact_examples = build_examples(
        min_value=args.min,
        max_value=args.max,
        include_compact=True,
        include_spaced=False,
        objective=args.objective,
        seed=args.seed,
    )
    arithmetic_examples = build_examples(
        min_value=args.min,
        max_value=args.max,
        include_compact=True,
        include_spaced=False,
        objective="addition",
        seed=args.seed,
    )
    start = time.time()
    global_step = 0
    seen_batches = 0
    running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        for raw_batch in loader:
            if global_step >= total_steps:
                break
            batch = move_batch(raw_batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            ce = answer_ce_loss(outputs.logits, batch["labels"])
            if args.kl_beta > 0 and args.kl_on != "none":
                with torch.no_grad(), model.disable_adapter():
                    base_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    ).logits
                kl = kl_loss(
                    outputs.logits,
                    base_logits,
                    batch["kl_logit_mask"],
                    temperature=args.kl_temperature,
                )
            else:
                kl = outputs.logits.new_zeros(())
            loss = args.ce_beta * ce + args.kl_beta * kl
            (loss / args.grad_accum).backward()
            seen_batches += 1
            running["loss"] += float(loss.detach().cpu())
            running["ce"] += float(ce.detach().cpu())
            running["kl"] += float(kl.detach().cpu())
            running["n"] += 1

            if seen_batches % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % args.eval_every == 0 or global_step == total_steps:
                denom = max(running["n"], 1)
                row = {
                    "step": global_step,
                    "loss": running["loss"] / denom,
                    "ce": running["ce"] / denom,
                    "kl": running["kl"] / denom,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - start,
                }
                model.eval()
                with torch.no_grad():
                    row["compact_exact"] = exact_accuracy(
                        model,
                        tokenizer,
                        compact_examples,
                        device=device,
                        batch_size=max(args.batch_size, 32),
                    )
                    if args.objective != "addition":
                        row["arithmetic_exact"] = exact_accuracy(
                            model,
                            tokenizer,
                            arithmetic_examples,
                            device=device,
                            batch_size=max(args.batch_size, 32),
                        )
                model.train()
                summary["logs"].append(row)
                with open(out_dir / "train_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                print(json.dumps(row), flush=True)
                running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "n": 0}

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    model.eval()
    final_eval = exact_accuracy(
        model,
        tokenizer,
        compact_examples,
        device=device,
        batch_size=max(args.batch_size, 32),
    )
    summary["final_eval"] = final_eval
    if args.objective != "addition":
        summary["final_arithmetic_eval"] = exact_accuracy(
            model,
            tokenizer,
            arithmetic_examples,
            device=device,
            batch_size=max(args.batch_size, 32),
        )
    summary["elapsed_s"] = time.time() - start
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] adapter={adapter_dir} accuracy={final_eval['accuracy']:.4%}", flush=True)


if __name__ == "__main__":
    main()
