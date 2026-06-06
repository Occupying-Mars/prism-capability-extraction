from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Sequence, TypeVar

import torch


T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    if size <= 0:
        raise ValueError(size)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def grouped_batches(
    items: Sequence[T], key_fn: Callable[[T], int], batch_size: int
) -> Iterable[Sequence[T]]:
    if batch_size <= 0:
        raise ValueError(batch_size)
    buckets: dict[int, list[T]] = defaultdict(list)
    for item in items:
        buckets[key_fn(item)].append(item)
    for key in sorted(buckets):
        bucket = buckets[key]
        yield from chunked(bucket, batch_size)


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return
    raise ValueError("Tokenizer has no pad/eos/unk token available for batching")


def greedy_generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    *,
    max_tokens: int,
    batch_size: int,
) -> list[str]:
    ensure_pad_token(tokenizer)
    prompts = list(prompts)
    out: list[str] = [""] * len(prompts)
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None

    by_length: dict[int, list[int]] = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        by_length[prompt_len].append(idx)

    for prompt_len in sorted(by_length):
        for batch_indices in chunked(by_length[prompt_len], batch_size):
            prompt_batch = [prompts[i] for i in batch_indices]
            batch_out = _greedy_generate_same_length_batch(
                model,
                tokenizer,
                prompt_batch,
                device,
                max_tokens=max_tokens,
                pad_id=pad_id,
            )
            for original_idx, pred in zip(batch_indices, batch_out):
                out[original_idx] = pred

    return out


def _greedy_generate_same_length_batch(
    model,
    tokenizer,
    prompt_batch: Sequence[str],
    device: str,
    *,
    max_tokens: int,
    pad_id: int,
) -> list[str]:
    """Greedy-generate a batch whose prompts have identical token length."""
    out: list[str] = []

    if not prompt_batch:
        return out

    enc = tokenizer(
        list(prompt_batch),
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
    digits = [[] for _ in range(input_ids.shape[0])]

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits

        last_pos = attention_mask.sum(dim=1) - 1
        next_ids = logits[
            torch.arange(input_ids.shape[0], device=device), last_pos
        ].argmax(dim=-1)
        decoded = [
            tokenizer.decode(int(tok_id)).strip() for tok_id in next_ids.tolist()
        ]

        prev_finished = finished.clone()
        append_ids = next_ids.clone()
        append_mask = (~prev_finished).to(attention_mask.dtype)

        for i, tok in enumerate(decoded):
            if prev_finished[i]:
                append_ids[i] = pad_id
                append_mask[i] = 0
                continue

            had_digits = bool(digits[i])
            if tok.isdigit():
                digits[i].append(tok)
            elif had_digits:
                finished[i] = True
                append_ids[i] = pad_id
                append_mask[i] = 0

        input_ids = torch.cat([input_ids, append_ids[:, None]], dim=1)
        attention_mask = torch.cat([attention_mask, append_mask[:, None]], dim=1)

        if bool(finished.all()):
            break

    out.extend("".join(ds) for ds in digits)

    return out
