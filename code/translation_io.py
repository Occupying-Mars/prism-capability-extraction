#!/usr/bin/env python3
"""Shared helpers for Hy-MT1.5 English -> Hindi translation experiments.

Centralizes prompt formatting, FLORES devtest loading, and basic
generation so eval, attribution, and training scripts agree on inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch


HY_MT_USER_TEMPLATE = "Translate the following segment into {target_language}, without additional explanation.\n\n{source}"
SARVAM_SYSTEM_TEMPLATE = "Translate the text below to {target_language}."
DEFAULT_TARGET_LANGUAGE = "Hindi"
DEFAULT_PROMPT_STYLE = "hy_mt"


@dataclass(frozen=True)
class Pair:
    src: str
    tgt: str


def build_messages(src: str, *, target_language: str = DEFAULT_TARGET_LANGUAGE,
                    prompt_style: str = DEFAULT_PROMPT_STYLE):
    """Return a chat-template messages list for this prompt style."""
    if prompt_style == "hy_mt":
        return [{"role": "user",
                 "content": HY_MT_USER_TEMPLATE.format(
                     target_language=target_language, source=src)}]
    if prompt_style == "sarvam":
        return [
            {"role": "system",
             "content": SARVAM_SYSTEM_TEMPLATE.format(target_language=target_language)},
            {"role": "user", "content": src},
        ]
    raise ValueError(f"unknown prompt_style={prompt_style!r}")


def build_user_prompt(src: str, target_language: str = DEFAULT_TARGET_LANGUAGE) -> str:
    return HY_MT_USER_TEMPLATE.format(target_language=target_language, source=src)


def apply_chat(tokenizer, src: str, *, target_language: str = DEFAULT_TARGET_LANGUAGE,
               add_generation_prompt: bool = True,
               prompt_style: str = DEFAULT_PROMPT_STYLE) -> str:
    msg = build_messages(src, target_language=target_language,
                          prompt_style=prompt_style)
    return tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=add_generation_prompt,
    )


def encode_chat(tokenizer, src: str, *, target_language: str = DEFAULT_TARGET_LANGUAGE,
                add_generation_prompt: bool = True,
                prompt_style: str = DEFAULT_PROMPT_STYLE) -> torch.Tensor:
    msg = build_messages(src, target_language=target_language,
                          prompt_style=prompt_style)
    return tokenizer.apply_chat_template(
        msg, tokenize=True, add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )


def encode_chat_with_target(tokenizer, src: str, tgt: str, *,
                             target_language: str = DEFAULT_TARGET_LANGUAGE,
                             prompt_style: str = DEFAULT_PROMPT_STYLE) -> Tuple[torch.Tensor, int]:
    """Return (full_ids, prompt_len) where ids[prompt_len:] are the target tokens."""
    prompt_text = apply_chat(tokenizer, src, target_language=target_language,
                             add_generation_prompt=True,
                             prompt_style=prompt_style)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    target_ids = tokenizer(tgt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    full_ids = torch.cat([prompt_ids, target_ids], dim=0)
    return full_ids, int(prompt_ids.shape[0])


# ────────────────────────────────────────────────────────────
# FLORES devtest loader (1012 sentences per pair)
# ────────────────────────────────────────────────────────────

def load_flores_devtest_any(*, src_lang: str = "eng_Latn", tgt_lang: str = "hin_Deva",
                             max_examples: Optional[int] = None) -> List[Pair]:
    """Load FLORES-200 devtest pairs from mteb/flores (one parquet column per
    language code; 1012 rows for devtest)."""
    from datasets import load_dataset
    ds = load_dataset("mteb/flores", split="devtest")
    if src_lang not in ds.column_names:
        raise ValueError(f"src lang {src_lang!r} missing from mteb/flores devtest columns")
    if tgt_lang not in ds.column_names:
        raise ValueError(f"tgt lang {tgt_lang!r} missing from mteb/flores devtest columns")
    pairs: List[Pair] = []
    for row in ds:
        s, t = row[src_lang], row[tgt_lang]
        if not s or not t:
            continue
        pairs.append(Pair(src=s, tgt=t))
        if max_examples is not None and len(pairs) >= max_examples:
            break
    return pairs


# ────────────────────────────────────────────────────────────
# Batched chat-template generation
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_translations(model, tokenizer, sources: List[str], *,
                          target_language: str = DEFAULT_TARGET_LANGUAGE,
                          batch_size: int = 8,
                          max_new_tokens: int = 512,
                          do_sample: bool = False,
                          temperature: float = 0.7,
                          top_k: int = 20, top_p: float = 0.6,
                          repetition_penalty: float = 1.05,
                          prompt_style: str = DEFAULT_PROMPT_STYLE,
                          device: Optional[str] = None) -> List[str]:
    if device is None:
        device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    outputs: List[str] = []
    for start in range(0, len(sources), batch_size):
        chunk = sources[start:start + batch_size]
        # Build chat-formatted strings
        chat_strs = [apply_chat(tokenizer, s, target_language=target_language,
                                  prompt_style=prompt_style) for s in chunk]
        enc = tokenizer(chat_strs, return_tensors="pt", padding=True,
                          add_special_tokens=False, padding_side="left")
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_id,
        )
        if do_sample:
            gen_kwargs.update(dict(top_k=top_k, top_p=top_p,
                                   temperature=temperature,
                                   repetition_penalty=repetition_penalty))
        gen = model.generate(**enc, **gen_kwargs)
        prompt_lens = enc["attention_mask"].sum(dim=1)
        for i, gids in enumerate(gen):
            # generated tokens after the input
            out_tokens = gids[enc["input_ids"].shape[1]:]
            text = tokenizer.decode(out_tokens, skip_special_tokens=True)
            outputs.append(text.strip())
    return outputs
