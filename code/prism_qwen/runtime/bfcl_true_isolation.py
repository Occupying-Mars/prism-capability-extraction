#!/usr/bin/env python3
"""Merge a BFCL LoRA adapter and run the physical MLP substrate only.

This differs from ``bfcl_direct_qwen3.py eval-mask``: that path computes the
full MLP and zeroes non-kept channels before ``down_proj``. This path replaces
each layer MLP with a smaller MLP containing only the selected channels, while
leaving attention, norms, residuals, embeddings, and lm_head intact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bfcl_direct_qwen3 import (
    messages_for_generation,
    normalized_prediction_ok,
    parse_tool_calls,
    prediction_ok,
    read_records,
    write_jsonl,
)


def torch_dtype(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"unknown torch dtype: {name}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"not a torch dtype: {name}")
    return dtype


def parse_max_memory(value: str | None) -> dict[int | str, str] | None:
    if not value:
        return None
    out: dict[int | str, str] = {}
    for item in value.split(","):
        key, mem = item.split(":", 1)
        key = key.strip()
        out[int(key) if key.isdigit() else key] = mem.strip()
    return out


def load_topk_mask(path: Path, k: int) -> dict[int, torch.Tensor]:
    scores = torch.tensor(np.load(path)["mlp_scores"])
    flat = scores.flatten()
    k = min(k, flat.numel())
    idx = torch.topk(flat, k=k).indices
    d_ffn = scores.shape[1]
    out = {layer: torch.zeros(d_ffn, dtype=torch.bool) for layer in range(scores.shape[0])}
    for item in idx.tolist():
        out[item // d_ffn][item % d_ffn] = True
    return out


def first_param_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def decoder_root(model: nn.Module) -> nn.Module:
    cur = model
    for _ in range(8):
        if hasattr(cur, "layers"):
            return cur
        for attr in ("model", "base_model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    raise AttributeError("could not find decoder layers")


class SubstrateMLP(nn.Module):
    """Qwen-style gated MLP sliced to the kept intermediate channels."""

    def __init__(self, original: nn.Module, keep_idx: torch.Tensor):
        super().__init__()
        self.act_fn = original.act_fn
        self.hidden_size = original.down_proj.out_features
        self.intermediate_size = int(keep_idx.numel())

        if self.intermediate_size == 0:
            self.register_buffer("_empty", torch.empty(0), persistent=False)
            return

        device = original.gate_proj.weight.device
        keep_idx = keep_idx.to(device=device, dtype=torch.long)
        self.gate_proj = nn.Linear(
            original.gate_proj.in_features,
            self.intermediate_size,
            bias=original.gate_proj.bias is not None,
            device=device,
            dtype=original.gate_proj.weight.dtype,
        )
        self.up_proj = nn.Linear(
            original.up_proj.in_features,
            self.intermediate_size,
            bias=original.up_proj.bias is not None,
            device=device,
            dtype=original.up_proj.weight.dtype,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            original.down_proj.out_features,
            bias=original.down_proj.bias is not None,
            device=device,
            dtype=original.down_proj.weight.dtype,
        )

        with torch.no_grad():
            self.gate_proj.weight.copy_(original.gate_proj.weight.index_select(0, keep_idx))
            self.up_proj.weight.copy_(original.up_proj.weight.index_select(0, keep_idx))
            self.down_proj.weight.copy_(original.down_proj.weight.index_select(1, keep_idx))
            if original.gate_proj.bias is not None:
                self.gate_proj.bias.copy_(original.gate_proj.bias.index_select(0, keep_idx))
            if original.up_proj.bias is not None:
                self.up_proj.bias.copy_(original.up_proj.bias.index_select(0, keep_idx))
            if original.down_proj.bias is not None:
                self.down_proj.bias.copy_(original.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.intermediate_size == 0:
            return torch.zeros((*x.shape[:-1], self.hidden_size), device=x.device, dtype=x.dtype)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    @classmethod
    def empty_like(cls, original: nn.Module, intermediate_size: int) -> "SubstrateMLP":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.act_fn = original.act_fn
        obj.hidden_size = original.down_proj.out_features
        obj.intermediate_size = int(intermediate_size)
        if obj.intermediate_size == 0:
            obj.register_buffer("_empty", torch.empty(0, device=original.down_proj.weight.device), persistent=False)
            return obj
        device = original.down_proj.weight.device
        dtype = original.down_proj.weight.dtype
        obj.gate_proj = nn.Linear(
            original.gate_proj.in_features,
            obj.intermediate_size,
            bias=original.gate_proj.bias is not None,
            device=device,
            dtype=dtype,
        )
        obj.up_proj = nn.Linear(
            original.up_proj.in_features,
            obj.intermediate_size,
            bias=original.up_proj.bias is not None,
            device=device,
            dtype=dtype,
        )
        obj.down_proj = nn.Linear(
            obj.intermediate_size,
            original.down_proj.out_features,
            bias=original.down_proj.bias is not None,
            device=device,
            dtype=dtype,
        )
        return obj


def install_physical_substrate(model: nn.Module, mask: dict[int, torch.Tensor]) -> dict[str, Any]:
    root = decoder_root(model)
    summary = {
        "layers": len(root.layers),
        "kept_total": 0,
        "kept_per_layer": {},
        "mode": "physical_mlp_substrate_full_attention",
        "attention": "unchanged full attention blocks",
    }
    for layer_idx, layer in enumerate(root.layers):
        original = layer.mlp
        keep = mask.get(layer_idx)
        if keep is None:
            keep = torch.zeros(original.gate_proj.out_features, dtype=torch.bool)
        keep_idx = torch.where(keep)[0]
        layer.mlp = SubstrateMLP(original, keep_idx)
        kept = int(keep_idx.numel())
        summary["kept_total"] += kept
        summary["kept_per_layer"][str(layer_idx)] = kept
    return summary


def install_empty_physical_substrate(model: nn.Module, kept_per_layer: dict[str, int]) -> dict[str, Any]:
    root = decoder_root(model)
    summary = {
        "layers": len(root.layers),
        "kept_total": 0,
        "kept_per_layer": {},
        "mode": "physical_mlp_substrate_bundle",
        "attention": "unchanged full attention blocks",
    }
    for layer_idx, layer in enumerate(root.layers):
        kept = int(kept_per_layer[str(layer_idx)])
        layer.mlp = SubstrateMLP.empty_like(layer.mlp, kept)
        summary["kept_total"] += kept
        summary["kept_per_layer"][str(layer_idx)] = kept
    return summary


def load_model_and_tokenizer(model_name: str, dtype: str, device_map: str | None, max_memory: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype(dtype),
        device_map=device_map,
        max_memory=parse_max_memory(max_memory),
    )
    model.eval()
    return model, tokenizer


def move_model_for_single_gpu(model: nn.Module, device_arg: str | None) -> nn.Module:
    if device_arg and device_arg not in ("auto", "cuda"):
        return model.to(device_arg)
    if torch.cuda.is_available():
        return model.to("cuda")
    return model


def merge_adapter(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    if args.output.exists():
        import shutil

        shutil.rmtree(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        max_memory=parse_max_memory(args.max_memory),
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(json.dumps({"merged_dir": str(args.output), "base_model": args.base_model, "adapter": str(args.adapter)}, indent=2))


def run_eval_loop(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    model: nn.Module,
    tokenizer: Any,
    input_device: torch.device,
) -> list[dict[str, Any]]:
    out_rows = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            encoded_items = [
                tokenizer.apply_chat_template(
                    messages_for_generation(
                        row,
                        bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt,
                    ),
                    tools=row.get("tools") or None,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    enable_thinking=args.enable_thinking,
                )
                for row in batch_rows
            ]
            encoded = tokenizer.pad(encoded_items, padding=True, return_tensors="pt").to(input_device)
            if args.decode_backend == "manual-greedy":
                output = manual_greedy_generate(
                    model,
                    encoded,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stop_token_ids=(
                        tokenizer.encode("</tool_call>", add_special_tokens=False) if args.stop_on_tool_call else None
                    ),
                )
            else:
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = encoded["input_ids"].shape[-1]
            for row, seq in zip(batch_rows, output):
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                pred = parse_tool_calls(text)
                raw_correct = prediction_ok(pred, row)
                normalized_correct = normalized_prediction_ok(pred, row)
                out_rows.append(
                    {
                        "id": row["id"],
                        "prediction_text": text,
                        "prediction_calls": pred,
                        "target": row.get("target"),
                        "reference_calls": row.get("reference_calls"),
                        "correct": normalized_correct if args.normalized else raw_correct,
                        "raw_correct": raw_correct,
                        "normalized_correct": normalized_correct,
                    }
                )
            if args.log_every and len(out_rows) % args.log_every == 0:
                print(f"evaluated {len(out_rows)}/{len(rows)}", flush=True)
    return out_rows


def manual_greedy_generate(
    model: nn.Module,
    encoded: dict[str, torch.Tensor],
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: int | list[int] | None,
    stop_token_ids: list[int] | None = None,
) -> torch.Tensor:
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    eos_ids = None
    if eos_token_id is not None:
        eos_values = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        eos_ids = torch.tensor(eos_values, device=input_ids.device, dtype=input_ids.dtype)
    stop_ids = None
    if stop_token_ids:
        stop_ids = torch.tensor(stop_token_ids, device=input_ids.device, dtype=input_ids.dtype)
    unfinished = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    model_kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": True,
    }
    generated = input_ids
    next_sequence_length: int | None = input_ids.shape[-1]
    past_key_values = None

    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            generated,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            is_first_iteration=step == 0,
            **model_kwargs,
        )
        outputs = model(**model_inputs, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        if eos_ids is not None:
            next_tokens = torch.where(unfinished, next_tokens, torch.full_like(next_tokens, pad_token_id))
            unfinished = unfinished & ~((next_tokens[:, None] == eos_ids[None, :]).any(dim=-1))
        generated = torch.cat([generated, next_tokens[:, None]], dim=-1)
        if stop_ids is not None and generated.shape[-1] >= stop_ids.numel():
            matched_stop = (generated[:, -stop_ids.numel() :] == stop_ids[None, :]).all(dim=-1)
            unfinished = unfinished & ~matched_stop
        next_position_ids = model_kwargs["attention_mask"].long().sum(dim=-1, keepdim=True)
        model_kwargs["attention_mask"] = torch.cat(
            [model_kwargs["attention_mask"], torch.ones_like(next_tokens[:, None])],
            dim=-1,
        )
        model_kwargs["position_ids"] = torch.cat([model_kwargs["position_ids"], next_position_ids], dim=-1)
        next_sequence_length = 1
        if not bool(unfinished.any()):
            break
    return generated


def write_eval_summary(
    args: argparse.Namespace,
    out_rows: list[dict[str, Any]],
    substrate_summary: dict[str, Any],
    extra: dict[str, Any],
    note: str,
) -> None:
    write_jsonl(args.output, out_rows)
    judged = len(out_rows)
    correct = sum(int(row["correct"]) for row in out_rows)
    raw_correct = sum(int(row["raw_correct"]) for row in out_rows)
    normalized_correct = sum(int(row["normalized_correct"]) for row in out_rows)
    summary = {
        "examples": judged,
        "exact_correct": correct,
        "exact_accuracy": correct / judged if judged else None,
        "raw_exact_correct": raw_correct,
        "raw_exact_accuracy": raw_correct / judged if judged else None,
        "normalized_exact_correct": normalized_correct,
        "normalized_exact_accuracy": normalized_correct / judged if judged else None,
        "reported_metric": "normalized_exact" if args.normalized else "raw_exact",
        "decode_backend": getattr(args, "decode_backend", "hf-generate"),
        "generations": str(args.output),
        "isolation": substrate_summary,
        "note": note,
    }
    summary.update(extra)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def eval_substrate(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device_map, args.max_memory)
    mask = load_topk_mask(args.attribution, args.topk)
    substrate_summary = install_physical_substrate(model, mask)
    out_rows = run_eval_loop(args, rows, model, tokenizer, input_device=model.device)
    write_eval_summary(
        args,
        out_rows,
        substrate_summary,
        extra={
            "model": args.model,
            "attribution": str(args.attribution),
            "mask_topk": args.topk,
        },
        note="physical MLP substrate only; full attention/norm/residual/lm_head retained",
    )


def export_substrate(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    if args.output.exists():
        import shutil

        shutil.rmtree(args.output)

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device_map, args.max_memory)
    mask = load_topk_mask(args.attribution, args.topk)
    isolation = install_physical_substrate(model, mask)
    args.output.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, args.output / "model.safetensors")
    metadata = {
        "source_model": args.model,
        "attribution": str(args.attribution),
        "topk": args.topk,
        "dtype": args.dtype,
        "format": "qwen_physical_mlp_substrate_v1",
        "isolation": isolation,
    }
    (args.output / "substrate_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"substrate_dir": str(args.output), **metadata}, indent=2))


def load_substrate_bundle(bundle: Path, dtype: str, device: str | None):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    metadata = json.loads((bundle / "substrate_metadata.json").read_text())
    config = AutoConfig.from_pretrained(bundle)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
        isolation = install_empty_physical_substrate(
            model,
            metadata["isolation"]["kept_per_layer"],
        )
    state = load_file(bundle / "model.safetensors", device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch missing={missing} unexpected={unexpected}")
    model = move_model_for_single_gpu(model, device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(bundle)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    isolation["mode"] = "loaded_physical_mlp_substrate_bundle"
    isolation["bundle"] = str(bundle)
    isolation["source_model"] = metadata.get("source_model")
    isolation["topk"] = metadata.get("topk")
    isolation["dtype"] = dtype
    return model, tokenizer, isolation


def eval_substrate_bundle(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer, substrate_summary = load_substrate_bundle(args.bundle, args.dtype, args.device)
    out_rows = run_eval_loop(
        args,
        rows,
        model,
        tokenizer,
        input_device=first_param_device(model.get_input_embeddings()),
    )
    write_eval_summary(
        args,
        out_rows,
        substrate_summary,
        extra={"bundle": str(args.bundle)},
        note="loaded physical MLP substrate bundle; full attention/norm/residual/lm_head retained",
    )


def eval_custom_stack(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from prism_qwen.runtime.qwen_substrate_stack import (
        QwenSubstrateLM,
        manual_greedy_generate_custom,
        maybe_compile_mlp,
    )

    args.decode_backend = "custom-greedy"
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.bundle)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QwenSubstrateLM.from_bundle(
        args.bundle,
        dtype=torch_dtype(args.dtype),
        device=args.device,
        mlp_padding_multiple=args.mlp_padding_multiple,
        mlp_impl=args.mlp_impl,
    )
    if args.compile_mlp:
        maybe_compile_mlp(model)

    out_rows = []
    input_device = first_param_device(model)
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            encoded_items = [
                tokenizer.apply_chat_template(
                    messages_for_generation(
                        row,
                        bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt,
                    ),
                    tools=row.get("tools") or None,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    enable_thinking=args.enable_thinking,
                )
                for row in batch_rows
            ]
            encoded = tokenizer.pad(encoded_items, padding=True, return_tensors="pt").to(input_device)
            output = manual_greedy_generate_custom(
                model,
                encoded["input_ids"],
                encoded["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stop_token_ids=(
                    tokenizer.encode("</tool_call>", add_special_tokens=False) if args.stop_on_tool_call else None
                ),
            )
            prompt_len = encoded["input_ids"].shape[-1]
            for row, seq in zip(batch_rows, output):
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                pred = parse_tool_calls(text)
                raw_correct = prediction_ok(pred, row)
                normalized_correct = normalized_prediction_ok(pred, row)
                out_rows.append(
                    {
                        "id": row["id"],
                        "prediction_text": text,
                        "prediction_calls": pred,
                        "target": row.get("target"),
                        "reference_calls": row.get("reference_calls"),
                        "correct": normalized_correct if args.normalized else raw_correct,
                        "raw_correct": raw_correct,
                        "normalized_correct": normalized_correct,
                    }
                )
            if args.log_every and len(out_rows) % args.log_every == 0:
                print(f"evaluated {len(out_rows)}/{len(rows)}", flush=True)

    write_eval_summary(
        args,
        out_rows,
        {
            "mode": "custom_qwen_substrate_stack",
            "attention": "custom sdpa full attention",
            "mlp": "packed gate/up reduced gated mlp",
            "mlp_impl": args.mlp_impl,
            "compile_mlp": args.compile_mlp,
            "mlp_padding_multiple": args.mlp_padding_multiple,
        },
        extra={"bundle": str(args.bundle)},
        note="custom qwen substrate runner; tokenizer only uses transformers",
    )


def inspect_substrate(args: argparse.Namespace) -> None:
    mask = load_topk_mask(args.attribution, args.topk)
    kept_per_layer = {str(layer): int(channels.sum().item()) for layer, channels in mask.items()}
    print(
        json.dumps(
            {
                "attribution": str(args.attribution),
                "topk": args.topk,
                "kept_total": sum(kept_per_layer.values()),
                "kept_per_layer": kept_per_layer,
            },
            indent=2,
        )
    )


def encode_generation_batches(args: argparse.Namespace, rows: list[dict[str, Any]], tokenizer) -> list[list[dict[str, Any]]]:
    batches = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batches.append(
            [
                tokenizer.apply_chat_template(
                    messages_for_generation(
                        row,
                        bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt,
                    ),
                    tools=row.get("tools") or None,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    enable_thinking=args.enable_thinking,
                )
                for row in batch_rows
            ]
        )
    return batches


def maybe_cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_memory_summary() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def benchmark(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device_map, args.max_memory)
    isolation = {"mode": "full_model"}
    if args.mode == "substrate":
        mask = load_topk_mask(args.attribution, args.topk)
        isolation = install_physical_substrate(model, mask)

    batches = encode_generation_batches(args, rows, tokenizer)
    input_device = first_param_device(model.get_input_embeddings())

    def run_once(measure: bool) -> dict[str, int]:
        examples = 0
        prompt_tokens = 0
        generated_tokens = 0
        for encoded_items in batches:
            encoded = tokenizer.pad(encoded_items, padding=True, return_tensors="pt").to(input_device)
            examples += int(encoded["input_ids"].shape[0])
            prompt_tokens += int(encoded["attention_mask"].sum().item())
            if args.kind == "forward":
                _ = model(**encoded, use_cache=False)
                continue
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_tokens += int((output.shape[-1] - encoded["input_ids"].shape[-1]) * output.shape[0])
        if measure:
            maybe_cuda_sync()
        return {
            "examples": examples,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
        }

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_once(measure=False)
        maybe_cuda_sync()
        reset_cuda_peak()
        start = time.perf_counter()
        totals = {"examples": 0, "prompt_tokens": 0, "generated_tokens": 0}
        for _ in range(args.repeats):
            step = run_once(measure=True)
            for key, value in step.items():
                totals[key] += value
        elapsed = time.perf_counter() - start

    summary = {
        "kind": args.kind,
        "mode": args.mode,
        "model": args.model,
        "pairs": str(args.pairs),
        "examples": totals["examples"],
        "prompt_tokens": totals["prompt_tokens"],
        "generated_tokens": totals["generated_tokens"],
        "elapsed_sec": elapsed,
        "examples_per_sec": totals["examples"] / elapsed if elapsed else None,
        "prompt_tokens_per_sec": totals["prompt_tokens"] / elapsed if elapsed else None,
        "generated_tokens_per_sec": (
            totals["generated_tokens"] / elapsed if elapsed and totals["generated_tokens"] else None
        ),
        "batch_size": args.batch_size,
        "limit": args.limit,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens if args.kind == "generate" else None,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "isolation": isolation,
        "cuda_memory": cuda_memory_summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("merge-adapter")
    p.add_argument("--base-model", default="/root/models/Qwen3-8B")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-memory")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=merge_adapter)

    p = sub.add_parser("export-substrate")
    p.add_argument("--model", required=True, help="merged model path or HF id")
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-memory")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=export_substrate)

    p = sub.add_parser("eval-substrate")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True, help="merged model path or HF id")
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-memory")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--decode-backend", choices=["hf-generate", "manual-greedy"], default="hf-generate")
    p.add_argument("--stop-on-tool-call", action="store_true")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.add_argument("--normalized", action="store_true")
    p.set_defaults(func=eval_substrate)

    p = sub.add_parser("eval-bundle")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--decode-backend", choices=["hf-generate", "manual-greedy"], default="hf-generate")
    p.add_argument("--stop-on-tool-call", action="store_true")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.add_argument("--normalized", action="store_true")
    p.set_defaults(func=eval_substrate_bundle)

    p = sub.add_parser("eval-custom")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--compile-mlp",
        action="store_true",
        help="Disabled fail-closed: torch.compile changed bf16 greedy outputs and was slower in BFCL.",
    )
    p.add_argument(
        "--mlp-padding-multiple",
        type=int,
        default=0,
        help="Disabled for values >1: nonzero padding changed bf16 greedy outputs in BFCL.",
    )
    p.add_argument(
        "--mlp-impl",
        default="packed-module",
        help=(
            "MLP runtime implementation for the custom substrate stack. Supports packed-module, "
            "packed-functional, triton-full, triton-full-m<M>n<N>k<K>, triton-full-padded128, "
            "triton-full-padded128-dyn, triton-full-padded128-decode, "
            "triton-full-padded128-decode-buffered, triton-full-padded128-decode-widthgated, "
            "triton-full-padded128-decode-widthgated-split, "
            "triton-full-padded<P>-decode-widthgated-s<S>l<L>-split, "
            "triton-full-padded<P>-decode-m<M>n<N>k<K>w<W>-widthgated-s<S>l<L>-split, "
            "triton-full-padded<P>-decode-m<M>n<N>k<K>w<W>, "
            "triton-full-padded<P>-decode-m<M>n<N>k<K>-widthgated-s<S>l<L>-split, "
            "triton-full-padded<P>-decode-m<M>n<N>k<K>, triton-silu-down-decode-widthgated-split, "
            "triton-silu-down-decode-m<M>n<N>k<K>w<W>-widthgated-s<S>l<L>-split, "
            "triton-silu-down-decode-m<M>n<N>k<K>w<W>-widthgated-s<S>-split, "
            "triton-silu-down-decode-widthgated-s<S>-split, and triton-full-padded<P>-m<M>n<N>k<K>."
        ),
    )
    p.add_argument("--stop-on-tool-call", action="store_true")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.add_argument("--normalized", action="store_true")
    p.set_defaults(func=eval_custom_stack)

    p = sub.add_parser("inspect-substrate")
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.set_defaults(func=inspect_substrate)

    p = sub.add_parser("benchmark")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True, help="merged model path or HF id")
    p.add_argument("--mode", choices=["full", "substrate"], required=True)
    p.add_argument("--kind", choices=["forward", "generate"], required=True)
    p.add_argument("--attribution", type=Path)
    p.add_argument("--topk", type=int, default=160000)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-memory")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    p.set_defaults(func=benchmark)

    args = parser.parse_args()
    if getattr(args, "mode", None) == "substrate" and args.attribution is None:
        parser.error("benchmark --mode substrate requires --attribution")
    args.func(args)


if __name__ == "__main__":
    main()
