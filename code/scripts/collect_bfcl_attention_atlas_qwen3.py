#!/usr/bin/env python3
"""Build a descriptive BFCL attention activation atlas for Qwen3.

This is activation data, not attribution and not training. It teacher-forces
prompt + gold tool-call continuation and records compact per-query stats for
q/k/v projection outputs plus the o_proj input (OV stream).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_common.wandb_utils import add_wandb_args, init_wandb_run, log_wandb_receipts
from scripts.bfcl_attention_qwen3 import (
    decoder_layers,
    input_device,
    install_mlp_keep_hooks,
    load_mlp_keep_mask,
    load_model_and_tokenizer,
    qwen_attention_shape,
)
from scripts.bfcl_direct_qwen3 import encode_prompt, format_tool_call_target, read_records

SEGMENTS = ("prompt", "target", "full")
STATS = ("mean_abs", "rms", "max_abs")
CHANNEL_SITES = ("q", "k", "v", "ov")
UNIT_SITES = ("q_head", "k_head", "v_head", "ov_head", "qk_group", "qkv_group")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    blob = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world, local_rank


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("eval_id", row.get("id")))


def pair_to_manifest_row(
    item: dict[str, Any],
    *,
    shard_rank: int,
    shard_world: int,
    shard_position: int,
    source_pairs: Path,
    source_pairs_sha256: str,
) -> dict[str, Any]:
    row = item["row"]
    refs = row.get("reference_calls") or row.get("target")
    return {
        "global_index": int(item["global_index"]),
        "shard_rank": int(shard_rank),
        "shard_world": int(shard_world),
        "shard_position": int(shard_position),
        "eval_id": row_id(row),
        "category": row.get("category"),
        "prompt_hash": stable_hash(row.get("messages") or []),
        "tools_hash": stable_hash(row.get("tools") or []),
        "reference_calls_hash": stable_hash(refs),
        "prompt_tokens": int(item["prompt_tokens"]),
        "target_tokens": int(item["target_tokens"]),
        "full_tokens": int(item["full_tokens"]),
        "source_pairs": str(source_pairs),
        "source_pairs_sha256": source_pairs_sha256,
    }


def tokenize_rows(tokenizer, rows: list[tuple[int, dict[str, Any]]], *, enable_thinking: bool) -> list[dict[str, Any]]:
    tokenized = []
    for global_index, row in rows:
        prompt = encode_prompt(tokenizer, row, enable_thinking=enable_thinking)
        target_text = format_tool_call_target(row)
        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        prompt_ids = prompt["input_ids"][0]
        input_ids = np.concatenate([prompt_ids.cpu().numpy(), target_ids.cpu().numpy()]).astype(np.int64)
        tokenized.append(
            {
                "global_index": global_index,
                "row": row,
                "input_ids": input_ids,
                "prompt_tokens": int(prompt_ids.shape[0]),
                "target_tokens": int(target_ids.shape[0]),
                "full_tokens": int(input_ids.shape[0]),
            }
        )
    return tokenized


def batches(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def reduce_activation_segments(
    act: torch.Tensor,
    out: torch.Tensor,
    *,
    layer_idx: int,
    lengths: list[tuple[int, int, int]],
) -> None:
    act = act.detach().to(torch.float32)
    for batch_idx, (prompt_tokens, target_tokens, full_tokens) in enumerate(lengths):
        spans = (
            (0, prompt_tokens),
            (prompt_tokens, prompt_tokens + target_tokens),
            (0, full_tokens),
        )
        for seg_idx, (start, end) in enumerate(spans):
            if end <= start:
                out[batch_idx, seg_idx, :, layer_idx, :].zero_()
                continue
            seg = act[batch_idx, start:end, :]
            abs_seg = seg.abs()
            out[batch_idx, seg_idx, 0, layer_idx, :] = abs_seg.mean(dim=0)
            out[batch_idx, seg_idx, 1, layer_idx, :] = torch.sqrt((seg * seg).mean(dim=0))
            out[batch_idx, seg_idx, 2, layer_idx, :] = abs_seg.max(dim=0).values


def decile_thresholds(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        return np.zeros(9, dtype=np.float32)
    return np.quantile(finite, np.arange(0.1, 1.0, 0.1)).astype(np.float32)


def decile_scores(values: np.ndarray, thresholds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        if thresholds is None:
            thresholds = np.zeros(9, dtype=np.float32)
        return np.ones(flat.shape, dtype=np.uint8).reshape(values.shape), thresholds
    if thresholds is None:
        thresholds = np.quantile(finite, np.arange(0.1, 1.0, 0.1)).astype(np.float32)
    scores = 1 + (flat[:, None] > thresholds[None, :]).sum(axis=1)
    return np.clip(scores, 1, 10).astype(np.uint8).reshape(values.shape), thresholds


def top_flat(values: np.ndarray, *, top_n: int, unit_name: str) -> list[dict[str, Any]]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    n = min(max(top_n, 0), flat.size)
    if n == 0:
        return []
    top_idx = np.argpartition(flat, -n)[-n:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    width = values.shape[-1]
    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        item = int(idx)
        rows.append(
            {
                "rank": rank,
                "layer": item // width,
                unit_name: item % width,
                "value": float(flat[item]),
            }
        )
    return rows


def score_array(
    stats: np.ndarray,
    *,
    local_path: Path,
    global_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    local_scores = np.lib.format.open_memmap(local_path, mode="w+", dtype=np.uint8, shape=stats.shape)
    global_scores = np.lib.format.open_memmap(global_path, mode="w+", dtype=np.uint8, shape=stats.shape)
    n_queries = stats.shape[0]
    local_thresholds = np.zeros((n_queries, len(SEGMENTS), len(STATS), 9), dtype=np.float32)
    global_thresholds = np.zeros((len(SEGMENTS), len(STATS), 9), dtype=np.float32)
    for seg_idx in range(len(SEGMENTS)):
        for stat_idx in range(len(STATS)):
            thresholds = decile_thresholds(stats[:, seg_idx, stat_idx, :, :])
            global_thresholds[seg_idx, stat_idx] = thresholds
            for qidx in range(n_queries):
                local, local_t = decile_scores(stats[qidx, seg_idx, stat_idx, :, :])
                glob, _ = decile_scores(stats[qidx, seg_idx, stat_idx, :, :], thresholds=thresholds)
                local_scores[qidx, seg_idx, stat_idx] = local
                global_scores[qidx, seg_idx, stat_idx] = glob
                local_thresholds[qidx, seg_idx, stat_idx] = local_t
            local_scores.flush()
            global_scores.flush()
    return local_thresholds, global_thresholds


def score_unit_arrays(unit_stats: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    local_scores: dict[str, np.ndarray] = {}
    global_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, np.ndarray] = {}
    for site, arr in unit_stats.items():
        local = np.zeros(arr.shape, dtype=np.uint8)
        glob = np.zeros(arr.shape, dtype=np.uint8)
        site_thresholds = np.zeros((len(SEGMENTS), len(STATS), 9), dtype=np.float32)
        for seg_idx in range(len(SEGMENTS)):
            for stat_idx in range(len(STATS)):
                t = decile_thresholds(arr[:, seg_idx, stat_idx, :, :])
                site_thresholds[seg_idx, stat_idx] = t
                for qidx in range(arr.shape[0]):
                    local[qidx, seg_idx, stat_idx], _ = decile_scores(arr[qidx, seg_idx, stat_idx, :, :])
                    glob[qidx, seg_idx, stat_idx], _ = decile_scores(arr[qidx, seg_idx, stat_idx, :, :], thresholds=t)
        local_scores[site] = local
        global_scores[site] = glob
        thresholds[site] = site_thresholds
    return local_scores, global_scores, thresholds


def collect(args: argparse.Namespace) -> None:
    rank, world, local_rank = distributed_context()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rows = list(enumerate(read_records(args.pairs)))
    if args.limit is not None:
        rows = rows[: args.limit]
    assigned = [item for item in rows if item[0] % world == rank]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_sha = sha256_file(args.pairs)

    model, tokenizer = load_model_and_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = input_device(model)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    n_layers, n_heads, hidden, head_dim = qwen_attention_shape(model)
    n_kv_heads = int(getattr(model.config, "num_key_value_heads", n_heads))
    kv_width = n_kv_heads * head_dim
    site_widths = {"q": hidden, "k": kv_width, "v": kv_width, "ov": hidden}

    mlp_keep, mlp_info = load_mlp_keep_mask(
        args.mlp_attribution,
        args.mlp_topk,
        (n_layers, int(model.config.intermediate_size)),
    )
    mlp_hooks = install_mlp_keep_hooks(model, mlp_keep)

    tokenized = tokenize_rows(tokenizer, assigned, enable_thinking=args.enable_thinking)
    if args.sort_by_length:
        tokenized.sort(key=lambda item: item["full_tokens"])

    shard_count = len(tokenized)
    shard_paths: dict[str, Path] = {}
    shard_mmaps: dict[str, np.ndarray] = {}
    for site, width in site_widths.items():
        path = args.output_dir / f"shard_rank{rank:02d}_{site}_channel_stats_float16.npy"
        shard_paths[site] = path
        shard_mmaps[site] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(shard_count, len(SEGMENTS), len(STATS), n_layers, width),
        )

    active_stats: dict[str, torch.Tensor] | None = None
    active_lengths: list[tuple[int, int, int]] = []
    hooks = []

    def output_hook(site: str, layer_idx: int):
        def hook(_module, _inputs, output):
            if active_stats is None:
                return None
            reduce_activation_segments(output, active_stats[site], layer_idx=layer_idx, lengths=active_lengths)
            return None

        return hook

    def pre_hook(site: str, layer_idx: int):
        def hook(_module, hook_args):
            if active_stats is not None:
                reduce_activation_segments(hook_args[0], active_stats[site], layer_idx=layer_idx, lengths=active_lengths)
            return None

        return hook

    for layer_idx, layer in enumerate(decoder_layers(model)):
        attn = layer.self_attn
        hooks.append(attn.q_proj.register_forward_hook(output_hook("q", layer_idx)))
        hooks.append(attn.k_proj.register_forward_hook(output_hook("k", layer_idx)))
        hooks.append(attn.v_proj.register_forward_hook(output_hook("v", layer_idx)))
        hooks.append(attn.o_proj.register_forward_pre_hook(pre_hook("ov", layer_idx)))

    run = init_wandb_run(
        args,
        default_project="prism-bfcl-attention",
        default_group=args.run_name,
        default_job_type="attention-atlas-collect",
        default_name=args.run_name,
        default_tags="bfcl,qwen3,attention,atlas",
        config={
            "cmd": "collect",
            "pairs": str(args.pairs),
            "model": args.model,
            "adapter": args.adapter,
            "examples": len(rows),
            "assigned_examples": shard_count,
            "rank": rank,
            "world_size": world,
            "segments": list(SEGMENTS),
            "stats": list(STATS),
            "sites": list(CHANNEL_SITES),
            "shape": {
                "layers": n_layers,
                "q_heads": n_heads,
                "kv_heads": n_kv_heads,
                "hidden": hidden,
                "head_dim": head_dim,
                "kv_width": kv_width,
            },
            "mlp_mask": mlp_info,
        },
    )

    query_manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    processed = 0
    started = time.time()
    try:
        for batch_start, batch in batches(tokenized, max(args.batch_size, 1)):
            batch_size = len(batch)
            max_len = max(item["full_tokens"] for item in batch)
            input_ids = torch.full(
                (batch_size, max_len),
                int(tokenizer.pad_token_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
            active_lengths = []
            for bidx, item in enumerate(batch):
                ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
                input_ids[bidx, : ids.shape[0]] = ids
                attention_mask[bidx, : ids.shape[0]] = 1
                active_lengths.append(
                    (
                        int(item["prompt_tokens"]),
                        int(item["target_tokens"]),
                        int(item["full_tokens"]),
                    )
                )

            active_stats = {
                site: torch.zeros(
                    (batch_size, len(SEGMENTS), len(STATS), n_layers, width),
                    dtype=torch.float32,
                    device=device,
                )
                for site, width in site_widths.items()
            }
            try:
                with torch.inference_mode():
                    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                for site in CHANNEL_SITES:
                    shard_mmaps[site][batch_start : batch_start + batch_size] = (
                        active_stats[site].to(torch.float16).cpu().numpy()
                    )
                    shard_mmaps[site].flush()
                for offset, item in enumerate(batch):
                    shard_position = batch_start + offset
                    query_manifest.append(
                        pair_to_manifest_row(
                            item,
                            shard_rank=rank,
                            shard_world=world,
                            shard_position=shard_position,
                            source_pairs=args.pairs,
                            source_pairs_sha256=pairs_sha,
                        )
                    )
                processed += batch_size
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    failures.append(
                        {
                            "global_index": int(item["global_index"]),
                            "eval_id": row_id(item["row"]),
                            "error": repr(exc),
                        }
                    )
                raise
            finally:
                active_stats = None
                del input_ids, attention_mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if processed % max(args.log_every, 1) == 0 or processed == shard_count:
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed > 0 else 0.0
                progress = {
                    "rank": rank,
                    "processed": processed,
                    "assigned": shard_count,
                    "elapsed_sec": round(elapsed, 2),
                    "queries_per_sec": round(rate, 4),
                }
                print(json.dumps(progress, sort_keys=True), flush=True)
                if run is not None:
                    run.log(
                        {
                            "collect/processed": processed,
                            "collect/assigned": shard_count,
                            "collect/elapsed_sec": elapsed,
                            "collect/queries_per_sec": rate,
                        },
                        step=processed,
                    )
    finally:
        for hook in hooks:
            hook.remove()
        for hook in mlp_hooks:
            hook.remove()

    query_manifest.sort(key=lambda row: row["shard_position"])
    query_manifest_path = args.output_dir / f"shard_rank{rank:02d}_query_manifest.jsonl"
    write_jsonl(query_manifest_path, query_manifest)

    elapsed = time.time() - started
    manifest = {
        "artifact": "bfcl_attention_activation_atlas_shard",
        "run_name": args.run_name,
        "model": args.model,
        "adapter": args.adapter or "",
        "dtype": args.dtype,
        "device": str(device),
        "gpu_name": gpu_name,
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "source_pairs": str(args.pairs),
        "source_pairs_sha256": pairs_sha,
        "rows_in_scope": len(rows),
        "assigned_queries": shard_count,
        "queries_processed": processed,
        "failed_queries": len(failures),
        "failures": failures,
        "segments": list(SEGMENTS),
        "stats": list(STATS),
        "channel_sites": list(CHANNEL_SITES),
        "array_order": ["query", "segment", "stat", "layer", "channel"],
        "layers": n_layers,
        "q_heads": n_heads,
        "kv_heads": n_kv_heads,
        "hidden_size": hidden,
        "head_dim": head_dim,
        "site_widths": site_widths,
        "total_channel_cells_per_query": int(n_layers * sum(site_widths.values())),
        "hook_points": {
            "q": "self_attn.q_proj output",
            "k": "self_attn.k_proj output",
            "v": "self_attn.v_proj output",
            "ov": "self_attn.o_proj input",
        },
        "teacher_forced_format": "prompt chat template + gold <tool_call> continuation",
        "descriptive_not_attribution": True,
        "mlp_mask": mlp_info,
        "batch_size": args.batch_size,
        "enable_thinking": args.enable_thinking,
        "sort_by_length": args.sort_by_length,
        "elapsed_sec": elapsed,
        "queries_per_sec": processed / elapsed if elapsed > 0 else None,
        "category_counts": dict(Counter(row.get("category") for _, row in rows)),
        "stats_paths": {site: str(path) for site, path in shard_paths.items()},
        "query_manifest_path": str(query_manifest_path),
    }
    manifest_path = args.output_dir / f"shard_rank{rank:02d}_manifest.json"
    json_dump(manifest_path, manifest)
    print(json.dumps(jsonable(manifest), indent=2, sort_keys=True), flush=True)
    if run is not None:
        log_wandb_receipts(run, name=f"{args.run_name}-attention-atlas-shard{rank:02d}", files=[manifest_path, query_manifest_path])
        run.finish()


def finalize(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest_paths = sorted(args.shard_dir.glob("shard_rank*_manifest.json"))
    if not shard_manifest_paths:
        raise FileNotFoundError(f"no shard manifests in {args.shard_dir}")
    shard_infos = [json.loads(path.read_text()) for path in shard_manifest_paths]
    first = shard_infos[0]
    n_layers = int(first["layers"])
    n_heads = int(first["q_heads"])
    n_kv_heads = int(first["kv_heads"])
    head_dim = int(first["head_dim"])
    site_widths = {site: int(width) for site, width in first["site_widths"].items()}
    kv_group_size = n_heads // n_kv_heads

    query_rows: list[dict[str, Any]] = []
    shard_arrays: dict[str, dict[int, np.ndarray]] = {site: {} for site in CHANNEL_SITES}
    for info in shard_infos:
        rank = int(info["rank"])
        qpath = Path(info["query_manifest_path"])
        if not qpath.is_absolute():
            qpath = args.shard_dir / qpath.name
        query_rows.extend(read_jsonl(qpath))
        for site in CHANNEL_SITES:
            spath = Path(info["stats_paths"][site])
            if not spath.is_absolute():
                spath = args.shard_dir / spath.name
            shard_arrays[site][rank] = np.load(spath, mmap_mode="r")

    query_rows.sort(key=lambda row: int(row["global_index"]))
    n_queries = len(query_rows)
    if n_queries == 0:
        raise ValueError("no queries found")
    duplicates = [eval_id for eval_id, count in Counter(row["eval_id"] for row in query_rows).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate eval_id entries: {duplicates[:10]}")

    query_manifest_path = args.output_dir / "query_manifest.jsonl"
    write_jsonl(query_manifest_path, query_rows)

    stats_paths: dict[str, Path] = {}
    local_score_paths: dict[str, Path] = {}
    global_score_paths: dict[str, Path] = {}
    threshold_payload: dict[str, np.ndarray] = {
        "segments": np.array(SEGMENTS),
        "stats": np.array(STATS),
    }
    channel_stats: dict[str, np.ndarray] = {}

    for site in CHANNEL_SITES:
        shape = (n_queries, len(SEGMENTS), len(STATS), n_layers, site_widths[site])
        stats_path = args.output_dir / f"{site}_channel_stats_float16.npy"
        stats_paths[site] = stats_path
        stats_mm = np.lib.format.open_memmap(stats_path, mode="w+", dtype=np.float16, shape=shape)
        for out_idx, row in enumerate(query_rows):
            rank = int(row["shard_rank"])
            shard_position = int(row["shard_position"])
            stats_mm[out_idx] = shard_arrays[site][rank][shard_position]
        stats_mm.flush()
        channel_stats[site] = np.load(stats_path, mmap_mode="r")

        local_path = args.output_dir / f"{site}_channel_scores_local_uint8.npy"
        global_path = args.output_dir / f"{site}_channel_scores_global_uint8.npy"
        local_t, global_t = score_array(channel_stats[site], local_path=local_path, global_path=global_path)
        local_score_paths[site] = local_path
        global_score_paths[site] = global_path
        threshold_payload[f"{site}_local_thresholds"] = local_t
        threshold_payload[f"{site}_global_thresholds"] = global_t

    np.savez_compressed(args.output_dir / "channel_decile_thresholds.npz", **threshold_payload)

    q_head = np.asarray(channel_stats["q"]).reshape(n_queries, len(SEGMENTS), len(STATS), n_layers, n_heads, head_dim).mean(axis=-1).astype(np.float16)
    k_head = np.asarray(channel_stats["k"]).reshape(n_queries, len(SEGMENTS), len(STATS), n_layers, n_kv_heads, head_dim).mean(axis=-1).astype(np.float16)
    v_head = np.asarray(channel_stats["v"]).reshape(n_queries, len(SEGMENTS), len(STATS), n_layers, n_kv_heads, head_dim).mean(axis=-1).astype(np.float16)
    ov_head = np.asarray(channel_stats["ov"]).reshape(n_queries, len(SEGMENTS), len(STATS), n_layers, n_heads, head_dim).mean(axis=-1).astype(np.float16)
    q_group = q_head.reshape(n_queries, len(SEGMENTS), len(STATS), n_layers, n_kv_heads, kv_group_size).mean(axis=-1)
    qk_group = ((q_group.astype(np.float32) + k_head.astype(np.float32)) / 2.0).astype(np.float16)
    qkv_group = ((q_group.astype(np.float32) + k_head.astype(np.float32) + v_head.astype(np.float32)) / 3.0).astype(np.float16)
    unit_stats = {
        "q_head": q_head,
        "k_head": k_head,
        "v_head": v_head,
        "ov_head": ov_head,
        "qk_group": qk_group,
        "qkv_group": qkv_group,
    }
    np.savez_compressed(
        args.output_dir / "attention_unit_stats_float16.npz",
        segments=np.array(SEGMENTS),
        stats=np.array(STATS),
        unit_sites=np.array(UNIT_SITES),
        **unit_stats,
    )
    unit_local, unit_global, unit_thresholds = score_unit_arrays(unit_stats)
    np.savez_compressed(
        args.output_dir / "attention_unit_scores_local_uint8.npz",
        segments=np.array(SEGMENTS),
        stats=np.array(STATS),
        **unit_local,
    )
    np.savez_compressed(
        args.output_dir / "attention_unit_scores_global_uint8.npz",
        segments=np.array(SEGMENTS),
        stats=np.array(STATS),
        **unit_global,
    )
    np.savez_compressed(
        args.output_dir / "attention_unit_decile_thresholds.npz",
        segments=np.array(SEGMENTS),
        stats=np.array(STATS),
        **unit_thresholds,
    )

    top_channel_rows = []
    for qidx, row in enumerate(query_rows):
        for segment_idx, segment in enumerate(SEGMENTS):
            for stat_idx, stat in enumerate(STATS):
                for site in CHANNEL_SITES:
                    top_channel_rows.append(
                        {
                            "eval_id": row["eval_id"],
                            "global_index": int(row["global_index"]),
                            "site": site,
                            "segment": segment,
                            "stat": stat,
                            "top": top_flat(
                                channel_stats[site][qidx, segment_idx, stat_idx],
                                top_n=args.top_n,
                                unit_name="channel",
                            ),
                        }
                    )
    top_channel_path = args.output_dir / "top_attention_channels_per_query.jsonl"
    write_jsonl(top_channel_path, top_channel_rows)

    top_unit_rows = []
    for qidx, row in enumerate(query_rows):
        for segment_idx, segment in enumerate(SEGMENTS):
            for stat_idx, stat in enumerate(STATS):
                for site, arr in unit_stats.items():
                    unit_name = "kv_group" if site.endswith("_group") else "head"
                    top_unit_rows.append(
                        {
                            "eval_id": row["eval_id"],
                            "global_index": int(row["global_index"]),
                            "site": site,
                            "segment": segment,
                            "stat": stat,
                            "top": top_flat(arr[qidx, segment_idx, stat_idx], top_n=args.top_n, unit_name=unit_name),
                        }
                    )
    top_unit_path = args.output_dir / "top_attention_units_per_query.jsonl"
    write_jsonl(top_unit_path, top_unit_rows)

    category_counts = dict(Counter(row.get("category") for row in query_rows))
    category_summary: dict[str, dict[str, Any]] = {}
    for category in sorted(category_counts):
        indices = [idx for idx, row in enumerate(query_rows) if row.get("category") == category]
        category_summary[str(category)] = {
            "query_count": len(indices),
            "unit_mean": {
                site: np.asarray(arr[indices]).mean(axis=(0, 1, 2, 3)).astype(np.float32).tolist()
                for site, arr in unit_stats.items()
            },
        }
    json_dump(args.output_dir / "bucket_summary_manifest.json", {"category_counts": category_counts})
    np.savez_compressed(
        args.output_dir / "bucket_summary_heatmaps.npz",
        categories=np.array([str(k) for k in category_summary]),
        **{
            f"category_{category}_{site}_unit_mean": np.array(values["unit_mean"][site], dtype=np.float32)
            for category, values in category_summary.items()
            for site in UNIT_SITES
        },
    )

    readme = args.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# BFCL attention activation atlas",
                "",
                "descriptive teacher-forced activation atlas for q/k/v projection outputs and ov stream.",
                "this is activation data for attribution/search prep, not a causal behavior claim.",
                "",
                f"- queries: `{n_queries}`",
                f"- segments: `{', '.join(SEGMENTS)}`",
                f"- stats: `{', '.join(STATS)}`",
                f"- channel sites: `{', '.join(CHANNEL_SITES)}`",
                f"- unit heatmaps: `{', '.join(UNIT_SITES)}`",
                f"- layers: `{n_layers}`",
                f"- q heads: `{n_heads}`",
                f"- kv heads/groups: `{n_kv_heads}`",
                "",
                "channel arrays use axis order `[query, segment, stat, layer, channel]`.",
                "unit arrays use axis order `[query, segment, stat, layer, head_or_group]`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    files_for_checksum = [
        query_manifest_path,
        readme,
        args.output_dir / "channel_decile_thresholds.npz",
        args.output_dir / "attention_unit_stats_float16.npz",
        args.output_dir / "attention_unit_scores_local_uint8.npz",
        args.output_dir / "attention_unit_scores_global_uint8.npz",
        args.output_dir / "attention_unit_decile_thresholds.npz",
        args.output_dir / "bucket_summary_manifest.json",
        args.output_dir / "bucket_summary_heatmaps.npz",
        top_channel_path,
        top_unit_path,
    ]
    files_for_checksum.extend(stats_paths.values())
    files_for_checksum.extend(local_score_paths.values())
    files_for_checksum.extend(global_score_paths.values())

    manifest = {
        "artifact": "bfcl_attention_activation_atlas",
        "model": first["model"],
        "adapter": first.get("adapter", ""),
        "source_pairs": first["source_pairs"],
        "source_pairs_sha256": first["source_pairs_sha256"],
        "queries_processed": n_queries,
        "failed_queries": sum(int(info.get("failed_queries", 0)) for info in shard_infos),
        "segments": list(SEGMENTS),
        "stats": list(STATS),
        "channel_sites": list(CHANNEL_SITES),
        "unit_sites": list(UNIT_SITES),
        "channel_array_order": ["query", "segment", "stat", "layer", "channel"],
        "unit_array_order": ["query", "segment", "stat", "layer", "head_or_group"],
        "score_rule": "uint8 1..10 deciles; local per query/site/segment/stat and global per site/segment/stat",
        "tie_rule": "score = 1 + count(value > decile_threshold); clipped to 1..10",
        "layers": n_layers,
        "q_heads": n_heads,
        "kv_heads": n_kv_heads,
        "kv_group_size": kv_group_size,
        "head_dim": head_dim,
        "site_widths": site_widths,
        "channel_shapes": {site: list(np.load(path, mmap_mode="r").shape) for site, path in stats_paths.items()},
        "unit_shapes": {site: list(arr.shape) for site, arr in unit_stats.items()},
        "mlp_mask": first.get("mlp_mask"),
        "descriptive_not_attribution": True,
        "includes_bfcl_eval_rows": True,
        "category_counts": category_counts,
        "files": {path.name: str(path) for path in files_for_checksum},
    }
    manifest_path = args.output_dir / "attention_atlas_manifest.json"
    json_dump(manifest_path, manifest)
    files_for_checksum.append(manifest_path)

    if args.write_checksums:
        lines = []
        for path in sorted(files_for_checksum, key=lambda p: p.name):
            lines.append(f"{sha256_file(path)}  {path.name}")
        (args.output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jsonable(manifest), indent=2, sort_keys=True))


def validate(args: argparse.Namespace) -> None:
    manifest_path = args.atlas_dir / "attention_atlas_manifest.json"
    query_manifest_path = args.atlas_dir / "query_manifest.jsonl"
    manifest = json.loads(manifest_path.read_text())
    query_rows = read_jsonl(query_manifest_path)
    expected_queries = args.expected_queries or int(manifest["queries_processed"])
    duplicates = [eval_id for eval_id, count in Counter(row["eval_id"] for row in query_rows).items() if count > 1]

    checks: dict[str, Any] = {
        "manifest_exists": manifest_path.exists(),
        "query_manifest_exists": query_manifest_path.exists(),
        "query_count": len(query_rows),
        "query_count_ok": len(query_rows) == expected_queries,
        "duplicate_eval_ids": duplicates,
        "manifest_failed_queries": manifest.get("failed_queries"),
        "channel_sites": list(CHANNEL_SITES),
        "unit_sites": list(UNIT_SITES),
        "channel_checks": {},
        "unit_files_exist": {
            "stats": (args.atlas_dir / "attention_unit_stats_float16.npz").exists(),
            "local_scores": (args.atlas_dir / "attention_unit_scores_local_uint8.npz").exists(),
            "global_scores": (args.atlas_dir / "attention_unit_scores_global_uint8.npz").exists(),
        },
        "top_channels_exists": (args.atlas_dir / "top_attention_channels_per_query.jsonl").exists(),
        "top_units_exists": (args.atlas_dir / "top_attention_units_per_query.jsonl").exists(),
    }
    for site in CHANNEL_SITES:
        stats_path = args.atlas_dir / f"{site}_channel_stats_float16.npy"
        local_path = args.atlas_dir / f"{site}_channel_scores_local_uint8.npy"
        global_path = args.atlas_dir / f"{site}_channel_scores_global_uint8.npy"
        stats = np.load(stats_path, mmap_mode="r")
        local = np.load(local_path, mmap_mode="r")
        glob = np.load(global_path, mmap_mode="r")
        site_ok = (
            stats.shape[0] == expected_queries
            and stats.dtype == np.float16
            and local.shape == stats.shape
            and glob.shape == stats.shape
            and local.dtype == np.uint8
            and glob.dtype == np.uint8
            and int(local.min()) >= 1
            and int(local.max()) <= 10
            and int(glob.min()) >= 1
            and int(glob.max()) <= 10
        )
        checks["channel_checks"][site] = {
            "ok": site_ok,
            "stats_shape": list(stats.shape),
            "stats_dtype": str(stats.dtype),
            "local_dtype": str(local.dtype),
            "global_dtype": str(glob.dtype),
            "local_min": int(local.min()),
            "local_max": int(local.max()),
            "global_min": int(glob.min()),
            "global_max": int(glob.max()),
        }

    unit_stats = np.load(args.atlas_dir / "attention_unit_stats_float16.npz")
    unit_local = np.load(args.atlas_dir / "attention_unit_scores_local_uint8.npz")
    unit_global = np.load(args.atlas_dir / "attention_unit_scores_global_uint8.npz")
    unit_checks = {}
    for site in UNIT_SITES:
        stats = unit_stats[site]
        local = unit_local[site]
        glob = unit_global[site]
        unit_checks[site] = {
            "ok": (
                stats.shape[0] == expected_queries
                and stats.dtype == np.float16
                and local.shape == stats.shape
                and glob.shape == stats.shape
                and local.dtype == np.uint8
                and glob.dtype == np.uint8
                and int(local.min()) >= 1
                and int(local.max()) <= 10
                and int(glob.min()) >= 1
                and int(glob.max()) <= 10
            ),
            "stats_shape": list(stats.shape),
            "local_min": int(local.min()),
            "local_max": int(local.max()),
            "global_min": int(glob.min()),
            "global_max": int(glob.max()),
        }
    checks["unit_checks"] = unit_checks
    checks["category_counts"] = dict(Counter(row.get("category") for row in query_rows))
    ok = (
        checks["query_count_ok"]
        and not duplicates
        and checks["manifest_failed_queries"] == 0
        and all(item["ok"] for item in checks["channel_checks"].values())
        and all(checks["unit_files_exist"].values())
        and all(item["ok"] for item in unit_checks.values())
        and checks["top_channels_exists"]
        and checks["top_units_exists"]
    )
    checks["ok"] = ok
    print(json.dumps(jsonable(checks), indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect_p = sub.add_parser("collect")
    collect_p.add_argument("--pairs", type=Path, required=True)
    collect_p.add_argument("--output-dir", type=Path, required=True)
    collect_p.add_argument("--model", required=True)
    collect_p.add_argument("--adapter")
    collect_p.add_argument("--tokenizer")
    collect_p.add_argument("--dtype", default="bfloat16")
    collect_p.add_argument("--device-map", default="auto")
    collect_p.add_argument("--local-files-only", action="store_true")
    collect_p.add_argument("--merge-adapter", action="store_true")
    collect_p.add_argument("--mlp-attribution", type=Path)
    collect_p.add_argument("--mlp-topk", type=int)
    collect_p.add_argument("--batch-size", type=int, default=8)
    collect_p.add_argument("--limit", type=int)
    collect_p.add_argument("--log-every", type=int, default=25)
    collect_p.add_argument("--enable-thinking", action="store_true")
    collect_p.add_argument("--sort-by-length", action="store_true")
    collect_p.add_argument("--run-name", default="bfcl-attention-atlas")
    add_wandb_args(
        collect_p,
        default_project="prism-bfcl-attention",
        default_job_type="attention-atlas-collect",
        default_tags="bfcl,qwen3,attention,atlas",
    )
    collect_p.set_defaults(func=collect)

    finalize_p = sub.add_parser("finalize")
    finalize_p.add_argument("--shard-dir", type=Path, required=True)
    finalize_p.add_argument("--output-dir", type=Path, required=True)
    finalize_p.add_argument("--top-n", type=int, default=16)
    finalize_p.add_argument("--write-checksums", action="store_true")
    finalize_p.set_defaults(func=finalize)

    validate_p = sub.add_parser("validate")
    validate_p.add_argument("--atlas-dir", type=Path, required=True)
    validate_p.add_argument("--expected-queries", type=int)
    validate_p.set_defaults(func=validate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
