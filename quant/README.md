# Qwen substrate quantization (issue #4)

Compress the **best-substrate Qwen** without losing its BFCL normalized-exact
score. Tracks survey issue #4 (quantization approaches for the BFCL MLP
substrate). Attention work (issue #3) runs in parallel and is untouched here.

## What the substrate is

```
Qwen/Qwen3-8B
  + b007 r32 rsLoRA adapter        # epsilon_repair, issue #6 tree search
  + issue #12 v13 MACE-90 mask     # keep-only on each mlp.down_proj input
```

| item | value |
|---|---|
| base | `Qwen/Qwen3-8B` (36 layers, d_ffn 12288, 442,368 MLP channels) |
| adapter | b007, rsLoRA r=32 α=64, targets all q/k/v/o + gate/up/down proj |
| mask (MACE-90) | `category_repair_java_r500_protect_tail_b140875_p10000.npz`, topk **140,875** (31.85% MLP) |
| score | 600 / 664 normalized-exact = **90.4% recovery** |
| full anchor | 664 / 1007 |

The intervention path is tokenbender's `scripts/bfcl_direct_qwen3.py eval-mask`
verbatim (keep-only hook on `mlp.down_proj` input). `quantize_substrate.py`
imports those helpers and only adds a weight-quant stage.

## Artifacts (from `TokenBender/circuit-discovery`, HF dataset repo)

`download_artifacts.py` pulls — skipping the heavy per-row eval results:
- b007 adapter weights + tokenizer + configs
- b007 ReLP attribution (`relp_full_collimated.npz`)
- issue #12 refined mask set (`*.npz`, ~422 MB) + frontier/threshold metadata
- BFCL single-call eval inputs (`pairs.jsonl`) to score quantized substrates
- tokenbender's BFCL harness `scripts/`

`substrate_meta/` holds the small JSON receipts (b007 summary, adapter config,
v13 frontier / threshold_hits / manifest) committed to git.

## Staging: attention first, MLP later

The b007 substrate already keeps only ~32% of MLP channels, so the MLP block is
sparse while **attention is dense** — the bigger remaining bit-volume. We
quantize in stages via `--target`:

1. `--target attn` — quantize `self_attn.{q,k,v,o}_proj`, leave the masked MLP
   substrate in bf16. **Current focus.**
2. `--target mlp` — quantize the kept MLP projections. Later stage.
3. `--target both` — full quant once each stage holds its score.

## Quant backends (issue #4 shortlist)

| `--method` | backend | notes |
|---|---|---|
| `nf4` (default) | bitsandbytes | NF4 4-bit + double-quant, QLoRA-style, LoRA stays bf16 |
| `int8` | bitsandbytes | LLM.int8() W8A8 |
| `int4wo` | torchao | Int4 weight-only, Marlin-friendly |
| `int8wo` | torchao | Int8 weight-only |
| `none` | — | bf16 baseline (sanity / anchor) |

`--target` selects modules per backend: bitsandbytes excludes the complement via
`llm_int8_skip_modules`; torchao filters by FQN.

## Run (on the pod)

```bash
# 1. one-time
bash setup_pod.sh

# 2. download artifacts (reads HF token from .env)
set -a; . ./.env; set +a; export HF_TOKEN="$hf_token"
.venv/bin/python download_artifacts.py --mode full --dest ./artifacts

# 3. quantize + eval (wandb on by default, keys from .env)
# bf16 anchor first (must reproduce ~600/664):
.venv/bin/python quantize_substrate.py --method none --eval
# attention-first quant:
.venv/bin/python quantize_substrate.py --target attn --method nf4   --eval
.venv/bin/python quantize_substrate.py --target attn --method int4wo --eval
# MLP later:
.venv/bin/python quantize_substrate.py --target mlp  --method nf4   --eval
```

Runs log to wandb (`prism-bfcl` / group `qwen-substrate-quant`); pass
`--no-wandb` to disable. The bf16 `--method none` full run is the correctness
anchor: it must reproduce ~600/664 normalized-exact (the issue #12 v13 MACE-90
score) before any quant delta is trusted.

Pod: Lium `qwen-quant-substrate` (1×RTX PRO 6000 Blackwell, 96 GB). Branch:
`krishna/qwen-substrate-quant` (off `main`). The parallel attention pod
(`noble-raven-bb`, A100) is not touched.
