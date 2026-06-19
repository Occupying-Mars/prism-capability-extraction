#!/usr/bin/env bash
# Attention-first quant sweep on the full 1007 BFCL pairs (wandb-logged).
#   1. bf16 anchor (correctness gate: must reproduce ~600/664 = v13 MACE-90)
#   2. attention-only NF4 (bitsandbytes)
#   3. attention-only int4 weight-only (torchao)
# MLP quant is a later stage (--target mlp). Run from /workspace/qwen-quant.
set -uo pipefail
cd /workspace/qwen-quant
set -a; . ./.env; set +a
export HF_TOKEN="$hf_token" WANDB_API_KEY="$wandb_api_key"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
mkdir -p reports

echo "=== [1/3] bf16 anchor (correctness) ==="
$PY quantize_substrate.py --method none --target both --eval \
    --report reports/anchor_none_full.json

echo "=== [2/3] attention-first NF4 ==="
$PY quantize_substrate.py --target attn --method nf4 --eval \
    --report reports/attn_nf4_full.json

echo "=== [3/3] attention-first int4wo (torchao) ==="
$PY quantize_substrate.py --target attn --method int4wo --eval \
    --report reports/attn_int4wo_full.json

echo "=== sweep done ==="
for r in reports/anchor_none_full.json reports/attn_nf4_full.json reports/attn_int4wo_full.json; do
  echo "--- $r ---"; cat "$r"; echo
done
