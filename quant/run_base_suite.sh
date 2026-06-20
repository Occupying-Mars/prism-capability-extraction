#!/usr/bin/env bash
# Full quant suite on VANILLA Qwen3-8B (no substrate / adapter / mask) — control
# vs the fragile substrate. bf16 before, then each quant, eval on held-out BFCL 1007.
# A100, vLLM 0.23 + transformers 5.12, flashinfer off (no nvcc on image).
set -uo pipefail
cd /workspace/qwen-dyn
set -a; . ./.env; set +a
export HF_TOKEN="$hf_token" WANDB_API_KEY="$wandb_api_key" HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_FLASHINFER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN
PY=/workspace/qwen-dyn/.venv/bin/python
BASE=models/qwen3-8b-base
TRAIN=data/train_mixed.jsonl
PAIRS=data/pairs.jsonl
mkdir -p reports out
EVAL() { $PY vllm_eval.py --pairs "$PAIRS" --enforce-eager "$@"; }

echo "=== [1/5] bf16 base (BEFORE) ==="
EVAL --model "$BASE" --report reports/base_bf16.json

echo "=== [2/5] AutoRound W4 (eval-aware) ==="
$PY autoround_quantize.py --model "$BASE" --train "$TRAIN" --out out/base-ar-w4 --nsamples 512 --iters 200 --seqlen 256
ARDIR=$(ls -d out/base-ar-w4/*/ | head -1)
EVAL --model "$ARDIR" --report reports/base_ar_w4.json

echo "=== [3/5] GPTQ W4A16 (control) ==="
$PY dynamic_quant_experiment.py --scheme W4A16 --model "$BASE" --train "$TRAIN" --out out/base-w4a16
EVAL --model out/base-w4a16 --quantization compressed-tensors --report reports/base_w4a16.json

echo "=== [4/5] W8A8 dynamic (int8) ==="
$PY dynamic_quant_experiment.py --scheme W8A8 --model "$BASE" --train "$TRAIN" --out out/base-w8a8
EVAL --model out/base-w8a8 --quantization compressed-tensors --report reports/base_w8a8.json

echo "=== [5/5] FP8 dynamic ==="
$PY dynamic_quant_experiment.py --scheme FP8_DYNAMIC --model "$BASE" --out out/base-fp8
EVAL --model out/base-fp8 --quantization compressed-tensors --report reports/base_fp8.json

echo "=== DONE ==="
for f in base_bf16 base_ar_w4 base_w4a16 base_w8a8 base_fp8; do
  echo "-- $f --"; cat "reports/$f.json" 2>/dev/null | grep -E 'normalized_exact_correct|recovery_vs'
done
