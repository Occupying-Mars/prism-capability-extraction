#!/usr/bin/env bash
# Set up the quant venv on the Lium pod (RTX PRO 6000 Blackwell / sm_120, CUDA 12.8).
# Always uv. Run from /workspace/qwen-quant.
set -euo pipefail
cd /workspace/qwen-quant
export PATH=/root/.local/bin:$PATH

uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# PyTorch built for CUDA 12.8 — required for Blackwell (sm_120) RTX PRO 6000.
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch

# HF stack + SOTA quant backends (issue #4 shortlist):
#   bitsandbytes -> NF4 4-bit QLoRA (LoRA-composable)
#   torchao      -> Int4/Int8 weight-only (PyTorch-native, Marlin-friendly)
uv pip install \
  "transformers>=4.53" "peft>=0.19.1" "accelerate>=1.0" \
  safetensors numpy "huggingface_hub>=0.34" datasets \
  bitsandbytes torchao wandb

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
import importlib
for m in ("transformers", "peft", "bitsandbytes", "torchao"):
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "IMPORT FAIL:", e)
PY
echo "[setup_pod] done"
