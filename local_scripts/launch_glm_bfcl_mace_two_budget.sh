#!/usr/bin/env bash
set -Eeuo pipefail

TARGET="${LIUM_TARGET:-krishna/prism-2xedition}"
REMOTE_REPO="${REMOTE_REPO:-/root/prism-capability-extraction}"
REMOTE_RUNS="${REMOTE_RUNS:-/root/prism-runs}"
REMOTE_LAUNCH="${REMOTE_RUNS}/glm_bfcl_mace_two_budget.sh"
SESSION="${TMUX_SESSION:-glm_bfcl_mace}"
WANDB_PROJECT_DEFAULT="${WANDB_PROJECT:-prism-bfcl}"
WANDB_GROUP_DEFAULT="${WANDB_GROUP:-glm-bfcl-mace}"
HF_REPO_ID="${HF_REPO_ID:-}"
UPLOAD_HF="${UPLOAD_HF:-0}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 2
  }
}

read_dotenv_value() {
  local key="$1"
  python3 - "$key" <<'PY'
import os
import shlex
import sys
from pathlib import Path

key = sys.argv[1]
path = Path(".env")
if not path.exists():
    raise SystemExit(0)
for raw in path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    if k.strip() != key:
        continue
    v = v.strip()
    if v and v[0] in {"'", '"'}:
        try:
            v = shlex.split(v)[0]
        except ValueError:
            v = v.strip("'\"")
    else:
        v = v.split(" #", 1)[0].strip()
    print(v)
    break
PY
}

need lium
need python3

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

if ! lium exec "$TARGET" "true" >/dev/null; then
  echo "pod target not found: $TARGET" >&2
  exit 3
fi

WANDB_MODE_INPUT="${WANDB_MODE:-$(read_dotenv_value WANDB_MODE)}"
WANDB_MODE_INPUT="${WANDB_MODE_INPUT:-auto}"
WANDB_API_KEY_INPUT="${WANDB_API_KEY:-$(read_dotenv_value WANDB_API_KEY)}"
WANDB_API_KEY_INPUT="${WANDB_API_KEY_INPUT:-$(read_dotenv_value wandb_api_key)}"
WANDB_ENTITY_INPUT="${WANDB_ENTITY:-$(read_dotenv_value WANDB_ENTITY)}"
WANDB_PROJECT_INPUT="${WANDB_PROJECT:-$(read_dotenv_value WANDB_PROJECT)}"
WANDB_GROUP_INPUT="${WANDB_GROUP:-$(read_dotenv_value WANDB_GROUP)}"
HF_TOKEN_INPUT="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-$(read_dotenv_value HF_TOKEN)}}"
HF_TOKEN_INPUT="${HF_TOKEN_INPUT:-$(read_dotenv_value HUGGING_FACE_HUB_TOKEN)}"
HF_TOKEN_INPUT="${HF_TOKEN_INPUT:-$(read_dotenv_value hf_token)}"
if [[ -z "$HF_TOKEN_INPUT" && -s "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN_INPUT="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
fi

WANDB_API_KEY_INPUT="$WANDB_API_KEY_INPUT" \
WANDB_ENTITY_INPUT="$WANDB_ENTITY_INPUT" \
WANDB_PROJECT_INPUT="${WANDB_PROJECT_INPUT:-$WANDB_PROJECT_DEFAULT}" \
WANDB_GROUP_INPUT="${WANDB_GROUP_INPUT:-$WANDB_GROUP_DEFAULT}" \
WANDB_MODE_INPUT="$WANDB_MODE_INPUT" \
HF_TOKEN_INPUT="$HF_TOKEN_INPUT" \
HF_REPO_ID="$HF_REPO_ID" \
UPLOAD_HF="$UPLOAD_HF" \
python3 - "$tmpdir/glm_bfcl_env" <<'PY'
import os
import shlex
import sys
from pathlib import Path

out = Path(sys.argv[1])
lines = {
    "WANDB_MODE": os.environ["WANDB_MODE_INPUT"],
    "WANDB_PROJECT": os.environ["WANDB_PROJECT_INPUT"],
    "WANDB_GROUP": os.environ["WANDB_GROUP_INPUT"],
    "HF_REPO_ID": os.environ["HF_REPO_ID"],
    "UPLOAD_HF": os.environ["UPLOAD_HF"],
}
if os.environ.get("WANDB_ENTITY_INPUT"):
    lines["WANDB_ENTITY"] = os.environ["WANDB_ENTITY_INPUT"]
if os.environ.get("WANDB_API_KEY_INPUT"):
    lines["WANDB_API_KEY"] = os.environ["WANDB_API_KEY_INPUT"]
if os.environ.get("HF_TOKEN_INPUT"):
    lines["HF_TOKEN"] = os.environ["HF_TOKEN_INPUT"]
    lines["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN_INPUT"]
out.write_text("".join(f"export {k}={shlex.quote(v)}\n" for k, v in lines.items()))
out.chmod(0o600)
PY

cat >"$tmpdir/glm_bfcl_mace_two_budget.sh" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_REPO="${REMOTE_REPO:-/root/prism-capability-extraction}"
REMOTE_RUNS="${REMOTE_RUNS:-/root/prism-runs}"
RUN_DIR="${REMOTE_REPO}/runs/glm_bfcl_mace_two_budget"
LOG_DIR="${REMOTE_RUNS}/glm_bfcl_mace_logs"
HF_STAGE="${REMOTE_RUNS}/hf_stage/glm_bfcl_mace_two_budget_v1"
FULL_LOG="${LOG_DIR}/glm_bfcl_mace_two_budget.log"
CONFIG="${REMOTE_REPO}/code/configs/bfcl_glm_mace_two_budget.json"
MODEL="${GLM_MODEL:-/root/models/zai-org/glm-4-9b-chat}"
TRAIN_JSONL="${REMOTE_REPO}/data/bfcl_strict_10k_mix/train.jsonl"
PAIRS_JSONL="${REMOTE_REPO}/data/bfcl_single_call/pairs.jsonl"

source /root/glm_bfcl_env
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REMOTE_REPO}/code:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-prism-bfcl}"
export WANDB_GROUP="${WANDB_GROUP:-glm-bfcl-mace}"
export WANDB_MODE="${WANDB_MODE:-auto}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

uv_bin() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi
  if [[ -x /root/uv-bootstrap/bin/uv ]]; then
    echo /root/uv-bootstrap/bin/uv
    return
  fi
  echo "uv not found on pod" >&2
  exit 5
}

setup_env() {
  cd "$REMOTE_REPO"
  local uv
  uv="$(uv_bin)"
  if [[ ! -x .venv/bin/python ]]; then
    "$uv" venv .venv
  fi
  source .venv/bin/activate
  "$uv" pip install -U pip setuptools wheel
  "$uv" pip install -e code
  "$uv" pip install -U \
    "peft==0.19.1" \
    "transformers==4.44.2" \
    "tokenizers==0.19.1" \
    "huggingface-hub==0.36.0" \
    "wandb>=0.15" \
    "safetensors" \
    "tiktoken>=0.7"
  python - <<'PY'
import os
import torch
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
print("wandb_mode", os.environ.get("WANDB_MODE"))
print("wandb_key", "present" if os.environ.get("WANDB_API_KEY") else "missing")
print("hf_token", "present" if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") else "missing")
PY
}

restore_model() {
  cd "$REMOTE_REPO"
  if [[ -s "${MODEL}/config.json" ]]; then
    return
  fi
  MODEL="$MODEL" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download("zai-org/glm-4-9b-chat", local_dir=os.environ["MODEL"])
PY
}

restore_data() {
  cd "$REMOTE_REPO"
  if [[ ! -s "$PAIRS_JSONL" ]]; then
    python code/scripts/bfcl_direct_qwen3.py download-bfcl-single-call \
      --output-dir data/bfcl_single_call \
      --output data/bfcl_single_call/pairs.jsonl \
      --manifest data/bfcl_single_call/manifest.json
  fi
  if [[ ! -s "$TRAIN_JSONL" ]]; then
    python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

items = [
    "data/bfcl_strict_10k_mix/train.jsonl",
    "data/bfcl_strict_10k_mix/manifest.json",
]
for filename in items:
    src = hf_hub_download(
        "Occupying-Mars/issue49-bfcl-repro-artifacts",
        filename,
        repo_type="dataset",
    )
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(Path(src).read_bytes())
    print(filename, out.stat().st_size)
PY
  fi
}

write_budget_manifest() {
  cd "$REMOTE_REPO"
  MODEL="$MODEL" RUN_DIR="$RUN_DIR" python - <<'PY'
import json
import os
from pathlib import Path
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(os.environ["MODEL"], trust_remote_code=True)
n_layers = int(cfg.num_layers)
d_ffn = int(cfg.ffn_hidden_size)
total = n_layers * d_ffn
budgets = [
    {"name": "p50", "fraction": 0.50, "topk": round(total * 0.50)},
    {"name": "p36", "fraction": 0.36, "topk": round(total * 0.36)},
]
out = {
    "model": os.environ["MODEL"],
    "n_layers": n_layers,
    "d_ffn": d_ffn,
    "total_mlp_channels": total,
    "budgets": budgets,
    "intervention": "selected GLM MLP channels pass; non-selected GLM MLP channels are zeroed",
}
path = Path(os.environ["RUN_DIR"]) / "budget_manifest.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
}

train_lora() {
  cd "$REMOTE_REPO"
  local out="${RUN_DIR}/r0_lora_r32"
  if [[ -s "${out}/train_summary.json" && -d "${out}/adapter" ]]; then
    echo "[skip] train exists"
    return
  fi
  python code/prism_glm/train_bfcl_full_glm.py \
    --model "$MODEL" \
    --train-jsonl "$TRAIN_JSONL" \
    --out-dir "$out" \
    --train-mode lora \
    --prompt-format glm_native \
    --target-format glm_native \
    --epochs 1.0 \
    --batch-size 1 \
    --grad-accum 16 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lr 2e-4 \
    --log-every 25 \
    --save-every 0 \
    --dtype bfloat16 \
    --wandb-mode "$WANDB_MODE" \
    --wandb-entity "${WANDB_ENTITY:-}" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-name glm_bfcl_r0_lora_r32 \
    --wandb-tags bfcl,glm,mace,r0,lora
}

eval_anchor() {
  cd "$REMOTE_REPO"
  local name="$1"
  local adapter="$2"
  local out="${RUN_DIR}/${name}.jsonl"
  if [[ -s "${out%.jsonl}.summary.json" ]]; then
    echo "[skip] eval exists ${name}"
    return
  fi
  local adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  fi
  python code/prism_glm/bfcl_direct_glm.py eval \
    --name "$name" \
    --pairs "$PAIRS_JSONL" \
    --model "$MODEL" \
    "${adapter_args[@]}" \
    --output "$out" \
    --prompt-format glm_native \
    --target-format glm_native \
    --batch-size 8 \
    --dtype bfloat16
}

merge_for_attribution() {
  cd "$REMOTE_REPO"
  local merged="${RUN_DIR}/r0_lora_r32/merged"
  if [[ -s "${merged}/config.json" ]]; then
    echo "[skip] merged target exists"
    return
  fi
  python code/prism_glm/merge_lora.py \
    --base "$MODEL" \
    --adapter "${RUN_DIR}/r0_lora_r32/adapter" \
    --output "$merged" \
    --dtype bfloat16 \
    --device-map auto
}

attribute() {
  cd "$REMOTE_REPO"
  local attr="${RUN_DIR}/r0_lora_r32/relp_full_collimated.npz"
  if [[ -s "$attr" ]]; then
    echo "[skip] attribution exists"
    return
  fi
  python code/prism_glm/bfcl_direct_glm.py relp-attribute \
    --pairs "$PAIRS_JSONL" \
    --output "$attr" \
    --model "${RUN_DIR}/r0_lora_r32/merged" \
    --dtype bfloat16 \
    --device-map auto \
    --prompt-format glm_native \
    --target-format glm_native \
    --log-every 10 \
    --report-topk 50
}

eval_budgets() {
  cd "$REMOTE_REPO"
  local attr="${RUN_DIR}/r0_lora_r32/relp_full_collimated.npz"
  RUN_DIR="$RUN_DIR" python - <<'PY' | while IFS=$'\t' read -r budget topk; do
import json
import os
from pathlib import Path
data = json.loads((Path(os.environ["RUN_DIR"]) / "budget_manifest.json").read_text())
for item in data["budgets"]:
    print(f"{item['name']}\t{item['topk']}")
PY
    local name="glm_r0_lora_r32_${budget}_k${topk}"
    local out="${RUN_DIR}/${name}.jsonl"
    if [[ ! -s "${out%.jsonl}.summary.json" ]]; then
      python code/prism_glm/bfcl_direct_glm.py eval \
        --name "$name" \
        --pairs "$PAIRS_JSONL" \
        --model "$MODEL" \
        --adapter "${RUN_DIR}/r0_lora_r32/adapter" \
        --attribution "$attr" \
        --topk "$topk" \
        --output "$out" \
        --prompt-format glm_native \
        --target-format glm_native \
        --batch-size 8 \
        --dtype bfloat16
    fi
    local bucket_dir="${RUN_DIR}/failure_buckets/${budget}_k${topk}"
    if [[ ! -d "$bucket_dir" ]]; then
      python code/scripts/build_bfcl_failure_buckets.py \
        --eval-jsonl "$out" \
        --pairs-jsonl "$PAIRS_JSONL" \
        --out-dir "$bucket_dir" \
        --run-name "$name"
    fi
  done
}

write_final_summary() {
  cd "$REMOTE_REPO"
  RUN_DIR="$RUN_DIR" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
summary = {
    "experiment_id": "glm_bfcl_mace_two_budget_v1",
    "config": "code/configs/bfcl_glm_mace_two_budget.json",
    "budget_manifest": json.loads((run_dir / "budget_manifest.json").read_text()),
    "results": {},
    "artifact_policy": "merged full model local only; adapter/attribution/evals/buckets are public-safe candidates",
}
for path in sorted(run_dir.glob("*.summary.json")):
    summary["results"][path.stem.removesuffix(".summary")] = json.loads(path.read_text())
for path in sorted((run_dir / "r0_lora_r32").glob("*.summary.json")):
    summary["results"][path.stem.removesuffix(".summary")] = json.loads(path.read_text())
(run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

stage_artifacts() {
  cd "$REMOTE_REPO"
  rm -rf "$HF_STAGE"
  mkdir -p "$HF_STAGE/run" "$HF_STAGE/configs" "$HF_STAGE/logs"
  cp -f "$CONFIG" "$HF_STAGE/configs/bfcl_glm_mace_two_budget.json"
  cp -f docs/experiments/bfcl_glm_mace_two_budget.md "$HF_STAGE/configs/bfcl_glm_mace_two_budget.md"
  cp -f "$FULL_LOG" "$HF_STAGE/logs/glm_bfcl_mace_two_budget.log" || true
  rsync -a \
    --exclude "r0_lora_r32/merged" \
    --exclude "checkpoint_step_*" \
    "$RUN_DIR/" "$HF_STAGE/run/"
  tar -C "$HF_STAGE" -czf "${REMOTE_RUNS}/glm_bfcl_mace_two_budget_artifacts.tgz" .
  sha256sum "${REMOTE_RUNS}/glm_bfcl_mace_two_budget_artifacts.tgz" \
    | tee "${REMOTE_RUNS}/glm_bfcl_mace_two_budget_artifacts.tgz.sha256"
}

upload_hf() {
  if [[ "${UPLOAD_HF:-0}" != "1" ]]; then
    echo "[skip] UPLOAD_HF != 1"
    return
  fi
  if [[ -z "${HF_REPO_ID:-}" ]]; then
    echo "UPLOAD_HF=1 requires HF_REPO_ID" >&2
    exit 6
  fi
  HF_STAGE="$HF_STAGE" python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    folder_path=os.environ["HF_STAGE"],
    path_in_repo="bfcl/glm_mace_two_budget_v1",
    commit_message="Add GLM BFCL MACE two-budget artifacts",
)
print("uploaded_to_hf", os.environ["HF_REPO_ID"], "bfcl/glm_mace_two_budget_v1")
PY
}

run_pipeline() {
  cd "$REMOTE_REPO"
  echo "== glm bfcl mace two-budget pipeline =="
  date
  setup_env
  source .venv/bin/activate
  restore_model
  restore_data
  write_budget_manifest
  eval_anchor glm_base_unmasked ""
  train_lora
  eval_anchor glm_r0_lora_r32_unmasked "${RUN_DIR}/r0_lora_r32/adapter"
  merge_for_attribution
  attribute
  eval_budgets
  write_final_summary
  stage_artifacts
  upload_hf
  echo "== done =="
  date
}

mode="${1:-run}"
case "$mode" in
  check)
    setup_env
    ;;
  run)
    exec > >(tee -a "$FULL_LOG") 2>&1
    run_pipeline
    ;;
  *)
    echo "usage: $0 [check|run]" >&2
    exit 2
    ;;
esac
REMOTE

lium exec "$TARGET" "mkdir -p '$REMOTE_RUNS'"
lium scp "$TARGET" "$tmpdir/glm_bfcl_env" /root/
lium scp "$TARGET" "$tmpdir/glm_bfcl_mace_two_budget.sh" "$REMOTE_LAUNCH"
lium exec "$TARGET" "chmod 600 /root/glm_bfcl_env && chmod +x '$REMOTE_LAUNCH' && tmux new -d -s '$SESSION' 'REMOTE_REPO=$REMOTE_REPO REMOTE_RUNS=$REMOTE_RUNS bash $REMOTE_LAUNCH run' && tmux ls"
