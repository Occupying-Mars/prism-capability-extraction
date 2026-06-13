#!/usr/bin/env bash
set -Eeuo pipefail

TARGET_HOST="${TARGET_HOST:-216.243.220.74}"
TARGET_PORT="${TARGET_PORT:-40099}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_REPO="${REMOTE_REPO:-/root/prism-capability-extraction}"
REMOTE_RUNS="${REMOTE_RUNS:-/root/prism-runs}"
SESSION="${TMUX_SESSION:-glm_tb_replica}"
REMOTE_SCRIPT="${REMOTE_RUNS}/glm_bfcl_tokenbender_replica.sh"

ssh_cmd() {
  ssh -i "$SSH_KEY" -p "$TARGET_PORT" root@"$TARGET_HOST" "$@"
}

scp_to() {
  scp -i "$SSH_KEY" -P "$TARGET_PORT" "$@"
}

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

cat >"$tmpdir/glm_bfcl_tokenbender_replica.sh" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_REPO="${REMOTE_REPO:-/root/prism-capability-extraction}"
REMOTE_RUNS="${REMOTE_RUNS:-/root/prism-runs}"
RUN_DIR="${REMOTE_REPO}/runs/glm_bfcl_tokenbender_replica"
DATA_DIR="${REMOTE_REPO}/data/glm_bfcl_tokenbender_replica"
LOG_DIR="${REMOTE_RUNS}/glm_tb_replica_logs"
LOG="${LOG_DIR}/glm_tb_replica.log"
MODEL="${GLM_MODEL:-/root/models/zai-org/glm-4-9b-chat}"
TRAIN_BASE="${REMOTE_REPO}/data/bfcl_strict_10k_mix/train.jsonl"
PAIRS="${REMOTE_REPO}/data/bfcl_single_call/pairs.jsonl"
MAX_BRANCH_ROUNDS="${MAX_BRANCH_ROUNDS:-20}"
PARALLEL_BRANCHES="${PARALLEL_BRANCHES:-2}"
BEAM_WIDTH="${BEAM_WIDTH:-3}"

source /root/glm_bfcl_env
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-prism-bfcl}"
export WANDB_ENTITY="${WANDB_ENTITY:-krishnapg2315}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="${REMOTE_REPO}/code:${PYTHONPATH:-}"

mkdir -p "$RUN_DIR" "$DATA_DIR" "$LOG_DIR"

setup_env() {
  cd "$REMOTE_REPO"
  if [[ ! -x .venv/bin/python ]]; then
    uv venv .venv
  fi
  source .venv/bin/activate
  uv pip install -U pip setuptools wheel
  uv pip install -e code
  uv pip install -U \
    "peft==0.19.1" \
    "transformers==4.44.2" \
    "tokenizers==0.19.1" \
    "huggingface-hub==0.36.0" \
    "wandb>=0.15" \
    "safetensors" \
    "tiktoken>=0.7"
}

write_budget_manifest() {
  cd "$REMOTE_REPO"
  MODEL="$MODEL" RUN_DIR="$RUN_DIR" python - <<'PY'
import json, os
from pathlib import Path
from transformers import AutoConfig

qwen_total = 36 * 12288
issue2_qwen_topks = [80000, 120000, 160000, 200000, 240000]
issue6_qwen_topks = [40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000]
cfg = AutoConfig.from_pretrained(os.environ["MODEL"], trust_remote_code=True)
total = int(cfg.num_layers) * int(cfg.ffn_hidden_size)
def map_topks(vals):
    return [{"qwen_topk": k, "fraction": k / qwen_total, "glm_topk": round(total * k / qwen_total)} for k in vals]
out = {
    "model": os.environ["MODEL"],
    "total_mlp_channels": total,
    "qwen_reference_total": qwen_total,
    "issue2_ladder": map_topks(issue2_qwen_topks),
    "issue6_ladder": map_topks(issue6_qwen_topks),
}
path = Path(os.environ["RUN_DIR"]) / "budget_manifest.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
}

topks_for() {
  local key="$1"
  RUN_DIR="$RUN_DIR" KEY="$key" python - <<'PY'
import json, os
from pathlib import Path
data = json.loads((Path(os.environ["RUN_DIR"]) / "budget_manifest.json").read_text())
for row in data[os.environ["KEY"]]:
    print(row["glm_topk"])
PY
}

json_get() {
  local path="$1"
  local key="$2"
  python - "$path" "$key" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

run_leak_audit() {
  local train_jsonl="$1"
  local out="$2"
  cd "$REMOTE_REPO"
  python code/scripts/audit_bfcl_train_eval_overlap.py \
    --train-jsonl "$train_jsonl" \
    --eval-jsonl "$PAIRS" \
    --output "$out" \
    --near-threshold 0.85 \
    --fail-on-overlap
}

round_dir() {
  echo "${RUN_DIR}/$1"
}

round_train_dir() {
  echo "$(round_dir "$1")/unmasked_r32"
}

round_train_jsonl() {
  local round="$1"
  if [[ "$round" == "r0" ]]; then
    echo "$TRAIN_BASE"
  elif [[ "$round" == r* ]]; then
    echo "${DATA_DIR}/${round}/train_mixed.jsonl"
  else
    echo "${DATA_DIR}/branches/${round}/train_mixed.jsonl"
  fi
}

branch_data_dir() {
  echo "${DATA_DIR}/branches/$1"
}

train_round() {
  local round="$1"
  local train_jsonl="$2"
  local out_dir
  out_dir="$(round_train_dir "$round")"
  cd "$REMOTE_REPO"
  if [[ -s "${out_dir}/train_summary.json" && -d "${out_dir}/adapter" ]]; then
    echo "[skip] train ${round}"
    return
  fi
  python code/prism_glm/train_bfcl_full_glm.py \
    --model "$MODEL" \
    --train-jsonl "$train_jsonl" \
    --out-dir "$out_dir" \
    --train-mode lora \
    --prompt-format glm_native \
    --target-format glm_native \
    --epochs 1.0 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-seq-length 1024 \
    --lr 2e-4 \
    --warmup-ratio 0.05 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.0 \
    --use-rslora \
    --policy-kl-beta 1.0 \
    --ce-beta 0.2 \
    --kl-temperature 1.0 \
    --log-every 25 \
    --save-every 0 \
    --dtype bfloat16 \
    --wandb-mode "$WANDB_MODE" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group glm-tokenbender-replica \
    --wandb-name "glm_tb_${round}_r32" \
    --wandb-tags "bfcl,glm,tokenbender-replica,${round}"
}

eval_full_round() {
  local round="$1"
  local out="${RUN_DIR}/glm_${round}_r32_unmasked.jsonl"
  cd "$REMOTE_REPO"
  if [[ -s "${out%.jsonl}.summary.json" ]]; then
    echo "[skip] full eval ${round}"
    return
  fi
  python code/prism_glm/bfcl_direct_glm.py eval \
    --name "glm_${round}_r32_unmasked" \
    --pairs "$PAIRS" \
    --model "$MODEL" \
    --adapter "$(round_train_dir "$round")/adapter" \
    --output "$out" \
    --prompt-format glm_native \
    --target-format glm_native \
    --batch-size 8 \
    --dtype bfloat16
}

merge_round() {
  local round="$1"
  local merged="$(round_train_dir "$round")/merged"
  cd "$REMOTE_REPO"
  if [[ -s "${merged}/config.json" ]]; then
    echo "[skip] merge ${round}"
    return
  fi
  python code/prism_glm/merge_lora.py \
    --base "$MODEL" \
    --adapter "$(round_train_dir "$round")/adapter" \
    --output "$merged" \
    --dtype bfloat16 \
    --device-map auto
}

attribute_round() {
  local round="$1"
  local attr="$(round_dir "$round")/relp_full_collimated.npz"
  cd "$REMOTE_REPO"
  if [[ -s "$attr" ]]; then
    echo "[skip] attr ${round}"
    return
  fi
  python code/prism_glm/bfcl_direct_glm.py relp-attribute \
    --pairs "$PAIRS" \
    --output "$attr" \
    --model "$(round_train_dir "$round")/merged" \
    --dtype bfloat16 \
    --device-map auto \
    --prompt-format glm_native \
    --target-format glm_native \
    --log-every 10 \
    --report-topk 50
}

eval_ladder_round() {
  local round="$1"
  local ladder="$2"
  local attr="$(round_dir "$round")/relp_full_collimated.npz"
  cd "$REMOTE_REPO"
  while read -r topk; do
    [[ -n "$topk" ]] || continue
    local out="${RUN_DIR}/glm_${round}_k${topk}_masked.jsonl"
    if [[ ! -s "${out%.jsonl}.summary.json" ]]; then
      python code/prism_glm/bfcl_direct_glm.py eval \
        --name "glm_${round}_k${topk}_masked" \
        --pairs "$PAIRS" \
        --model "$MODEL" \
        --adapter "$(round_train_dir "$round")/adapter" \
        --attribution "$attr" \
        --topk "$topk" \
        --output "$out" \
        --prompt-format glm_native \
        --target-format glm_native \
        --batch-size 8 \
        --dtype bfloat16
    fi
    local bucket_dir="$(round_dir "$round")/failure_buckets_k${topk}"
    if [[ ! -d "$bucket_dir" ]]; then
      python code/scripts/build_bfcl_failure_buckets.py \
        --eval-jsonl "$out" \
        --pairs-jsonl "$PAIRS" \
        --out-dir "$bucket_dir" \
        --run-name "glm_${round}_k${topk}"
    fi
  done < <(topks_for "$ladder")
}

write_round_summary() {
  local round="$1"
  RUN_DIR="$RUN_DIR" ROUND="$round" python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["RUN_DIR"])
round_id = os.environ["ROUND"]
summary = {"round": round_id, "evals": {}}
for p in sorted(root.glob(f"glm_{round_id}_*.summary.json")):
    summary["evals"][p.stem.removesuffix(".summary")] = json.loads(p.read_text())
td = root / round_id / "unmasked_r32" / "train_summary.json"
if td.exists():
    summary["train_summary"] = str(td)
ad = root / round_id / "relp_full_collimated.summary.json"
if ad.exists():
    summary["attribution_summary"] = json.loads(ad.read_text())
(root / round_id / "round_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

failure_dirs_for_round() {
  local round="$1"
  find "$(round_dir "$round")" -maxdepth 1 -type d -name 'failure_buckets_k*' | sort
}

failure_dirs_for_issue2_ladder() {
  local round="$1"
  local topk dir
  while read -r topk; do
    dir="$(round_dir "$round")/failure_buckets_k${topk}"
    [[ -d "$dir" ]] && printf '%s\n' "$dir"
  done < <(topks_for issue2_ladder)
}

build_nearmiss_round() {
  local round="$1"
  local parent="$2"
  local data_dir="${DATA_DIR}/${round}"
  mkdir -p "$data_dir" "$(round_dir "$round")"
  mapfile -t failure_dirs < <(failure_dirs_for_issue2_ladder "$parent")
  python "$REMOTE_REPO/code/scripts/build_bfcl_nearmiss_curriculum.py" \
    --base-train-jsonl "$(round_train_jsonl "$parent")" \
    --eval-jsonl "$PAIRS" \
    --failure-bucket-dirs "${failure_dirs[@]}" \
    --edge-output "${data_dir}/edge.jsonl" \
    --mixed-output "${data_dir}/train_mixed.jsonl" \
    --manifest "${data_dir}/manifest.json" \
    --round-id "$round" \
    --augmentation-ratio 0.20 \
    --max-augmentation-ratio 0.20 \
    --seed "$((600 + ${round#r}))" \
    --fail-on-leak
  run_leak_audit "${data_dir}/train_mixed.jsonl" "$(round_dir "$round")/mixed_overlap_audit.json"
}

build_branch_data() {
  local branch="$1"
  local parent="$2"
  local profile="$3"
  local seed="$4"
  local data_dir="${DATA_DIR}/branches/${branch}"
  mkdir -p "$data_dir" "$(round_dir "$branch")"
  mapfile -t failure_dirs < <(failure_dirs_for_issue2_ladder "$parent")
  python "$REMOTE_REPO/code/scripts/build_bfcl_tree_branch_curriculum.py" \
    --base-train-jsonl "$(round_train_jsonl "$parent")" \
    --eval-jsonl "$PAIRS" \
    --failure-bucket-dirs "${failure_dirs[@]}" \
    --edge-output "${data_dir}/edge.jsonl" \
    --mixed-output "${data_dir}/train_mixed.jsonl" \
    --manifest "${data_dir}/manifest.json" \
    --branch-id "$branch" \
    --parent-id "$parent" \
    --branch-profile "$profile" \
    --seed "$seed" \
    --fail-on-leak
  run_leak_audit "${data_dir}/train_mixed.jsonl" "$(round_dir "$branch")/mixed_overlap_audit.json"
}

stage_round() {
  local round="$1"
  local train_jsonl="$2"
  local ladder="${3:-issue2_ladder}"
  train_round "$round" "$train_jsonl"
  eval_full_round "$round"
  merge_round "$round"
  attribute_round "$round"
  eval_ladder_round "$round" "$ladder"
  write_round_summary "$round"
}

write_branch_summary() {
  local spec="$1"
  local branch
  branch="$(json_get "$spec" branch_id)"
  RUN_DIR="$RUN_DIR" DATA_DIR="$DATA_DIR" BRANCH="$branch" SPEC="$spec" python - <<'PY'
import json, math, os
from pathlib import Path

root = Path(os.environ["RUN_DIR"])
spec = json.loads(Path(os.environ["SPEC"]).read_text())
branch = os.environ["BRANCH"]
run_dir = root / branch
data_dir = Path(os.environ["DATA_DIR"]) / "branches" / branch
anchor_path = root / "glm_r0_r32_unmasked.summary.json"
anchor = 0
if anchor_path.exists():
    anchor = int(json.loads(anchor_path.read_text()).get("normalized_exact_correct") or 0)
thresholds_abs = {label: math.ceil(anchor * frac) if anchor else None for label, frac in {"80": 0.80, "85": 0.85, "90": 0.90}.items()}
summary = {
    **spec,
    "train_jsonl": str(data_dir / "train_mixed.jsonl"),
    "manifest": str(data_dir / "manifest.json"),
    "leak_audit": str(run_dir / "mixed_overlap_audit.json"),
    "train_summary": str(run_dir / "unmasked_r32" / "train_summary.json"),
    "attribution": str(run_dir / "relp_full_collimated.npz"),
    "full_anchor_normalized_correct": anchor or None,
    "evals": {},
    "thresholds": {},
}
train_path = data_dir / "train_mixed.jsonl"
if train_path.exists():
    summary["train_rows"] = sum(1 for line in train_path.read_text().splitlines() if line.strip())
for p in sorted(root.glob(f"glm_{branch}_k*_masked.summary.json")):
    topk = p.name.split("_k", 1)[1].split("_masked", 1)[0]
    row = json.loads(p.read_text())
    correct = row.get("normalized_exact_correct")
    row["behavior_recovery_vs_full_anchor"] = correct / anchor if anchor and isinstance(correct, int) else None
    summary["evals"][topk] = row
for label, threshold in thresholds_abs.items():
    hits = []
    if threshold is not None:
        for topk, row in summary["evals"].items():
            correct = row.get("normalized_exact_correct")
            if isinstance(correct, int) and correct >= threshold:
                hits.append(int(topk))
    summary["thresholds"][f"smallest_k_ge_{label}_percent"] = min(hits) if hits else None
scores = [row.get("normalized_exact_correct") for row in summary["evals"].values()]
scores = [x for x in scores if isinstance(x, int)]
summary["main_frontier_best_correct"] = max(scores) if scores else None
summary["survivor"] = bool(
    summary["thresholds"].get("smallest_k_ge_80_percent") is not None
    or (summary["main_frontier_best_correct"] is not None and anchor and summary["main_frontier_best_correct"] >= math.ceil(anchor * 0.75))
)
(run_dir / "branch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

write_tree_state() {
  RUN_DIR="$RUN_DIR" MAX_BRANCH_ROUNDS="$MAX_BRANCH_ROUNDS" python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["RUN_DIR"])
summaries = [json.loads(p.read_text()) for p in sorted(root.glob("b*/branch_summary.json"))]
best_by_topk = {}
for row in summaries:
    for topk, ev in row.get("evals", {}).items():
        correct = ev.get("normalized_exact_correct")
        if isinstance(correct, int) and (topk not in best_by_topk or correct > best_by_topk[topk]["correct"]):
            best_by_topk[topk] = {"correct": correct, "branch_id": row["branch_id"], "branch_profile": row["branch_profile"]}
survivors = [row for row in summaries if row.get("survivor")]
state = {
    "experiment_id": "glm_bfcl_tokenbender_issue6_replica",
    "branches_completed": len(summaries),
    "max_branch_rounds": int(os.environ["MAX_BRANCH_ROUNDS"]),
    "best_by_topk": best_by_topk,
    "survivor_count": len(survivors),
    "survivors": [
        {
            "branch_id": row["branch_id"],
            "parent_id": row["parent_id"],
            "depth": row.get("depth", 1),
            "branch_profile": row["branch_profile"],
            "main_frontier_best_correct": row.get("main_frontier_best_correct"),
            "thresholds": row.get("thresholds", {}),
        }
        for row in survivors
    ],
    "should_continue": len(summaries) < int(os.environ["MAX_BRANCH_ROUNDS"]),
}
(root / "tree_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
print(json.dumps(state, indent=2, sort_keys=True))
PY
}

plan_next_wave() {
  local wave_dir="$1"
  mkdir -p "$wave_dir"
  RUN_DIR="$RUN_DIR" WAVE_DIR="$wave_dir" MAX_BRANCH_ROUNDS="$MAX_BRANCH_ROUNDS" PARALLEL_BRANCHES="$PARALLEL_BRANCHES" BEAM_WIDTH="$BEAM_WIDTH" python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["RUN_DIR"])
wave_dir = Path(os.environ["WAVE_DIR"])
max_rounds = int(os.environ["MAX_BRANCH_ROUNDS"])
parallel = int(os.environ["PARALLEL_BRANCHES"])
beam_width = int(os.environ["BEAM_WIDTH"])
profiles_first = ["conservative_nearmiss", "bucket_balanced", "teacher_ranked", "schema_stratified", "compression_biased", "hardcase_replay"]
profiles_next = ["epsilon_repair", "pareto_trim", "compression_biased", "teacher_ranked", "bucket_balanced", "schema_stratified", "conservative_nearmiss", "hardcase_replay"]
summaries = [json.loads(p.read_text()) for p in sorted(root.glob("b*/branch_summary.json"))]
remaining = max_rounds - len(summaries)
if remaining <= 0:
    print("planned 0")
    raise SystemExit(0)
specs = []
if len(summaries) < len(profiles_first):
    for profile in profiles_first[len(summaries): len(summaries) + min(parallel, remaining)]:
        idx = len(summaries) + len(specs) + 1
        specs.append({"branch_id": f"b{idx:03d}", "parent_id": "r0", "depth": 1, "branch_profile": profile, "seed": 606 + idx * 17})
else:
    def score(row):
        best = int(row.get("main_frontier_best_correct") or 0)
        thresholds = row.get("thresholds", {})
        bonus = 0
        for label, weight in (("90", 1_000_000), ("85", 100_000), ("80", 10_000)):
            k = thresholds.get(f"smallest_k_ge_{label}_percent")
            if k is not None:
                bonus += weight - int(k)
        return bonus + best
    parents = [row for row in summaries if row.get("survivor")] or summaries
    parents = sorted(parents, key=score, reverse=True)[:beam_width]
    used = {(row.get("parent_id"), row.get("branch_profile"), row.get("depth")) for row in summaries}
    next_idx = len(summaries) + 1
    for parent in parents:
        for profile in profiles_next:
            if len(specs) >= min(parallel, remaining):
                break
            depth = int(parent.get("depth", 1)) + 1
            key = (parent["branch_id"], profile, depth)
            if key in used:
                continue
            specs.append({"branch_id": f"b{next_idx:03d}", "parent_id": parent["branch_id"], "depth": depth, "branch_profile": profile, "seed": 606 + next_idx * 17})
            next_idx += 1
        if len(specs) >= min(parallel, remaining):
            break
for spec in specs:
    (wave_dir / f"{spec['branch_id']}.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
print("planned", len(specs))
PY
}

run_branch_spec() {
  local spec="$1"
  local gpu="$2"
  local branch parent profile seed
  branch="$(json_get "$spec" branch_id)"
  parent="$(json_get "$spec" parent_id)"
  profile="$(json_get "$spec" branch_profile)"
  seed="$(json_get "$spec" seed)"
  echo "== branch ${branch} parent=${parent} profile=${profile} gpu=${gpu} =="
  date
  export CUDA_VISIBLE_DEVICES="$gpu"
  build_branch_data "$branch" "$parent" "$profile" "$seed"
  stage_round "$branch" "$(round_train_jsonl "$branch")" issue6_ladder
  write_branch_summary "$spec"
  rm -rf "$(round_train_dir "$branch")/merged"
  echo "== branch ${branch} done =="
  date
}

run_wave() {
  local wave="$1"
  local wave_dir="${RUN_DIR}/plans/wave_${wave}"
  rm -rf "$wave_dir"
  mkdir -p "$wave_dir"
  plan_next_wave "$wave_dir"
  mapfile -t specs < <(find "$wave_dir" -maxdepth 1 -type f -name 'b*.json' | sort)
  [[ "${#specs[@]}" -gt 0 ]] || return 1
  local pids=()
  local i=0
  for spec in "${specs[@]}"; do
    local branch gpu log
    branch="$(json_get "$spec" branch_id)"
    gpu="$((i % PARALLEL_BRANCHES))"
    log="${LOG_DIR}/${branch}.log"
    echo "[wave ${wave}] launching ${branch} gpu=${gpu}"
    (run_branch_spec "$spec" "$gpu") >"$log" 2>&1 &
    pids+=("$!")
    i="$((i + 1))"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || exit 30
  write_tree_state
}

run_pipeline() {
  cd "$REMOTE_REPO"
  echo "== glm tokenbender replica =="
  date
  setup_env
  source .venv/bin/activate
  write_budget_manifest

  run_leak_audit "$TRAIN_BASE" "${RUN_DIR}/r0/leak_audit.json"
  mkdir -p "${RUN_DIR}/r0"
  stage_round r0 "$TRAIN_BASE" issue2_ladder

  local parent=r0
  for round in r1 r2 r3; do
    build_nearmiss_round "$round" "$parent"
    stage_round "$round" "$(round_train_jsonl "$round")" issue2_ladder
    parent="$round"
  done

  mkdir -p "${RUN_DIR}/plans"
  write_tree_state
  local wave=1
  while true; do
    local completed
    completed="$(RUN_DIR="$RUN_DIR" python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["RUN_DIR"]) / "tree_state.json"
print((json.loads(p.read_text()) if p.exists() else {}).get("branches_completed", 0))
PY
)"
    [[ "$completed" -lt "$MAX_BRANCH_ROUNDS" ]] || break
    run_wave "$wave" || break
    wave="$((wave + 1))"
  done
  write_tree_state

  RUN_DIR="$RUN_DIR" python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["RUN_DIR"])
out = {"rounds": {}, "branches": {}, "tree_state": None}
for p in sorted(root.glob("*/round_summary.json")):
    row = json.loads(p.read_text())
    if p.parent.name.startswith("b"):
        out["branches"][p.parent.name] = row
    else:
        out["rounds"][p.parent.name] = row
state = root / "tree_state.json"
if state.exists():
    out["tree_state"] = json.loads(state.read_text())
(root / "final_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
  echo "== done =="
  date
}

exec > >(tee -a "$LOG") 2>&1
run_pipeline
REMOTE

ssh_cmd "mkdir -p '$REMOTE_RUNS'"
scp_to "$tmpdir/glm_bfcl_tokenbender_replica.sh" root@"$TARGET_HOST":"$REMOTE_SCRIPT"
ssh_cmd "chmod +x '$REMOTE_SCRIPT' && tmux kill-session -t '$SESSION' 2>/dev/null || true; tmux new -d -s '$SESSION' 'REMOTE_REPO=$REMOTE_REPO REMOTE_RUNS=$REMOTE_RUNS bash $REMOTE_SCRIPT'; tmux ls"
