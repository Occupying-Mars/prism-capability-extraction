#!/usr/bin/env python3
"""Download the best-substrate Qwen artifacts from tokenbender's release.

Source: TokenBender/circuit-discovery (HF *dataset* repo).

Substrate = Qwen/Qwen3-8B + the issue#6 tree-search `b007` recovery-specialist
rsLoRA adapter, masked to a kept-channel MLP substrate. issue#12 recursive
co-activation MACE further refines the kept-channel masks on top of b007.

We pull, by design:
  - the b007 r32 rsLoRA adapter weights + tokenizer + configs
  - the b007 ReLP full-collimated attribution (.npz) -> derive top-k masks
  - issue#12 refined mask set (all candidate/canonical .npz masks, ~422 MB)
  - frontier / threshold / manifest metadata (small JSON)
  - tokenbender's BFCL harness scripts + configs + docs
  - the BFCL single-call eval *inputs* (needed to score a quantized substrate)

We deliberately SKIP the dataset *results*: per-row eval `*.jsonl`, training
mixes, `detailed_scores.json`, `*.log`, `*.csv`. Use --mode full on the GPU pod
(adapter + masks) and --mode code locally (scripts + metadata only).

Auth: reads HF token from $HF_TOKEN or $hf_token (the repo .env key).
"""
import argparse
import os
import sys
from huggingface_hub import snapshot_download

REPO = "TokenBender/circuit-discovery"
I6 = "bfcl/issue6_tree_search_v1"
I12 = "bfcl/issue12_recursive_coactivation_mace_v1"
B007 = f"{I6}/run/branches/b007"

# Small, version-controllable: scripts, configs, metadata, adapter *config*.
CODE_PATTERNS = [
    f"{I12}/code/*",
    f"{I12}/docs/*",
    f"{I12}/*.json",
    f"{I12}/*.md",
    f"{I12}/SHA256SUMS",
    f"{I6}/configs/*",
    f"{B007}/branch_summary.json",
    f"{B007}/unmasked_r32/run_config.json",
    f"{B007}/unmasked_r32/train_summary.json",
    f"{B007}/unmasked_r32/adapter/adapter_config.json",
    # issue12 mask metadata (small JSON only, not detailed_scores)
    f"{I12}/runs/*frontier.json",
    f"{I12}/runs/*threshold_hits.json",
    f"{I12}/runs/*candidate_manifest.json",
    f"{I12}/runs/*category_floor_audit*.json",
]

# Heavy substrate artifacts: adapter weights + all refined masks. Pod only.
FULL_PATTERNS = CODE_PATTERNS + [
    f"{B007}/unmasked_r32/adapter/*",        # adapter_model.safetensors + tokenizer
    f"{B007}/relp_full_collimated*",          # ReLP attribution npz/summary
    f"{I12}/runs/*.npz",                       # all issue12 refined masks (~422 MB)
    f"{I12}/data/bfcl_single_call/*",          # BFCL eval inputs (to score quant)
]

# Never pull these even if a pattern would match (results / bulky logs).
IGNORE_PATTERNS = [
    "*/eval_*.jsonl",
    "*/detailed_scores.json",
    "*.log",
    "*.csv",
    "*/train_mixed.jsonl",
    "*/candidate_masks.jsonl",
    "*/all_failures.jsonl",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["code", "full"], required=True)
    ap.add_argument("--dest", required=True, help="local target dir")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")
    if not token:
        sys.exit("ERROR: set HF_TOKEN or hf_token (see repo .env)")

    patterns = CODE_PATTERNS if args.mode == "code" else FULL_PATTERNS
    print(f"[download] repo={REPO} mode={args.mode} -> {args.dest}")
    path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=args.dest,
        token=token,
        allow_patterns=patterns,
        ignore_patterns=IGNORE_PATTERNS,
        max_workers=8,
    )
    print(f"[download] done: {path}")


if __name__ == "__main__":
    main()
