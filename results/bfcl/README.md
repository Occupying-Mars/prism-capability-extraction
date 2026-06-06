# BFCL / Function-Calling Receipts

This directory contains small BFCL/function-calling receipts included directly
in the git release. Large generated artifacts, adapters, datasets, and full
model outputs are released on Hugging Face and pinned in
`docs/ARTIFACT_MANIFEST.json`.

## Included Local Receipts

- `eval_masked_summary.json`: k160 masked eval summary.
- `eval_unmasked_summary.json`: unmasked merged-model eval summary.
- `gkd_aopd_hybrid_k160_v0__eval_k160_hybrid_v0_masked.summary.json`: k160
  hybrid masked eval summary.
- `gkd_aopd_hybrid_k160_v0__run_config.json`: k160 hybrid run config.
- `gkd_aopd_hybrid_k160_v0__train_summary.json`: k160 hybrid train summary.
- `r32_more_online_repro_summary.json`: reproduction summary for the uploaded
  full-model artifact.

## Public Artifact Pointers

| artifact | repository | revision |
|---|---|---|
| BFCL data and reproduction artifacts | `Occupying-Mars/issue49-bfcl-repro-artifacts` | `303db0bddcfb04bebaf07ab4a4dc4c089240c545` |
| k160 rank-32 adapter | `Occupying-Mars/issue49-k160-r32-len1024-adapter` | `e5104eee6e9dd0fff11f377b743330429970d672` |
| k240 rank-16 adapter | `Occupying-Mars/issue49-k240-r16-adapter` | `b9018cc3090b856df701240fd73f9f98c627917c` |
| r32 more-online full model | `TokenBender/issue51-r32-more-online-codex-repro-551-full` | `ff4daac9e49a8f927153c9a04daa9faba2fb5a66` |

