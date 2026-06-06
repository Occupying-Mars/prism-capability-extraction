# Public Data and Model Sources

This repository does not bundle large public datasets or model checkpoints.
Use the public upstream sources and the Hugging Face artifact repositories
listed in `ARTIFACT_MANIFEST.json`.

## Models

- `Qwen/Qwen2.5-Math-1.5B`: arithmetic experiments.
- `Qwen/Qwen2.5-1.5B`: cross-model arithmetic control.
- `Qwen/Qwen3-8B`: BFCL/function-calling experiments.
- `tencent/HY-MT1.5-1.8B`: EN-PT translation experiments.

## Evaluation Data

- Two-digit addition: generated exhaustively by the arithmetic scripts.
- NTREX-128: public translation evaluation source; build with
  `code/build_ntrex_en2pt_jsonl.py`.
- BFCL v3: public Berkeley Function Calling Leaderboard JSON files. The helper
  script is `code/scripts/bfcl_direct_qwen3.py`.

## Training Data

- Arithmetic: generated from integer pairs by the local scripts.
- Translation: generated teacher-labeled EN-PT data is hosted at
  [TokenBender/synth-data-en-pt-circuit](https://huggingface.co/datasets/TokenBender/synth-data-en-pt-circuit);
  public NTREX is used for held-out evaluation.
- Function calling: strict single-call training mix derived from public
  ToolMind and Argilla/APIGen-style sources, filtered to BFCL-compatible rows.
  The release artifact is hosted at
  [Occupying-Mars/issue49-bfcl-repro-artifacts](https://huggingface.co/datasets/Occupying-Mars/issue49-bfcl-repro-artifacts).

## Large Artifacts

The following large artifacts are released on Hugging Face:

- arithmetic generated data, masks, and run receipts:
  `TokenBender/circuit-discovery` dataset
- arithmetic checkpoints and adapters:
  `TokenBender/circuit-discovery` model
- EN-PT generated data and run artifacts:
  `TokenBender/synth-data-en-pt-circuit`
- EN-PT adapters and masks:
  `Occupying-Mars/hy-lora-conditions`
- BFCL raw attribution and data receipts:
  `Occupying-Mars/issue49-bfcl-repro-artifacts`
- k160 rank-32 adapter:
  `Occupying-Mars/issue49-k160-r32-len1024-adapter`
- k240 rank-16 adapter:
  `Occupying-Mars/issue49-k240-r16-adapter`
- full-model reproduction artifact:
  `TokenBender/issue51-r32-more-online-codex-repro-551-full`
