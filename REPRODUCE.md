# Reproduction Notes

This repository contains code and small receipts for the released Prism
experiments. Large model weights, generated datasets, raw attribution files, and
full-model artifacts are hosted on Hugging Face and referenced from
`docs/ARTIFACT_MANIFEST.json`.

## Environment

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ./code
```

Install CUDA-specific `torch` builds as appropriate for your hardware.

## Arithmetic

The arithmetic task uses generated two-digit addition pairs.

Representative commands:

```bash
cd code
python3 train_lora_2digit_kl.py --model Qwen/Qwen2.5-Math-1.5B --out-dir runs/arithmetic_lora
python3 evaluate_full_answer_masks.py --model Qwen/Qwen2.5-Math-1.5B --help
python3 evaluate_full_answer_generation_masks.py --model Qwen/Qwen2.5-Math-1.5B --help
```

Included receipts are under `results/arithmetic/`.

## Translation

Build the held-out NTREX EN-PT evaluation file:

```bash
cd code
python3 build_ntrex_en2pt_jsonl.py
```

Representative commands:

```bash
python3 train_masked_kl_conditioning.py --help
python3 evaluate_translation_adapter_masks.py --help
python3 evaluate_translation_masks.py --help
```

Included receipts are under `results/translation/`.

## BFCL / Function Calling

Download and prepare public BFCL v3 single-call rows:

```bash
cd code
python3 scripts/bfcl_direct_qwen3.py download-bfcl-single-call --help
```

Filter public ToolMind and Argilla/APIGen-style data into strict
BFCL-compatible single-call rows:

```bash
python3 scripts/filter_toolmind_bfcl_strict.py --help
python3 scripts/filter_argilla_apigen_bfcl_strict.py --help
python3 scripts/build_bfcl_strict_10k_mix.py --help
```

Condition and evaluate:

```bash
python3 scripts/train_bfcl_masked_lora.py --help
python3 scripts/train_bfcl_masked_policy_distill.py --help
python3 scripts/train_bfcl_prime_opd_sampled_lora.py --help
python3 scripts/bfcl_direct_qwen3.py eval-mask --help
```

Small BFCL receipts are under `results/bfcl/`. Complete BFCL data, adapters,
and the full-model reproduction are available from the Hugging Face artifact
repositories listed in `docs/ARTIFACT_MANIFEST.json`.
