# Release Boundary

This repository is the public release package for the Prism paper experiments.

## Scope

The repository is intentionally flattened into a single public code snapshot.
It contains the paper-facing experiment scripts, helper modules, small result
receipts, paper PDF, figure assets, and documentation needed to inspect and
rerun the released experiment families:

- arithmetic capability extraction;
- EN-PT translation rescue;
- BFCL/function-calling substrate conditioning.

Large generated artifacts are released separately on Hugging Face and pinned by
immutable revision SHA in `ARTIFACT_MANIFEST.json`.

## Artifact Revisions

- `TokenBender/circuit-discovery` dataset:
  `4b9fb53fef92550042d8576fe011e99270fdca8b`
- `TokenBender/circuit-discovery` model:
  `f75ca2f123ce6aaca0e8096918df1ddb34b5d546`
- `TokenBender/synth-data-en-pt-circuit`:
  `36ee2512bcabf32f224c34792db4fb1907d711c3`
- `Occupying-Mars/hy-lora-conditions`:
  `8139bf31538727c87c04f9a88b0b0ccaeacb8832`
- `Occupying-Mars/issue49-bfcl-repro-artifacts`:
  `303db0bddcfb04bebaf07ab4a4dc4c089240c545`
- `Occupying-Mars/issue49-k160-r32-len1024-adapter`:
  `e5104eee6e9dd0fff11f377b743330429970d672`
- `Occupying-Mars/issue49-k240-r16-adapter`:
  `b9018cc3090b856df701240fd73f9f98c627917c`
- `TokenBender/issue51-r32-more-online-codex-repro-551-full`:
  `ff4daac9e49a8f927153c9a04daa9faba2fb5a66`

## Exclusions

The git repository does not include large model checkpoints, generated
datasets, raw attribution arrays, or full-model outputs. Those artifacts are
hosted in the pinned public Hugging Face repositories above.
