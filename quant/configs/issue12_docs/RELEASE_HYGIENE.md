# Release Hygiene

This repository was built as a fresh single-commit release package with copied
final code, the paper PDF, figure assets, and selected result summaries. It does
not preserve development history.

## Release Construction

- Development history is flattened into one public commit.
- Large generated artifacts are referenced by Hugging Face repository and
  immutable revision SHA.
- Local machine paths, pod-local working directories, tokens, and non-release URLs
  were excluded from release documentation where they are not part of an
  artifact receipt.
- Non-release exploratory code is excluded from the release package.

## Intentionally Retained Names

The following public model, dataset, package, or method names are retained:

- Qwen
- HY-MT
- NTREX
- BFCL
- ToolMind
- Argilla/APIGen
- Hugging Face repository names used for the public artifacts

## Verification

The package was scanned before the initial commit for stale missing-artifact
language, extra novice-paper files, common build artifacts, local home paths,
and secret-like tokens.
