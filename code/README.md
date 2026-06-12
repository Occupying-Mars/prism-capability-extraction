# Prism Experiment Code

This directory contains the runnable experiment scripts and shared utilities
used by the public Prism capability-extraction release.

Python 3.12+ is recommended. Full reruns require large public models, public
datasets, and GPU hardware.

The Qwen physical-substrate runtime lives in `prism_qwen/runtime/`. It is kept
separate from the existing public experiment scripts so the release tree stays
small and model-specific runtime code is easy to audit.

From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ./code
```
