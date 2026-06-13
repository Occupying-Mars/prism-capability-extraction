# GLM BFCL MACE Two-Budget Run

This is the GLM version of the tokenbender #2 pattern:

```text
unmasked LoRA collimation -> merge local attribution target -> ReLP attribution -> selected-MLP-channel eval
```

The run intentionally evaluates only two GLM selected-MLP-channel budgets:

- `p50`: 50% of GLM MLP channels
- `p36`: 36% of GLM MLP channels

The evaluator keeps the full transformer live. Only non-selected GLM MLP
intermediate channels are zeroed.

## Fixed Recipe

- Model: `zai-org/glm-4-9b-chat`
- Prompt and target: GLM native chat/tool template
- Train set: `data/bfcl_strict_10k_mix/train.jsonl`
- Eval set: BFCL single-call 1007 pairs
- Training: unmasked r32/alpha64 rsLoRA, one epoch
- Attribution target: local merged LoRA model
- Reported metric: normalized exact

## Artifacts

The launcher stages:

- config and this doc
- train summary and adapter
- attribution `.npz` and summary
- base and adapter full-model anchor evals
- p50 and p36 masked evals
- failure buckets
- final summary and logs

Merged full-model weights are local-only for attribution and are excluded from
public artifact staging.
