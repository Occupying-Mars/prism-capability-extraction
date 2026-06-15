# OLMo BFCL Code

OLMo-specific BFCL base evaluation, ReLP attribution, and LoRA training code.

This package is intentionally separate from Qwen and GLM paths. The initial
OLMo target is `allenai/OLMo-7B-hf`, which is a base model without the Qwen
tool chat template or ChatGLM remote-code layout. BFCL prompts therefore use a
manual JSON call target format, while attribution and masking use the OLMo
decoder/MLP layout. The 0724 instruct checkpoint naturally emits
`<function_call>{...}</function_call>`, so r32 LoRA runs use that tag by
default.
