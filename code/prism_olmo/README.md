# OLMo BFCL Code

OLMo-specific BFCL base evaluation and ReLP attribution code.

This package is intentionally separate from Qwen and GLM paths. The initial
OLMo target is `allenai/OLMo-7B-hf`, which is a base model without the Qwen
tool chat template or ChatGLM remote-code layout. BFCL prompts therefore use a
manual `<tool_call>` JSON target format, while attribution and masking use the
OLMo decoder/MLP layout.
