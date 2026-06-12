# GLM BFCL Code

GLM-specific BFCL attribution, evaluation, and masked-LoRA conditioning code.

This package is intentionally separate from the existing Qwen scripts because
GLM-4 uses ChatGLM remote code and different MLP projection names:
`dense_h_to_4h` and `dense_4h_to_h`.

