from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_qwen.runtime.qwen_substrate_stack import (
    PackedGatedMLP,
    PaddedTritonFullPackedGatedMLP,
    QwenSubstrateLM,
    _parse_triton_mlp_impl,
    maybe_compile_mlp,
    triton,
)


def test_packed_gated_mlp_matches_separate_gate_up() -> None:
    torch.manual_seed(0)
    hidden = 7
    kept = 5
    padded = 8
    x = torch.randn(3, 4, hidden)
    gate = torch.randn(kept, hidden)
    up = torch.randn(kept, hidden)
    down = torch.randn(hidden, kept)

    mlp = PackedGatedMLP(hidden, kept, padding_multiple=padded)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(PackedGatedMLP.pack_gate_up_weights(gate, up, mlp.intermediate_size))
        mlp.down_proj.weight.copy_(PackedGatedMLP.pack_down_weights(down, mlp.intermediate_size))

    expected = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    torch.testing.assert_close(mlp(x), expected, atol=1e-6, rtol=1e-6)


def test_packed_gated_mlp_zero_channels_returns_zero() -> None:
    mlp = PackedGatedMLP(hidden_size=7, intermediate_size=0)
    x = torch.randn(2, 3, 7)
    out = mlp(x)
    assert out.shape == x.shape
    assert out.abs().sum() == 0


def test_padded_triton_gate_up_layout_uses_padded_up_offset() -> None:
    gate = torch.ones(3, 2)
    up = torch.full((3, 2), 2.0)
    packed = PaddedTritonFullPackedGatedMLP.pack_gate_up_weights(gate, up, padded_intermediate_size=5)

    torch.testing.assert_close(packed[:3], gate)
    assert packed[3:5].abs().sum() == 0
    torch.testing.assert_close(packed[5:8], up)
    assert packed[8:10].abs().sum() == 0


def test_padded_triton_torch_prefill_matches_packed_formula() -> None:
    if triton is None:
        pytest.skip("triton is not installed")
    torch.manual_seed(2)
    hidden = 7
    kept = 3
    padded = 5
    x = torch.randn(4, hidden)
    gate = torch.randn(kept, hidden)
    up = torch.randn(kept, hidden)
    down = torch.randn(hidden, kept)

    mlp = PaddedTritonFullPackedGatedMLP(
        hidden,
        kept,
        padding_multiple=padded,
        torch_prefill=True,
        decode_workspace_tokens=4,
    )
    assert mlp.gate_up_buffer.shape == (4, 2 * mlp.intermediate_size)
    assert mlp.out_buffer.shape == (4, hidden)
    with torch.no_grad():
        mlp.gate_up_weight.copy_(PaddedTritonFullPackedGatedMLP.pack_gate_up_weights(gate, up, mlp.intermediate_size))
        assert mlp.prefill_gate_up_weight is not None
        mlp.prefill_gate_up_weight.copy_(PackedGatedMLP.pack_gate_up_weights(gate, up, kept))
        mlp.down_weight.copy_(PackedGatedMLP.pack_down_weights(down, mlp.intermediate_size))
        assert mlp.prefill_down_weight is not None
        mlp.prefill_down_weight.copy_(down)

    expected = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    torch.testing.assert_close(mlp._torch_forward(x), expected, atol=1e-6, rtol=1e-6)


def test_padded_triton_forward_residual_matches_separate_add() -> None:
    if triton is None or not torch.cuda.is_available():
        pytest.skip("triton cuda is not available")
    torch.manual_seed(4)
    hidden = 16
    kept = 7
    x = torch.randn(2, 1, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gate = torch.randn(kept, hidden, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(kept, hidden, device="cuda", dtype=torch.bfloat16)
    down = torch.randn(hidden, kept, device="cuda", dtype=torch.bfloat16)

    mlp = PaddedTritonFullPackedGatedMLP(
        hidden,
        kept,
        padding_multiple=8,
        block_m=2,
        block_n=8,
        block_k=16,
        num_warps=1,
        torch_prefill=True,
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        mlp.gate_up_weight.copy_(PaddedTritonFullPackedGatedMLP.pack_gate_up_weights(gate, up, mlp.intermediate_size))
        mlp.down_weight.copy_(PackedGatedMLP.pack_down_weights(down, mlp.intermediate_size))
        assert mlp.prefill_gate_up_weight is not None
        mlp.prefill_gate_up_weight.copy_(PackedGatedMLP.pack_gate_up_weights(gate, up, kept))
        assert mlp.prefill_down_weight is not None
        mlp.prefill_down_weight.copy_(down)

    with torch.inference_mode():
        expected = residual + mlp(x)
        actual = mlp.forward_residual(x, residual)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_padded_triton_forward_post_norm_residual_matches_separate_ops() -> None:
    if triton is None or not torch.cuda.is_available():
        pytest.skip("triton cuda is not available")
    torch.manual_seed(5)
    hidden = 16
    kept = 7
    eps = 1e-6
    x = torch.randn(2, 1, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(kept, hidden, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(kept, hidden, device="cuda", dtype=torch.bfloat16)
    down = torch.randn(hidden, kept, device="cuda", dtype=torch.bfloat16)

    mlp = PaddedTritonFullPackedGatedMLP(
        hidden,
        kept,
        padding_multiple=8,
        block_m=2,
        block_n=8,
        block_k=16,
        num_warps=1,
        torch_prefill=True,
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        mlp.gate_up_weight.copy_(PaddedTritonFullPackedGatedMLP.pack_gate_up_weights(gate, up, mlp.intermediate_size))
        mlp.down_weight.copy_(PackedGatedMLP.pack_down_weights(down, mlp.intermediate_size))
        assert mlp.prefill_gate_up_weight is not None
        mlp.prefill_gate_up_weight.copy_(PackedGatedMLP.pack_gate_up_weights(gate, up, kept))
        assert mlp.prefill_down_weight is not None
        mlp.prefill_down_weight.copy_(down)

    with torch.inference_mode():
        normed = x.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = norm_weight * normed.to(x.dtype)
        expected = residual + mlp(normed)
        actual = mlp.forward_post_norm_residual(x, norm_weight, eps, residual)
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.05)


def _write_tiny_bundle(path: Path) -> None:
    torch.manual_seed(1)
    hidden = 8
    vocab = 13
    layers = 2
    heads = 2
    kv_heads = 1
    head_dim = 4
    kept = {"0": 3, "1": 5}
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "head_dim": head_dim,
                "rms_norm_eps": 1e-6,
                "vocab_size": vocab,
                "rope_theta": 10000.0,
            }
        )
    )
    (path / "substrate_metadata.json").write_text(
        json.dumps(
            {
                "source_model": "tiny-random",
                "topk": sum(kept.values()),
                "dtype": "float32",
                "isolation": {"kept_per_layer": kept},
            }
        )
    )
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.randn(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        layer_kept = int(kept[str(layer)])
        state[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden)
        state[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden)
        state[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(heads * head_dim, hidden)
        state[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(kv_heads * head_dim, hidden)
        state[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(kv_heads * head_dim, hidden)
        state[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden, heads * head_dim)
        state[f"{prefix}.self_attn.q_norm.weight"] = torch.randn(head_dim)
        state[f"{prefix}.self_attn.k_norm.weight"] = torch.randn(head_dim)
        state[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(layer_kept, hidden)
        state[f"{prefix}.mlp.up_proj.weight"] = torch.randn(layer_kept, hidden)
        state[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden, layer_kept)
    save_file(state, path / "model.safetensors")


def test_bundle_padding_fails_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_tiny_bundle(bundle)
    with pytest.raises(RuntimeError, match="mlp-padding-multiple is disabled"):
        QwenSubstrateLM.from_bundle(bundle, dtype=torch.float32, device="cpu", mlp_padding_multiple=4)


def test_unknown_mlp_impl_fails_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_tiny_bundle(bundle)
    with pytest.raises(ValueError, match="unknown mlp implementation"):
        QwenSubstrateLM.from_bundle(bundle, dtype=torch.float32, device="cpu", mlp_impl="triton-full-paddedx")


def test_tuned_silu_down_impl_parser() -> None:
    parsed = _parse_triton_mlp_impl("triton-silu-down-decode-m8n32k64w1-widthgated-s1800l9000-split")
    assert parsed == ("silu-down-split", None, 8, 32, 64, None, None, None, 1, True, 0, 1800, 9000)

    parsed = _parse_triton_mlp_impl("triton-silu-down-decode-m8n64k64w1-widthgated-s1800-split")
    assert parsed == ("silu-down-split", None, 8, 64, 64, None, None, None, 1, True, 0, 1800, None)


def test_prefill_decode_cache_matches_full_sequence(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_tiny_bundle(bundle)
    model = QwenSubstrateLM.from_bundle(bundle, dtype=torch.float32, device="cpu", mlp_padding_multiple=0)

    input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)

    with torch.no_grad():
        full_logits, _ = model(input_ids, attention_mask, position_ids)
        past = model.init_cache(input_ids.shape[0], max_length=input_ids.shape[1], device=torch.device("cpu"))
        _, past = model(input_ids[:, :4], attention_mask[:, :4], position_ids[:, :4], past, cache_position=0)
        decode_logits, _ = model(
            input_ids[:, 4:],
            attention_mask,
            position_ids[:, 4:],
            past,
            cache_position=4,
        )

    torch.testing.assert_close(decode_logits[:, -1], full_logits[:, -1], atol=1e-5, rtol=1e-5)


def test_packed_functional_matches_packed_module_logits(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_tiny_bundle(bundle)
    module_model = QwenSubstrateLM.from_bundle(
        bundle,
        dtype=torch.float32,
        device="cpu",
        mlp_impl="packed-module",
    )
    functional_model = QwenSubstrateLM.from_bundle(
        bundle,
        dtype=torch.float32,
        device="cpu",
        mlp_impl="packed-functional",
    )

    input_ids = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)

    with torch.no_grad():
        module_logits, _ = module_model(input_ids, attention_mask, position_ids)
        functional_logits, _ = functional_model(input_ids, attention_mask, position_ids)

    torch.testing.assert_close(functional_logits, module_logits, atol=0.0, rtol=0.0)


def test_compile_mlp_fails_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_tiny_bundle(bundle)
    model = QwenSubstrateLM.from_bundle(bundle, dtype=torch.float32, device="cpu", mlp_padding_multiple=0)
    with pytest.raises(RuntimeError, match="compile-mlp is disabled"):
        maybe_compile_mlp(model)
