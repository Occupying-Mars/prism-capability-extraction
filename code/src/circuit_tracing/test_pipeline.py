#!/usr/bin/env python3
"""
End-to-end tests for the circuit tracing pipeline.

Run:
    cd code
    python -m pytest src/circuit_tracing/test_pipeline.py -v
    # or directly:
    python src/circuit_tracing/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import torch
import pytest

MODEL_NAME = os.environ.get("PRISM_TEST_MODEL", "Qwen/Qwen3-0.6B")
TEST_DEVICE = os.environ.get(
    "PRISM_TEST_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
TEST_DTYPE = os.environ.get(
    "PRISM_TEST_DTYPE",
    "float32",
)
RUN_SLOW_TESTS = os.environ.get("PRISM_RUN_SLOW_TESTS") == "1"


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _model_device(model) -> str:
    return str(next(model.parameters()).device)

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=_torch_dtype(TEST_DTYPE), attn_implementation="eager",
    ).to(TEST_DEVICE).eval()
    return model, tokenizer


# ── Tests ─────────────────────────────────────────────────────────────

class TestTasks:
    def test_generate_examples(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        from .tasks import TwoDigitAdditionTask
        task = TwoDigitAdditionTask(tokenizer, seed=0)
        examples = task.generate_examples(10)

        assert len(examples) == 10
        for ex in examples:
            # answers should be valid sums
            parts = ex.input_text.split(" + ")
            a = int(parts[0])
            b_str = parts[1].replace(" =", "").strip()
            b = int(b_str)
            assert ex.target_text == str(a + b)

    def test_metric_fn_runs(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        from .tasks import TwoDigitAdditionTask
        task = TwoDigitAdditionTask(tokenizer)
        ex = task.generate_examples(1)[0]

        ids = task.tokenize(ex.input_text).to(_model_device(model))
        metric_fn = task.make_metric_fn(ex)
        with torch.no_grad():
            logits = model(ids).logits
        val = metric_fn(logits)
        assert val.shape == ()  # scalar
        assert torch.isfinite(val)


class TestRelP:
    def test_attributions_nonzero(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        from .relp import ReLPAttributor
        from .tasks import TwoDigitAdditionTask

        task = TwoDigitAdditionTask(tokenizer)
        ex = task.generate_examples(1)[0]
        ids = task.tokenize(ex.input_text)
        metric_fn = task.make_metric_fn(ex)

        attributor = ReLPAttributor(model, tokenizer, device=_model_device(model))
        # only test on 2 layers for speed
        scores = attributor.attribute(ids, metric_fn, layers=[0, 1])

        assert 0 in scores and 1 in scores
        assert scores[0].shape[-1] == model.config.intermediate_size
        # at least some attributions should be nonzero
        assert scores[0].abs().sum() > 0, "all attributions are zero"

    def test_linearised_forward_matches_original(self, model_and_tokenizer):
        """Linearised forward should produce the same logits as original."""
        model, tokenizer = model_and_tokenizer
        from .relp import ReLPAttributor

        attributor = ReLPAttributor(model, tokenizer, device=_model_device(model))
        ids = tokenizer("Hello world", return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(_model_device(model))

        with torch.no_grad():
            logits_orig = model(ids).logits.clone()

        with torch.no_grad():
            with attributor._linearised_ctx():
                logits_lin = model(ids).logits.clone()

        # forward values must match (linearisation only changes backward)
        torch.testing.assert_close(logits_orig, logits_lin, atol=1e-4, rtol=1e-4)

    def test_attribute_dataset(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        from .relp import ReLPAttributor
        from .tasks import TwoDigitAdditionTask

        task = TwoDigitAdditionTask(tokenizer)
        examples = task.generate_examples(3)
        attributor = ReLPAttributor(model, tokenizer, device=_model_device(model))
        agg = attributor.attribute_dataset(examples, task, layers=[0])
        assert 0 in agg
        assert agg[0].shape == (model.config.intermediate_size,)


class TestMeanCache:
    def test_build(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        from .ablation import MeanCache
        from .tasks import TwoDigitAdditionTask

        task = TwoDigitAdditionTask(tokenizer)
        examples = task.generate_examples(5)
        device = _model_device(model)
        calib_ids = [task.tokenize(ex.input_text).to(device) for ex in examples]

        mc = MeanCache.build(model, calib_ids, device=device, layers=[0, 1])
        assert 0 in mc.means and 1 in mc.means
        assert mc.means[0].shape == (model.config.intermediate_size,)
        # means should be finite
        assert torch.isfinite(mc.means[0]).all()


class TestCircuit:
    def test_extract(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        from .circuit import extract_circuit

        d_ffn = model.config.intermediate_size
        fake_attr = {
            0: torch.randn(d_ffn),
            1: torch.randn(d_ffn),
        }
        circuit = extract_circuit(fake_attr, k=20)
        assert circuit.size == 20

    def test_evaluate_smoke(self, model_and_tokenizer):
        """Smoke test: evaluate a tiny random circuit."""
        model, tokenizer = model_and_tokenizer
        from .circuit import Circuit, evaluate_circuit
        from .ablation import MeanCache
        from .tasks import TwoDigitAdditionTask

        task = TwoDigitAdditionTask(tokenizer)
        examples = task.generate_examples(5)
        device = _model_device(model)
        calib_ids = [task.tokenize(ex.input_text).to(device) for ex in examples[:3]]
        mc = MeanCache.build(model, calib_ids, device=device, layers=[0, 1])

        circuit = Circuit(nodes={(0, 0), (0, 1), (1, 0)})
        results = evaluate_circuit(
            model, circuit, mc, examples[:2], task, device=device,
        )
        assert "faithfulness" in results
        assert "completeness" in results
        assert results["n_examples"] == 2


class TestAttentionFreezing:
    def test_linearised_attn_forward_matches_original(self, model_and_tokenizer):
        """Attention freezing should not change forward-pass values."""
        model, tokenizer = model_and_tokenizer
        from .relp import ReLPAttributor

        attributor = ReLPAttributor(model, tokenizer, device=_model_device(model))
        ids = tokenizer("The capital of Texas is",
                        return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(_model_device(model))

        with torch.no_grad():
            logits_orig = model(ids).logits.clone()

        with torch.no_grad():
            with attributor._linearised_ctx():
                logits_lin = model(ids).logits.clone()

        torch.testing.assert_close(logits_orig, logits_lin, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not RUN_SLOW_TESTS,
    reason="set PRISM_RUN_SLOW_TESTS=1 to run edge attribution tests",
)
class TestEdgeAttribution:
    def test_edge_scores_nonzero(self, model_and_tokenizer):
        """Edge attribution should produce some nonzero scores."""
        model, tokenizer = model_and_tokenizer
        from .relp import ReLPAttributor
        from .tasks import TwoDigitAdditionTask

        task = TwoDigitAdditionTask(tokenizer)
        ex = task.generate_examples(1)[0]
        ids = task.tokenize(ex.input_text)
        metric_fn = task.make_metric_fn(ex)

        attributor = ReLPAttributor(model, tokenizer, device=_model_device(model))
        # small set: 5 neurons in layer 0, 5 in layer 1
        sources = [(0, i) for i in range(5)]
        targets = [(1, i) for i in range(5)]

        edges = attributor.attribute_edges(ids, metric_fn, sources, targets)
        assert len(edges) == 25  # 5 sources * 5 targets
        assert any(abs(v) > 0 for v in edges.values()), "all edge scores are zero"

    def test_edge_layer_ordering(self, model_and_tokenizer):
        """All edges must have source_layer < target_layer."""
        model, tokenizer = model_and_tokenizer
        from .relp import ReLPAttributor
        from .tasks import TwoDigitAdditionTask

        task = TwoDigitAdditionTask(tokenizer)
        ex = task.generate_examples(1)[0]
        ids = task.tokenize(ex.input_text)
        metric_fn = task.make_metric_fn(ex)

        attributor = ReLPAttributor(model, tokenizer, device=_model_device(model))
        neurons = [(0, 0), (1, 0), (2, 0)]  # one per layer
        edges = attributor.attribute_edges(ids, metric_fn, neurons, neurons)

        for (src, tgt), _ in edges.items():
            assert src[0] < tgt[0], f"Edge {src} -> {tgt} violates layer ordering"

    def test_extract_circuit_with_edges(self, model_and_tokenizer):
        """Edge-pruned circuit should remove isolated neurons."""
        from .circuit import extract_circuit_with_edges

        # small controlled attributions — 10 neurons per layer
        fake_attr = {
            0: torch.zeros(10),
            5: torch.zeros(10),
            10: torch.zeros(10),
        }
        # make specific neurons the top-scoring ones
        fake_attr[0][0] = 10.0
        fake_attr[5][3] = 8.0
        fake_attr[10][0] = 9.0

        # only connect layer 0 neuron 0 <-> layer 10 neuron 0
        fake_edges = {
            ((0, 0), (10, 0)): 5.0,
        }
        circuit = extract_circuit_with_edges(
            fake_attr, fake_edges, k_nodes=30, k_edges=10,
        )
        assert circuit.n_edges >= 1
        assert (0, 0) in circuit.nodes
        assert (10, 0) in circuit.nodes
        # layer 5 neuron 3 has no edges and is not in min/max layer → pruned
        assert (5, 3) not in circuit.nodes


@pytest.mark.skipif(
    not RUN_SLOW_TESTS,
    reason="set PRISM_RUN_SLOW_TESTS=1 to run end-to-end pipeline tests",
)
class TestEndToEnd:
    def test_full_pipeline(self, model_and_tokenizer):
        """Run the full pipeline with tiny settings."""
        from .run import main
        results = main([
            "--model", MODEL_NAME,
            "--device", TEST_DEVICE,
            "--dtype", TEST_DTYPE,
            "--circuit-size", "10",
            "--n-attr", "3",
            "--n-calib", "3",
            "--n-eval", "2",
            "--sweep-ks", "5", "10",
            "--no-wandb",
            "--no-edges",
        ])
        assert results is not None
        assert "per_property" in results

    def test_full_pipeline_with_edges(self, model_and_tokenizer):
        """Run the full pipeline including edge attribution."""
        from .run import main
        results = main([
            "--model", MODEL_NAME,
            "--device", TEST_DEVICE,
            "--dtype", TEST_DTYPE,
            "--circuit-size", "10",
            "--n-attr", "3",
            "--n-calib", "3",
            "--n-eval", "2",
            "--n-edge-attr", "2",
            "--edge-k-nodes", "20",
            "--edge-k-edges", "50",
            "--sweep-ks", "5", "10",
            "--no-wandb",
        ])
        assert results is not None
        assert "per_property" in results


# ── Direct execution ──────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
