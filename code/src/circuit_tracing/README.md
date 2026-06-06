# Circuit Tracing in the MLP Neuron Basis

Implementation of **"Language Model Circuits Are Sparse in the Neuron Basis"** (Arora, Wu, Steinhardt & Schwettmann, 2026) applied to **Qwen 3 0.6B**.

## Key Idea

MLP activations (the pre-down-projection hidden states in SwiGLU MLPs) form a **sparse, interpretable feature basis** for circuit tracing — as sparse as SAE features — when combined with **RelP attribution** (a single linearised backward pass). No additional training or dictionary learning is needed.

## What This Implements

| Paper Contribution | Module | Status |
|---|---|---|
| MLP activations as feature basis | `relp.py` (hooks on `down_proj` input) | Done |
| RelP attribution (linearised backward) | `relp.py` | Done |
| Circuit extraction (top-k neurons) | `circuit.py` | Done |
| Faithfulness / completeness evaluation | `circuit.py` + `ablation.py` | Done |
| Mean ablation interventions | `ablation.py` (PyTorch hooks + pyvene) | Done |
| Task framework (2-digit addition placeholder) | `tasks.py` | Placeholder |

## Architecture

```
circuit_tracing/
├── __init__.py        # Package exports
├── relp.py            # RelP attribution engine
├── circuit.py         # Circuit extraction & evaluation
├── tasks.py           # Task definitions (2-digit addition placeholder)
├── ablation.py        # Mean activation cache & ablation hooks
├── run.py             # End-to-end pipeline CLI
├── test_pipeline.py   # Tests (9 tests, all passing)
└── README.md          # This file
```

## RelP Linearisation Rules (Table 2 from paper)

The key innovation is a **single modified backward pass** that linearises nonlinearities:

| Operation | Original | Linearised (backward only) |
|---|---|---|
| **RMSNorm** | `x / sqrt(var + eps)` | `x / freeze(sqrt(var + eps))` — freeze denominator |
| **SiLU** | `x * sigmoid(x)` | `x * freeze(sigmoid(x))` — freeze sigmoid gate |
| **Gated MLP** | `silu(gate) * up` | Half-rule: each factor gets `0.5 × gradient` |
| **Attention** | `softmax(QK/√d) · V` | `freeze(attn_weights) · V` — not implemented in this release |

The forward pass produces **identical outputs** to the original model. Only the gradient computation changes, giving cleaner attribution signals.

### Implementation Detail

```python
# Linearised SiLU: gradient only flows through x, not through sigmoid
def linearised_silu(x):
    return x * torch.sigmoid(x).detach()

# Half-rule for gate*up multiplication (preserves relevance conservation)
intermediate = 0.5 * (gate * up.detach() + gate.detach() * up)
# Forward value: gate*up (unchanged)
# Backward: each path receives half the gradient
```

## How It Works

### 1. Mean Activation Cache
Run the model on a calibration dataset, collect mean MLP activations at each layer. These means serve as the "baseline" for ablation.

### 2. RelP Attribution
For each task example:
- Monkey-patch the model with linearised operations
- Forward pass (identical outputs to original)
- Backward pass through linearised graph
- Attribution score = `activation × gradient` per neuron

### 3. Circuit Extraction
- Aggregate attributions across examples (absolute value, summed over positions)
- Select top-k neurons by attribution score → this is the circuit

### 4. Evaluation (Marks et al., 2025)
- **Faithfulness**: ablate complement of circuit → metric should match full model
- **Completeness**: ablate circuit itself → metric should drop to baseline
- A perfect circuit: faithfulness ≈ 1, completeness ≈ 0

## Usage

### Quick Start

```bash
cd code

# Activate the venv
source ../.venv/bin/activate

# Run the full pipeline
python -m src.circuit_tracing.run

# With custom settings
python -m src.circuit_tracing.run \
    --model Qwen/Qwen3-0.6B \
    --circuit-size 100 \
    --n-examples 50 \
    --n-calib 30 \
    --n-eval 20 \
    --output results.json
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace model name |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |
| `--dtype` | `float32` | `float32`, `float16`, `bfloat16` |
| `--circuit-size` | `50` | Number of neurons in the circuit (k) |
| `--n-examples` | `20` | Task examples for attribution |
| `--n-calib` | `30` | Calibration examples for mean cache |
| `--n-eval` | `10` | Evaluation examples for metrics |
| `--seed` | `42` | Random seed |
| `--output` | `None` | Path to save JSON results |

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.circuit_tracing import (
    ReLPAttributor, TwoDigitAdditionTask,
    MeanCache, extract_circuit, evaluate_circuit,
)

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Setup task
task = TwoDigitAdditionTask(tokenizer)
examples = task.generate_examples(50)

# Build mean cache
calib_ids = [task.tokenize(ex.input_text) for ex in examples[:20]]
mean_cache = MeanCache.build(model, calib_ids)

# Run RelP attribution
attributor = ReLPAttributor(model, tokenizer)
attributions = attributor.attribute_dataset(examples[:20], task)

# Extract circuit
circuit = extract_circuit(attributions, k=50)
print(circuit)  # Circuit(size=50, layers=[...])

# Evaluate
results = evaluate_circuit(model, circuit, mean_cache, examples[20:30], task)
print(f"Faithfulness: {results['faithfulness']:.4f}")
print(f"Completeness: {results['completeness']:.4f}")
```

### Adding a New Task

```python
from src.circuit_tracing.tasks import TaskExample

class MyTask:
    def __init__(self, tokenizer, seed=42):
        self.tokenizer = tokenizer

    def generate_examples(self, n):
        return [TaskExample(
            input_text="The cat sat on the",
            target_text="mat",
            cf_input_text="The dog sat on the",
            cf_target_text="floor",
        )] * n

    def make_metric_fn(self, example):
        tid = self.tokenizer.encode(example.target_text, add_special_tokens=False)[0]
        cid = self.tokenizer.encode(example.cf_target_text, add_special_tokens=False)[0]
        def metric_fn(logits):
            return logits[0, -1, tid] - logits[0, -1, cid]
        return metric_fn

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
```

## Running Tests

```bash
cd code
python -m pytest src/circuit_tracing/test_pipeline.py -v
```

By default the tests use `Qwen/Qwen3-0.6B`, float32, and CUDA when available.
Override with `PRISM_TEST_MODEL`, `PRISM_TEST_DEVICE`, or `PRISM_TEST_DTYPE`.
Edge attribution and end-to-end pipeline tests are slower; include them with
`PRISM_RUN_SLOW_TESTS=1`.

The suite covers:
- `TestTasks::test_generate_examples` — validates task data generation
- `TestTasks::test_metric_fn_runs` — metric produces finite scalars
- `TestRelP::test_attributions_nonzero` — RelP produces non-trivial scores
- `TestRelP::test_linearised_forward_matches_original` — linearised forward == original forward
- `TestRelP::test_attribute_dataset` — dataset-level aggregation works
- `TestMeanCache::test_build` — mean cache produces finite means
- `TestCircuit::test_extract` — top-k extraction selects correct count
- `TestCircuit::test_evaluate_smoke` — evaluation returns valid metrics
- `TestAttentionFreezing` and `TestEdgeAttribution` — attention-freezing and
  edge-scoring smoke coverage
- `TestEndToEnd` — opt-in full pipeline coverage

## Model Details

### Qwen 3 0.6B Architecture
- **28 layers**, d_model=1024, d_ffn=3072
- **16 attention heads** (8 KV heads, GQA)
- **SwiGLU MLP**: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`
- **RMSNorm** (same linearisation as Llama)
- Total MLP neurons: 86,016 (28 × 3,072)
- Runs comfortably on 24GB Mac in float32

### Why MLP Activations (Not Outputs)?
The MLP activation vector (pre-`down_proj`, dimension d_ffn) is a **privileged basis** because the SiLU nonlinearity operates element-wise on it. The MLP output (post-`down_proj`, dimension d_model) mixes neurons through the projection, destroying sparsity. This is the paper's key insight: MLP activations yield 100× sparser circuits than MLP outputs.

## Dependencies

- `torch` (>=2.0)
- `transformers` (>=4.40)
- `pyvene` (>=0.1.8) — intervention utilities
- `pytest` — for tests

## Limitations & Future Work

- **Attention linearisation**: Not yet implemented (paper's Table 2 includes freezing attention weights). Currently only MLP + RMSNorm are linearised.
- **Position-specific circuits**: Current implementation aggregates over token positions. The paper treats each (layer, position, neuron) as a separate node.
- **Edge-level tracing**: The paper also traces neuron-to-neuron edges. Only node-level circuits are implemented.
- **Task placeholder**: The 2-digit addition task is a placeholder. Will be refined with specific counterfactual pairs.

## References

- Arora, Wu, Steinhardt & Schwettmann. "Language Model Circuits Are Sparse in the Neuron Basis." arXiv:2601.22594, 2026.
- Marks et al. "Sparse Feature Circuits." NeurIPS, 2025.
- Jafari et al. "RelP: Relevance Propagation." 2025.
- Gao et al. "Weight-Sparse Transformers Have Interpretable Circuits." 2025.
