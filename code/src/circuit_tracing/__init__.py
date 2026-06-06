"""
Circuit tracing in the MLP neuron basis using RelP attribution.

Implements the methods from:
"Language Model Circuits Are Sparse in the Neuron Basis" (Arora et al., 2026)
"""

from .relp import ReLPAttributor
from .circuit import Circuit, extract_circuit, extract_circuit_with_edges, evaluate_circuit
from .tasks import TwoDigitAdditionTask, StructuredAdditionTask, FullAdditionTask, TaskExample
from .ablation import MeanCache, mean_ablate_run
from .dataset import generate_dataset, save_dataset, load_dataset

__all__ = [
    "ReLPAttributor",
    "Circuit",
    "extract_circuit",
    "extract_circuit_with_edges",
    "evaluate_circuit",
    "TwoDigitAdditionTask",
    "StructuredAdditionTask",
    "FullAdditionTask",
    "TaskExample",
    "MeanCache",
    "mean_ablate_run",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
]
