# Terminology

This file records project terms that affect experiment design, issue language,
and result interpretation.

## MACE

`MACE` means **minimum acceptable capability extract**.

Source alignment: this repository follows the terminology defined in
`Occupying-Mars/circuit-shotting` issue #54:

```text
https://github.com/Occupying-Mars/circuit-shotting/issues/54
```

This source alignment is for the term definition. It does not import older
circuit-shotting experiment-guide practice into this repository. Any future
experiment guide for this repo should be PRISM-native: derived from this repo,
the paper, the public release surface, and the newer issue-led BFCL sequence in
`tokenbender/prism-capability-extraction`.

MACE replaces older `MVC` / `minimum viable circuit` language as the governing
abstraction for extraction work. The extracted object is not assumed to be a
classical circuit. It can be any minimum acceptable capability-bearing extract
or sparse stack.

A MACE candidate may include one or more of:

- selected MLP channels
- attention heads
- selected layers
- sparse weights
- numerical sparsity
- activation gates
- low-rank structure
- bundle/export paths
- combinations of the above

Operationally, MACE is a hill-climb over extract size and dependency footprint
under an explicit acceptable behavior threshold:

```text
minimize      extract size, cost, dependency footprint, and entanglement
subject to    task behavior >= explicit acceptable threshold
              guardrail/category floors hold
              regression stays within the declared epsilon
              stability checks pass
```

For BFCL/function-calling work, recursive co-activation search can propose
candidate MACE units, but co-activation alone does not prove MACE. Candidate
extracts must be tested by masked-model evaluation. In this repository, causal
masked evaluation is the validation method for a MACE candidate, not the
expansion of the acronym.

Do not expand MACE as `minimum acceptable causal extraction`.

## Jagged-Substrate Inference Quality

`Jagged-substrate inference quality` is the companion objective to MACE.

Once a MACE-style extract is found, the retained structure may be irregular:
different MLP slices, heads, layers, numerical formats, sparse weights, gates,
or adapters may survive in a pattern that dense inference runtimes are not
designed to exploit.

This objective asks whether the extracted stack can actually run well. It
tracks latency, generated tokens/sec, prompt tokens/sec, peak memory, batching
behavior, and quality/parity for the irregular extract. A MACE claim should not
be treated as practically complete unless its runtime path and retained
dependencies are explicit.
