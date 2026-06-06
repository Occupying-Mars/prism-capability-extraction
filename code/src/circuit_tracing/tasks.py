"""
Task definitions for circuit tracing evaluation.

Each task provides paired (original, counterfactual) examples and a
logit-difference metric, following the evaluation protocol from
Marks et al. (2025) and Arora et al. (2026).
"""

from __future__ import annotations

import random
import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TaskExample:
    """A single paired example for circuit tracing."""
    input_text: str
    target_text: str          # correct continuation
    cf_input_text: str        # counterfactual input
    cf_target_text: str       # counterfactual continuation
    # structured fields (populated by StructuredAdditionTask)
    target_digit: Optional[str] = None
    cf_digit: Optional[str] = None
    meta: Optional[dict] = None


def metric_input_ids(task, example: TaskExample) -> torch.Tensor:
    tokenize_for_metric = getattr(task, "tokenize_for_metric", None)
    if callable(tokenize_for_metric):
        return tokenize_for_metric(example)
    return task.tokenize(example.input_text)


def metric_cf_input_ids(task, example: TaskExample) -> torch.Tensor:
    tokenize_cf_for_metric = getattr(task, "tokenize_cf_for_metric", None)
    if callable(tokenize_cf_for_metric):
        return tokenize_cf_for_metric(example)
    return task.tokenize(example.cf_input_text)


# ──────────────────────────────────────────────────────────────────────
# Structured addition task  (uses the procedural dataset)
# ──────────────────────────────────────────────────────────────────────

class StructuredAdditionTask:
    """
    Task that loads examples from the procedural dataset
    (see dataset.py) and provides a proper logit-difference metric.

    Each example has explicit target_digit / cf_digit fields and
    a metric_position that determines which answer token to compare.
    """

    def __init__(self, tokenizer, examples: List[dict]):
        self.tokenizer = tokenizer
        self._raw = examples
        self._examples = [self._to_task_example(e) for e in examples]

    @staticmethod
    def _to_task_example(raw: dict) -> TaskExample:
        return TaskExample(
            input_text=raw["input_text"],
            target_text=raw["answer"],
            cf_input_text=raw["cf_input_text"],
            cf_target_text=raw["cf_answer"],
            target_digit=raw["target_digit"],
            cf_digit=raw["cf_digit"],
            meta=raw,
        )

    @property
    def examples(self) -> List[TaskExample]:
        return self._examples

    def generate_examples(self, n: Optional[int] = None) -> List[TaskExample]:
        if n is None or n >= len(self._examples):
            return list(self._examples)
        return list(self._examples[:n])

    def make_metric_fn(self, example: TaskExample):
        """
        Logit difference:  logit(target_digit) - logit(cf_digit)
        at the last token position.
        """
        td = example.target_digit or example.target_text[0]
        cd = example.cf_digit or example.cf_target_text[0]

        tid = self.tokenizer.encode(td, add_special_tokens=False)[0]
        cid = self.tokenizer.encode(cd, add_special_tokens=False)[0]

        def metric_fn(logits: torch.Tensor) -> torch.Tensor:
            return logits[0, -1, tid] - logits[0, -1, cid]

        return metric_fn

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]


# ──────────────────────────────────────────────────────────────────────
# Legacy placeholder  (kept for backward compat with tests)
# ──────────────────────────────────────────────────────────────────────

class TwoDigitAdditionTask:
    """Simple random 2-digit addition task (legacy placeholder)."""

    def __init__(self, tokenizer, seed: int = 42):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)

    def generate_examples(self, n: int = 100) -> List[TaskExample]:
        examples: List[TaskExample] = []
        for _ in range(n):
            a = self.rng.randint(10, 49)
            b = self.rng.randint(10, 49)
            answer = a + b

            delta = self.rng.choice([10, -10])
            b_cf = b + delta
            if b_cf < 10 or b_cf > 49:
                b_cf = b - delta
            answer_cf = a + b_cf

            examples.append(TaskExample(
                input_text=f"{a} + {b} =",
                target_text=str(answer),
                cf_input_text=f"{a} + {b_cf} =",
                cf_target_text=str(answer_cf),
                target_digit=str(answer // 10),
                cf_digit=str(answer_cf // 10),
            ))
        return examples

    def make_metric_fn(self, example: TaskExample):
        td = example.target_digit or example.target_text[0]
        cd = example.cf_digit or example.cf_target_text[0]
        tid = self.tokenizer.encode(td, add_special_tokens=False)[0]
        cid = self.tokenizer.encode(cd, add_special_tokens=False)[0]

        def metric_fn(logits: torch.Tensor) -> torch.Tensor:
            return logits[0, -1, tid] - logits[0, -1, cid]

        return metric_fn

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]


# ──────────────────────────────────────────────────────────────────────
# Full 2-digit addition task  (single unified circuit)
# ──────────────────────────────────────────────────────────────────────

class FullAdditionTask:
    """
    Trace one circuit that does the *entire* 2-digit addition,
    not per-property.  The metric is the sum of logit-differences
    for both answer digits (tens + ones), giving a single scalar
    that rewards getting the whole answer right.

    Each example is paired with a counterfactual that changes *b*
    so the sum differs in both digits when possible.
    """

    def __init__(self, tokenizer, seed: int = 42, n: int = 200,
                 pairs: Optional[List[tuple]] = None):
        """
        Parameters
        ----------
        pairs : optional list of (a, b) ints. If provided, examples are
                built from these instead of randomly generated. Sum must be
                in [20, 99] (2-digit answer) — pairs outside that range are
                skipped. Each pair still gets a counterfactual b'.
        """
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        if pairs is not None:
            self._examples = self._from_pairs(pairs)
        else:
            self._examples = self._generate(n)

    def _from_pairs(self, pairs: List[tuple]) -> List[TaskExample]:
        examples: List[TaskExample] = []
        for a, b in pairs:
            s = a + b
            if not (20 <= s <= 99):
                continue
            b2 = None
            for delta in [11, -11, 13, -13, 9, -9, 7, -7, 22, -22]:
                cand = b + delta
                if 10 <= cand <= 99 and 20 <= a + cand <= 99 and (a + cand) != s:
                    s2 = a + cand
                    if str(s)[0] != str(s2)[0] and str(s)[1] != str(s2)[1]:
                        b2 = cand
                        break
            if b2 is None:
                for delta in [10, -10, 5, -5, 2, -2, 1, -1]:
                    cand = b + delta
                    if 10 <= cand <= 99 and 20 <= a + cand <= 99 and (a + cand) != s:
                        b2 = cand
                        break
            if b2 is None:
                continue
            s2 = a + b2
            examples.append(TaskExample(
                input_text=f"{a} + {b} =",
                target_text=str(s),
                cf_input_text=f"{a} + {b2} =",
                cf_target_text=str(s2),
                target_digit=str(s),
                cf_digit=str(s2),
                meta={"a": a, "b": b, "b_cf": b2},
            ))
        self.rng.shuffle(examples)
        return examples

    def _generate(self, n: int) -> List[TaskExample]:
        examples: List[TaskExample] = []
        seen = set()
        attempts = 0
        while len(examples) < n and attempts < n * 20:
            attempts += 1
            a = self.rng.randint(10, 49)
            b = self.rng.randint(10, 49)
            s = a + b
            if s > 99 or (a, b) in seen:
                continue
            seen.add((a, b))

            # pick a counterfactual b that changes *both* digits of the sum
            for delta in [11, -11, 13, -13, 9, -9, 7, -7]:
                b2 = b + delta
                s2 = a + b2
                if 10 <= b2 <= 49 and 20 <= s2 <= 99 and s2 != s:
                    if str(s)[0] != str(s2)[0] and str(s)[1] != str(s2)[1]:
                        break
            else:
                b2 = b + (10 if b + 10 <= 49 else -10)
                s2 = a + b2

            examples.append(TaskExample(
                input_text=f"{a} + {b} =",
                target_text=str(s),
                cf_input_text=f"{a} + {b2} =",
                cf_target_text=str(s2),
                target_digit=str(s),
                cf_digit=str(s2),
                meta={"a": a, "b": b, "b_cf": b2},
            ))
        return examples

    @property
    def examples(self) -> List[TaskExample]:
        return self._examples

    def generate_examples(self, n: int | None = None) -> List[TaskExample]:
        if n is None or n >= len(self._examples):
            return list(self._examples)
        return list(self._examples[:n])

    def make_metric_fn(self, example: TaskExample):
        """
        Metric = teacher-forced logit_diff(tens) + logit_diff(ones).

        Rewards getting the whole 2-digit answer right in one scalar.
        Callers must run the model on ``tokenize_for_metric(example)``, which
        appends the gold tens digit so the ones digit is scored at its own
        prediction position.
        """
        ans = example.target_text
        cf_ans = example.cf_target_text
        if len(ans) != 2 or len(cf_ans) != 2:
            raise ValueError("FullAdditionTask expects 2-digit target/cf answers")

        t_tens = self.tokenizer.encode(ans[0], add_special_tokens=False)[0]
        c_tens = self.tokenizer.encode(cf_ans[0], add_special_tokens=False)[0]
        t_ones = self.tokenizer.encode(ans[1], add_special_tokens=False)[0]
        c_ones = self.tokenizer.encode(cf_ans[1], add_special_tokens=False)[0]
        prompt_len = int(self.tokenize(example.input_text).shape[1])
        metric_len = int(self.tokenize_for_metric(example).shape[1])
        tens_pos = prompt_len - 1
        ones_pos = metric_len - 1

        def metric_fn(logits: torch.Tensor) -> torch.Tensor:
            if logits.shape[1] <= ones_pos:
                raise ValueError(
                    "FullAdditionTask metric requires logits from "
                    "tokenize_for_metric(example)"
                )
            tens_diff = logits[0, tens_pos, t_tens] - logits[0, tens_pos, c_tens]
            ones_diff = logits[0, ones_pos, t_ones] - logits[0, ones_pos, c_ones]
            return tens_diff + ones_diff

        return metric_fn

    def tokenize_for_metric(self, example: TaskExample) -> torch.Tensor:
        return self.tokenize_prompt_answer_prefix(
            example.input_text, example.target_text
        )

    def tokenize_cf_for_metric(self, example: TaskExample) -> torch.Tensor:
        return self.tokenize_prompt_answer_prefix(
            example.cf_input_text, example.cf_target_text
        )

    def tokenize_prompt_answer_prefix(
        self, prompt: str, answer: str
    ) -> torch.Tensor:
        if len(answer) < 2:
            raise ValueError("FullAdditionTask metric needs at least two answer digits")
        return self.tokenizer(
            prompt + answer[:-1], return_tensors="pt", add_special_tokens=False,
        )["input_ids"]

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
        )["input_ids"]
