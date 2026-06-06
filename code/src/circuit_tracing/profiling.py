from __future__ import annotations

import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler


def add_profile_args(ap) -> None:
    ap.add_argument("--profile-trace-dir", default=None)
    ap.add_argument("--profile-jsonl", default=None)
    ap.add_argument("--profile-warmup-steps", type=int, default=1)
    ap.add_argument("--profile-active-steps", type=int, default=3)
    ap.add_argument("--profile-repeat", type=int, default=1)


class PhaseProfiler:
    def __init__(
        self,
        *,
        trace_dir: str | None,
        jsonl_path: str | None,
        device: str,
        warmup_steps: int,
        active_steps: int,
        repeat: int,
    ) -> None:
        self._step = 0
        self._jsonl = None
        self._prof = None
        self._device = device

        if jsonl_path:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = path.open("a", encoding="utf-8")

        if trace_dir:
            acts = [ProfilerActivity.CPU]
            if device == "cuda" and torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)
            trace_path = Path(trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
            self._prof = profile(
                activities=acts,
                schedule=schedule(
                    wait=0,
                    warmup=max(warmup_steps, 0),
                    active=max(active_steps, 1),
                    repeat=max(repeat, 1),
                ),
                on_trace_ready=tensorboard_trace_handler(str(trace_path)),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self._prof.__enter__()

    @property
    def enabled(self) -> bool:
        return self._prof is not None or self._jsonl is not None

    def _write(self, payload: dict[str, Any]) -> None:
        if self._jsonl is None:
            return
        self._jsonl.write(json.dumps(payload, sort_keys=True) + "\n")
        self._jsonl.flush()

    @contextmanager
    def phase(self, name: str, **meta: Any):
        ctx = record_function(name) if self._prof is not None else nullcontext()
        t0 = time.perf_counter()
        with ctx:
            yield
        dt = time.perf_counter() - t0
        self._write(
            {
                "type": "phase",
                "name": name,
                "step": self._step,
                "duration_s": dt,
                **meta,
            }
        )

    def step(self, **meta: Any) -> None:
        self._step += 1
        if self._prof is not None:
            self._prof.step()
        self._write({"type": "step", "step": self._step, **meta})

    def close(self) -> None:
        if self._prof is not None:
            self._prof.__exit__(None, None, None)
            self._prof = None
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None


def make_profiler(args, device: str) -> PhaseProfiler:
    return PhaseProfiler(
        trace_dir=args.profile_trace_dir,
        jsonl_path=args.profile_jsonl,
        device=device,
        warmup_steps=args.profile_warmup_steps,
        active_steps=args.profile_active_steps,
        repeat=args.profile_repeat,
    )
