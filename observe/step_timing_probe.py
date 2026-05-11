"""
observe/step_timing_probe.py — per-step wall-clock + throughput.

Records the wall-clock duration of every training step, computes steps/sec
and tokens/sec (when token count is supplied), and emits a short rolling
window so the runner can print mid-training progress without re-deriving.

## Why this is separate from TimingProbe

`observe/timing_probe.py` is keyed by *phase* (preflight, model_load,
compile, …) — coarse and one-shot per phase. Training has a long inner
*step* loop that the phase-level probe doesn't see. This probe sits below
the phase abstraction and records every iteration.

## Contract

Receives `before_step(step)` and `after_step(step, metrics)`. The metrics
dict MAY contain `tokens_in_batch` (or `samples_in_batch`); when present
the probe folds it into a tokens/sec rolling figure.

## Output shape

    {
      "n_steps": 200,
      "first_step_s": 5.20,        # cold compile + first step
      "median_step_s": 0.030,      # excluding the first 5 steps (warmup)
      "p95_step_s": 0.034,
      "p99_step_s": 0.036,
      "median_throughput_samples_sec": 1066.7,
      "median_throughput_tokens_sec": 273170.0,
      "step_durations_s": [5.20, 0.041, 0.032, ...],   # full list
      "rolling_window_size": 20
    }
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.probe import Probe

_WARMUP_STEPS = 5  # excluded from steady-state percentiles
_ROLLING_WINDOW = 20


class StepTimingProbe(Probe):
    """Per-step wall-clock + samples/sec + tokens/sec."""

    name = "step_timing"

    def __init__(self, rolling_window: int = _ROLLING_WINDOW) -> None:
        self._rolling_window = rolling_window
        self._step_start: Optional[float] = None
        self._durations: List[float] = []
        self._samples_per_step: List[int] = []
        self._tokens_per_step: List[int] = []

    def before_step(self, step: int) -> None:
        self._step_start = time.perf_counter()

    def after_step(self, step: int, metrics: Dict[str, Any]) -> None:
        if self._step_start is None:
            return
        dur = time.perf_counter() - self._step_start
        self._durations.append(dur)

        samples = metrics.get("samples_in_batch")
        tokens = metrics.get("tokens_in_batch")
        if isinstance(samples, (int, float)):
            self._samples_per_step.append(int(samples))
        if isinstance(tokens, (int, float)):
            self._tokens_per_step.append(int(tokens))

        self._step_start = None

    def rolling_throughput(self) -> Dict[str, Optional[float]]:
        """Return the trailing-window samples/sec + tokens/sec, or None if no data."""
        n = self._rolling_window
        if not self._durations:
            return {"samples_sec": None, "tokens_sec": None}
        recent_durs = self._durations[-n:]
        total_s = sum(recent_durs)
        if total_s <= 0:
            return {"samples_sec": None, "tokens_sec": None}
        out: Dict[str, Optional[float]] = {}
        if self._samples_per_step:
            out["samples_sec"] = sum(self._samples_per_step[-n:]) / total_s
        else:
            out["samples_sec"] = None
        if self._tokens_per_step:
            out["tokens_sec"] = sum(self._tokens_per_step[-n:]) / total_s
        else:
            out["tokens_sec"] = None
        return out

    def write_log(self) -> Optional[Dict[str, Any]]:
        if not self._durations:
            return {
                "n_steps": 0,
                "first_step_s": None,
                "median_step_s": None,
                "p95_step_s": None,
                "p99_step_s": None,
                "median_throughput_samples_sec": None,
                "median_throughput_tokens_sec": None,
                "step_durations_s": [],
                "rolling_window_size": self._rolling_window,
            }

        first = self._durations[0]
        steady = self._durations[_WARMUP_STEPS:] or self._durations[1:] or self._durations
        sorted_steady = sorted(steady)
        n = len(sorted_steady)

        def _pct(p: float) -> float:
            # Linear-interpolation percentile so small n doesn't snap to extremes.
            if n == 1:
                return sorted_steady[0]
            idx = (n - 1) * p
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            frac = idx - lo
            return sorted_steady[lo] * (1 - frac) + sorted_steady[hi] * frac

        # samples/sec and tokens/sec — only meaningful when batch sizes were
        # supplied. Use median per-step rate to avoid skew from the cold first
        # step (which includes compile time on JIT'd loops).
        per_step_samples_sec: List[float] = []
        per_step_tokens_sec: List[float] = []
        for i in range(min(len(self._durations), len(self._samples_per_step))):
            d = self._durations[i]
            if d > 0:
                per_step_samples_sec.append(self._samples_per_step[i] / d)
        for i in range(min(len(self._durations), len(self._tokens_per_step))):
            d = self._durations[i]
            if d > 0:
                per_step_tokens_sec.append(self._tokens_per_step[i] / d)

        median_samples = (
            statistics.median(per_step_samples_sec[_WARMUP_STEPS:] or per_step_samples_sec)
            if per_step_samples_sec
            else None
        )
        median_tokens = (
            statistics.median(per_step_tokens_sec[_WARMUP_STEPS:] or per_step_tokens_sec)
            if per_step_tokens_sec
            else None
        )

        return {
            "n_steps": len(self._durations),
            "first_step_s": first,
            "median_step_s": statistics.median(steady),
            "p95_step_s": _pct(0.95),
            "p99_step_s": _pct(0.99),
            "median_throughput_samples_sec": median_samples,
            "median_throughput_tokens_sec": median_tokens,
            "step_durations_s": list(self._durations),
            "rolling_window_size": self._rolling_window,
        }
