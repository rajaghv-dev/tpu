"""
Trace analyzer — turns a `ProfilerTrace` into the structured numbers the
bottleneck report uses.

Two outputs:

  * `compute_breakdown(trace)` — what fraction of step time was compile,
    device, input wait, collective, host overhead.

  * `step_summary(trace)` — per-step durations, useful for percentile
    stats and "slow first step" detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .profiler_trace import ProfilerTrace, TraceEvent


_KNOWN_CATS = (
    "compile", "device", "collective",
    "input_pipeline", "host", "checkpoint",
)


@dataclass
class Breakdown:
    compile_s: float
    device_s: float
    collective_s: float
    input_pipeline_s: float
    host_s: float
    checkpoint_s: float
    other_s: float

    def total(self) -> float:
        return sum(getattr(self, f"{c}_s") for c in _KNOWN_CATS) + self.other_s

    def fractions(self) -> Dict[str, float]:
        t = max(self.total(), 1e-9)
        return {c: getattr(self, f"{c}_s") / t for c in _KNOWN_CATS}


def compute_breakdown(trace: ProfilerTrace) -> Breakdown:
    by_cat = trace.breakdown_by_cat()
    known_sum = sum(by_cat.get(c, 0.0) for c in _KNOWN_CATS)
    other = sum(v for k, v in by_cat.items() if k not in _KNOWN_CATS)
    return Breakdown(
        compile_s=by_cat.get("compile", 0.0),
        device_s=by_cat.get("device", 0.0),
        collective_s=by_cat.get("collective", 0.0),
        input_pipeline_s=by_cat.get("input_pipeline", 0.0),
        host_s=by_cat.get("host", 0.0),
        checkpoint_s=by_cat.get("checkpoint", 0.0),
        other_s=other,
    )


@dataclass
class StepSummary:
    n_steps: int
    step_durations_s: List[float]
    first_step_s: float
    median_step_s: float
    p95_step_s: float


def step_summary(trace: ProfilerTrace) -> StepSummary:
    durations: List[float] = []
    for e in trace.events:
        if e.cat == "device" and e.name.startswith("step"):
            durations.append(e.dur_us / 1e6)
    if not durations:
        return StepSummary(0, [], 0.0, 0.0, 0.0)
    sd = sorted(durations[1:]) or sorted(durations)
    p95_idx = max(int(len(sd) * 0.95) - 1, 0)
    return StepSummary(
        n_steps=len(durations),
        step_durations_s=durations,
        first_step_s=durations[0],
        median_step_s=sd[len(sd) // 2] if sd else 0.0,
        p95_step_s=sd[p95_idx] if sd else 0.0,
    )
