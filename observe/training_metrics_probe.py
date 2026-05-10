"""
observe/training_metrics_probe.py — capture per-step training metrics.

Records loss, grad_norm, learning_rate, and any other scalar emitted by the
training runner's `after_step(step, metrics)` callback. The output JSON is
keyed by step number; the dashboard joins it against the step timing probe
to produce loss-vs-time and loss-vs-tokens curves.

## Contract

The training runner (train/runner.py) calls fanout_after_step(step, metrics)
exactly once per training step, where `metrics` is a dict of scalars. This
probe stores them as-is — what gets logged is whatever the runner emits.

## Why a probe and not a logger

Same reason every other observability artefact is a probe: the runner stays
small and uniform; new metric collectors plug in without touching the loop.
Per-step OTel spans, W&B sync, custom CSVs — all of those become 50-line
probes that subscribe to the same `after_step` event.

## Output shape

    {
      "n_steps": 200,
      "metric_keys": ["loss", "lr", "grad_norm"],
      "history": [
        {"step": 0,   "loss": 4.21, "lr": 0.0,        "grad_norm": 1.34},
        {"step": 1,   "loss": 4.18, "lr": 1.0e-7,     "grad_norm": 1.31},
        ...
      ],
      "ad_hoc": [
        {"name": "eval_loss",     "value": 0.42, "step": 100},
        {"name": "eval_accuracy", "value": 0.89, "step": 100}
      ]
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from observe.probe import Probe


class TrainingMetricsProbe(Probe):
    """Record per-step scalar metrics emitted by the training runner."""

    name = "training_metrics"

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []
        self._ad_hoc: List[Dict[str, Any]] = []
        self._metric_keys: Set[str] = set()

    def after_step(self, step: int, metrics: Dict[str, Any]) -> None:
        # We materialise scalars only — JAX/Torch tensors must be cast by the
        # caller before reaching us. Keeping this strict prevents accidental
        # device-host syncs at metric-record time.
        row: Dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            if isinstance(v, (bool, int, float, str)) or v is None:
                row[k] = v
                self._metric_keys.add(k)
            else:
                # Best-effort float cast — silently drop on failure rather
                # than raise. The probe must not fail the training loop.
                try:
                    row[k] = float(v)
                    self._metric_keys.add(k)
                except (TypeError, ValueError):
                    pass
        self._history.append(row)

    def record_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
    ) -> None:
        if isinstance(value, (bool, int, float, str)) or value is None:
            cast_value: Any = value
        else:
            try:
                cast_value = float(value)
            except (TypeError, ValueError):
                return
        self._ad_hoc.append({"name": name, "value": cast_value, "step": step})

    def write_log(self) -> Optional[Dict[str, Any]]:
        return {
            "n_steps": len(self._history),
            "metric_keys": sorted(self._metric_keys),
            "history": list(self._history),
            "ad_hoc": list(self._ad_hoc),
        }
