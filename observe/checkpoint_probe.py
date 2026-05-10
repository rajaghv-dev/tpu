"""
observe/checkpoint_probe.py — record checkpoint sizes and write times.

Checkpointing is one of the most expensive non-compute parts of training.
A single 7B model BF16 checkpoint is ~14 GB; on a slow disk this can stall
the step loop for 30+ seconds. This probe captures the cost so the
dashboard can answer "how much wall time was actually spent saving?".

## Contract

The training runner calls `record_metric("checkpoint_write", duration_s, step)`
each time it writes a checkpoint, and additionally `record_metric(
"checkpoint_size_bytes", n_bytes, step)`. This probe collects all such
records and emits a per-checkpoint summary plus a total.

The probe also walks the run_log_dir's `checkpoints/` subdir at write_log
time so the JSON includes file sizes even if the runner forgot to record
them (defensive — checkpoints visible on disk are real regardless).

## Output

    {
      "n_checkpoints": 3,
      "total_write_time_s": 12.4,
      "total_size_bytes": 41943040,
      "events": [
        {"step": 100, "duration_s": 4.2, "size_bytes": 13981013, "path": "ckpt-100"},
        {"step": 200, ...},
        {"step": 300, ...}
      ],
      "discovered_files": [
        {"path": "checkpoints/ckpt-100", "size_bytes": 13981013}, ...
      ]
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.probe import Probe


class CheckpointProbe(Probe):
    """Track checkpoint write events emitted via record_metric."""

    name = "checkpoint"

    def __init__(self) -> None:
        self._log_dir: Optional[Path] = None
        self._events: List[Dict[str, Any]] = []
        # Buffer in-progress event so we can pair size_bytes with duration_s
        # by step number when they arrive in either order.
        self._pending_by_step: Dict[int, Dict[str, Any]] = {}

    def before_run(self, run_id: str, config: Any, log_dir: Path) -> None:
        self._log_dir = log_dir

    def record_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
    ) -> None:
        # Only react to checkpoint-shaped metrics. Anything else is
        # transparently ignored — other probes (TrainingMetricsProbe) will
        # capture it.
        if name not in ("checkpoint_write", "checkpoint_size_bytes", "checkpoint_path"):
            return
        if step is None:
            step = -1
        bucket = self._pending_by_step.setdefault(step, {"step": step})
        if name == "checkpoint_write":
            try:
                bucket["duration_s"] = float(value)
            except (TypeError, ValueError):
                return
        elif name == "checkpoint_size_bytes":
            try:
                bucket["size_bytes"] = int(value)
            except (TypeError, ValueError):
                return
        elif name == "checkpoint_path":
            bucket["path"] = str(value)
        # We deliberately do NOT flush on the fly. The runner may emit
        # duration / size / path in any order, so we accumulate per-step
        # buckets and emit them all at write_log. Checkpoints are rare
        # within a run; the memory cost is negligible.

    def write_log(self) -> Optional[Dict[str, Any]]:
        # Flush every pending bucket — partial or complete.
        for step, bucket in sorted(
            self._pending_by_step.items(), key=lambda kv: kv[0]
        ):
            self._events.append(dict(bucket))
        self._pending_by_step.clear()

        # On-disk discovery — independent of what the runner reported.
        discovered: List[Dict[str, Any]] = []
        if self._log_dir is not None:
            ckpt_root = self._log_dir / "checkpoints"
            if ckpt_root.exists():
                for p in sorted(ckpt_root.rglob("*")):
                    if p.is_file():
                        try:
                            discovered.append({
                                "path": str(p.relative_to(self._log_dir)),
                                "size_bytes": p.stat().st_size,
                            })
                        except OSError:
                            pass

        total_dur = sum(e.get("duration_s") or 0.0 for e in self._events)
        total_size = sum(e.get("size_bytes") or 0 for e in self._events)
        if total_size == 0 and discovered:
            total_size = sum(d["size_bytes"] for d in discovered)

        return {
            "n_checkpoints": len(self._events),
            "total_write_time_s": total_dur,
            "total_size_bytes": total_size,
            "events": list(self._events),
            "discovered_files": discovered,
        }
