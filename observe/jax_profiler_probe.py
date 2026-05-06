"""
observe/jax_profiler_probe.py — wrap the latency phase in a jax.profiler trace.

Captures one full `jax.profiler` trace covering the latency phase. The
trace directory is recorded in the probe log so it can be opened in
TensorBoard or Perfetto after the run.

## Behaviour

- `jax.profiler` is imported lazily — the probe tolerates JAX not being
  installed (returns `available=False`).
- `before_phase("latency")` starts the trace; failures (e.g. profiler
  not supported on the current backend) are caught so they cannot fail
  the benchmark.
- Both `after_phase("latency")` and `on_error("latency", ...)` stop the
  trace, otherwise the trace directory is left half-written and
  unparseable by downstream tools.
- The trace files (`xspace.pb`, `events.json.gz`, …) are not parsed
  here — they are dense protobuf or compressed-JSON binaries intended
  for visualisation tools.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from observe.probe import Probe

if TYPE_CHECKING:
    from benchmarks.runner import ExperimentConfig

_log = logging.getLogger(__name__)


class JaxProfilerProbe(Probe):
    """Wrap the `latency` phase in a single `jax.profiler` trace."""

    name = "jax_profiler"

    def __init__(self) -> None:
        self._trace_dir: Optional[Path] = None
        self._available: bool = False
        self._reason: Optional[str] = None
        self._started: bool = False
        self._stopped: bool = False
        self._jax_profiler: Any = None

    # ── Lifecycle ────────────────────────────────────────────────────────
    def before_run(
        self,
        run_id: str,
        config: "ExperimentConfig",
        log_dir: Path,
    ) -> None:
        self._trace_dir = (log_dir / "jax_profiler").resolve()
        self._trace_dir.mkdir(parents=True, exist_ok=True)

        # Lazy import — jax may not be installed in test envs.
        try:
            import jax.profiler as jp  # noqa: WPS433 — lazy by design
        except Exception as exc:  # noqa: BLE001 — profiler import is fragile
            self._available = False
            self._reason = f"jax.profiler unavailable: {type(exc).__name__}: {exc}"
            self._jax_profiler = None
            return
        self._jax_profiler = jp
        self._available = True

    def before_phase(self, phase_name: str) -> None:
        if phase_name != "latency":
            return
        if not self._available or self._jax_profiler is None:
            return
        if self._trace_dir is None:
            return
        try:
            self._jax_profiler.start_trace(str(self._trace_dir))
            self._started = True
        except Exception as exc:  # noqa: BLE001 — profiler start can fail in many ways
            self._available = False
            self._reason = f"start_trace failed: {type(exc).__name__}: {exc}"
            self._started = False
            _log.warning("jax_profiler start_trace failed: %s", exc)

    def after_phase(self, phase_name: str, duration_s: float) -> None:
        if phase_name != "latency":
            return
        self._safe_stop()

    def on_error(self, phase_name: str, exc: BaseException) -> None:
        # If the latency phase raised mid-trace we still need to stop
        # the trace, otherwise the directory is corrupt and TensorBoard
        # will refuse to open it.
        if phase_name != "latency":
            return
        self._safe_stop()

    # ── Output ───────────────────────────────────────────────────────────
    def write_log(self) -> Optional[Dict[str, Any]]:
        n_files = 0
        total_bytes = 0
        if self._trace_dir is not None and self._trace_dir.exists():
            try:
                for p in self._trace_dir.rglob("*"):
                    if p.is_file():
                        n_files += 1
                        try:
                            total_bytes += p.stat().st_size
                        except OSError:
                            pass
            except OSError as exc:
                _log.warning("could not walk trace dir: %s", exc)

        return {
            "trace_dir": str(self._trace_dir) if self._trace_dir else None,
            "n_files": n_files,
            "total_bytes": total_bytes,
            "available": self._available,
            "started": self._started,
            "stopped": self._stopped,
            "reason": self._reason,
        }

    # ── Internals ────────────────────────────────────────────────────────
    def _safe_stop(self) -> None:
        """Stop the trace if it was started; never raises."""
        if not self._started or self._stopped:
            return
        if self._jax_profiler is None:
            return
        try:
            self._jax_profiler.stop_trace()
            self._stopped = True
        except Exception as exc:  # noqa: BLE001 — must not fail the benchmark
            _log.warning("jax_profiler stop_trace failed: %s", exc)
            # Mark stopped anyway so we don't try again from on_error.
            self._stopped = True
            self._reason = f"stop_trace failed: {type(exc).__name__}: {exc}"
