"""
observe/memory_probe.py — host-RSS / VMS snapshot probe.

Snapshots host process memory at every phase boundary using `psutil`. The
output JSON looks like:

    {
      "available": true,
      "baseline_rss_mb": 142.3,
      "baseline_vms_mb": 1820.4,
      "snapshots": [
        {"phase": "preflight", "when": "before", "rss_mb": 142.3, "vms_mb": ...},
        {"phase": "preflight", "when": "after",  "rss_mb": 142.4, "vms_mb": ...},
        ...
      ]
    }

If `psutil` is not installed, the probe degrades to a no-op and writes
`{"available": false, ...}` so dashboards can show "memory probe unavailable
on this host" instead of silently missing the file.

IMPORTANT — what this probe does NOT measure:

  * TPU HBM usage. psutil only sees the host (CPU) process; TPU memory lives
    in a separate device address space and is not visible via /proc.
  * GPU VRAM. Same reason.

For accelerator memory, see `cloud_monitoring_probe` (Stage 4) which polls
the GCP Monitoring API for per-chip HBM bytes-in-use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.probe import Probe


class MemoryProbe(Probe):
    """
    Sample host RSS + VMS at every phase boundary.

    Each phase produces two snapshots (`when="before"` and `when="after"`).
    On error a third snapshot tagged `when="on_error"` is appended.
    """

    name = "memory"

    def __init__(self) -> None:
        self._available: bool = False
        self._psutil = None
        self._proc = None
        self._baseline_rss_mb: Optional[float] = None
        self._baseline_vms_mb: Optional[float] = None
        self._snapshots: List[Dict[str, Any]] = []

        # Lazy import so a missing psutil doesn't hard-fail at module-load.
        try:
            import psutil  # type: ignore
            self._psutil = psutil
            self._proc = psutil.Process()
            self._available = True
        except Exception:  # noqa: BLE001 — any import failure → degrade
            self._available = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _snapshot(self) -> Optional[Dict[str, float]]:
        """Return current rss_mb/vms_mb, or None if psutil isn't usable."""
        if not self._available or self._proc is None:
            return None
        try:
            mi = self._proc.memory_info()
        except Exception:  # noqa: BLE001 — process gone or permission denied
            return None
        return {
            "rss_mb": mi.rss / (1024.0 * 1024.0),
            "vms_mb": mi.vms / (1024.0 * 1024.0),
        }

    # ── lifecycle ────────────────────────────────────────────────────────────

    def before_run(self, run_id: str, config: Any, log_dir: Path) -> None:
        snap = self._snapshot()
        if snap is not None:
            self._baseline_rss_mb = snap["rss_mb"]
            self._baseline_vms_mb = snap["vms_mb"]

    def before_phase(self, phase_name: str) -> None:
        snap = self._snapshot()
        if snap is None:
            return
        self._snapshots.append({
            "phase": phase_name,
            "when": "before",
            "rss_mb": snap["rss_mb"],
            "vms_mb": snap["vms_mb"],
        })

    def after_phase(self, phase_name: str, duration_s: float) -> None:
        snap = self._snapshot()
        if snap is None:
            return
        self._snapshots.append({
            "phase": phase_name,
            "when": "after",
            "rss_mb": snap["rss_mb"],
            "vms_mb": snap["vms_mb"],
        })

    def on_error(self, phase_name: str, exc: BaseException) -> None:
        snap = self._snapshot()
        if snap is None:
            return
        self._snapshots.append({
            "phase": phase_name,
            "when": "on_error",
            "rss_mb": snap["rss_mb"],
            "vms_mb": snap["vms_mb"],
        })

    def after_run(self, run_id: str, result: Optional[Dict[str, Any]]) -> None:
        # Per-phase samples are already the artefact; nothing to flush.
        return

    # ── output ───────────────────────────────────────────────────────────────

    def write_log(self) -> Optional[Dict[str, Any]]:
        return {
            "available": self._available,
            "baseline_rss_mb": self._baseline_rss_mb,
            "baseline_vms_mb": self._baseline_vms_mb,
            "snapshots": list(self._snapshots),
        }
