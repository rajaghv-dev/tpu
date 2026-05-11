"""
observe/cloud_monitoring_probe.py — Cloud Monitoring TPU silicon probe.

Polls Google Cloud Monitoring once a second during a benchmark run for
TPU-chip-level metrics (MXU util, HBM BW, HBM capacity, host CPU, network)
and emits a per-phase min/mean/max summary plus the raw timeline.

The polled metric types are per-TPU-chip, exported by the TPU runtime to
Cloud Monitoring (see Google Cloud TPU monitoring docs). They are sampled
on the GCP control plane, NOT on-host — there is a ~30-90s ingestion lag,
so the timeline rows are GCP-timestamps, not local clock.

This probe is OPTIONAL. If `google-cloud-monitoring` is not installed, or
auth/project/instance can't be resolved, the probe degrades to a no-op and
emits `{"available": false, ...}` so callers always get a stable artifact.

Stdlib only besides the (lazy, optional) `google.cloud.monitoring_v3` import.
"""
from __future__ import annotations

import logging
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from observe.probe import Probe

_log = logging.getLogger(__name__)

# Per-TPU-chip metrics published by the TPU runtime to Cloud Monitoring.
# Keys are short labels we use in the summary; values are the full GCP
# metric type strings.
_METRICS: Dict[str, str] = {
    "mxu_utilization":              "tpu.googleapis.com/tpu/mxu_utilization",
    "network_sent_bytes_count":     "tpu.googleapis.com/tpu/network/sent_bytes_count",
    "memory_bandwidth_utilization": "tpu.googleapis.com/tpu/memory_bandwidth_utilization",
    "memory_utilization":           "tpu.googleapis.com/tpu/memory_utilization",
    "cpu_utilization":              "tpu.googleapis.com/tpu/cpu/utilization",
}

_POLL_INTERVAL_S = 1.0
_MAX_SAMPLES = 7200  # 2h cap; oldest dropped on overflow.
_DRAIN_TIMEOUT_S = 2.0
_GCLOUD_TIMEOUT_S = 2.0
_STATE_ENV_PATH = Path(__file__).resolve().parent.parent / ".tpu-bench-state" / "state.env"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_state_env_value(key: str, path: Path = _STATE_ENV_PATH) -> Optional[str]:
    """
    Best-effort read of a `KEY=value` line from state.env. Returns None on any
    failure (missing file, bad encoding, key absent). Quotes are stripped.
    """
    try:
        text = path.read_text()
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def _gcloud_default_project() -> Optional[str]:
    """Return `gcloud config get-value project` or None on any failure."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, timeout=_GCLOUD_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    val = (result.stdout or "").strip()
    return val or None


def aggregate_per_phase_summary(
    samples: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Pure aggregation: list of {ts, phase, metric, value} -> nested
    {phase: {metric: {min, mean, max}}}.

    Factored out so unit tests can verify aggregation without spinning up
    the polling thread.
    """
    by_phase_metric: Dict[Tuple[str, str], List[float]] = {}
    for row in samples:
        try:
            phase = row["phase"]
            metric = row["metric"]
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        by_phase_metric.setdefault((phase, metric), []).append(value)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (phase, metric), values in by_phase_metric.items():
        if not values:
            continue
        summary.setdefault(phase, {})[metric] = {
            "min": float(min(values)),
            "mean": float(statistics.fmean(values)),
            "max": float(max(values)),
        }
    return summary


# ── Probe ─────────────────────────────────────────────────────────────────────

class CloudMonitoringProbe(Probe):
    """
    Polls Cloud Monitoring once per second for TPU silicon metrics, tagging
    each sample with the current benchmark phase. No-op fallback if GCP
    libs/creds/instance metadata are absent.
    """

    name = "cloud_monitoring"

    def __init__(
        self,
        project: Optional[str] = None,
        tpu_name: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> None:
        # Public-ish state (read by tests and write_log).
        self._available: bool = False
        self._current_phase: str = "setup"
        self._samples: List[Dict[str, Any]] = []
        # Thread plumbing.
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # Lazy GCP import handle.
        self._monitoring_v3 = None
        self._client = None

        self._project: Optional[str] = (
            project or os.environ.get("GCP_PROJECT") or _gcloud_default_project()
        )
        self._tpu_name: Optional[str] = (
            tpu_name or os.environ.get("TPU_NAME") or _read_state_env_value("TPU_NAME")
        )
        self._zone: Optional[str] = (
            zone or os.environ.get("TPU_ZONE") or _read_state_env_value("TPU_ZONE")
        )

        if not (self._project and self._tpu_name and self._zone):
            _log.info(
                "CloudMonitoringProbe disabled: missing project/tpu_name/zone "
                "(project=%r, tpu=%r, zone=%r)",
                self._project, self._tpu_name, self._zone,
            )
            return

        # Lazy, graceful import. Missing package == no-op.
        try:
            from google.cloud import monitoring_v3  # type: ignore
        except Exception as exc:  # noqa: BLE001
            _log.info(
                "CloudMonitoringProbe disabled: google.cloud.monitoring_v3 "
                "import failed: %s", exc,
            )
            return

        try:
            client = monitoring_v3.MetricServiceClient()
        except Exception as exc:  # noqa: BLE001
            _log.info(
                "CloudMonitoringProbe disabled: client init failed (likely "
                "missing ADC): %s", exc,
            )
            return

        self._monitoring_v3 = monitoring_v3
        self._client = client
        self._available = True

    # ── Lifecycle hooks ──────────────────────────────────────────────────

    def before_run(self, run_id, config, log_dir: Path) -> None:  # type: ignore[override]
        if not self._available:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="cloud-mon-probe", daemon=True,
        )
        self._thread.start()

    def before_phase(self, phase_name: str) -> None:  # type: ignore[override]
        self._current_phase = phase_name

    def after_phase(self, phase_name: str, duration_s: float) -> None:  # type: ignore[override]
        # Stay tagged with the just-finished phase until the next before_phase.
        # This keeps inter-phase samples attributed to the most recent phase.
        self._current_phase = phase_name

    def on_error(self, phase_name: str, exc: BaseException) -> None:  # type: ignore[override]
        self._current_phase = phase_name

    def after_run(self, run_id: str, result) -> None:  # type: ignore[override]
        if not self._available:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_DRAIN_TIMEOUT_S)
            self._thread = None

    def write_log(self) -> Optional[Dict[str, Any]]:  # type: ignore[override]
        with self._lock:
            samples_snapshot = list(self._samples)
        return {
            "available": self._available,
            "polling_interval_s": _POLL_INTERVAL_S,
            "n_samples": len(samples_snapshot),
            "samples": samples_snapshot,
            "per_phase_summary": aggregate_per_phase_summary(samples_snapshot),
            "tpu_name": self._tpu_name,
            "zone": self._zone,
            "project": self._project,
        }

    # ── Polling thread ───────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """
        Run until stop_event is set. Each iteration does ONE batch query for
        all metrics. Tolerates transient API failures with a single retry +
        warning. Never raises out (probes must not crash the benchmark).
        """
        while not self._stop_event.is_set():
            cycle_start = time.time()
            try:
                self._poll_once(cycle_start)
            except Exception as exc:  # noqa: BLE001 — defensive top-level
                _log.warning(
                    "cloud_monitoring_probe poll cycle failed unexpectedly: %s: %s",
                    type(exc).__name__, exc,
                )
            # Sleep the remainder of the interval; wake early on stop.
            elapsed = time.time() - cycle_start
            remaining = max(0.0, _POLL_INTERVAL_S - elapsed)
            if self._stop_event.wait(remaining):
                break

    def _poll_once(self, ts: float) -> None:
        """One full sweep across all metrics. ts = start-of-cycle epoch sec."""
        for label, metric_type in _METRICS.items():
            value = self._fetch_metric_with_retry(metric_type)
            if value is None:
                continue
            self._append_sample({
                "ts": ts,
                "phase": self._current_phase,
                "metric": label,
                "value": value,
            })

    def _fetch_metric_with_retry(self, metric_type: str) -> Optional[float]:
        """One retry on failure; warns and returns None if both attempts fail."""
        for attempt in (1, 2):
            try:
                return self._fetch_metric(metric_type)
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    _log.warning(
                        "cloud_monitoring_probe metric %s failed after retry: "
                        "%s: %s",
                        metric_type, type(exc).__name__, exc,
                    )
                    return None
                # First failure: brief backoff, swallow and retry.
                time.sleep(0.1)
        return None

    def _fetch_metric(self, metric_type: str) -> Optional[float]:
        """
        Issue a list_time_series call for the last 5 minutes (Cloud Monitoring
        ingestion lag is up to ~90s; smaller windows risk empty responses) and
        return the most recent point's double_value, or None if no point.
        """
        assert self._monitoring_v3 is not None and self._client is not None
        m_v3 = self._monitoring_v3

        now = time.time()
        interval = m_v3.TimeInterval({
            "end_time":   {"seconds": int(now)},
            "start_time": {"seconds": int(now - 300)},
        })
        # Filter on metric type AND this TPU instance. The TPU runtime
        # tags samples with `resource.label.instance_id` (not name) on some
        # TPU generations; name-matching via metric labels is the portable
        # path. See Google Cloud TPU metrics docs.
        flt = (
            f'metric.type = "{metric_type}" '
            f'AND resource.labels.zone = "{self._zone}" '
            f'AND (resource.labels.node_id = "{self._tpu_name}" '
            f'OR metric.labels.instance_name = "{self._tpu_name}")'
        )
        request = {
            "name": f"projects/{self._project}",
            "filter": flt,
            "interval": interval,
            "view": m_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
        series_iter = self._client.list_time_series(request=request)

        latest_ts = -1
        latest_val: Optional[float] = None
        for series in series_iter:
            for pt in getattr(series, "points", []) or []:
                end_time = getattr(getattr(pt, "interval", None), "end_time", None)
                pt_ts = int(getattr(end_time, "seconds", 0) or 0)
                val = getattr(pt, "value", None)
                # TPU metrics are doubles; fall back to int64 just in case.
                num = (
                    getattr(val, "double_value", None)
                    if val is not None else None
                )
                if num is None or num == 0.0:
                    int_val = getattr(val, "int64_value", None) if val is not None else None
                    if int_val is not None:
                        num = float(int_val)
                if num is None:
                    continue
                if pt_ts > latest_ts:
                    latest_ts = pt_ts
                    latest_val = float(num)
        return latest_val

    def _append_sample(self, row: Dict[str, Any]) -> None:
        """Append with a 7200-row cap (drop oldest on overflow)."""
        with self._lock:
            self._samples.append(row)
            overflow = len(self._samples) - _MAX_SAMPLES
            if overflow > 0:
                # Drop oldest in one slice — O(n) but n is bounded.
                del self._samples[:overflow]
