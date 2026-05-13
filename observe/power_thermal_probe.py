"""
observe/power_thermal_probe.py — background sampler for power, temperature,
and utilization across GPUs, TPUs (best-effort), and the host.

## Why this probe matters

Throughput regressions during training are most often blamed on the model code
or the data pipeline, when in fact the silicon itself has throttled. The two
common silent failure modes are:

  1. **Thermal throttling.** Both NVIDIA GPUs and Google TPUs reduce clock
     speeds once junction temperature passes an internal threshold (commonly
     ~83 C on consumer NVIDIA, ~85 C on TPU v5e/v6e). A 10 C rise above that
     threshold can halve sustained throughput — silently. Training loss and
     gradients look fine; steps just take longer. Without a temperature trace
     you have no way to attribute a slowdown to thermals vs. data starvation
     vs. code. Watch `max` temp_c per device across the train_loop phase.

  2. **Power capping.** On hosted environments (Colab, Kaggle, some Lambda
     pods) the accelerator's power limit is set well below the device TDP. A
     GPU that should pull 350 W might be capped at 180 W. Power draw is a
     much faster signal than utilization counters — utilization can read 99%
     while the device is power-limited and underperforming. Compare `mean`
     power_w against the device's nameplate TDP. If the ceiling is well below
     spec across the whole run, you are power-capped and no amount of code
     tuning will help.

A third use case is **host-bottleneck detection.** If host CPU load_avg_1m is
consistently above the number of cores, or RAM% is above ~85, your data
pipeline or tokenizer is the bottleneck — the accelerator will sit idle
regardless of model size. Look for high host load with simultaneously low
util_gpu_pct samples to confirm.

## Tools used (all optional; the probe degrades per-source)

  * **nvidia-smi** — ubiquitous on NVIDIA hosts. Returns power.draw (W),
    temperature.gpu (C, junction), utilization.gpu (% of SM busy in last
    sample window), utilization.memory (% of memory controller busy, NOT %
    of VRAM used), memory.used, memory.total. Sampled in CSV-noheader form
    for cheap parsing. Caveat: utilization.gpu is the fraction of time at
    least one kernel was running; it does not reflect SM occupancy. A kernel
    that touches one SM still reads 100% util. Use power_w as the
    higher-fidelity signal.

  * **tpu-info** — ships with `libtpu` / Cloud TPU VM images on some recent
    releases. Not present on Colab TPU runtimes or older v3-8 pods. When
    available it returns per-chip duty cycle and HBM bytes-in-use. This is
    the best-effort source for TPU thermal/util data; Stage 2 will replace
    this with a real libtpu binding.

  * **rocm-smi** — AMD's equivalent of nvidia-smi. Same shape of data.
    Included for completeness; almost no project hosts have it.

  * **psutil** — host CPU%, load average, RAM%. cpu_percent(interval=None)
    is the non-blocking variant; the first call is meaningless (returns 0.0)
    but every subsequent call returns the % since the previous call, which
    fits the 1 Hz sampling cadence well.

  * **/sys/class/thermal/thermal_zone*/temp** — Linux thermal subsystem.
    Each zone is a sensor; you typically see CPU package, NVMe, ambient,
    and on laptops the chassis. Values in millidegrees C. Useful for
    confirming whether a thermal event is the accelerator (nvidia-smi) or
    the host (these zones).

## How to read the output

`write_log()` returns a flat structure with three pieces:

  * `tools.*` — which sources actually worked on this host. Use this to
    explain "why is gpus empty?" without surprise.
  * `samples[]` — the raw 1 Hz trace, each entry tagged with the phase
    name (preflight / model_load / compile / warmup / train_loop / ...) and,
    inside the training loop, the current step. Phase tagging lets you ask
    "what were temps during compile vs steady-state training?" Joining on
    `ts_mono` (perf_counter) lets you align with the timing probe's
    timestamps without clock-skew issues.
  * `summary.gpu_<i>` — pre-computed mean / max / p95 of power_w and
    mean / max of temp_c, computed in `after_run` over the retained samples.
    This is the quick-glance triage view; the dashboard renders these as a
    single row per device. The full trace stays in `samples[]` for anyone
    who wants to plot it.

## Operational notes

  * Sampling happens in a daemon background thread, NEVER on the training
    hot path. At 1 Hz the sampling overhead is dominated by subprocess
    fork+exec of nvidia-smi (~5–10 ms wall, near-zero CPU on a 32-core host).
  * Samples are capped at 10 000 (~2 h 47 m at 1 Hz). Older samples are
    dropped; the count of dropped samples is recorded.
  * Tools that fail once are marked dead and skipped on subsequent samples,
    so a missing nvidia-smi does not pay the 2 s timeout every second.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from observe.probe import Probe

_log = logging.getLogger(__name__)

_SUBPROC_TIMEOUT_S = 2.0
_MAX_SAMPLES = 10_000
_THERMAL_ZONE_GLOB = "/sys/class/thermal"


class PowerThermalProbe(Probe):
    """
    Background sampler for power draw, temperature, and utilization.

    Lifecycle:
      * `before_run` starts the worker thread.
      * Worker loop ticks at `sample_rate_hz` (default 1 Hz, clamped to
        [0.1, 10.0]).
      * `before_phase` / `before_step` / `after_step` update tagging state so
        each sample carries its `phase` and (during the training loop) `step`.
      * `after_run` signals shutdown, joins with a 2 s timeout, then computes
        the per-device summary stats.
      * `write_log` returns the assembled payload.
    """

    name = "power_thermal"

    def __init__(self, sample_rate_hz: float = 1.0) -> None:
        # Clamp to sane range. Faster than 10 Hz starts to perturb the host;
        # slower than 0.1 Hz means you miss short throttling events.
        if sample_rate_hz < 0.1:
            sample_rate_hz = 0.1
        elif sample_rate_hz > 10.0:
            sample_rate_hz = 10.0
        self._sample_rate_hz: float = float(sample_rate_hz)
        self._period_s: float = 1.0 / self._sample_rate_hz

        # Shared state. The worker thread only writes to _samples /
        # _samples_dropped / _tools_dead; the main thread only writes to the
        # phase/step tags. Python's GIL makes single-attribute reads/writes
        # atomic enough for this purpose — no lock needed.
        self._samples: Deque[Dict[str, Any]] = deque(maxlen=_MAX_SAMPLES)
        self._samples_dropped: int = 0
        self._tools_dead: Set[str] = set()
        self._tools_seen_ok: Set[str] = set()

        # Phase / step tagging — updated from the main thread, read from the
        # worker thread. Atomic single-assignment is fine.
        self._current_phase: Optional[str] = None
        self._current_step: Optional[int] = None

        # Thread control.
        self._stop_evt: threading.Event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # Whether psutil is usable. Discovered lazily on first sample so the
        # probe construction itself is free.
        self._psutil = None
        self._psutil_checked: bool = False

        # Cached list of thermal-zone files. Discovered once.
        self._thermal_zone_files: Optional[List[str]] = None

        # Final summary, populated in after_run.
        self._summary: Dict[str, Any] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def before_run(self, run_id: str, config: Any, log_dir: Path) -> None:
        self._current_phase = None
        self._current_step = None
        self._stop_evt.clear()
        self._worker = threading.Thread(
            target=self._run_worker,
            name="power_thermal_probe",
            daemon=True,
        )
        self._worker.start()

    def before_phase(self, phase_name: str) -> None:
        self._current_phase = phase_name

    def after_phase(self, phase_name: str, duration_s: float) -> None:
        # Leave _current_phase as-is; the next before_phase will overwrite it.
        # This means samples taken between phases get tagged with the phase
        # that just ended, which is the most useful default.
        return

    def on_error(self, phase_name: str, exc: BaseException) -> None:
        # Same as after_phase — keep the tag so post-mortem samples are
        # attributable to the phase that crashed.
        return

    def before_step(self, step: int) -> None:
        self._current_step = step

    def after_step(self, step: int, metrics: Dict[str, Any]) -> None:
        # Clear so samples taken between training steps (e.g. an eval phase)
        # are not falsely attributed to the last seen step.
        self._current_step = None

    def after_run(self, run_id: str, result: Optional[Dict[str, Any]]) -> None:
        self._stop_evt.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        # Compute summary now, outside the worker, so write_log is pure.
        self._summary = self._compute_summary()

    # ── worker thread ───────────────────────────────────────────────────────

    def _run_worker(self) -> None:
        """
        Sample loop. Every iteration: take one sample, sleep the remainder of
        the period. The loop body is wrapped in a broad try/except so a bug
        in one sampler never kills the thread — the probe is observability,
        it must not raise into the runner.
        """
        while not self._stop_evt.is_set():
            t_start = time.perf_counter()
            try:
                sample = self._sample()
                if len(self._samples) == _MAX_SAMPLES:
                    self._samples_dropped += 1
                self._samples.append(sample)
            except Exception as exc:  # noqa: BLE001 — must never crash thread
                _log.warning("power_thermal sample raised %s: %s",
                             type(exc).__name__, exc)
            elapsed = time.perf_counter() - t_start
            remaining = self._period_s - elapsed
            if remaining > 0:
                self._stop_evt.wait(remaining)

    # ── sampling ────────────────────────────────────────────────────────────

    def _sample(self) -> Dict[str, Any]:
        """Take one full sample across all enabled sources."""
        gpus = self._sample_nvidia()
        tpus = self._sample_tpu()
        amd_gpus = self._sample_amd()
        host = self._sample_host()

        # Merge AMD GPUs into the same list under the gpus key with a vendor
        # marker so dashboards don't need a separate code path. Empty lists
        # are kept (not omitted) so consumers can rely on the key existing.
        all_gpus: List[Dict[str, Any]] = []
        all_gpus.extend(gpus)
        all_gpus.extend(amd_gpus)

        return {
            "ts": time.time(),
            "ts_mono": time.perf_counter(),
            "phase": self._current_phase,
            "step": self._current_step,
            "gpus": all_gpus,
            "tpus": tpus,
            "host": host,
        }

    # ── NVIDIA ──────────────────────────────────────────────────────────────

    def _sample_nvidia(self) -> List[Dict[str, Any]]:
        if "nvidia_smi" in self._tools_dead:
            return []
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,power.draw,temperature.gpu,"
            "utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = self._run_cmd(cmd, "nvidia_smi")
        if out is None:
            return []
        results: List[Dict[str, Any]] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            try:
                results.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "power_w": _to_float(parts[2]),
                    "temp_c": _to_float(parts[3]),
                    "util_gpu_pct": _to_float(parts[4]),
                    "util_mem_pct": _to_float(parts[5]),
                    "mem_used_mib": _to_float(parts[6]),
                    "mem_total_mib": _to_float(parts[7]),
                })
            except (ValueError, IndexError):
                # One malformed row should not kill the whole sample. Skip.
                continue
        if results:
            self._tools_seen_ok.add("nvidia_smi")
        return results

    # ── TPU ─────────────────────────────────────────────────────────────────

    def _sample_tpu(self) -> List[Dict[str, Any]]:
        """
        Best-effort TPU sampling via the `tpu-info` CLI. This is a stop-gap
        for Stage 1; Stage 2 will add a libtpu-backed sampler that reads
        device counters directly without forking a subprocess.
        """
        if "tpu_info" in self._tools_dead:
            return []
        if shutil.which("tpu-info") is None:
            self._tools_dead.add("tpu_info")
            return []
        out = self._run_cmd(["tpu-info", "--json"], "tpu_info")
        if out is None:
            return []
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            # Some versions of tpu-info ignore --json and emit a text table.
            # Treat as dead so we don't burn a 2 s timeout every second.
            self._tools_dead.add("tpu_info")
            return []
        # Shape is version-dependent; we only commit to "list of dicts" and
        # pass it through. Downstream dashboards normalise per version.
        if isinstance(parsed, list):
            chips = parsed
        elif isinstance(parsed, dict) and "chips" in parsed:
            chips = parsed.get("chips") or []
        else:
            chips = []
        if chips:
            self._tools_seen_ok.add("tpu_info")
        return list(chips)

    # ── AMD ─────────────────────────────────────────────────────────────────

    def _sample_amd(self) -> List[Dict[str, Any]]:
        if "rocm_smi" in self._tools_dead:
            return []
        if shutil.which("rocm-smi") is None:
            self._tools_dead.add("rocm_smi")
            return []
        out = self._run_cmd(["rocm-smi", "--json"], "rocm_smi")
        if out is None:
            return []
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            self._tools_dead.add("rocm_smi")
            return []
        # rocm-smi --json returns a dict keyed by "card0", "card1", ...
        results: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            for key, payload in parsed.items():
                if not isinstance(payload, dict):
                    continue
                results.append({
                    "index": _strip_card_prefix(key),
                    "name": payload.get("Card series") or payload.get("name"),
                    "power_w": _to_float(payload.get("Average Graphics Package Power (W)")),
                    "temp_c": _to_float(payload.get("Temperature (Sensor edge) (C)")),
                    "util_gpu_pct": _to_float(payload.get("GPU use (%)")),
                    "raw": payload,
                })
        if results:
            self._tools_seen_ok.add("rocm_smi")
        return results

    # ── Host ────────────────────────────────────────────────────────────────

    def _sample_host(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not self._psutil_checked:
            self._psutil_checked = True
            try:
                import psutil  # type: ignore
                self._psutil = psutil
            except Exception:  # noqa: BLE001 — any import failure → degrade
                self._psutil = None
        psutil = self._psutil
        if psutil is not None:
            try:
                # interval=None — non-blocking, returns % since last call.
                out["cpu_pct"] = float(psutil.cpu_percent(interval=None))
            except Exception:  # noqa: BLE001
                out["cpu_pct"] = None
            try:
                la = psutil.getloadavg()
                out["load_avg_1m"] = float(la[0])
            except (AttributeError, OSError):
                # Windows lacks getloadavg.
                out["load_avg_1m"] = None
            try:
                out["ram_pct"] = float(psutil.virtual_memory().percent)
            except Exception:  # noqa: BLE001
                out["ram_pct"] = None
            self._tools_seen_ok.add("psutil")

        # Thermal zones (Linux only). Cheap enough to do every sample once
        # the file list is cached.
        zones = self._read_thermal_zones()
        if zones is not None:
            out["thermal_zones_c"] = zones
            self._tools_seen_ok.add("thermal_zones")
        return out

    def _read_thermal_zones(self) -> Optional[List[float]]:
        """Read /sys/class/thermal/thermal_zone*/temp. None if unavailable."""
        if "thermal_zones" in self._tools_dead:
            return None
        if self._thermal_zone_files is None:
            try:
                base = _THERMAL_ZONE_GLOB
                if not os.path.isdir(base):
                    self._tools_dead.add("thermal_zones")
                    self._thermal_zone_files = []
                    return None
                files: List[str] = []
                for entry in sorted(os.listdir(base)):
                    if not entry.startswith("thermal_zone"):
                        continue
                    p = os.path.join(base, entry, "temp")
                    if os.path.isfile(p):
                        files.append(p)
                self._thermal_zone_files = files
            except OSError:
                self._tools_dead.add("thermal_zones")
                self._thermal_zone_files = []
                return None
        if not self._thermal_zone_files:
            return None
        values: List[float] = []
        for path in self._thermal_zone_files:
            try:
                with open(path, "r") as f:
                    raw = f.read().strip()
                # Sysfs reports millidegrees C.
                values.append(round(int(raw) / 1000.0, 1))
            except (OSError, ValueError):
                # Per-zone failure is fine; just skip the zone.
                continue
        return values

    # ── subprocess helper ──────────────────────────────────────────────────

    def _run_cmd(self, cmd: List[str], tool_key: str) -> Optional[str]:
        """
        Run cmd with a 2 s timeout. Returns stdout on success, None on any
        failure. The first failure marks the tool dead so subsequent samples
        skip it cheaply.
        """
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_SUBPROC_TIMEOUT_S,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            self._tools_dead.add(tool_key)
            return None
        if res.returncode != 0:
            self._tools_dead.add(tool_key)
            return None
        try:
            return res.stdout.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            self._tools_dead.add(tool_key)
            return None

    # ── summary ────────────────────────────────────────────────────────────

    def _compute_summary(self) -> Dict[str, Any]:
        """
        Aggregate per-GPU power/temp stats across retained samples. Restricted
        to NVIDIA + AMD entries (anything in `gpus`). TPU summary is left for
        Stage 2 once the schema stabilises.
        """
        # Bucket: gpu_index -> {"power_w": [...], "temp_c": [...]}.
        per_gpu: Dict[int, Dict[str, List[float]]] = {}
        for sample in self._samples:
            for g in sample.get("gpus") or []:
                idx = g.get("index")
                if idx is None:
                    continue
                bucket = per_gpu.setdefault(int(idx), {"power_w": [], "temp_c": []})
                if isinstance(g.get("power_w"), (int, float)):
                    bucket["power_w"].append(float(g["power_w"]))
                if isinstance(g.get("temp_c"), (int, float)):
                    bucket["temp_c"].append(float(g["temp_c"]))

        out: Dict[str, Any] = {}
        for idx, bucket in sorted(per_gpu.items()):
            out[f"gpu_{idx}"] = {
                "power_w": _scalar_stats(bucket["power_w"], with_p95=True),
                "temp_c": _scalar_stats(bucket["temp_c"], with_p95=False),
            }
        return out

    # ── output ─────────────────────────────────────────────────────────────

    def write_log(self) -> Optional[Dict[str, Any]]:
        available = bool(self._tools_seen_ok)
        return {
            "available": available,
            "sample_rate_hz": self._sample_rate_hz,
            "tools": {
                "nvidia_smi": "nvidia_smi" in self._tools_seen_ok,
                "tpu_info": "tpu_info" in self._tools_seen_ok,
                "rocm_smi": "rocm_smi" in self._tools_seen_ok,
                "psutil": "psutil" in self._tools_seen_ok,
                "thermal_zones": "thermal_zones" in self._tools_seen_ok,
            },
            "n_samples": len(self._samples),
            "samples_dropped_overflow": self._samples_dropped,
            "samples": list(self._samples),
            "summary": self._summary,
        }


# ── small helpers ────────────────────────────────────────────────────────────


def _to_float(v: Any) -> Optional[float]:
    """
    Permissive float coercion. nvidia-smi sometimes emits "[N/A]" or
    "[Not Supported]" for power/util on lower-end SKUs — those become None
    rather than corrupting downstream stats.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.startswith("["):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _strip_card_prefix(key: str) -> Any:
    """rocm-smi keys look like 'card0' → 0. Fall back to the raw key."""
    if key.startswith("card"):
        tail = key[4:]
        try:
            return int(tail)
        except ValueError:
            return key
    return key


def _scalar_stats(values: List[float], with_p95: bool) -> Dict[str, Optional[float]]:
    """Mean / max (/ p95) of a list of floats. Returns Nones for empty input."""
    if not values:
        return {"mean": None, "max": None, **({"p95": None} if with_p95 else {})}
    n = len(values)
    mean = sum(values) / n
    mx = max(values)
    out: Dict[str, Optional[float]] = {
        "mean": round(mean, 3),
        "max": round(mx, 3),
    }
    if with_p95:
        # Nearest-rank percentile — avoids a numpy dep in the probe thread.
        ordered = sorted(values)
        # rank index is ceil(0.95 * n) - 1, clamped to [0, n-1].
        k = int(0.95 * n + 0.9999999)  # ceil without importing math
        k = max(0, min(n - 1, k - 1))
        out["p95"] = round(ordered[k], 3)
    return out
