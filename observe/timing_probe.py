"""
observe/timing_probe.py — wall-clock duration probe.

Records the duration of every benchmark phase plus the overall run, producing a
per-run JSON artefact at `results/run_logs/<run_id>/timing.json`.

The runner already times its own phases internally (see `benchmarks/runner.py`
phase context-manager), but those numbers are folded into the final result
dict; they are NOT broken out per-phase. This probe bridges that gap so the
dashboard can show "where did the 90 seconds go?" without re-running.

A few design notes:

  * `time.perf_counter()` is used everywhere — it is the highest-resolution
    monotonic clock available on every platform and is the same source the
    runner uses. The per-phase deltas this probe emits will agree to within
    a few microseconds of the runner's internal numbers.
  * `phase_summary` is keyed by phase name; since each phase appears at most
    once per run, this is mostly a re-keying of the timeline — the dashboard
    finds it convenient.
  * On error, we still record the partial duration and a `error: True` flag
    so the timeline is faithful even for failed runs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.probe import Probe


class TimingProbe(Probe):
    """
    Record wall-clock timings for every phase boundary.

    Hook contract (matches `Probe`):
        before_run         → stamp run start.
        before_phase(p)    → stamp the start of phase `p`.
        after_phase(p, d)  → record `{phase, duration_s, ts}`.
        on_error(p, exc)   → record `{phase, duration_s, error: True, ts}`.
        after_run          → stamp run end.
        write_log()        → return total + timeline + summary.
    """

    name = "timing"

    def __init__(self) -> None:
        self._run_start: Optional[float] = None
        self._run_end: Optional[float] = None
        self._phase_starts: Dict[str, float] = {}
        self._timeline: List[Dict[str, Any]] = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    def before_run(self, run_id: str, config: Any, log_dir: Path) -> None:
        self._run_start = time.perf_counter()

    def after_run(self, run_id: str, result: Optional[Dict[str, Any]]) -> None:
        self._run_end = time.perf_counter()

    def before_phase(self, phase_name: str) -> None:
        self._phase_starts[phase_name] = time.perf_counter()

    def after_phase(self, phase_name: str, duration_s: float) -> None:
        # Prefer the runner's own duration_s — it's measured immediately after
        # the phase body, before any probe fan-out, so it's slightly more
        # accurate than `now - _phase_starts[p]` (which includes other probes'
        # before_phase work). Fall back to our own clock if needed.
        if duration_s is None and phase_name in self._phase_starts:
            duration_s = time.perf_counter() - self._phase_starts[phase_name]
        self._timeline.append({
            "phase": phase_name,
            "duration_s": duration_s,
            "ts": time.perf_counter(),
        })

    def on_error(self, phase_name: str, exc: BaseException) -> None:
        start = self._phase_starts.get(phase_name)
        duration_s = (time.perf_counter() - start) if start is not None else None
        self._timeline.append({
            "phase": phase_name,
            "duration_s": duration_s,
            "error": True,
            "ts": time.perf_counter(),
        })

    # ── output ───────────────────────────────────────────────────────────────

    def write_log(self) -> Optional[Dict[str, Any]]:
        if self._run_start is None:
            # before_run was never called — nothing useful to write.
            return None
        end = self._run_end if self._run_end is not None else time.perf_counter()
        total_run_s = end - self._run_start

        # phase_summary: re-keyed timeline. Each phase name should appear at
        # most once in a normal run; if a phase somehow repeats we keep the
        # last occurrence (it's the more interesting one for triage).
        phase_summary: Dict[str, Dict[str, Any]] = {}
        for entry in self._timeline:
            phase_summary[entry["phase"]] = entry

        return {
            "total_run_s": total_run_s,
            "timeline": list(self._timeline),
            "phase_summary": phase_summary,
        }
