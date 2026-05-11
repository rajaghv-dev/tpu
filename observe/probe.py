"""
observe/probe.py — extension hooks for the benchmark runner.

A Probe is an object that gets called at well-defined points in
`benchmarks.runner.run_experiment` and writes artefacts to
`results/run_logs/<run_id>/<probe_name>.json`. Probes are how Stages 2-6 of the
project plan add observability without modifying the runner each time.

## Lifecycle

For each `run_experiment(cfg)` call, every active probe receives:

    1.  before_run(run_id, config, log_dir)        once, before phase 1
    2.  before_phase("preflight")                  once per phase
    2a. after_phase("preflight", duration_s)       on success of that phase
    2b. on_error("preflight", exception)           on failure (after_phase NOT called)
    3.  ... repeated for: model_load, compile, warmup, latency, throughput, postflight
    4.  after_run(run_id, result_or_None)          once, even on failure (result=None then)

After `after_run`, each probe's `write_log()` is called. If it returns a dict,
that dict is written to `<log_dir>/<probe_name>.json`. Returning `None` skips
the file (useful for probes that write their own files at custom paths during
the run, e.g. raw_timings.jsonl streamed line-by-line).

## Implementing a probe

Subclass `Probe`, set `name` (used as the JSON filename), override the hooks
you need. Every hook has a no-op default — only override what you care about.
Probes MUST tolerate exceptions inside their own hooks; the runner catches and
logs but does not propagate, so a buggy probe never fails a benchmark.

## Registering a probe

Two ways:

    from observe.probe import register_probe
    register_probe(MyProbe())

    # or, more useful for tests:
    from observe.probe import set_active_probes
    set_active_probes([MyProbe()])

The runner reads from `get_active_probes()` at run start. The default set is
empty — Stage 1 ships zero default probes. Stage 2+ scripts (or the harness
CLI) can register the ones they need.

## Why a class hierarchy and not just callbacks

Probes typically carry state across hooks (start times, accumulators,
per-phase memory snapshots). A class is the natural fit. The base class also
gives us a single place to add cross-cutting concerns later (timeouts,
tracing of probes themselves, ordering semantics).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # avoid runtime import cycle with runner.py
    from benchmarks.runner import ExperimentConfig

_log = logging.getLogger(__name__)


class Probe:
    """
    Base class. All hooks have no-op defaults so subclasses override only what
    they need. `name` MUST be a filesystem-safe identifier — it becomes the
    output filename `<name>.json`.
    """

    #: Used as `<name>.json` when the runner writes the probe's log. Subclass
    #: MUST override; collisions raise at register time.
    name: str = ""

    # ── Lifecycle hooks ──────────────────────────────────────────────────
    # Every method below has a no-op default so subclasses can implement only
    # the events they care about. Method signatures are stable contract — do
    # not change parameter names (probes may call them by keyword).

    def before_run(
        self,
        run_id: str,
        config: "ExperimentConfig",
        log_dir: Path,
    ) -> None:
        """Called once at run start. log_dir = results/run_logs/<run_id>/."""

    def after_run(
        self,
        run_id: str,
        result: Optional[Dict[str, Any]],
    ) -> None:
        """
        Called once at run end. result is the harness result dict on success
        or None on failure. Probes that need to flush state should do so here.
        """

    def before_phase(self, phase_name: str) -> None:
        """Called immediately before each phase body executes."""

    def after_phase(self, phase_name: str, duration_s: float) -> None:
        """Called after the phase body completes successfully."""

    def on_error(self, phase_name: str, exc: BaseException) -> None:
        """
        Called when the phase raised. `after_phase` is NOT called for that
        phase. Probes typically want to record the partial state here.
        """

    def write_log(self) -> Optional[Dict[str, Any]]:
        """
        Optional. Return a JSON-serialisable dict to be written to
        `<log_dir>/<name>.json` by the runner after `after_run`. Return None
        to skip the write (use this when the probe writes its own files).
        """
        return None

    # ── Step-level hooks (training only) ─────────────────────────────────
    # The benchmark/inference runner does NOT call these — they exist for
    # the training runner (train/runner.py) which has a per-step inner loop.
    # Inference probes that don't override these stay no-ops; training
    # probes that need per-step data override only what they need.
    #
    # Contract:
    #   before_step(step)        — called before each training step body.
    #   after_step(step, metrics)— called after each step. metrics is a
    #                              dict like {"loss": 1.23, "lr": 2e-5,
    #                              "grad_norm": 0.41}; the runner is free
    #                              to add or omit keys.
    #   record_metric(...)       — ad-hoc scalar record outside steps
    #                              (e.g. eval loss at epoch end).

    def before_step(self, step: int) -> None:
        """Called immediately before each training step body."""

    def after_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Called after each successful training step body."""

    def record_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
    ) -> None:
        """Ad-hoc metric — eval loss at epoch end, etc. Not tied to a step."""


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: List[Probe] = []


def register_probe(probe: Probe) -> None:
    """Append a probe to the active set. Raises if name collides."""
    if not probe.name:
        raise ValueError(f"Probe {type(probe).__name__} has empty name")
    for existing in _REGISTRY:
        if existing.name == probe.name:
            raise ValueError(f"Probe name '{probe.name}' already registered")
    _REGISTRY.append(probe)


def set_active_probes(probes: List[Probe]) -> None:
    """Replace the active set wholesale. Mostly used by tests."""
    global _REGISTRY
    _REGISTRY = list(probes)


def get_active_probes() -> List[Probe]:
    """Return a snapshot of the currently active probe list."""
    return list(_REGISTRY)


def clear_probes() -> None:
    """Remove all registered probes. Mostly used by tests."""
    global _REGISTRY
    _REGISTRY = []


# ── Hook fan-out helpers (called by runner.py) ────────────────────────────────

def _safe_call(probe: Probe, method_name: str, *args, **kwargs) -> None:
    """
    Invoke probe.<method_name>(*args, **kwargs), catching any exception.

    Probes are observability code; they must not fail the benchmark. We log
    and swallow. If a probe is genuinely broken, it'll show up as a missing
    output file in run_logs, which is recoverable.
    """
    try:
        getattr(probe, method_name)(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        _log.warning("probe %r %s raised %s: %s",
                     probe.name, method_name, type(exc).__name__, exc)


def fanout_before_run(run_id: str, config: "ExperimentConfig", log_dir: Path) -> None:
    """Called by runner.run_experiment exactly once at run start."""
    log_dir.mkdir(parents=True, exist_ok=True)
    for p in _REGISTRY:
        _safe_call(p, "before_run", run_id, config, log_dir)


def fanout_after_run(
    run_id: str,
    result: Optional[Dict[str, Any]],
    log_dir: Path,
) -> None:
    """Called by runner.run_experiment exactly once at run end. Writes per-probe logs."""
    import json
    for p in _REGISTRY:
        _safe_call(p, "after_run", run_id, result)
        try:
            payload = p.write_log()
        except Exception as exc:  # noqa: BLE001
            _log.warning("probe %r write_log raised %s: %s",
                         p.name, type(exc).__name__, exc)
            continue
        if payload is None:
            continue
        target = log_dir / f"{p.name}.json"
        try:
            target.write_text(json.dumps(payload, indent=2, default=str))
        except OSError as exc:
            _log.warning("could not write %s: %s", target, exc)


def fanout_before_phase(phase_name: str) -> None:
    for p in _REGISTRY:
        _safe_call(p, "before_phase", phase_name)


def fanout_after_phase(phase_name: str, duration_s: float) -> None:
    for p in _REGISTRY:
        _safe_call(p, "after_phase", phase_name, duration_s)


def fanout_on_error(phase_name: str, exc: BaseException) -> None:
    for p in _REGISTRY:
        _safe_call(p, "on_error", phase_name, exc)


# ── Step-level fan-outs (training only) ───────────────────────────────────────

def fanout_before_step(step: int) -> None:
    for p in _REGISTRY:
        _safe_call(p, "before_step", step)


def fanout_after_step(step: int, metrics: Dict[str, Any]) -> None:
    for p in _REGISTRY:
        _safe_call(p, "after_step", step, metrics)


def fanout_record_metric(name: str, value: Any, step: Optional[int] = None) -> None:
    for p in _REGISTRY:
        _safe_call(p, "record_metric", name, value, step)
