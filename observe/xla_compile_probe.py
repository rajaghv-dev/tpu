"""
observe/xla_compile_probe.py — JAX/XLA compilation observability probe.

## What this probe captures

This probe is a focused window onto the *compile-time* behaviour of a JAX/XLA
training or inference run. It records:

  * The JAX configuration values active at run start (`jax_config_snapshot`),
    so you can later correlate "why was this run slow?" with x64 mode,
    matmul precision, persistent-cache dir, etc.
  * The raw `XLA_FLAGS` and `LIBTPU_INIT_ARGS` env vars, parsed into lists.
    These are the single biggest hidden lever on TPU performance — a flag
    like `--xla_tpu_megacore_fusion=true` can swing throughput by tens of
    percent, and we want them faithfully recorded for every run.
  * Per-phase wall-clock compile timing (`compile_timing`) — separate from
    the global `timing_probe`, this view zooms in on phases where the cost
    is dominated by `jax.jit` tracing + lowering + XLA codegen.
  * Persistent compilation cache state (`compile_cache`) — was the cache
    enabled, and how many entries / bytes did this run *add* to it. A cache
    that never grows is a cache that never helps; a cache that grows every
    run probably has a non-deterministic input shape and is invalidating
    itself, which we want to flag.
  * A count of compile events scraped from JAX's INFO-level compile log
    (`compile_counter`) — how many distinct XLA compilations actually fired,
    tagged by the phase they happened in.
  * A boolean `recompile_warning` that fires if more than two compile events
    occurred in any "should-be-stable" phase (warmup, train_loop, eval).

## Why XLA compilation is the largest single cost in many runs

The first time a `jax.jit`-decorated function is called with new input
shapes / dtypes / sharding, JAX traces the Python function with abstract
values, lowers the resulting IR to HLO, hands HLO to XLA, and XLA performs
fusion, layout assignment, autotuning, register allocation, and code
generation for the target hardware. On TPU, this can take 10–120 seconds
even for small functions because XLA's autotuner explores many fusion and
tiling options. Subsequent calls with the *same* shape/dtype signature hit
an in-process cache and skip all of this — runtime drops from "seconds" to
"microseconds". This means the *first step* of a run is usually 10–1000×
slower than every other step. If you forget to warm the cache before
timing, every benchmark you collect is dominated by compile, not compute.

## What "silent recompile" means and why it is a bug

The JIT cache is keyed by the abstract signature of the inputs:
`(shape, dtype, sharding, weak_type, ...)` for every positional and keyword
argument, plus the static_argnums values. If *any* of those change between
calls — for instance, your input pipeline produces a tail batch of size 7
when every prior batch was size 8, or you pass `learning_rate` as a Python
float that rounds slightly differently from step to step — JAX considers
this a brand-new function instantiation and triggers a *fresh* XLA compile.
The user sees a hang of seconds-to-minutes mid-training, with no error.
This is a "silent recompile". The fixes vary (`static_argnums`,
`jax.tree_util.Partial`, padding to a fixed shape, pulling Python scalars
into device arrays) but you cannot fix what you cannot see. This probe
makes silent recompiles visible by setting `jax_log_compiles=True` and
counting compile events per phase. Any phase that should be steady-state
(warmup, train_loop, eval) but logged >2 compiles produces a
`recompile_warning`.

## The persistent compilation cache (JAX_COMPILATION_CACHE_DIR)

JAX supports a persistent on-disk cache for XLA executables, controlled by
the env var / config key `JAX_COMPILATION_CACHE_DIR`. When set, the FIRST
run pays the full compile cost and writes the compiled executable to that
directory; SUBSEQUENT runs (in any process) hit the disk cache and skip
XLA codegen entirely — first-step time drops by 1-2 orders of magnitude.
This makes the persistent cache the single highest-ROI knob for iterative
development. The catch: if your shapes / dtypes / JAX version / TPU
software stack change, cache keys miss, and you silently pay the full
compile cost again. This probe records the cache directory and counts how
many entries / bytes were added during the run, so you can tell at a
glance whether you got cache hits or cache misses.

## Why this probe is complementary to `hlo_dump_probe`

`hlo_dump_probe` captures *what XLA produced* (the HLO IR text, op counts,
fusion counts) — it answers "what does the compiled graph look like".
This probe captures *what XLA did* (timing, count, cache state) — it
answers "how much did compilation cost me, and is the cache helping".
They share no output keys and can be enabled independently.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from observe.probe import Probe

if TYPE_CHECKING:  # pragma: no cover — avoid import cycle with runner
    from benchmarks.runner import ExperimentConfig

_log = logging.getLogger(__name__)


# JAX config keys we want to snapshot. Recorded in this exact order so the
# output dict is reproducible across runs.
_JAX_CONFIG_KEYS: List[str] = [
    "jax_enable_x64",
    "jax_default_matmul_precision",
    "jax_default_dtype_bits",
    "jax_disable_jit",
    "jax_log_compiles",
    "jax_traceback_filtering",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_compilation_cache_dir",
]

# Phases for which we record compile_timing entries. Anything outside this
# set is ignored — we don't want a noisy timing dict cluttered with
# "preflight" / "postflight" / etc.
_DEFAULT_TIMED_PHASES: Set[str] = {
    "compile",
    "model_load",
    "warmup",
    "train_loop",
    "eval",
}

# Phases where a recompile is almost certainly a bug. The compile phase is
# *expected* to compile (that is literally its job); model_load may compile
# during parameter init on some backends. Everything else should be cache
# hits.
_NO_RECOMPILE_PHASES: Set[str] = {"warmup", "train_loop", "eval"}

# Cap on filesystem traversal — see _count_cache_entries.
_FS_WALK_TIMEOUT_S: float = 1.0


class _CompileLogHandler(logging.Handler):
    """
    Logging handler that scrapes JAX's compile log for compile events.

    JAX, when `jax_log_compiles` is True, emits INFO records on loggers
    `jax._src.compiler` / `jax._src.dispatch` whose messages contain
    "Compiling" or "Finished tracing + transforming". We attach this
    handler at INFO level for the duration of the run and count records.

    Each captured event is appended to `events` as a small dict; the owning
    probe maintains the current phase pointer and stamps it in.
    """

    def __init__(self, probe: "XlaCompileProbe") -> None:
        super().__init__(level=logging.INFO)
        self._probe = probe
        self.events: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — malformed log records mustn't kill the probe
            return
        if "Compiling" not in msg and "Finished tracing + transforming" not in msg:
            return
        # We can't reliably parse the module name out of every JAX version
        # — the format has shifted. Best-effort: take the first quoted /
        # bracketed token after "Compiling", fall back to the logger name.
        module_name = record.name
        try:
            if "Compiling" in msg:
                # Typical shape: "Compiling foo for ..." → take the next token.
                after = msg.split("Compiling", 1)[1].strip()
                if after:
                    module_name = after.split()[0].strip(":,.;'\"")
        except Exception:  # noqa: BLE001
            pass
        self.events.append({
            "module_name": module_name,
            "time": time.perf_counter(),
            "phase": self._probe._current_phase,
        })


class XlaCompileProbe(Probe):
    """
    Capture JAX/XLA compile-time observability for a single run.

    See module docstring for the full rationale. Outputs a single JSON
    file `xla_compile.json` per run; degrades gracefully when JAX is
    missing, when the compile log is unavailable, or when the persistent
    cache dir is unset.
    """

    name = "xla_compile"

    def __init__(
        self,
        timed_phases: Optional[Set[str]] = None,
        no_recompile_phases: Optional[Set[str]] = None,
    ) -> None:
        self._timed_phases: Set[str] = (
            set(timed_phases) if timed_phases is not None else set(_DEFAULT_TIMED_PHASES)
        )
        self._no_recompile_phases: Set[str] = (
            set(no_recompile_phases) if no_recompile_phases is not None else set(_NO_RECOMPILE_PHASES)
        )

        # Populated in before_run / hooks.
        self._jax = None  # the jax module, if importable
        self._jax_config_snapshot: Dict[str, Any] = {}
        self._xla_flags: List[str] = []
        self._libtpu_init_args: List[str] = []

        # Per-phase timing — populated in before_phase / after_phase.
        self._phase_starts: Dict[str, float] = {}
        self._compile_timing: Dict[str, Dict[str, float]] = {}

        # Compilation-cache accounting — taken in before_run / after_run.
        self._cache_dir: Optional[str] = None
        self._cache_before: Optional[Dict[str, int]] = None
        self._cache_after: Optional[Dict[str, int]] = None

        # Compile-log scraping.
        self._log_handler: Optional[_CompileLogHandler] = None
        self._log_targets: List[logging.Logger] = []
        self._prev_log_levels: Dict[str, int] = {}
        self._prev_jax_log_compiles: Any = None  # the prior config value, restored in after_run
        self._current_phase: Optional[str] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def before_run(
        self,
        run_id: str,
        config: "ExperimentConfig",
        log_dir: Path,
    ) -> None:
        """Snapshot JAX config + env, install compile-log handler, count cache files."""
        # 1) Try to import jax. A missing jax is fine — every downstream
        # field will be null / "unavailable".
        try:
            import jax  # type: ignore
            self._jax = jax
        except Exception:  # noqa: BLE001 — degrade silently
            self._jax = None

        # 2) Snapshot jax.config values. Newer JAX exposes `jax.config.values`
        # (a dict); older JAX requires `jax.config.read(key)` per key.
        self._jax_config_snapshot = self._snapshot_jax_config()

        # 3) Parse env flags.
        self._xla_flags = self._split_env("XLA_FLAGS")
        self._libtpu_init_args = self._split_env("LIBTPU_INIT_ARGS")

        # 4) Resolve & count cache dir.
        self._cache_dir = self._resolve_cache_dir()
        if self._cache_dir is not None:
            self._cache_before = self._count_cache_entries(self._cache_dir)

        # 5) Turn on compile logging and install our log handler.
        self._install_compile_log_handler()

    def before_phase(self, phase_name: str) -> None:
        self._current_phase = phase_name
        if phase_name in self._timed_phases:
            self._phase_starts[phase_name] = time.perf_counter()

    def after_phase(self, phase_name: str, duration_s: float) -> None:
        if phase_name in self._timed_phases:
            start = self._phase_starts.get(phase_name)
            now = time.perf_counter()
            # Prefer the runner's duration_s when present; fall back to our
            # own clock. Either way, record before_ts / after_ts so the
            # consumer can place the phase on a global timeline.
            if start is None:
                start = now - (duration_s or 0.0)
            self._compile_timing[phase_name] = {
                "duration_s": float(duration_s) if duration_s is not None else (now - start),
                "before_ts": float(start),
                "after_ts": float(now),
            }
        # Note: we deliberately keep `_current_phase` pointing at this phase
        # until the next before_phase fires. Late compile log lines (e.g.
        # async dispatch finishing just after the phase body) are still
        # most-accurately attributed to the phase we just left.

    def on_error(self, phase_name: str, exc: BaseException) -> None:
        # Mirror after_phase's bookkeeping with an `error` tag so the
        # consumer sees partial timing for failed phases.
        if phase_name in self._timed_phases:
            start = self._phase_starts.get(phase_name)
            now = time.perf_counter()
            if start is None:
                start = now
            self._compile_timing[phase_name] = {
                "duration_s": now - start,
                "before_ts": float(start),
                "after_ts": float(now),
                "error": True,
            }

    def after_run(self, run_id: str, result: Optional[Dict[str, Any]]) -> None:
        # 1) Count cache entries again so we can report delta.
        if self._cache_dir is not None:
            self._cache_after = self._count_cache_entries(self._cache_dir)

        # 2) Tear down the compile-log handler. Do this last so any
        # compile-log lines emitted during shutdown still get counted.
        self._uninstall_compile_log_handler()

    # ── output ───────────────────────────────────────────────────────────────

    def write_log(self) -> Optional[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if self._log_handler is not None:
            events = list(self._log_handler.events)

        # Compile-count by phase, plus a recompile warning if any
        # supposed-to-be-stable phase saw >2 compiles.
        by_phase: Dict[str, int] = {}
        for ev in events:
            phase = ev.get("phase") or "unknown"
            by_phase[phase] = by_phase.get(phase, 0) + 1

        recompile_warning = any(
            by_phase.get(p, 0) > 2 for p in self._no_recompile_phases
        )

        # compile_cache section.
        if self._cache_dir is None:
            compile_cache: Dict[str, Any] = {
                "enabled": False,
                "reason": "JAX_COMPILATION_CACHE_DIR unset",
            }
        else:
            before = self._cache_before or {"n_files": 0, "total_bytes": 0}
            after = self._cache_after or before
            compile_cache = {
                "enabled": True,
                "cache_dir": self._cache_dir,
                "entries_before": before.get("n_files", 0),
                "entries_after": after.get("n_files", 0),
                "entries_added": after.get("n_files", 0) - before.get("n_files", 0),
                "bytes_added": after.get("total_bytes", 0) - before.get("total_bytes", 0),
            }

        return {
            "captured_at": _dt.datetime.utcnow().isoformat() + "Z",
            "jax_config_snapshot": self._jax_config_snapshot,
            "xla_flags": list(self._xla_flags),
            "libtpu_init_args": list(self._libtpu_init_args),
            "compile_timing": dict(self._compile_timing),
            "compile_cache": compile_cache,
            "compile_counter": {
                "total_events": len(events),
                "by_phase": by_phase,
                "events": events,
            },
            "recompile_warning": bool(recompile_warning),
            # Reserved for forward-compat — see module docstring. The
            # runner currently does not expose params to the probe layer.
            "param_footprint": {
                "available": False,
                "reason": "param tree not exposed to probe",
            },
        }

    # ── internals: JAX config snapshot ───────────────────────────────────────

    def _snapshot_jax_config(self) -> Dict[str, Any]:
        """
        Read every key in _JAX_CONFIG_KEYS, preferring `jax.config.values`
        (a dict on newer JAX) but falling back to `jax.config.read(key)`
        per-key for older versions. Missing keys → None.
        """
        out: Dict[str, Any] = {k: None for k in _JAX_CONFIG_KEYS}
        if self._jax is None:
            return out

        # Path A: dict-style. `jax.config.values` is a dict on modern JAX.
        values: Dict[str, Any] = {}
        try:
            cfg = self._jax.config  # type: ignore[attr-defined]
            cand = getattr(cfg, "values", None)
            if isinstance(cand, dict):
                values = cand
        except Exception:  # noqa: BLE001
            values = {}

        for key in _JAX_CONFIG_KEYS:
            if key in values:
                out[key] = self._coerce_jsonable(values[key])
                continue
            # Path B: read(key) per-key. Tolerate AttributeError, KeyError,
            # and a raised "unknown config option" from older JAX.
            try:
                cfg = self._jax.config  # type: ignore[attr-defined]
                reader = getattr(cfg, "read", None)
                if reader is None:
                    continue
                out[key] = self._coerce_jsonable(reader(key))
            except Exception:  # noqa: BLE001 — missing key → leave None
                out[key] = None
        return out

    @staticmethod
    def _coerce_jsonable(value: Any) -> Any:
        """
        Normalise a config value to something json.dumps will accept.
        Most jax config values are already primitive; this guards against
        the occasional enum / object.
        """
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        try:
            return str(value)
        except Exception:  # noqa: BLE001
            return None

    # ── internals: env parsing ───────────────────────────────────────────────

    @staticmethod
    def _split_env(var_name: str) -> List[str]:
        raw = os.environ.get(var_name, "") or ""
        return [tok for tok in raw.split() if tok]

    # ── internals: persistent cache accounting ───────────────────────────────

    def _resolve_cache_dir(self) -> Optional[str]:
        """
        Return the persistent-cache directory as set via jax.config or the
        env var, or None if neither is set. Prefer the config (it's
        authoritative if the user called `jax.config.update(...)` at
        import time).
        """
        cfg_val = self._jax_config_snapshot.get("jax_compilation_cache_dir")
        if cfg_val:
            return str(cfg_val)
        env_val = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        if env_val:
            return env_val
        return None

    @staticmethod
    def _count_cache_entries(cache_dir: str) -> Dict[str, int]:
        """
        Walk `cache_dir` counting files and total bytes, bounded by
        _FS_WALK_TIMEOUT_S so a giant cache or a hung network mount cannot
        stall the run. On timeout, return whatever we counted so far with
        a `partial: True` marker.
        """
        out: Dict[str, int] = {"n_files": 0, "total_bytes": 0}
        path = Path(cache_dir)
        if not path.exists():
            return out
        start = time.perf_counter()
        try:
            for entry in path.rglob("*"):
                if time.perf_counter() - start > _FS_WALK_TIMEOUT_S:
                    out["partial"] = 1
                    break
                try:
                    if entry.is_file():
                        out["n_files"] += 1
                        out["total_bytes"] += entry.stat().st_size
                except OSError:
                    # Permission denied / vanished mid-walk — skip.
                    continue
        except OSError as exc:
            _log.warning("xla_compile cache walk of %s failed: %s", cache_dir, exc)
        return out

    # ── internals: compile-log handler ───────────────────────────────────────

    def _install_compile_log_handler(self) -> None:
        """
        Flip `jax_log_compiles` on (preserving prior value) and attach a
        logging.Handler to the JAX compile-related loggers at INFO level.
        Tolerate missing jax, missing loggers, and config update failures
        — every step is wrapped so the probe never raises.
        """
        # Flip jax_log_compiles via jax.config.update so JAX actually emits
        # the records we want to scrape.
        if self._jax is not None:
            try:
                cfg = self._jax.config  # type: ignore[attr-defined]
                reader = getattr(cfg, "read", None)
                if reader is not None:
                    try:
                        self._prev_jax_log_compiles = reader("jax_log_compiles")
                    except Exception:  # noqa: BLE001
                        self._prev_jax_log_compiles = None
                updater = getattr(cfg, "update", None)
                if updater is not None:
                    try:
                        updater("jax_log_compiles", True)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass

        # Attach the handler to both candidate loggers — different JAX
        # versions emit on different names, and attaching to both is cheap.
        handler = _CompileLogHandler(self)
        self._log_handler = handler
        for logger_name in ("jax._src.compiler", "jax._src.dispatch"):
            try:
                lg = logging.getLogger(logger_name)
                self._prev_log_levels[logger_name] = lg.level
                if lg.level == logging.NOTSET or lg.level > logging.INFO:
                    lg.setLevel(logging.INFO)
                lg.addHandler(handler)
                self._log_targets.append(lg)
            except Exception:  # noqa: BLE001
                continue

    def _uninstall_compile_log_handler(self) -> None:
        """Reverse of _install_compile_log_handler. Tolerant of partial state."""
        if self._log_handler is not None:
            for lg in self._log_targets:
                try:
                    lg.removeHandler(self._log_handler)
                except Exception:  # noqa: BLE001
                    pass
                # Restore prior level if we changed it.
                prev = self._prev_log_levels.get(lg.name)
                if prev is not None:
                    try:
                        lg.setLevel(prev)
                    except Exception:  # noqa: BLE001
                        pass
        self._log_targets = []

        # Restore jax_log_compiles to whatever it was before before_run.
        if self._jax is not None:
            try:
                cfg = self._jax.config  # type: ignore[attr-defined]
                updater = getattr(cfg, "update", None)
                if updater is not None and self._prev_jax_log_compiles is not None:
                    try:
                        updater("jax_log_compiles", self._prev_jax_log_compiles)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
