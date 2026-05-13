"""
observe/determinism_probe.py — runtime-determinism settings snapshot probe.

This probe is COMPLEMENTARY to `observe/lineage.py`. Lineage records *what
code and data went into a run* (git SHA, package versions, HF model revision,
input seed). DeterminismProbe records *what runtime settings were active
that influence numerical outcomes* — the knobs that decide whether two runs
of the same code on the same data will produce bit-identical results, drift
within numerical tolerance, or diverge entirely.

## What "determinism" actually means in JAX / XLA

It is tempting to think that setting every seed (NumPy, JAX PRNG, Python
hash seed, framework eval seed) is sufficient for reproducibility. It is
not. Two runs with identical seeds on identical hardware can still produce
different floating-point outputs because of:

  * **Operation ordering.** Floating-point addition is not associative:
    `(a + b) + c != a + (b + c)` in general. A reduction performed across
    N devices may sum partial results in whatever order the collective
    backend (NCCL on GPU, ICI on TPU) happens to deliver them on a given
    run. The numerical answer differs by a few ULPs per reduction.

  * **Reduction tree shapes.** XLA's reduction lowering chooses a tree
    shape based on autotuning and shape heuristics. Different shapes
    (binary tree vs. linear scan vs. block-then-tree) give different
    last-bit results from the same inputs.

  * **Async collectives.** `xla_gpu_enable_async_collectives` and friends
    allow XLA to overlap communication with compute, but the order in
    which partial results arrive at the reducer is no longer guaranteed.

  * **Parallel atomicAdd on GPU.** Many CUDA kernels (e.g. scatter,
    embedding-grad, some softmax variants) use atomicAdd to accumulate
    into a shared output buffer. The order of atomic accumulations is
    determined by hardware scheduling and varies run-to-run. cuDNN
    autotuning picks among such kernels; pinning it (TF_CUDNN_DETERMINISTIC,
    xla_gpu_deterministic_ops) forces the slower deterministic path.

  * **TF32 vs fp32 matmul.** On Ampere+ GPUs, JAX's default matmul precision
    is "default" which on supported devices lowers to TF32 (10-bit mantissa)
    rather than full fp32 (23-bit). TF32 is faster but gives different
    numerical results — and the cut-off where TF32 is used vs fp32 is itself
    shape-dependent, so two runs of the same code at slightly different
    batch sizes may take different precision paths.

  * **cuBLAS workspace allocation.** Without CUBLAS_WORKSPACE_CONFIG set
    to :4096:8 (or :16:8), cuBLAS is free to pick among algorithms
    differently between calls.

  * **Kernel autotuning.** The first call to a kernel benchmarks several
    implementations and caches the winner. The winner depends on the
    state of the GPU at autotune time — clock speeds, thermal throttling,
    co-resident processes. Different winners → different last bits.

## Bit-deterministic vs statistically-reproducible

It is important to distinguish two reproducibility regimes:

  * **Bit-deterministic**: two runs produce byte-identical tensors. Required
    for debugging numerical regressions, for cryptographic-style audit, and
    for proving that a refactor changed nothing. Requires *all* knobs in
    the recommended-settings table to be active, plus identical hardware
    and identical input ordering. Usually 2-10x slower than the fast path.

  * **Statistically reproducible**: two runs produce different bits but
    converge to indistinguishable downstream metrics (loss curve, eval
    accuracy) within run-to-run noise. This is what most ML practitioners
    actually want; it is achieved by setting seeds and using stable input
    ordering, even without the determinism flags.

The `deterministic_score` in this probe's output reports how close the
current process is to the bit-deterministic regime. A score of 0.0 means
the run is firmly in the "fast, non-reproducible" path; 1.0 means every
recommended knob is set and bit-identical outputs are plausible (modulo
hardware-level non-determinism like ECC scrubs or thermal throttling).

## How to read the `missing_recommendations` list

Each entry is `{key, current, recommended, why}`:

  * `key`         — namespaced setting name (e.g. `env.NCCL_DETERMINISTIC`).
  * `current`     — the value observed in this process (None if unset).
  * `recommended` — the value the determinism community recommends.
  * `why`         — a one-line explanation of what this setting does.

The list is informational. Some recommendations have steep performance
costs (TF32 disable can halve training throughput on A100) and you may
deliberately choose not to set them. The probe does not advise; it only
reports the gap.

## Output schema

    {
      "captured_at":      "<ISO-8601 UTC timestamp>",
      "seeds":            {"input_seed": ..., "init_seed": ..., "eval_seed": ...,
                           "pythonhashseed": "..."},
      "numpy":            {"random_state_hash": "<md5 hex>"} | {"available": false},
      "jax_flags":        {"jax_default_matmul_precision": "...", ...} |
                          {"available": false, "reason": "..."},
      "xla_flags_parsed": {"xla_gpu_deterministic_ops": "true", ...},
      "env_flags":        {"PYTHONHASHSEED": "0", ...},
      "analysis":         {
                            "deterministic_score": 0.375,
                            "missing_recommendations": [...],
                            "summary_text": "..."
                          },
      "process":          {"pid": ..., "ppid": ...}
    }

The probe is one-shot: only `before_run` populates the snapshot and only
`write_log` is meaningful. The other hooks are inherited no-ops.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.probe import Probe


# ── Recommended-determinism reference table ──────────────────────────────────
#
# Keys are namespaced:
#   env.<NAME>  → os.environ.get(NAME)
#   jax.<NAME>  → jax.config.read(NAME)
#   xla.<NAME>  → parsed from XLA_FLAGS
#
# Each value is a (recommended_value, why_explanation) tuple. The probe
# walks this table and flags every entry whose observed value differs.
_RECOMMENDED_DETERMINISM: Dict[str, tuple] = {
    "env.PYTHONHASHSEED": (
        "0",
        "stabilises dict iteration order; some workloads still depend on it",
    ),
    "env.CUBLAS_WORKSPACE_CONFIG": (
        ":4096:8",
        "required for deterministic cuBLAS GEMMs on CUDA >= 10.2",
    ),
    "env.NCCL_DETERMINISTIC": (
        "1",
        "forces NCCL collectives into deterministic algorithm choice",
    ),
    "env.TF_DETERMINISTIC_OPS": (
        "1",
        "legacy — affects any TF ops still in graph",
    ),
    "env.TF_CUDNN_DETERMINISTIC": (
        "1",
        "forces cuDNN into deterministic algorithm path",
    ),
    "env.NVIDIA_TF32_OVERRIDE": (
        "0",
        "disables TF32 — needed for bitwise-equal fp32 across runs",
    ),
    "jax.jax_default_matmul_precision": (
        "highest",
        "matmuls use full fp32 (not TF32 fast path)",
    ),
    "xla.xla_gpu_deterministic_ops": (
        "true",
        "forces XLA-GPU into deterministic algorithm path",
    ),
}


# ── Env-var snapshot list ────────────────────────────────────────────────────
#
# Environment variables we record verbatim into env_flags. These are the
# inputs to the determinism analysis; we record them whether or not they
# appear in the recommended table.
_ENV_FLAGS_TO_CAPTURE: List[str] = [
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_DETERMINISTIC",
    "NVIDIA_TF32_OVERRIDE",
    "TF_DETERMINISTIC_OPS",
    "TF_CUDNN_DETERMINISTIC",
    "JAX_DEBUG_NANS",
    "JAX_DISABLE_JIT",
    "JAX_ENABLE_X64",
    "JAX_THREEFRY_PARTITIONABLE",
]


# ── jax.config keys we snapshot ──────────────────────────────────────────────
_JAX_FLAGS_TO_CAPTURE: List[str] = [
    "jax_default_matmul_precision",
    "jax_enable_x64",
    "jax_default_dtype_bits",
    "jax_disable_jit",
    "jax_threefry_partitionable",
    "jax_xla_backend",
    "jax_log_compiles",
    "jax_persistent_cache_min_compile_time_secs",
]


def _safe_env(name: str) -> Optional[str]:
    """Return os.environ[name] or None if unset. Never raises."""
    try:
        val = os.environ.get(name)
        return val if val is not None else None
    except Exception:  # noqa: BLE001 — paranoid; os.environ.get shouldn't raise
        return None


def _parse_xla_flags(raw: str) -> Dict[str, Any]:
    """
    Split XLA_FLAGS into a dict.

    `--xla_gpu_deterministic_ops=true` → {"xla_gpu_deterministic_ops": "true"}
    `--xla_gpu_enable_async_collectives` → {"xla_gpu_enable_async_collectives": True}

    Tokens that do not begin with `--` are silently ignored. Never raises.
    """
    out: Dict[str, Any] = {}
    if not raw:
        return out
    try:
        for tok in raw.split():
            if not tok.startswith("--"):
                continue
            body = tok[2:]
            if "=" in body:
                key, _, value = body.partition("=")
                if key:
                    out[key] = value
            else:
                if body:
                    out[body] = True
    except Exception:  # noqa: BLE001 — defensive; split shouldn't raise
        return out
    return out


def _snapshot_numpy() -> Dict[str, Any]:
    """
    Hash NumPy's global random state into an md5 hex digest.

    A run with identical seeds will produce identical hashes here even
    though the raw state tuple is large and unstable to dump verbatim.
    Returns {"available": false} if numpy cannot be imported.
    """
    try:
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001 — numpy missing or broken install
        return {"available": False}
    try:
        state = np.random.get_state()
        h = hashlib.md5(repr(state).encode("utf-8", errors="replace")).hexdigest()
        return {"random_state_hash": h}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _snapshot_jax_flags() -> Dict[str, Any]:
    """
    Read each key in _JAX_FLAGS_TO_CAPTURE via jax.config.read.

    Per-key failures (unknown flag in this jax version) yield a `null`
    entry rather than killing the whole snapshot. A missing jax install
    yields `{"available": false, "reason": "..."}`.
    """
    try:
        import jax  # type: ignore
    except Exception as exc:  # noqa: BLE001 — jax not installed
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    flags: Dict[str, Any] = {}
    for key in _JAX_FLAGS_TO_CAPTURE:
        try:
            flags[key] = jax.config.read(key)
        except Exception:  # noqa: BLE001 — flag missing on this jax version
            flags[key] = None
    return flags


def _lookup_current(
    key: str,
    env_flags: Dict[str, Any],
    jax_flags: Dict[str, Any],
    xla_flags_parsed: Dict[str, Any],
) -> Any:
    """
    Resolve a namespaced recommendation key to its observed value.

    env.<NAME> → env_flags[NAME]
    jax.<NAME> → jax_flags[NAME]
    xla.<NAME> → xla_flags_parsed[NAME]

    Returns None if the key namespace is unknown or the value is absent.
    """
    if "." not in key:
        return None
    ns, _, name = key.partition(".")
    if ns == "env":
        return env_flags.get(name)
    if ns == "jax":
        # jax_flags may be {"available": false, ...} when import failed
        if jax_flags.get("available") is False:
            return None
        return jax_flags.get(name)
    if ns == "xla":
        return xla_flags_parsed.get(name)
    return None


def _values_match(current: Any, recommended: str) -> bool:
    """
    Recommended values are always strings. The observed value may be a
    bool/int/None/str depending on source. Normalise to string before
    comparing; None never matches.
    """
    if current is None:
        return False
    return str(current).lower() == str(recommended).lower()


def _build_analysis(
    env_flags: Dict[str, Any],
    jax_flags: Dict[str, Any],
    xla_flags_parsed: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Walk _RECOMMENDED_DETERMINISM, compare current vs recommended values,
    build a structured assessment.
    """
    missing: List[Dict[str, Any]] = []
    for key, (recommended, why) in _RECOMMENDED_DETERMINISM.items():
        current = _lookup_current(key, env_flags, jax_flags, xla_flags_parsed)
        if not _values_match(current, recommended):
            missing.append(
                {
                    "key": key,
                    "current": current,
                    "recommended": recommended,
                    "why": why,
                }
            )

    total = len(_RECOMMENDED_DETERMINISM)
    score = 1.0 - (len(missing) / total) if total else 1.0

    if not missing:
        summary = (
            f"all {total} recommended determinism settings active — "
            "runs likely bit-reproducible on identical hardware"
        )
    else:
        summary = (
            f"{len(missing)}/{total} recommended determinism settings inactive — "
            "runs likely non-bit-reproducible across hardware "
            "(numerical drift expected)"
        )

    return {
        "deterministic_score": score,
        "missing_recommendations": missing,
        "summary_text": summary,
    }


class DeterminismProbe(Probe):
    """
    One-shot probe that captures every runtime input affecting reproducibility.

    Populated entirely in `before_run`; emitted as JSON in `write_log`.
    Complementary to `observe.lineage`: lineage covers *what code and data
    was used*, this probe covers *what runtime knobs decided the numerics*.

    Never raises out of any hook — all external lookups are wrapped.
    """

    name = "determinism"

    def __init__(self) -> None:
        self._payload: Optional[Dict[str, Any]] = None

    # ── hooks ───────────────────────────────────────────────────────────────

    def before_run(
        self,
        run_id: str,
        config: Any,
        log_dir: Path,
    ) -> None:
        """Capture the determinism snapshot. Never raises."""
        try:
            self._payload = self._capture(config)
        except Exception as exc:  # noqa: BLE001 — probe must not fail the run
            # Best-effort: record the failure rather than dropping the file.
            self._payload = {
                "captured_at": _now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def write_log(self) -> Optional[Dict[str, Any]]:
        """Return the captured payload, or None if before_run never ran."""
        return self._payload

    # ── capture implementation ──────────────────────────────────────────────

    def _capture(self, config: Any) -> Dict[str, Any]:
        seeds: Dict[str, Any] = {
            "input_seed": _safe_getattr(config, "input_seed"),
            "init_seed": _safe_getattr(config, "init_seed"),
            "eval_seed": _safe_getattr(config, "eval_seed"),
            "pythonhashseed": _safe_env("PYTHONHASHSEED"),
        }

        numpy_snapshot = _snapshot_numpy()
        jax_flags = _snapshot_jax_flags()
        xla_flags_parsed = _parse_xla_flags(os.environ.get("XLA_FLAGS", "") or "")

        env_flags: Dict[str, Any] = {
            name: _safe_env(name) for name in _ENV_FLAGS_TO_CAPTURE
        }

        analysis = _build_analysis(env_flags, jax_flags, xla_flags_parsed)

        try:
            pid = os.getpid()
        except Exception:  # noqa: BLE001
            pid = None
        try:
            ppid = os.getppid()
        except Exception:  # noqa: BLE001
            ppid = None

        return {
            "captured_at": _now_iso(),
            "seeds": seeds,
            "numpy": numpy_snapshot,
            "jax_flags": jax_flags,
            "xla_flags_parsed": xla_flags_parsed,
            "env_flags": env_flags,
            "analysis": analysis,
            "process": {"pid": pid, "ppid": ppid},
        }


# ── tiny helpers ─────────────────────────────────────────────────────────────


def _safe_getattr(obj: Any, name: str) -> Any:
    """getattr that never raises and returns None on miss."""
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 — pathological __getattr__
        return None


def _now_iso() -> str:
    """ISO-8601 UTC timestamp, second precision, with 'Z' suffix."""
    try:
        return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:  # noqa: BLE001
        return ""
