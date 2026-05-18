#!/usr/bin/env python3
"""
Real-TPU JAX runner — the flagship Cloud TPU entry point.

Runs ON a TPU VM. One workload (matmul) — small, fully instrumented,
incrementable. To add another workload later, add a builder to `_BUILDERS`.

Instrumentation captured every run:
  * `jax.profiler.start_trace(...)` wraps the loop → XProf trace
  * Each step timed with `block_until_ready()` for honest device time
  * `jax.devices()[0].memory_stats()` sampled at post-init / post-compile /
    post-final → real HBM (not estimated)
  * HLO dumps via `XLA_FLAGS=--xla_dump_to=...` (set by the orchestrator)
  * libtpu firmware logs via `TPU_STDERR_LOG_LEVEL=0` (set by the orchestrator)

Artifacts written to `<output_dir>/`:
        run_<trace_id>.jsonl   — OCT events (promtail picks up)
        run_<trace_id>.csv     — Prometheus-shaped metrics (exporter scrapes)
        run_<trace_id>.json    — Chrome/Perfetto trace
        run_<trace_id>.md      — human-readable run report
        hlo/                   — HLO IR dumps (set by XLA_FLAGS)
        xprof/                 — jax.profiler output

CPU-safe: `--help` works without JAX. `jax.devices()` will fail on a host
with no TPU; that's intended — use this on a TPU VM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Allow running from inside the TPU VM at ~/cloud_tpu_lab/.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from cloud_tpu_lab.src.common.config import WorkloadConfig
from cloud_tpu_lab.src.common.cost import CostInputs, estimate_cost
from cloud_tpu_lab.src.common.trace import (
    TraceContext, new_executable_id, new_step_id, new_trace_id, reset_counters,
)
from cloud_tpu_lab.src.observability.logger import JsonlLogger
from cloud_tpu_lab.src.observability.metrics import MetricStream
from cloud_tpu_lab.src.observability.report import render_run_report, write_report
from cloud_tpu_lab.src.profiling.bottleneck_report import (
    diagnose, empty_hbm_stats, hbm_stats_from_jax,
)
from cloud_tpu_lab.src.profiling.profiler_trace import ProfilerTrace
from cloud_tpu_lab.src.profiling.trace_analyzer import (
    compute_breakdown, step_summary,
)


def _hbm_stats_or_empty() -> Dict[str, Any]:
    """Read `jax.devices()[0].memory_stats()` if available, else empty."""
    try:
        import jax
        dev = jax.devices()[0]
        if hasattr(dev, "memory_stats"):
            return hbm_stats_from_jax(dev.memory_stats())
    except Exception:
        pass
    return empty_hbm_stats()


def _dtype_from_precision(precision: str):
    import jax.numpy as jnp
    return jnp.bfloat16 if precision == "bf16" else jnp.float32


# ── Workloads ────────────────────────────────────────────────────────────────
# A builder returns (step_fn, init_state_fn, samples_per_step).
#   step_fn(state)       -> (new_state, scalar_metric)
#   init_state_fn(key)   -> initial state pytree
# To add a workload: write a builder, register it in `_BUILDERS`. That's it.


def build_matmul_workload(args) -> Tuple[Callable, Callable, int]:
    """Single jit'd N×N @ N×N matmul. Compute-bound; the simplest thing that
    exercises the MXU and shows up cleanly in HLO + XProf."""
    import jax
    import jax.numpy as jnp

    dt = _dtype_from_precision(args.precision)
    N = args.hidden_size

    @jax.jit
    def matmul(a, b):
        return a @ b

    def init_state(key):
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (N, N), dtype=dt)
        b = jax.random.normal(k2, (N, N), dtype=dt)
        return (a, b)

    def step(state):
        a, b = state
        c = matmul(a, b)
        # Feed c into the next step so XLA can't constant-fold the loop away.
        return (a, c), c.sum()

    return step, init_state, args.batch_size


_BUILDERS: Dict[str, Callable] = {
    "matmul": build_matmul_workload,
}


# ── Main run loop ────────────────────────────────────────────────────────────


def run(args) -> Dict[str, Any]:
    """Execute the workload and write all artifacts. Returns summary."""
    import jax  # imported here so `--help` works without JAX installed

    reset_counters()
    trace = TraceContext(trace_id=new_trace_id(),
                         executable_id=new_executable_id())
    log = JsonlLogger()
    metrics = MetricStream()
    pt = ProfilerTrace(trace_id=trace.trace_id)
    pt.start()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    xprof_dir = out_dir / "xprof"
    xprof_dir.mkdir(parents=True, exist_ok=True)

    # Static identity fields written into every event so the Prometheus
    # exporter's safe-label set is populated (framework / tpu_version /
    # workload_name / run_mode). Without these the exporter falls back to
    # "cpu_sim" defaults — wrong for real-TPU runs.
    _identity = {
        "workload_name": f"real_{args.workload}",
        "framework": "jax",
        "tpu_version": args.tpu_version,
        "run_mode": "cloud_tpu_vm",
    }

    def emit(layer: str, event: str, **fields: Any) -> None:
        # Caller may override the trace (e.g. per-step trace via with_step()).
        t = fields.pop("trace", trace)
        for k, v in _identity.items():
            fields.setdefault(k, v)
        log.log(layer=layer, event=event, trace=t, **fields)

    # ── Banner ──────────────────────────────────────────────────────────────
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"[run_jax_real_tpu] jax        : {jax.__version__}")
    print(f"[run_jax_real_tpu] backend    : {backend}")
    print(f"[run_jax_real_tpu] devices    : {devices}")
    print(f"[run_jax_real_tpu] trace_id   : {trace.trace_id}")
    print(f"[run_jax_real_tpu] output_dir : {out_dir}")
    if backend != "tpu":
        print(f"[run_jax_real_tpu] WARNING: backend is '{backend}', not 'tpu'. "
              f"Results will not reflect TPU performance.")

    emit("runtime", "jax.init",
         message=f"jax {jax.__version__} backend={backend}",
         jax_version=jax.__version__, backend=backend,
         devices=[str(d) for d in devices], n_devices=len(devices),
         workload=args.workload)

    def _hbm_event_fields(stats: Dict[str, Any]) -> Dict[str, Any]:
        # Flatten the HBM stats dict into the field names the metrics
        # exporter expects (hbm_used_bytes / hbm_capacity_bytes / etc.).
        return {
            "hbm_used_bytes": int(stats.get("used_bytes", 0)),
            "hbm_capacity_bytes": int(stats.get("capacity_bytes", 0)),
            "hbm_utilization_ratio": float(stats.get("utilization", 0.0)),
            "hbm_peak_bytes": int(stats.get("peak_bytes", 0)),
            "oom_events": int(stats.get("oom_events", 0)),
        }

    # ── HBM snapshot AFTER init ─────────────────────────────────────────────
    hbm_post_init = _hbm_stats_or_empty()
    emit("hbm", "hbm.snapshot", message="post-init", phase="post_init",
         metrics=hbm_post_init, **_hbm_event_fields(hbm_post_init))

    # ── Build workload + state ──────────────────────────────────────────────
    builder = _BUILDERS[args.workload]
    step_fn, init_state_fn, samples_per_step = builder(args)
    key = jax.random.PRNGKey(0)
    state = init_state_fn(key)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        state,
    )
    emit("runtime", "workload.ready",
         workload=args.workload, samples_per_step=samples_per_step,
         batch_size=args.batch_size, hidden_size=args.hidden_size,
         precision=args.precision)

    # ── jax.profiler.trace wraps the whole loop ─────────────────────────────
    profiler_started = False
    try:
        jax.profiler.start_trace(str(xprof_dir))
        profiler_started = True
    except Exception as exc:
        emit("profiler", "profiler.start_failed", level="WARN",
             message=str(exc), error_type=type(exc).__name__)

    compile_time_s = 0.0
    step_durations: List[float] = []
    try:
        # ── Step 0: dominated by JIT compile; time separately ──────────────
        step_trace = trace.with_step(new_step_id())
        t0 = time.perf_counter()
        state, loss = step_fn(state)
        if hasattr(loss, "block_until_ready"):
            loss.block_until_ready()
        compile_step_s = time.perf_counter() - t0
        step_durations.append(compile_step_s)

        hbm_post_compile = _hbm_stats_or_empty()
        emit("hbm", "hbm.snapshot", message="post-compile",
             phase="post_compile", metrics=hbm_post_compile,
             **_hbm_event_fields(hbm_post_compile))

        compile_time_s = compile_step_s
        pt.add_event("xla.compile", "compile", dur_s=compile_time_s, tid=0,
                     args={"trace_id": trace.trace_id,
                           "executable_id": trace.executable_id})
        pt.add_event("step.0", "device", dur_s=compile_step_s, tid=2,
                     args={"step": 0, "trace_id": step_trace.trace_id,
                           "step_id": step_trace.step_id})
        emit("xla", "xla.compile", trace=step_trace,
             compile_time_s=compile_time_s, cache_hit=False,
             executable_id=trace.executable_id)
        emit("runtime", "runtime.step", trace=step_trace,
             step=0, step_id=step_trace.step_id,
             step_time_s=compile_step_s,
             device_execution_time_s=compile_step_s,
             samples_per_step=samples_per_step,
             samples_per_second=samples_per_step / max(compile_step_s, 1e-9),
             compile_step=True,
             loss=float(loss) if hasattr(loss, "__float__") else None)
        metrics.record(
            step=0,
            cloud_tpu_step_time_seconds=compile_step_s,
            cloud_tpu_compile_time_seconds=compile_time_s,
            cloud_tpu_device_execution_time_seconds=compile_step_s,
            cloud_tpu_samples_per_second=samples_per_step / max(compile_step_s, 1e-9),
            cloud_tpu_hbm_used_bytes=int(hbm_post_compile.get("used_bytes", 0)),
            cloud_tpu_hbm_capacity_bytes=int(hbm_post_compile.get("capacity_bytes", 0)),
            cloud_tpu_hbm_utilization_ratio=float(hbm_post_compile.get("utilization", 0.0)),
        )

        # ── Steady-state steps 1..n_steps-1 ────────────────────────────────
        for step in range(1, args.n_steps):
            step_trace = trace.with_step(new_step_id())
            t0 = time.perf_counter()
            state, loss = step_fn(state)
            if hasattr(loss, "block_until_ready"):
                loss.block_until_ready()
            dt = time.perf_counter() - t0
            step_durations.append(dt)
            pt.add_event(f"step.{step}", "device", dur_s=dt, tid=2,
                         args={"step": step, "trace_id": step_trace.trace_id,
                               "step_id": step_trace.step_id})
            emit("runtime", "runtime.step", trace=step_trace,
                 step=step, step_id=step_trace.step_id,
                 step_time_s=dt,
                 device_execution_time_s=dt,
                 samples_per_step=samples_per_step,
                 samples_per_second=samples_per_step / max(dt, 1e-9),
                 loss=float(loss) if hasattr(loss, "__float__") else None)
            metrics.record(
                step=step,
                cloud_tpu_step_time_seconds=dt,
                cloud_tpu_compile_time_seconds=0.0,
                cloud_tpu_device_execution_time_seconds=dt,
                cloud_tpu_samples_per_second=samples_per_step / max(dt, 1e-9),
                cloud_tpu_hbm_used_bytes=int(hbm_post_compile.get("used_bytes", 0)),
                cloud_tpu_hbm_capacity_bytes=int(hbm_post_compile.get("capacity_bytes", 0)),
                cloud_tpu_hbm_utilization_ratio=float(hbm_post_compile.get("utilization", 0.0)),
            )
    finally:
        if profiler_started:
            try:
                jax.profiler.stop_trace()
            except Exception as exc:
                emit("profiler", "profiler.stop_failed", level="WARN",
                     message=str(exc))

    hbm_post_final = _hbm_stats_or_empty()
    emit("hbm", "hbm.snapshot", message="post-final", phase="post_final",
         metrics=hbm_post_final, **_hbm_event_fields(hbm_post_final))

    # ── Cost (median steady-state step) ─────────────────────────────────────
    steady = sorted(step_durations[1:]) if len(step_durations) > 1 else step_durations
    median_step = steady[len(steady) // 2] if steady else 0.0
    cost = estimate_cost(CostInputs(
        chip_count=len(devices),
        n_steps=args.n_steps,
        step_time_s=median_step,
        hourly_usd_per_chip=args.hourly_usd_per_chip,
        samples_per_step=samples_per_step,
        utilization=0.85,
    ))
    emit("cost", "cost.estimated", metrics=cost.to_dict())

    breakdown = compute_breakdown(pt)
    summary = step_summary(pt)
    findings = diagnose(breakdown=breakdown, hbm=hbm_post_final,
                        n_chips=len(devices),
                        total_run_usd=cost.total_run_usd)

    workload = WorkloadConfig(
        name=f"real_{args.workload}",
        framework="jax",
        model_kind=args.workload,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        n_steps=args.n_steps,
        precision=args.precision,
        tpu_version=args.tpu_version,
        chip_count=len(devices),
        hourly_usd_per_chip=args.hourly_usd_per_chip,
    )

    log_path = out_dir / f"run_{trace.trace_id}.jsonl"
    metrics_path = out_dir / f"run_{trace.trace_id}.csv"
    trace_path = out_dir / f"run_{trace.trace_id}.json"
    report_path = out_dir / f"run_{trace.trace_id}.md"

    log.flush(log_path)
    metrics.write_csv(metrics_path)
    pt.write_chrome_json(trace_path)
    report_md = render_run_report(
        trace_id=trace.trace_id, config=workload,
        breakdown=breakdown, summary=summary, hbm=hbm_post_final,
        compile_stats={"compile_time_s": compile_time_s, "recompile_count": 0},
        cost=cost, findings=findings,
    )
    write_report(report_path, report_md)

    summary_dict = {
        "trace_id": trace.trace_id,
        "backend": backend,
        "workload": args.workload,
        "n_steps": args.n_steps,
        "compile_time_s": compile_time_s,
        "median_step_s": median_step,
        "samples_per_step": samples_per_step,
        "total_run_usd": cost.total_run_usd,
        "hbm_used_bytes": int(hbm_post_final.get("used_bytes", 0)),
        "hbm_capacity_bytes": int(hbm_post_final.get("capacity_bytes", 0)),
        "hbm_utilization": float(hbm_post_final.get("utilization", 0.0)),
        "n_findings": len(findings),
        "artifacts": {
            "log":     str(log_path),
            "metrics": str(metrics_path),
            "trace":   str(trace_path),
            "report":  str(report_path),
            "xprof":   str(xprof_dir),
            "hlo":     os.environ.get("XLA_FLAGS", ""),
        },
    }
    print(json.dumps(summary_dict, indent=2))
    print(f"\nRun report: {report_path}")
    return summary_dict


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_jax_real_tpu",
        description="Real-TPU JAX runner. One workload (matmul); small + "
                    "fully instrumented + incrementable.",
    )
    p.add_argument("--workload", default="matmul",
                   choices=sorted(_BUILDERS.keys()),
                   help="The workload to run. Currently only 'matmul'; "
                        "add more by registering builders in _BUILDERS.")
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Used as samples_per_step for cost / throughput math.")
    p.add_argument("--hidden-size", type=int, default=512,
                   help="For matmul, this is the N in the N×N @ N×N product.")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--tpu-version", default="v5e",
                   choices=["v4", "v5e", "v5p", "v6e"],
                   help="Used for cost / roofline references only.")
    p.add_argument("--hourly-usd-per-chip", type=float, default=1.20,
                   help="Placeholder; confirm at cloud.google.com/tpu/pricing")
    p.add_argument("--output-dir", default="./cloud_tpu_lab_artifacts",
                   help="Directory on the TPU VM for run artifacts.")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
