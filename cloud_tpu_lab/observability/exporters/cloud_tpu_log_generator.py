"""
cloud_tpu_log_generator — write synthetic JSONL events for testing.

Produces lines that match the schema written by
`src/observability/logger.py`:

    {
      "timestamp": "...Z",
      "app": "cloud_tpu_lab",
      "level": "INFO",
      "layer": "xla",
      "event": "xla.compile",
      "message": "...",
      "metrics": {...},
      "trace_id": "TRACE-XXXX",
      ...extra fields
    }

The point of this helper is to exercise the local stack
(`docker compose up -d`) without needing to run the actual simulation
demo. It produces a believable mix of compile / step / HBM / collective
/ checkpoint events.

Usage
-----
    python cloud_tpu_log_generator.py \\
        --out cloud_tpu_lab/artifacts/logs/run_TEST-0001.jsonl \\
        --steps 200 \\
        --tpu-version v5p \\
        --workload-name synthetic_mlp \\
        --framework jax
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _line(
    *,
    layer: str,
    event: str,
    level: str = "INFO",
    message: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    trace_id: str,
    **extra: Any,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "app": "cloud_tpu_lab",
        "level": level,
        "layer": layer,
        "event": event,
        "message": message,
        "metrics": metrics or {},
        "trace_id": trace_id,
    }
    for k, v in extra.items():
        out[k] = v
    return out


def generate(
    *,
    steps: int,
    workload_name: str,
    framework: str,
    tpu_version: str,
    run_mode: str,
    trace_id: str,
    seed: int = 0,
    error_probability: float = 0.02,
    oom_probability: float = 0.005,
) -> Iterable[Dict[str, Any]]:
    rng = random.Random(seed)

    common = {
        "workload_name": workload_name,
        "framework": framework,
        "tpu_version": tpu_version,
        "run_mode": run_mode,
        "trace_id": trace_id,
    }

    hbm_capacity = {
        "v4": 32 * 1024**3,
        "v5e": 16 * 1024**3,
        "v5p": 95 * 1024**3,
        "v6e": 32 * 1024**3,
        "cpu_sim": 1 * 1024**3,
    }.get(tpu_version, 16 * 1024**3)

    # Compile once at the start (cache miss).
    compile_time = max(0.5, rng.gauss(2.0, 0.5))
    yield _line(
        layer="xla",
        event="xla.compile",
        message="first-step compile",
        metrics={"compile_time_s": compile_time},
        executable_id="EXE-0001",
        cache_hit=False,
        op_count=rng.randint(20, 200),
        **common,
    )

    for step in range(steps):
        step_id = f"STEP-{step:06d}"

        # Realistic-ish step time with a slow trend.
        device_exec = max(0.01, rng.gauss(0.08 + 0.0001 * step, 0.01))
        input_wait = max(0.0, rng.gauss(0.02, 0.01))
        host_overhead = max(0.0, rng.gauss(0.005, 0.002))
        step_time = device_exec + input_wait + host_overhead

        yield _line(
            layer="input_pipeline",
            event="input_pipeline.load_batch",
            metrics={"input_wait_s": input_wait},
            step_id=step_id,
            **common,
        )

        yield _line(
            layer="pjrt",
            event="pjrt.step",
            metrics={
                "step_time_s": step_time,
                "device_execution_time_s": device_exec,
                "host_overhead_s": host_overhead,
                "tokens_per_second": 1024.0 / max(step_time, 1e-3),
                "samples_per_second": 32.0 / max(step_time, 1e-3),
            },
            step_id=step_id,
            **common,
        )

        # HBM snapshot every 10 steps.
        if step % 10 == 0:
            util = min(0.99, 0.55 + 0.001 * step + rng.uniform(-0.02, 0.02))
            used = int(util * hbm_capacity)
            yield _line(
                layer="hbm",
                event="hbm.snapshot",
                metrics={
                    "hbm_used_bytes": used,
                    "hbm_capacity_bytes": hbm_capacity,
                    "hbm_utilization_ratio": util,
                    "memory_stall_s": max(0.0, rng.gauss(0.01, 0.005)),
                },
                operation_type="all",
                step_id=step_id,
                **common,
            )

        # Collectives roughly every 5 steps.
        if step % 5 == 0:
            ar = max(0.001, rng.gauss(0.01, 0.003))
            ag = max(0.001, rng.gauss(0.008, 0.002))
            rs = max(0.001, rng.gauss(0.006, 0.002))
            yield _line(
                layer="sharding",
                event="collective.all_reduce",
                metrics={"time_s": ar, "collective_time_s": ar + ag + rs},
                step_id=step_id,
                **common,
            )
            yield _line(
                layer="sharding",
                event="collective.all_gather",
                metrics={"time_s": ag, "collective_time_s": ar + ag + rs},
                step_id=step_id,
                **common,
            )
            yield _line(
                layer="sharding",
                event="collective.reduce_scatter",
                metrics={"time_s": rs, "collective_time_s": ar + ag + rs},
                step_id=step_id,
                **common,
            )

        # Profiler summary every 25 steps.
        if step % 25 == 0:
            yield _line(
                layer="profiler",
                event="profiler.summary",
                metrics={
                    "matrix_unit_utilization_ratio": min(
                        0.95, max(0.05, rng.gauss(0.6, 0.1))
                    ),
                },
                step_id=step_id,
                **common,
            )

        # Cost estimate every 50 steps.
        if step % 50 == 0:
            cost_per_step = step_time * 0.0000125  # placeholder
            yield _line(
                layer="cost",
                event="cost.estimate",
                metrics={
                    "cost_per_step": cost_per_step,
                    "cost_per_token": cost_per_step / max(1.0, 1024.0),
                },
                step_id=step_id,
                **common,
            )

        # Occasional OOM (only if HBM near full).
        if rng.random() < oom_probability:
            yield _line(
                layer="hbm",
                event="hbm.oom",
                level="ERROR",
                message="HBM allocation failed",
                step_id=step_id,
                **common,
            )

        # Occasional generic error.
        if rng.random() < error_probability:
            yield _line(
                layer="runtime",
                event="error.slow_step",
                level="ERROR",
                message=f"slow step detected: {step_time:.3f}s",
                metrics={"step_time_s": step_time},
                step_id=step_id,
                **common,
            )

    # Final checkpoint.
    yield _line(
        layer="checkpoint",
        event="checkpoint.save",
        metrics={"checkpoint_time_s": max(0.1, rng.gauss(1.5, 0.3))},
        **common,
    )


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic cloud_tpu_lab JSONL logs")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("cloud_tpu_lab/artifacts/logs/run_SYNTH-0001.jsonl"),
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--workload-name", default="synthetic_workload")
    p.add_argument("--framework", default="cpu_sim",
                   choices=["jax", "torch_xla", "tf", "cpu_sim"])
    p.add_argument("--tpu-version", default="v5p",
                   choices=["v4", "v5e", "v5p", "v6e", "cpu_sim"])
    p.add_argument("--run-mode", default="local_cpu",
                   choices=["local_cpu", "colab", "cloud_tpu_vm"])
    p.add_argument("--trace-id", default="TRACE-SYNTH-0001")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stream", action="store_true",
                   help="Append one line at a time with a small delay (for live tail demos).")
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    events = generate(
        steps=args.steps,
        workload_name=args.workload_name,
        framework=args.framework,
        tpu_version=args.tpu_version,
        run_mode=args.run_mode,
        trace_id=args.trace_id,
        seed=args.seed,
    )

    mode = "a" if args.stream else "w"
    with open(args.out, mode, encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
            if args.stream:
                fh.flush()
                time.sleep(0.05)

    print(f"wrote synthetic JSONL to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
