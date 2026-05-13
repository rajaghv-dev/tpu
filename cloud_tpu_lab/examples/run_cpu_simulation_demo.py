#!/usr/bin/env python3
"""
End-to-end CPU simulation demo — the vertical slice the rest of the repo
builds on. Runs with stdlib only (no jax, no torch, no tensorflow).

What it does, in order:

  1. Build a tiny MLP graph
  2. Lower to fake HLO
  3. Compile to fake PJRT executable (cache miss on step 0, hit after)
  4. Place onto a fake TPU device (catalog-driven specs)
  5. Allocate HBM for params, optimizer state, activations, workspace
  6. Build a 1-D mesh + simulate sharding the parameters
  7. Run N training steps:
       - per-op execution via PJRT runtime
       - per-step all-reduce on gradients (zero for 1-chip)
       - per-step input-pipeline wait
       - emit profiler trace events
       - emit JSONL log line and CSV metric row
  8. Generate cost report
  9. Generate bottleneck report
 10. Write artefacts to cloud_tpu_lab/artifacts/

Run from repo root:
    python -m cloud_tpu_lab.examples.run_cpu_simulation_demo
or:
    cd cloud_tpu_lab && python -m examples.run_cpu_simulation_demo
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow `python examples/run_cpu_simulation_demo.py` from inside cloud_tpu_lab
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from cloud_tpu_lab.src.common.config import SimulationConfig, WorkloadConfig
from cloud_tpu_lab.src.common.cost import CostInputs, estimate_cost
from cloud_tpu_lab.src.common.trace import (
    TraceContext, new_step_id, new_trace_id, reset_counters,
)
from cloud_tpu_lab.src.memory.activation_memory import (
    estimate_mlp_activations, total_bytes,
)
from cloud_tpu_lab.src.memory.hbm_sim import make_hbm_for_spec
from cloud_tpu_lab.src.model_examples.tiny_mlp_jax import build_tiny_mlp_graph
from cloud_tpu_lab.src.observability.logger import JsonlLogger
from cloud_tpu_lab.src.observability.metrics import MetricStream
from cloud_tpu_lab.src.observability.report import render_run_report, write_report
from cloud_tpu_lab.src.pjrt_sim.device import TpuDevice
from cloud_tpu_lab.src.pjrt_sim.executable import CompiledExecutable
from cloud_tpu_lab.src.pjrt_sim.runtime import PjrtRuntime
from cloud_tpu_lab.src.profiling.bottleneck_report import diagnose
from cloud_tpu_lab.src.profiling.profiler_trace import ProfilerTrace
from cloud_tpu_lab.src.profiling.trace_analyzer import (
    compute_breakdown, step_summary,
)
from cloud_tpu_lab.src.sharding.all_reduce import all_reduce_time
from cloud_tpu_lab.src.sharding.mesh import PartitionSpec, make_1d_mesh
from cloud_tpu_lab.src.sharding.partitioner import partition_tensor
from cloud_tpu_lab.src.tpu_versions.cloud_tpu_catalog import get_spec
from cloud_tpu_lab.src.tpu_versions.version_compare import render_table
from cloud_tpu_lab.src.xla_sim.compile_cache import CompileCache, compile_hlo
from cloud_tpu_lab.src.xla_sim.lowering import lower_to_hlo
from cloud_tpu_lab.src.input_pipeline.dataloader_sim import simulate_input_pipeline


def run_demo(
    workload: WorkloadConfig,
    sim: SimulationConfig,
    quiet: bool = False,
) -> dict:
    """Run the whole pipeline. Returns a small dict of run summary fields."""
    reset_counters()
    trace = TraceContext(trace_id=new_trace_id())
    log = JsonlLogger()
    metrics = MetricStream()
    pt = ProfilerTrace(trace_id=trace.trace_id)
    pt.start()

    def emit(event: str, fields: dict) -> None:
        log.log(layer=fields.pop("layer", "runtime"),
                event=event, trace=trace, **fields)

    if not quiet:
        print(f"[cloud_tpu_lab] trace_id = {trace.trace_id}")
        print(f"[cloud_tpu_lab] tpu_version = {workload.tpu_version}, "
              f"chips = {workload.chip_count}")

    # ── 1-2. Build + lower ──────────────────────────────────────────────────
    graph = build_tiny_mlp_graph(
        batch_size=workload.batch_size,
        input_dim=workload.input_dim,
        hidden_size=workload.hidden_size,
        output_dim=workload.output_dim,
        num_layers=workload.num_layers,
        dtype=workload.precision,
    )
    module = lower_to_hlo(graph)
    emit("xla.lowered", {
        "layer": "xla", "model_name": graph.name,
        "n_ops": len(module.ops),
        "op_count_by_kind": module.op_count_by_kind(),
        "total_flops": module.total_flops(),
    })

    # ── 3. Compile (cache miss first time) ──────────────────────────────────
    cache = CompileCache()
    mesh = make_1d_mesh(workload.chip_count, name="data")
    compile_result = compile_hlo(module, cache, mesh_shape=mesh.shape,
                                 dtype=workload.precision)
    trace.executable_id = compile_result.executable_id
    pt.add_event("xla.compile", "compile", compile_result.compile_time_s,
                 tid=0, args={
                     "trace_id": trace.trace_id,
                     "executable_id": compile_result.executable_id,
                     "cache_hit": compile_result.cache_hit,
                 })
    emit("xla.compile", {
        "layer": "xla", "compile_time_s": compile_result.compile_time_s,
        "executable_id": compile_result.executable_id,
        "cache_hit": compile_result.cache_hit, "op_count": compile_result.op_count,
    })
    executable = CompiledExecutable(
        executable_id=compile_result.executable_id, module=module,
        mesh_shape=mesh.shape, dtype=workload.precision,
    )

    # ── 4. Devices + runtime ────────────────────────────────────────────────
    spec = get_spec(workload.tpu_version)
    devices = [TpuDevice(device_id=i, spec=spec) for i in range(workload.chip_count)]
    runtime = PjrtRuntime(devices=devices)
    emit("pjrt.devices_ready", {"layer": "pjrt", "n_devices": len(devices),
                                "tpu_version": workload.tpu_version})

    # ── 5. HBM allocations ─────────────────────────────────────────────────
    hbm = make_hbm_for_spec(spec)
    elem = 2 if workload.precision == "bf16" else 4
    # Parameters: rough estimate from layer shapes.
    param_bytes = sum(
        op.bytes_in - op.bytes_out for op in module.ops if op.bytes_in > op.bytes_out
    ) // 2 or 1024
    hbm.allocate("params", "parameters", param_bytes)
    hbm.allocate("opt_state", "optimizer", param_bytes * 2)  # Adam m + v
    # Activations: from the activation estimator.
    acts = estimate_mlp_activations(
        workload.batch_size, workload.hidden_size,
        workload.num_layers, dtype=workload.precision,
    )
    hbm.allocate("activations", "activations", total_bytes(acts))
    hbm.allocate("workspace", "workspace", 1 * 1024 * 1024)
    emit("hbm.allocated", {
        "layer": "hbm",
        "used_bytes": hbm.used_bytes(),
        "capacity_bytes": hbm.capacity_bytes,
        "utilization": hbm.utilization(),
        "by_category": hbm.by_category(),
    })

    # ── 6. Sharding the params on the mesh ─────────────────────────────────
    shards = partition_tensor(
        logical_shape=(workload.hidden_size, workload.hidden_size),
        spec=PartitionSpec(("data", None)),
        mesh=mesh,
    )
    emit("sharding.partitioned", {
        "layer": "sharding", "n_shards": len(shards),
        "shard_shape": shards[0].shard_shape if shards else None,
    })

    # ── 7. Train loop ──────────────────────────────────────────────────────
    samples_per_step = workload.batch_size
    ici_bw_bytes_s = spec.ici_bandwidth_gbps * 1e9
    for step in range(workload.n_steps):
        step_trace = trace.with_step(new_step_id())

        # Input pipeline wait
        ip_cost = simulate_input_pipeline(
            n_steps=1, batch_size=workload.batch_size,
            bytes_per_sample=workload.input_dim * elem,
            device_step_time_s=0.005,
        )[0]
        pt.add_event("input.load_batch", "input_pipeline",
                     ip_cost.effective_wait_s, tid=1, args={
                         "step": step, "trace_id": step_trace.trace_id,
                     })

        # Device execution
        dev_t = runtime.execute(executable, step_trace,
                                log_event=lambda ev, fields: log.log(
                                    layer="runtime", event=ev, trace=step_trace,
                                    metrics=fields))
        pt.add_event(f"step.{step}", "device", dev_t, tid=2, args={
            "step": step, "trace_id": step_trace.trace_id,
        })

        # All-reduce gradients (zero for 1-chip)
        coll = all_reduce_time(
            payload_bytes=param_bytes, n_chips=workload.chip_count,
            ici_bandwidth_bytes_s=ici_bw_bytes_s,
        )
        if coll.sim_duration_s > 0:
            pt.add_event("collective.all_reduce", "collective",
                         coll.sim_duration_s, tid=3, args={
                             "step": step, "collective_id": coll.collective_id,
                             "trace_id": step_trace.trace_id,
                             "payload_bytes": coll.payload_bytes,
                         })

        total_step_s = ip_cost.effective_wait_s + dev_t + coll.sim_duration_s

        emit("train.step", {
            "layer": "runtime",
            "step": step,
            "step_id": step_trace.step_id,
            "step_time_s": total_step_s,
            "device_s": dev_t,
            "input_wait_s": ip_cost.effective_wait_s,
            "collective_s": coll.sim_duration_s,
            "samples_per_step": samples_per_step,
        })

        metrics.record(
            step=step,
            cloud_tpu_step_time_seconds=total_step_s,
            cloud_tpu_device_execution_time_seconds=dev_t,
            cloud_tpu_input_wait_time_seconds=ip_cost.effective_wait_s,
            cloud_tpu_collective_time_seconds=coll.sim_duration_s,
            cloud_tpu_all_reduce_time_seconds=coll.sim_duration_s,
            cloud_tpu_hbm_used_bytes=hbm.used_bytes(),
            cloud_tpu_hbm_capacity_bytes=hbm.capacity_bytes,
            cloud_tpu_hbm_utilization_ratio=hbm.utilization(),
            cloud_tpu_samples_per_second=samples_per_step / max(total_step_s, 1e-9),
            cloud_tpu_compile_time_seconds=(
                compile_result.compile_time_s if step == 0 else 0.0
            ),
        )

    # ── 8. Cost ────────────────────────────────────────────────────────────
    # Use median step time (excluding cold first step) as the steady-state step time.
    durations = [r.fields["cloud_tpu_step_time_seconds"] for r in metrics.rows]
    steady = sorted(durations[1:]) if len(durations) > 1 else durations
    median_step = steady[len(steady) // 2] if steady else 0.0
    cost = estimate_cost(CostInputs(
        chip_count=workload.chip_count,
        n_steps=workload.n_steps,
        step_time_s=median_step,
        hourly_usd_per_chip=workload.hourly_usd_per_chip,
        samples_per_step=samples_per_step,
        tokens_per_step=0,
        utilization=0.85,
    ))
    emit("cost.estimated", {"layer": "cost", **cost.to_dict()})

    # ── 9. Analysis + report ───────────────────────────────────────────────
    breakdown = compute_breakdown(pt)
    summary = step_summary(pt)
    findings = diagnose(breakdown=breakdown, hbm=hbm,
                        n_chips=workload.chip_count,
                        total_run_usd=cost.total_run_usd)

    # ── 10. Write artefacts ────────────────────────────────────────────────
    log_path = Path(sim.log_dir) / f"run_{trace.trace_id}.jsonl"
    metrics_path = Path(sim.metrics_dir) / f"run_{trace.trace_id}.csv"
    trace_path = Path(sim.trace_dir) / f"run_{trace.trace_id}.json"
    report_path = Path(sim.report_dir) / f"run_{trace.trace_id}.md"

    log.flush(log_path)
    metrics.write_csv(metrics_path)
    pt.write_chrome_json(trace_path)
    report_md = render_run_report(
        trace_id=trace.trace_id, config=workload,
        breakdown=breakdown, summary=summary, hbm=hbm,
        compile_cache=cache, cost=cost, findings=findings,
    )
    write_report(report_path, report_md)

    summary_dict = {
        "trace_id": trace.trace_id,
        "n_log_events": len(log.buffer),
        "n_metric_rows": len(metrics.rows),
        "n_trace_events": len(pt.events),
        "compile_time_s": compile_result.compile_time_s,
        "median_step_s": median_step,
        "total_run_usd": cost.total_run_usd,
        "hbm_utilization": hbm.utilization(),
        "n_findings": len(findings),
        "artifacts": {
            "log":     str(log_path),
            "metrics": str(metrics_path),
            "trace":   str(trace_path),
            "report":  str(report_path),
        },
    }
    if not quiet:
        print(json.dumps(summary_dict, indent=2))
        print(f"\nRun report: {report_path}")
    return summary_dict


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cloud_tpu_lab_demo",
        description="End-to-end CPU simulation of a Cloud TPU training step.")
    p.add_argument("--name", default="tiny_mlp_demo")
    p.add_argument("--tpu-version", default="v5e",
                   choices=["v4", "v5e", "v5p", "v6e"])
    p.add_argument("--chip-count", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--hourly-usd-per-chip", type=float, default=1.20,
                   help="Placeholder; update from cloud.google.com/tpu/pricing")
    p.add_argument("--show-versions", action="store_true",
                   help="Print the TPU version comparison table and exit")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.show_versions:
        print(render_table())
        return 0

    workload = WorkloadConfig(
        name=args.name,
        tpu_version=args.tpu_version,
        chip_count=args.chip_count,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        n_steps=args.n_steps,
        precision=args.precision,
        hourly_usd_per_chip=args.hourly_usd_per_chip,
    )
    sim = SimulationConfig()
    run_demo(workload, sim, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
