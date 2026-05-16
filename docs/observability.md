# Observability

Generated: 2026-05-16.

## Overview

Three-level observability model. All levels are independent — you can use any combination.

## Level 1: Structured Results (always on)

Every run appends to `results/runs.jsonl`. Every failure writes `results/run_logs/<run_id>/error.json`. Every run writes `results/run_logs/<run_id>/lineage.json`.

View results: `python3 scripts/render_results.py` → regenerates `results/RESULTS.md`.

## Level 2: Probe Outputs (pluggable)

Each probe writes `results/run_logs/<run_id>/<probe_name>.json` after the run. Register probes before running:

```python
from observe.probe import register_probe
from observe.timing_probe import TimingProbe
from observe.memory_probe import MemoryProbe

register_probe(TimingProbe())   # writes timing.json
register_probe(MemoryProbe())   # writes memory.json
```

Or via CLI: `--probes default` (TimingProbe + MemoryProbe + InputFingerprintProbe).

### Probe reference

| Probe | File | Output | What it measures |
|---|---|---|---|
| TimingProbe | observe/timing_probe.py | timing.json | Wall-clock per phase |
| MemoryProbe | observe/memory_probe.py | memory.json | psutil RSS/VMS at phase boundaries |
| InputFingerprintProbe | observe/input_fingerprint.py | input_fingerprint.json | SHA-256 of synthetic inputs |
| HloDumpProbe | observe/hlo_dump_probe.py | hlo_dump.json + hlo/*.txt | XLA HLO IR text |
| JaxProfilerProbe | observe/jax_profiler_probe.py | jax_profiler.json + *.pb | JAX profiler trace |
| CloudMonitoringProbe | observe/cloud_monitoring_probe.py | cloud_monitoring.json | GCP TPU MXU%, power, thermal |
| OTelProbe | observe/otel_probe.py | otel.json | OpenTelemetry spans |
| DeterminismProbe | observe/determinism_probe.py | determinism.json | Runtime determinism settings snapshot |
| DeviceInfoProbe | observe/device_info_probe.py | device_info.json | HW/SW stack snapshot |
| PowerThermalProbe | observe/power_thermal_probe.py | power_thermal.json | Power, temperature, utilization |
| XlaCompileProbe | observe/xla_compile_probe.py | xla_compile.json | XLA config, flags, compile timing |
| TrainingMetricsProbe | observe/training_metrics_probe.py | training_metrics.json | Per-step loss/lr/grad_norm |
| StepTimingProbe | observe/step_timing_probe.py | step_timing.json | Per-step wall-clock, samples/sec |
| CheckpointProbe | observe/checkpoint_probe.py | checkpoint.json | Checkpoint write events |

## Level 3: OpenTelemetry (opt-in)

Enable: `export TPU_BENCH_OTEL=otlp` (or `=file` for disk output).

```bash
# Send to local OTel Collector:
export TPU_BENCH_OTEL=otlp
export TPU_BENCH_OTEL_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=tpu-bench

# Or write OTLP-JSON to disk (TPU VM → pull to laptop):
export TPU_BENCH_OTEL=file
export TPU_BENCH_OTEL_DIR=results/otel/
```

## Local Grafana stack

Start: `make otel-view` → Grafana at http://localhost:3000
Stop: `make otel-down`

Requires Docker. Uses `infra/docker-compose.yml` with `grafana/otel-lgtm` all-in-one image.

After a TPU run: `make otel-collect` pulls `results/otel/` from the TPU VM, then `make otel-view` replays it.

## Grafana dashboards

5 pre-built dashboards in `results/dashboard/grafana/` and `infra/grafana/dashboards/`:
- Experiment Timeline
- Latency Distribution
- Throughput vs Precision
- Compile Breakdown
- MXU Heatmap

Import via Grafana UI: Dashboards → Import → upload JSON file.

## Logs

All logs are append-only. Never delete `results/runs.jsonl`. Per-run logs in `results/run_logs/<run_id>/` are regeneratable from probe re-run (not currently supported but planned).

## Metrics exported via OTel

When OTel is enabled, the harness records:
- `tpu_bench.latency_ms` (histogram)
- `tpu_bench.throughput_samples_sec` (histogram)
- `tpu_bench.compile_cold_s` (gauge)
- `tpu_bench.compile_warm_s` (gauge)
- `tpu_bench.latency_cv_pct` (gauge)
- `tpu_bench.cost_per_1k_usd` (gauge)

## Health checks

No HTTP health endpoints. Verify the harness is alive:
- `./scripts/90_status.sh` — prints current TPU burn rate
- `./scripts/92_idle_check.sh` — flags VMs running >2h

## Gaps

| Gap | Stage | Notes |
|---|---|---|
| No per-phase OTel spans | Stage 2 | Currently only metrics, no trace spans |
| No flops counter | Stage 3 | observe/flops_counter.py not implemented yet |
| No numerics validation | Stage 6 | observe/numerics.py not implemented yet |
| No system monitor (GPU SM%) | Stage 2 | observe/system_monitor.py not implemented yet |
