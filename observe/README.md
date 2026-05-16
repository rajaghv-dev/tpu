# observe/ — Probes for the TPU Benchmark

The observability layer for the TPU benchmark. Pluggable probes fire at every
phase boundary in `benchmarks.runner.run_experiment` and write per-run JSON
artefacts into `results/run_logs/<run_id>/<probe_name>.json`.

## Why probes vs callbacks vs monkey-patching

Stages 2–6 of the project plan add new measurements (HLO dumps, JAX profiler
traces, Cloud Monitoring polls, OTel export, …). Each is independent of the
core inference loop, but each needs hooks into the same lifecycle: run start,
phase boundaries, error paths, run end. Plain callbacks would force every
hook into a free function; monkey-patching the runner would couple every
extension to an internal API. A small `Probe` ABC gives each measurement its
own class with state, isolated failure handling (one bad probe never kills a
benchmark), and a uniform "one JSON file per run per probe" output contract.
See `DECISIONS.md` ADR-014 (probe architecture) and ADR-015 (probe output
schema).

## Lifecycle

```
runner.run_experiment(cfg)
  │
  ├── fanout_before_run(run_id, config, log_dir)
  │     for p in probes: p.before_run(...)
  │
  │   for phase in [preflight, model_load, compile, warmup,
  │                 latency, throughput, postflight]:
  │
  │     ├── fanout_before_phase(phase)
  │     │     for p in probes: p.before_phase(phase)
  │     │
  │     ├── <phase body executes>
  │     │
  │     ├── on success:                       on failure:
  │     │     fanout_after_phase(             fanout_on_error(
  │     │       phase, duration_s)              phase, exc)
  │     │     for p in probes:                for p in probes:
  │     │       p.after_phase(...)              p.on_error(...)
  │     │                                     # after_phase is NOT called
  │
  └── fanout_after_run(run_id, result_or_None, log_dir)
        for p in probes:
          p.after_run(run_id, result_or_None)
          payload = p.write_log()
          if payload is not None:
            <log_dir>/<p.name>.json ← json.dumps(payload)
```

Every fanout call goes through `_safe_call` (`probe.py:155`), which catches
any exception inside a probe hook, logs a warning, and continues. Probes can
fail without affecting the benchmark or other probes.

## The Probe ABC contract

From `observe/probe.py`:

```python
class Probe:
    name: str = ""          # filesystem-safe identifier; <name>.json output

    def before_run(self, run_id, config, log_dir): ...
    def after_run(self, run_id, result):           ...   # result=None on failure
    def before_phase(self, phase_name):            ...
    def after_phase(self, phase_name, duration_s): ...   # success path only
    def on_error(self, phase_name, exc):           ...   # failure path only
    def write_log(self) -> Optional[dict]:         ...   # None → skip file
```

Every hook has a no-op default — subclasses override only what they need.
`name` MUST be unique across registered probes; collisions raise at
`register_probe()` time. Method signatures are stable contract: probes are
allowed to call them by keyword.

Registry helpers (also in `probe.py`):

- `register_probe(probe)` — append one probe; raises on empty/colliding name.
- `set_active_probes([p1, p2, …])` — replace the set wholesale (tests).
- `get_active_probes()` — snapshot of the current list.
- `clear_probes()` — drop all probes (tests).

The runner reads from `get_active_probes()` at run start. The default set is
empty — Stage 1 ships zero default probes, so existing harness scripts keep
working unchanged.

## Built-in probes

| Name                | Captures                                                        | Optional dep                | Output                                     | Typical use                    |
| ------------------- | --------------------------------------------------------------- | --------------------------- | ------------------------------------------ | ------------------------------ |
| `timing`            | Wall-clock per phase + total run                                | (stdlib)                    | `timing.json`                              | "Where did the 90 s go?"       |
| `memory`            | Host RSS / VMS at each phase boundary                           | `psutil`                    | `memory.json`                              | Host-side leak detection       |
| `input_fingerprint` | SHA-256 of synthetic latency inputs                             | `numpy`                     | `input_fingerprint.json`                   | Cross-run determinism check    |
| `hlo_dump`          | XLA HLO IR text dump + op histogram                             | `XLA_FLAGS` writable        | `hlo_dump.json` + `hlo/*.txt`              | Compile-pipeline triage        |
| `jax_profiler`      | One `jax.profiler` trace covering the latency phase             | `jax`                       | `jax_profiler.json` + `jax_profiler/*.pb`  | Kernel-level perf in TensorBoard |
| `cloud_monitoring`  | Per-TPU-chip MXU / HBM / network at 1 Hz                        | `google-cloud-monitoring`   | `cloud_monitoring.json`                    | Silicon utilisation timeline   |
| `otel`              | OTel spans (run + per-phase) + metrics (latency / throughput / cost) | `opentelemetry-sdk` + OTLP exporter | `otel.json` (counts only)         | Live dashboards via collector  |
| `determinism`       | Runtime determinism settings snapshot at run start              | (stdlib)                    | `determinism.json`                         | Verify reproducibility config  |
| `device_info`       | One-shot hardware/software stack snapshot at run start          | `psutil`, `jax`             | `device_info.json`                         | System inventory per run       |
| `power_thermal`     | Background power, temperature, and utilization sampler          | `psutil`, `nvidia-smi`, `tpu-info` | `power_thermal.json`                | Thermal/power budget tracking  |
| `xla_compile`       | XLA compilation config snapshot, flags, timing, cache state     | (stdlib)                    | `xla_compile.json`                         | XLA compile-path debugging     |
| `training_metrics`  | Per-step loss, learning rate, gradient norm, and custom scalars | (stdlib)                    | `training_metrics.json`                    | Training convergence tracking  |
| `step_timing`       | Per-step wall-clock timing with jit-compile vs steady-state split | (stdlib)                   | `step_timing.json`                         | Step throughput and warmup cost |
| `checkpoint`        | Checkpoint save/load events: path, size, duration, success      | (stdlib)                    | `checkpoint.json`                          | Checkpoint health auditing     |

### `timing` — `observe/timing_probe.py`

Per-phase wall-clock timings using `time.perf_counter()`. Records a partial
duration with `error: True` when a phase raises. Output:

```json
{
  "total_run_s": 87.41,
  "timeline":      [{"phase": "preflight", "duration_s": 0.12, "ts": ...}, ...],
  "phase_summary": {"preflight": {...}, "model_load": {...}, ...}
}
```

No optional deps; cannot fail. Tested in `tests/test_app_probes.py::TestTimingProbe`.
Not directly tied to a single Grafana panel; `latency_violins.json` consumes
the OTel-equivalent metric.

### `memory` — `observe/memory_probe.py`

Two snapshots per phase (`when="before"`, `when="after"`) of host process
RSS/VMS. On error, appends a third snapshot tagged `when="on_error"`. Does
NOT see TPU HBM or GPU VRAM — host CPU only. If `psutil` is missing the
probe degrades to `{"available": false, "snapshots": []}`. Tested in
`tests/test_app_probes.py::TestMemoryProbe`.

### `input_fingerprint` — `observe/input_fingerprint.py`

In `before_phase("latency")`, regenerates the bs=1 synthetic inputs that
`benchmarks.runner.make_synthetic_inputs` would produce for this config and
writes a 16-hex-char SHA-256 prefix. Two runs with the same config must
produce the same fingerprint, otherwise a non-determinism has crept into the
input pipeline. Output stores the digest plus per-input shapes/dtypes — never
the raw bytes (a single fp32 image batch is ~600 KB and we run thousands of
experiments). Reads `config.input_seed`. If `numpy` import or input
construction fails, fields are `None`. Tested in
`tests/test_app_probes.py::TestInputFingerprintProbe`.

### `hlo_dump` — `observe/hlo_dump_probe.py`

In `before_run`, sets `XLA_FLAGS=--xla_dump_to=<log_dir>/hlo
--xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*` (preserving any prior
value, restored in `after_run`). After `compile`, snapshots file count + total
bytes. After the run, parses the largest `.txt` for instruction count, fusion
count, and a top-10 op histogram via two regexes (`_HLO_INSTR_RE`,
`_HLO_OP_RE`). Output:

```json
{
  "available": true,
  "n_files": 42, "total_bytes": 12345678,
  "largest_file": "...", "largest_file_bytes": 9876543,
  "instruction_count_estimate": 5210,
  "fusion_count_estimate": 38,
  "top_ops": {"add": 412, "broadcast": 305, ...},
  "xla_flags_set": "..."
}
```

Critical caveat: XLA reads `XLA_FLAGS` once when the JAX backend initialises.
**Register `HloDumpProbe` before any `jax.jit` runs in the process** — typically
at the top of your harness script. If JIT has already happened, the dump
directory will be empty and `available=false` with a `reason` string.
Tested in `tests/test_compiler_probes.py`.

### `jax_profiler` — `observe/jax_profiler_probe.py`

Wraps the `latency` phase in a single `jax.profiler.start_trace()` /
`stop_trace()` pair. Trace files (`xspace.pb`, `events.json.gz`, …) land in
`<log_dir>/jax_profiler/` and are intended for TensorBoard or Perfetto, not
in-process parsing. The probe stops the trace from BOTH `after_phase` and
`on_error` — half-written trace dirs are unparseable by downstream tools.
If `jax.profiler` is missing or `start_trace` fails (e.g. unsupported
backend) the probe degrades to `available=false` with a `reason`.
Tested in `tests/test_compiler_probes.py`.

### `cloud_monitoring` — `observe/cloud_monitoring_probe.py`

Polls Google Cloud Monitoring once per second (`_POLL_INTERVAL_S = 1.0`) on a
daemon thread for five per-TPU-chip metric types: `mxu_utilization`,
`network_sent_bytes_count`, `memory_bandwidth_utilization`,
`memory_utilization`, `cpu_utilization`. Each sample is tagged with the
current phase. Sample buffer is capped at 7200 entries (drop-oldest).

Required config — at least one of each must resolve:

- Project: `project=` arg, `$GCP_PROJECT`, or `gcloud config get-value project`.
- TPU name: `tpu_name=` arg, `$TPU_NAME`, or `TPU_NAME=` in `.tpu-bench-state/state.env`.
- Zone: `zone=` arg, `$TPU_ZONE`, or `TPU_ZONE=` in `.tpu-bench-state/state.env`.

If any of those are missing, or if `google.cloud.monitoring_v3` is not
installed, or if client init fails (no ADC), the probe disables itself with
`available=false`. Per-metric retry-once-then-warn. Output includes the raw
timeline plus a `per_phase_summary` of `{min, mean, max}` per (phase,
metric). Powers the `mxu_heatmap.json` Grafana panel. Tested in
`tests/test_cloud_monitoring_probe.py`.

### `otel` — `observe/otel_probe.py`

Emits one parent span (`benchmark.run`) per run and one child span
(`phase.<name>`) per phase. Records histograms for phase duration, latency
quantiles, throughput, and experiment cost; counter for phase errors.
Reads `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`) and
`OTEL_SERVICE_NAME` (default `tpu-bench`). Three-tier degradation:

1. SDK + OTLP HTTP exporter present → spans/metrics shipped via OTLP.
2. SDK present, OTLP exporter missing → falls back to `ConsoleSpanExporter`.
3. SDK missing → no-op probe; `write_log()` still emits a stable
   `{available: false, spans_emitted: 0, …}` so users can detect the gap
   from the run log without re-running.

The `otel.json` payload stores only counts — actual span data goes to the
collector. This probe feeds every Grafana panel in
`results/dashboard/grafana/` that uses Prometheus/Mimir as a datasource:
`roofline.json`, `latency_violins.json`, `failures.json`, `cost.json`.
Tested in `tests/test_otel_probe.py`.

## Writing your own probe

1. Subclass `Probe`.
2. Set `name` to a unique, filesystem-safe identifier (it becomes the JSON
   filename).
3. Implement only the hooks you need — every default is a no-op.
4. Optionally override `write_log()` to return a JSON-serialisable dict.
5. Register before `run_experiment` is called.

Minimal example (host load average per phase, ~20 lines):

```python
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.probe import Probe, register_probe


class LoadAvgProbe(Probe):
    name = "loadavg"

    def __init__(self) -> None:
        self._samples: List[Dict[str, Any]] = []

    def before_phase(self, phase_name: str) -> None:
        one, five, fifteen = os.getloadavg()
        self._samples.append({"phase": phase_name, "1m": one,
                              "5m": five, "15m": fifteen})

    def write_log(self) -> Optional[Dict[str, Any]]:
        return {"samples": self._samples}


register_probe(LoadAvgProbe())
```

Tradeoffs: this only samples at `before_phase`, so a phase that takes 60 s
gets one sample even though load can change inside it. A polling-thread
variant (cf. `cloud_monitoring_probe`) would give a true timeline but adds
threading complexity not warranted at this level of fidelity. Also,
`os.getloadavg()` is Unix-only — on Windows the probe would need a guard.

## Registering probes from CLI / harness

The Stage 1 harness (`benchmarks.harness.run_suite`) does NOT auto-register
probes — kept clean for backward compatibility. The recommended pattern is a
small entrypoint script:

```python
# scripts/run_with_probes.py
from observe.probe import register_probe
from observe.timing_probe import TimingProbe
from observe.memory_probe import MemoryProbe
from observe.otel_probe import OTelProbe

# Register BEFORE any jax import / jit if HloDumpProbe is in the list.
register_probe(TimingProbe())
register_probe(MemoryProbe())
register_probe(OTelProbe())

from benchmarks.harness import run_suite  # noqa: E402
run_suite(...)
```

Future work could add a `--probes` CLI flag that names probes by their
class/`name`. Today, register explicitly in Python.

## Performance considerations

Each fanout call costs ~10–50 µs per probe (no-op overhead — pure Python
dispatch through `_safe_call`). With 7 probes and 7 phases plus run-level
hooks, that's ~3 ms of probe overhead per run. A smoke run is ~8 minutes, so
overhead is well under 0.001 % — acceptable.

Two probes do real work and are exceptions:

- `cloud_monitoring` runs a 1 Hz polling thread independent of phases. CPU
  cost is dominated by the GCP RPC, not the probe.
- `hlo_dump` parses the largest HLO `.txt` once in `after_run`. Files are
  typically a few MB; parsing is line-based and finishes in well under a
  second.

If a probe is suspected of slowing things down, look at `probe.py:_safe_call`
— that's the single dispatch point. Adding a `time.perf_counter()` wrapper
there is the easiest way to attribute overhead to a specific probe.

## Testing probes

Five test files cover the probes:

- `tests/test_app_probes.py` — `TimingProbe`, `MemoryProbe`, `InputFingerprintProbe`.
- `tests/test_compiler_probes.py` — `HloDumpProbe`, `JaxProfilerProbe`.
- `tests/test_cloud_monitoring_probe.py` — `CloudMonitoringProbe`
  (incl. `aggregate_per_phase_summary`).
- `tests/test_otel_probe.py` — `OTelProbe` happy path + no-op fallback.

Test pattern: instantiate the probe, drive its hooks manually with synthetic
args, assert on the dict returned by `write_log()`. **Do not** stand up a
real TPU, OTLP collector, or GCP project — every probe has an `available`
flag or an equivalent fallback that lets unit tests run in CI without
external services.

## References

- ABC and registry: `observe/probe.py`
- Probes: `observe/timing_probe.py`, `observe/memory_probe.py`,
  `observe/input_fingerprint.py`, `observe/hlo_dump_probe.py`,
  `observe/jax_profiler_probe.py`, `observe/cloud_monitoring_probe.py`,
  `observe/otel_probe.py`, `observe/determinism_probe.py`,
  `observe/device_info_probe.py`, `observe/power_thermal_probe.py`,
  `observe/xla_compile_probe.py`, `observe/training_metrics_probe.py`,
  `observe/step_timing_probe.py`, `observe/checkpoint_probe.py`
- Tests: `tests/test_app_probes.py`, `tests/test_compiler_probes.py`,
  `tests/test_cloud_monitoring_probe.py`, `tests/test_otel_probe.py`
- Architecture decisions: `DECISIONS.md` ADR-014 (probe architecture),
  ADR-015 (probe output schema).
- Dashboards: `results/dashboard/grafana/README.md`.
- Runner integration: `benchmarks/runner.py` (search for `fanout_`).
