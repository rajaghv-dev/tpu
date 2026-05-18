> **Note:** this doc predates the real-TPU pivot. References to `src/xla_sim/`, `src/pjrt_sim/`, `src/sharding/`, `src/memory/`, `src/input_pipeline/`, and `examples/run_cpu_simulation_demo.py` are historical — those modules were removed. The TPU architecture / XLA / observability concepts below are still accurate. Current run flow lives in [README.md](../README.md) and [16_runbook_real_tpu.md](16_runbook_real_tpu.md).

# 13 — OCT Metrics Dictionary

> **Learning goal:** become fluent in every signal the lab emits. For each
> canonical metric: what produces it, what it means, what a healthy value
> looks like, what a sick value implies, and which labels are safe to slice
> by. For each JSONL field: the schema, the producer, and what it's for.
> For each OpenTelemetry span name: which pipeline layer it covers.

The OCT model — **Observability, Controllability, Traceability** — is
implemented as three artefact streams (metrics CSV, JSONL logs, span
traces) joined by a shared correlation ID. The producer of all three is
`cloud_tpu_lab`. The metric name registry is the source of truth in
`src/observability/metrics.py`; this document explains every entry there.

For wiring these signals into Prometheus / Loki / Tempo, see
`12_observability_with_grafana_prometheus.md`. For methodology around what
the numbers should look like, see `14_benchmarking_playbook.md`.

---

## 1. Metric naming conventions

- All names are lowercase, prefixed `cloud_tpu_`, snake_case.
- Time-valued metrics end in `_seconds`. Never `_ms`, never `_us`.
- Byte-valued metrics end in `_bytes`.
- Ratios end in `_ratio` and are in `[0, 1]`.
- Counters end in `_total` and are monotonic.
- Histograms are not enumerated by name here — they reuse the seconds /
  bytes base name with a `_bucket` / `_count` / `_sum` suffix per
  Prometheus convention.

## 2. Label hygiene — the rule

The single rule that prevents cardinality blowup:

```python
# src/observability/metrics.py
SAFE_LABELS      = ("workload_name", "framework", "tpu_version",
                    "run_mode", "operation_type", "error_type")
DANGEROUS_LABELS = ("trace_id", "step_id", "hlo_op_id",
                    "executable_id", "tensor_id", "shard_id")
```

- Safe labels can be applied to **any** metric. They are bounded.
- Dangerous labels go on **logs** (Loki) and **spans** (Tempo) — never on
  Prometheus metrics. See section 12 of
  `12_observability_with_grafana_prometheus.md` for why.

The per-metric "Dangerous-vs-safe labels" line below repeats this for
emphasis.

---

## 3. Metric definitions

### 3.1 `cloud_tpu_step_time_seconds`

- **Unit:** seconds.
- **Type:** gauge (per-step) and histogram (over a window).
- **Produced by:** the outer training-loop wrapper. The simulator emits it
  from `examples/run_cpu_simulation_demo.py`; the real-TPU path emits it
  per step in your training script.
- **Why it matters:** the headline number. End-to-end wall time for a
  single training step. Almost every cost and throughput derivation starts
  here.
- **Healthy:** stable to within a few percent after warmup; first step
  excluded (it includes compile).
- **Unhealthy:** jumpy (compile loop), trending up (memory or input
  starvation), bimodal (host stall vs hot path).
- **Related metrics:** `cloud_tpu_compile_time_seconds`,
  `cloud_tpu_device_execution_time_seconds`,
  `cloud_tpu_input_wait_time_seconds`, `cloud_tpu_collective_time_seconds`.
- **Safe labels:** `workload_name`, `framework`, `tpu_version`, `run_mode`.
- **Dangerous labels:** never `trace_id` or `step_id`.

### 3.2 `cloud_tpu_compile_time_seconds`

- **Unit:** seconds.
- **Type:** gauge (per event) + histogram.
- **Produced by:** `src/xla_sim/` (sim) and the framework's XLA path on
  real TPU (JAX trace, Torch-XLA compile, TF AOT/JIT).
- **Why it matters:** the most common source of "first run feels slow,
  second run is fine". Persistent compile time on every step means a
  recompile loop, which is a methodology bug, not a hardware limit.
- **Healthy:** large on first step, ~0 thereafter when a compile cache is
  used.
- **Unhealthy:** non-zero on every step → dynamic shapes. Repeated bursts
  → a shape control-flow branch hits a new case.
- **Related metrics:** `cloud_tpu_recompile_count_total` is the discrete
  counterpart.
- **Safe labels:** `framework`, `tpu_version`, `workload_name`.
- **Dangerous labels:** never `hlo_op_id` or `executable_id` (put those on
  the span / log, not the metric).

### 3.3 `cloud_tpu_recompile_count_total`

- **Unit:** dimensionless count.
- **Type:** counter (monotonic).
- **Produced by:** compile-cache layer in `src/xla_sim/`. Real-TPU path
  increments on `jax.config.jax_log_compiles`-style events.
- **Why it matters:** a flat line is the goal. Any upward slope after
  warmup is a methodology smell. Triggers the `compile` finding in
  `src/profiling/bottleneck_report.py`.
- **Healthy:** small fixed value after warmup; doesn't grow with step
  count.
- **Unhealthy:** rises monotonically with steps. One recompile per step =
  shapes are not stable.
- **Related metrics:** `cloud_tpu_compile_time_seconds`.
- **Safe labels:** `framework`, `workload_name`.
- **Dangerous labels:** never per-op IDs.

### 3.4 `cloud_tpu_input_wait_time_seconds`

- **Unit:** seconds (per step).
- **Type:** gauge + histogram.
- **Produced by:** `src/input_pipeline/`. Counts the time the device sat
  idle waiting for the next batch.
- **Why it matters:** if this is >10% of step time you are bottlenecked on
  data loading, not on the chip. The bottleneck rule fires `warn` at >10%
  and `high` at >25% (see `bottleneck_report.py`).
- **Healthy:** near zero with autotuned prefetch.
- **Unhealthy:** material fraction of step time → preprocess too heavy,
  prefetch too shallow, decoding done host-side per step.
- **Related metrics:** `cloud_tpu_step_time_seconds`,
  `cloud_tpu_host_overhead_seconds`.
- **Safe labels:** `workload_name`, `framework`.
- **Dangerous labels:** never per-batch IDs.

### 3.5 `cloud_tpu_device_execution_time_seconds`

- **Unit:** seconds (per step).
- **Type:** gauge + histogram.
- **Produced by:** `src/pjrt_sim/` for the simulator; on real TPU this is
  the device-side compute time inside a step.
- **Why it matters:** the part of step time that is actually "doing useful
  work on the chip". Compare to `cloud_tpu_step_time_seconds` to see how
  much of the step is *not* this — overhead in any other layer.
- **Healthy:** dominates the step (>70%) once warmup is over and the input
  pipeline is fed.
- **Unhealthy:** small fraction → bottleneck is elsewhere; check the other
  metrics in this section.
- **Related metrics:** all the other time metrics; the breakdown in
  `src/profiling/trace_analyzer.py` is built from these.
- **Safe labels:** `tpu_version`, `framework`, `workload_name`.
- **Dangerous labels:** never `device_event_id`.

### 3.6 `cloud_tpu_hbm_used_bytes`

- **Unit:** bytes.
- **Type:** gauge.
- **Produced by:** `src/memory/hbm_sim.py`. On real TPU you get equivalent
  numbers from the framework profiler.
- **Why it matters:** the absolute "how full is the chip's high-bandwidth
  memory". Combine with `cloud_tpu_hbm_capacity_bytes` to derive the
  ratio.
- **Healthy:** stable across steps after warmup; no monotonic growth that
  would indicate a leak.
- **Unhealthy:** grows step by step (likely a Python-side accumulator); or
  approaches capacity (see ratio).
- **Related metrics:** `cloud_tpu_hbm_capacity_bytes`,
  `cloud_tpu_hbm_utilization_ratio`.
- **Safe labels:** `tpu_version`, `workload_name`.
- **Dangerous labels:** never `tensor_id` or `shard_id` on the metric;
  attach those to the log line.

### 3.7 `cloud_tpu_hbm_capacity_bytes`

- **Unit:** bytes.
- **Type:** gauge.
- **Produced by:** `src/tpu_versions/` catalog; constant per SKU.
- **Why it matters:** denominator of the utilisation ratio. Mostly used as
  a label / metadata in the dashboard.
- **Healthy:** equal to the catalog value for the SKU.
- **Unhealthy:** unexpected value → wrong SKU selected for this run.
- **Related metrics:** `cloud_tpu_hbm_utilization_ratio`.
- **Safe labels:** `tpu_version`.
- **Dangerous labels:** none.

### 3.8 `cloud_tpu_hbm_utilization_ratio`

- **Unit:** ratio in `[0, 1]`.
- **Type:** gauge.
- **Produced by:** `src/memory/hbm_sim.py`; equals
  `cloud_tpu_hbm_used_bytes / cloud_tpu_hbm_capacity_bytes`.
- **Why it matters:** the most actionable HBM signal. Bottleneck rule
  fires `high` at >0.85.
- **Healthy:** comfortable headroom (0.5–0.8).
- **Unhealthy:** >0.85 → close to OOM. Enable activation
  checkpointing, lower batch, shard params, or step down a generation in
  precision.
- **Related metrics:** `cloud_tpu_hbm_used_bytes`,
  `cloud_tpu_hbm_capacity_bytes`. The simulator also tracks
  `hbm.oom_events`, which feeds into the report.
- **Safe labels:** `tpu_version`, `workload_name`.
- **Dangerous labels:** never per-tensor IDs.

### 3.9 `cloud_tpu_collective_time_seconds`

- **Unit:** seconds.
- **Type:** gauge + histogram.
- **Produced by:** `src/sharding/collectives.py`. Sum of time spent in any
  cross-chip collective.
- **Why it matters:** large at small batch on multi-chip configs;
  bottleneck rule fires `high` at >30% of step time when `n_chips > 1`.
- **Healthy:** small fraction of step time on small chip counts; grows
  with chip count but should also be amortised by larger batch.
- **Unhealthy:** dominates step time → communication-bound. Consider
  shard-model instead of shard-data, larger global batch, or moving to a
  higher-bandwidth SKU.
- **Related metrics:** `cloud_tpu_all_reduce_time_seconds`,
  `cloud_tpu_all_gather_time_seconds`,
  `cloud_tpu_reduce_scatter_time_seconds`.
- **Safe labels:** `tpu_version`, `operation_type`.
- **Dangerous labels:** never `collective_id`.

### 3.10 `cloud_tpu_all_reduce_time_seconds`

- **Unit:** seconds.
- **Type:** gauge + histogram.
- **Produced by:** `src/sharding/collectives.py` `all_reduce` path.
- **Why it matters:** the most common gradient-sync collective; sensitive
  to global batch.
- **Healthy:** scales sub-linearly with chip count when batch grows.
- **Unhealthy:** super-linear growth with chip count → topology or
  saturation issue.
- **Related metrics:** see 3.9.
- **Safe labels:** `tpu_version`, `operation_type`.
- **Dangerous labels:** none.

### 3.11 `cloud_tpu_all_gather_time_seconds`

- **Unit:** seconds.
- **Type:** gauge + histogram.
- **Produced by:** `src/sharding/collectives.py` `all_gather` path.
- **Why it matters:** dominant in fully-sharded data parallelism (FSDP-
  style sharded params get gathered before use).
- **Healthy:** proportional to the size of the gathered tensors and chip
  count.
- **Unhealthy:** dominates when batch is too small to amortise.
- **Related metrics:** see 3.9.
- **Safe labels:** `tpu_version`, `operation_type`.
- **Dangerous labels:** none.

### 3.12 `cloud_tpu_reduce_scatter_time_seconds`

- **Unit:** seconds.
- **Type:** gauge + histogram.
- **Produced by:** `src/sharding/collectives.py` `reduce_scatter` path.
- **Why it matters:** the complement of `all_gather` in many sharded
  setups.
- **Healthy / unhealthy:** see 3.11.
- **Related metrics:** see 3.9.
- **Safe labels:** `tpu_version`, `operation_type`.
- **Dangerous labels:** none.

### 3.13 `cloud_tpu_matrix_unit_utilization_ratio`

- **Unit:** ratio in `[0, 1]`.
- **Type:** gauge.
- **Produced by:** sim approximation of MXU utilisation; real TPU gets it
  from XProf / framework profiler.
- **Why it matters:** the closest signal to "are you actually using the
  chip's matmul fabric". Indirect proxy for MFU (see
  `14_benchmarking_playbook.md`).
- **Healthy:** 0.4–0.7 for well-tuned dense workloads at large batch.
- **Unhealthy:** <0.2 → small ops, padding, or non-matmul-dominated
  workload.
- **Related metrics:** `cloud_tpu_tokens_per_second` is the throughput
  view of the same chip-fabric utilisation.
- **Safe labels:** `tpu_version`, `workload_name`.
- **Dangerous labels:** none.

### 3.14 `cloud_tpu_memory_stall_time_seconds`

- **Unit:** seconds.
- **Type:** gauge + histogram.
- **Produced by:** `src/memory/hbm_sim.py` when bandwidth is saturated.
- **Why it matters:** distinguishes "compute-bound" from "memory-bound" at
  the chip level. Compute-bound workloads can be tuned by precision /
  fusion; memory-bound need batch / layout / locality changes.
- **Healthy:** small fraction of step time.
- **Unhealthy:** material → check op layout, fusion boundaries, dtype.
- **Related metrics:** `cloud_tpu_matrix_unit_utilization_ratio`.
- **Safe labels:** `tpu_version`.
- **Dangerous labels:** never `hlo_op_id`.

### 3.15 `cloud_tpu_host_overhead_seconds`

- **Unit:** seconds.
- **Type:** gauge + histogram.
- **Produced by:** the simulator's host wrapper; real-TPU equivalent is
  framework-side Python overhead.
- **Why it matters:** time spent in Python / driver between TPU dispatches.
  Bottleneck rule fires `warn` at >15% of step time.
- **Healthy:** low single-digit percent.
- **Unhealthy:** material → move loss / metric reduction *into* the JIT'd
  step; avoid per-step Python work.
- **Related metrics:** `cloud_tpu_step_time_seconds`,
  `cloud_tpu_input_wait_time_seconds`.
- **Safe labels:** `framework`.
- **Dangerous labels:** none.

### 3.16 `cloud_tpu_checkpoint_time_seconds`

- **Unit:** seconds.
- **Type:** gauge.
- **Produced by:** the checkpoint-write path in `src/memory/` /
  `src/model_examples/`.
- **Why it matters:** a slow checkpoint blocks the training loop. Spikes
  here look like step-time jumps if you don't slice the dashboard right.
- **Healthy:** comparable to a few step times for the first save; smaller
  for subsequent.
- **Unhealthy:** seconds per step → checkpointing too often, or doing it
  on the hot path instead of async.
- **Related metrics:** `cloud_tpu_step_time_seconds` (the jumpy panel).
- **Safe labels:** `workload_name`, `framework`.
- **Dangerous labels:** never per-tensor IDs.

### 3.17 `cloud_tpu_tokens_per_second`

- **Unit:** tokens per second.
- **Type:** gauge.
- **Produced by:** the workload wrapper from
  `tokens_per_step / step_time_seconds`.
- **Why it matters:** the throughput KPI for sequence workloads. Together
  with cost-per-step gives cost-per-token, the unit of "how much does it
  cost to train one more token".
- **Healthy:** stable after warmup; scales with batch and chip count to a
  point.
- **Unhealthy:** falls when batch grows → another resource (collective,
  HBM, host) is the new bottleneck.
- **Related metrics:** `cloud_tpu_samples_per_second`,
  `cloud_tpu_cost_per_token`.
- **Safe labels:** `workload_name`, `framework`, `tpu_version`.
- **Dangerous labels:** never `step_id`.

### 3.18 `cloud_tpu_samples_per_second`

- **Unit:** samples per second.
- **Type:** gauge.
- **Produced by:** as 3.17 but `samples_per_step / step_time_seconds`.
- **Why it matters:** the throughput KPI for non-token workloads (vision,
  tabular).
- **Healthy / unhealthy:** as 3.17.
- **Related metrics:** `cloud_tpu_tokens_per_second`,
  `cloud_tpu_cost_per_step`.
- **Safe labels:** as 3.17.
- **Dangerous labels:** as 3.17.

### 3.19 `cloud_tpu_cost_per_step`

- **Unit:** USD per step (the lab assumes the unit supplied to
  `--hourly-usd-per-chip`).
- **Type:** gauge.
- **Produced by:** `src/common/cost.py`. Pricing is **never** hardcoded —
  you pass `--hourly-usd-per-chip` from
  https://cloud.google.com/tpu/pricing.
- **Why it matters:** the cost KPI for short runs. Adding it to dashboards
  next to step time makes the cost / throughput trade explicit.
- **Healthy:** stable after warmup; rises only when step time rises.
- **Unhealthy:** rises while throughput drops → both bills are getting
  worse; investigate the timing metrics.
- **Related metrics:** `cloud_tpu_cost_per_token`,
  `cloud_tpu_step_time_seconds`.
- **Safe labels:** `tpu_version`, `workload_name`.
- **Dangerous labels:** none.

### 3.20 `cloud_tpu_cost_per_token`

- **Unit:** USD per token.
- **Type:** gauge.
- **Produced by:** `cloud_tpu_cost_per_step / tokens_per_step`.
- **Why it matters:** the most portable cost KPI across hardware
  generations. The thing you should optimise.
- **Healthy:** decreases as you scale until a collective or HBM ceiling
  kicks in.
- **Unhealthy:** rises with chip count → you've crossed the scaling break
  point; pick a smaller config.
- **Related metrics:** see 3.19.
- **Safe labels:** as 3.19.
- **Dangerous labels:** none.

### 3.21 `cloud_tpu_error_count_total`

- **Unit:** dimensionless count.
- **Type:** counter.
- **Produced by:** every layer increments on caught exceptions and OOM
  events.
- **Why it matters:** the canary. A non-zero value means "did something
  fail" — combine with the JSONL stream to get the message.
- **Healthy:** zero.
- **Unhealthy:** any positive value; check the log stream filtered to the
  same `trace_id`.
- **Related metrics:** none directly.
- **Safe labels:** `error_type`, `layer`.
- **Dangerous labels:** never `trace_id` (use Loki for the join).

---

## 4. JSONL log schema

The JSONL writer is `src/observability/logger.py`. Each line is one event.
The schema is intentionally flat (one level of nesting at most) so Loki,
`jq`, and pandas all consume it cleanly. Per `JsonlLogger.log()` plus the
`TraceContext.as_log_fields()` injection, the fields are:

| Field             | Type           | Producer                                  | Meaning |
|-------------------|----------------|-------------------------------------------|---------|
| `timestamp`       | string (ISO-8601, ms precision) | `utc_now_iso()` in `src/common/trace.py` | Event wall-clock. |
| `app`             | string         | `JsonlLogger.app` (defaults to `cloud_tpu_lab`). | Loki stream label. |
| `level`           | string         | `log(... level=...)`; one of `INFO`, `WARN`, `ERROR`. | Filter for alerts. |
| `layer`           | string         | Caller. Examples: `xla`, `pjrt`, `hbm`, `sharding`, `input_pipeline`, `profiler`, `cost`. | Matches the `bottleneck_report.py` taxonomy. |
| `event`           | string         | Caller. E.g. `compile`, `execute`, `all_reduce`, `oom`, `step_start`. | Filterable in LogQL. |
| `message`         | string         | Caller. Human-readable text. | For grep; not used by code. |
| `metrics`         | object         | Caller. Map of metric name → numeric value. | Same names as section 3. |
| `trace_id`        | string         | `TraceContext.trace_id`. | Top-level join key. |
| `step_id`         | string or null | `TraceContext.step_id`. | Per-step join key. |
| `model_layer_id`  | string or null | `TraceContext.model_layer_id`. | Per model-graph layer. |
| `hlo_op_id`       | string or null | `TraceContext.hlo_op_id`. | Per HLO op (high cardinality). |
| `executable_id`   | string or null | `TraceContext.executable_id`. | Per compiled XLA module. |
| `device_event_id` | string or null | `TraceContext.device_event_id`. | Per device dispatch. |
| `tensor_id`       | string or null | `TraceContext.tensor_id`. | Per logical tensor. |
| `shard_id`        | string or null | `TraceContext.shard_id`. | Per shard if sharded. |
| `collective_id`   | string or null | `TraceContext.collective_id`. | Per collective op. |
| `<extra>`         | any            | `**extra` kwargs. Coerced via `_jsonable`. | Caller-specific extension. |

Notes:

- All the ID fields are **strings of the form `PREFIX-NNNN`** — see
  `src/common/trace.py`. They are designed to be grep-friendly, not UUID-
  random.
- The `metrics` object is the bridge to the Prometheus exporter — keys
  here are the metric names in section 3.
- Loki labels are usually just `app` and `level`; everything else stays in
  the body and is parsed at query time with `| json`.

Example line:

```json
{"timestamp":"2025-08-12T10:14:22.731Z","app":"cloud_tpu_lab","level":"INFO",
 "layer":"xla","event":"compile","message":"compile cache miss",
 "metrics":{"cloud_tpu_compile_time_seconds":1.83},
 "trace_id":"TRACE-0001","step_id":"STEP-0001","hlo_op_id":"HLO-0042",
 "executable_id":"EXE-0007"}
```

---

## 5. OpenTelemetry span names

The OTel emitter mirrors the pipeline layers in `src/`. Span names are
stable so dashboards / TraceQL queries don't break across runs.

| Span name             | Producer (src dir)       | Attributes (always safe) | Attributes (high-cardinality, OK in Tempo) |
|-----------------------|--------------------------|--------------------------|--------------------------------------------|
| `model.step`          | `model_examples/`        | `workload_name`, `framework` | `trace_id`, `step_id` |
| `model.layer`         | `model_examples/`        | `workload_name`          | `model_layer_id` |
| `xla.lower`           | `xla_sim/`               | `framework`              | `hlo_op_id` |
| `xla.compile`         | `xla_sim/`               | `framework`              | `hlo_op_id`, `executable_id` |
| `pjrt.execute`        | `pjrt_sim/`              | `framework`              | `executable_id` |
| `device.execute`      | `pjrt_sim/`              | `tpu_version`            | `device_event_id` |
| `hbm.alloc`           | `memory/`                | `tpu_version`            | `tensor_id`, `shard_id` |
| `hbm.free`            | `memory/`                | `tpu_version`            | `tensor_id`, `shard_id` |
| `sharding.partition`  | `sharding/`              | `tpu_version`            | `tensor_id`, `shard_id` |
| `collective.all_reduce`     | `sharding/`        | `tpu_version`, `operation_type` | `collective_id` |
| `collective.all_gather`     | `sharding/`        | `tpu_version`, `operation_type` | `collective_id` |
| `collective.reduce_scatter` | `sharding/`        | `tpu_version`, `operation_type` | `collective_id` |
| `input_pipeline.batch` | `input_pipeline/`       | `workload_name`          | `step_id` |
| `profiler.window`     | `profiling/`             | `workload_name`          | `trace_id` |
| `cost.estimate`       | `common/cost.py`         | `tpu_version`            | `trace_id` |

The parent / child structure is implicit in the OCT ID hierarchy in
`src/common/trace.py`:

```
trace_id
└── step_id
    ├── model_layer_id
    │   └── hlo_op_id
    │       └── executable_id
    │           └── device_event_id
    │               ├── tensor_id → shard_id
    │               └── collective_id
```

A `model.step` span will have `xla.compile`, `pjrt.execute`,
`device.execute`, and possibly `collective.*` spans as descendants for a
single step. Joining a Prometheus spike to its causing span is then a
`trace_id` lookup in Tempo.

---

## 6. Cross-references

- `12_observability_with_grafana_prometheus.md` — wiring these signals
  into the local stack.
- `14_benchmarking_playbook.md` — methodology for collecting and
  interpreting them.
- `src/observability/metrics.py` — the canonical `METRIC_NAMES` tuple.
- `src/observability/logger.py` — the JSONL schema source.
- `src/common/trace.py` — the correlation-ID hierarchy.
- `src/profiling/bottleneck_report.py` — thresholds used to flag findings.

---

## 7. Exercises / TODOs

1. For each metric in section 3, find the place in `src/` that produces it
   and add a code-comment cross-reference back to this dictionary.
2. Run `python3 examples/run_cpu_simulation_demo.py` and pull the JSONL
   for a single `trace_id`. Verify every field listed in section 4 is
   present somewhere in the stream.
3. Force a high `cloud_tpu_collective_time_seconds` by simulating
   `--chip-count 16` on `v5e`. Confirm the bottleneck report fires the
   `collective` finding.
4. Add one **new** metric — propose a name, justify it against the
   conventions in section 1, classify its labels as safe/dangerous, then
   wire it through `metrics.py` (without committing, just sketch the
   patch). The exercise is design discipline, not code.
5. Build a Grafana row that shows every metric in section 3 in a small
   grid, with thresholds aligned to the rules in
   `src/profiling/bottleneck_report.py`.
