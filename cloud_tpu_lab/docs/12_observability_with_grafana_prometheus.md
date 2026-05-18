> **Note:** this doc predates the real-TPU pivot. References to `src/xla_sim/`, `src/pjrt_sim/`, `src/sharding/`, `src/memory/`, `src/input_pipeline/`, and `examples/run_cpu_simulation_demo.py` are historical — those modules were removed. The TPU architecture / XLA / observability concepts below are still accurate. Current run flow lives in [README.md](../README.md) and [16_runbook_real_tpu.md](16_runbook_real_tpu.md).

# 12 — Observability with Grafana / Prometheus / Loki / Tempo

> **Learning goal:** turn the OCT signals emitted by `cloud_tpu_lab`
> (metrics, structured logs, traces) into something you can browse, query,
> and alert on. Understand which signal goes into which store, how
> cardinality decisions affect cost, and how to wire a fresh laptop or a
> Cloud TPU VM into the lab's local Grafana stack.

`cloud_tpu_lab` ships **two observability modes**. Both work with the same
producer code; only the consumer changes.

| Mode             | Install cost          | Use when                                |
|------------------|-----------------------|------------------------------------------|
| **No-install**   | Python stdlib only    | First runs, CI, Colab, demo notebooks   |
| **Local stack**  | `docker compose up`   | Real benchmarking, dashboard exploration |

You can start in no-install mode, ship runs to disk, and only fire up the
local stack when you actually want to slice and dice.

---

## 1. No-install mode

This is the default and it is the only mode used by `make smoke`.

The producers are:

- `src/observability/logger.py` — buffered JSONL writer.
- `src/observability/metrics.py` — CSV per-step writer + name registry that
  is shared with the Prometheus exporter.
- `src/observability/report.py` (and the bottleneck rules in
  `src/profiling/bottleneck_report.py`) — Markdown report.

The consumers are:

- `jq` / `grep` over `artifacts/logs/run_<trace_id>.jsonl`.
- `pandas` / `awk` / a spreadsheet over `artifacts/metrics/run_<trace_id>.csv`.
- A Markdown viewer (your editor, GitHub) over
  `artifacts/reports/run_<trace_id>.md`.
- The Chrome trace viewer (`chrome://tracing` or `https://ui.perfetto.dev`)
  over `artifacts/traces/run_<trace_id>.json`.

Quick start:

```bash
python3 examples/run_cpu_simulation_demo.py
ls artifacts/{logs,metrics,traces,reports}
```

Slice the JSONL with `jq` — every line is one event, flat schema:

```bash
# Every HBM event for a trace, sorted by step.
jq -c 'select(.layer=="hbm")' artifacts/logs/run_TRACE-0001.jsonl

# All warnings.
jq -c 'select(.level=="WARN")' artifacts/logs/run_TRACE-0001.jsonl

# Per-step compile time.
jq -r 'select(.event=="compile") | [.step_id, .metrics.cloud_tpu_compile_time_seconds] | @tsv' \
  artifacts/logs/run_TRACE-0001.jsonl
```

The JSONL schema is described field-by-field in
`13_oct_metrics_dictionary.md`. The producer is
`src/observability/logger.py`.

---

## 2. Local stack mode — what's in the box

The `observability/` directory contains a `docker-compose.yml` that brings
up four services and a sidecar collector:

| Service     | Role in OCT                                                       |
|-------------|-------------------------------------------------------------------|
| Prometheus  | Stores numeric **metrics** scraped from the lab's exporter.       |
| Grafana     | Dashboards / explore UI over Prometheus, Loki, Tempo.             |
| Loki        | Stores the **structured logs** (the JSONL stream).                |
| Tempo       | Stores **distributed traces** (OpenTelemetry spans).              |
| Promtail    | Tails `artifacts/logs/*.jsonl` and forwards lines to Loki.        |

The split is deliberate:

- **Numeric, low-cardinality time series → Prometheus.** Step time,
  compile time, HBM ratio, throughput.
- **High-cardinality text / IDs → Loki.** trace_id, hlo_op_id, tensor_id,
  shard_id all go here as log labels or fields. Putting them in Prometheus
  would explode the index. See section 4 below.
- **Span-shaped causal data → Tempo.** Parent / child spans across the
  simulated pipeline.

Bring it up:

```bash
cd cloud_tpu_lab/observability
docker compose up -d

# Inspect.
docker compose ps
docker compose logs -f prometheus grafana loki tempo
```

Once everything is healthy:

- Grafana: http://localhost:3000  (default admin / admin → change it)
- Prometheus: http://localhost:9090
- Loki: http://localhost:3100  (queried via Grafana, not by hand)
- Tempo: http://localhost:3200  (queried via Grafana)

Tear it down:

```bash
docker compose down              # stop containers, keep volumes
docker compose down -v           # also drop the data volumes
```

The local stack itself is free — no GCP. But if you are also running on a
Cloud TPU VM, the VM is still PAID. The stack is on your laptop; the TPU is
in the cloud. Cleanup of the TPU is in `11_cleanup_and_cost_safety.md`.

---

## 3. Wiring the Prometheus exporter

The lab's exporter lives in `observability/exporters/`. It re-uses the
canonical metric names from `src/observability/metrics.py` so the names you
see in CSV are the names you see in Prometheus.

There are two integration patterns.

### 3a. Inside the lab process (pull-based)

The exporter starts a local HTTP server. Prometheus scrapes it.

```python
# Inside your training / simulation script.
from prometheus_client import start_http_server
from observability.exporters.cloud_tpu_metrics_exporter import register_all_metrics

register_all_metrics()
start_http_server(9464)         # /metrics on :9464
# ... run training / simulation; the exporter updates the metrics inline.
```

`prometheus.yml` scrape config:

```yaml
scrape_configs:
  - job_name: cloud_tpu_lab
    static_configs:
      - targets: ['host.docker.internal:9464']  # Linux: 'localhost:9464'
    scrape_interval: 5s
```

### 3b. Off-process (CSV-replay)

If the producing job has already finished — common with TPU VM runs that
you `scp` back to your laptop — replay the CSV through the exporter:

```bash
python3 -m observability.exporters.replay \
  --csv artifacts/metrics/run_TRACE-0001.csv \
  --port 9464
```

Prometheus scrapes the same endpoint. The replay walks the rows at a fixed
rate so the dashboard shows a meaningful time axis.

### 3c. Sanity-check the exporter

```bash
curl -s localhost:9464/metrics | grep cloud_tpu_

# Or in Prometheus's expression browser:
#   cloud_tpu_step_time_seconds
#   rate(cloud_tpu_recompile_count_total[5m])
```

You should see one line per metric in `METRIC_NAMES` from
`src/observability/metrics.py`.

---

## 4. Cardinality discipline — the most important section

The single biggest way to break a Prometheus stack is to put a
high-cardinality string in a label. The lab's metric module enumerates the
safe and dangerous sets explicitly:

```python
# src/observability/metrics.py
SAFE_LABELS      = ("workload_name", "framework", "tpu_version",
                    "run_mode", "operation_type", "error_type")
DANGEROUS_LABELS = ("trace_id", "step_id", "hlo_op_id",
                    "executable_id", "tensor_id", "shard_id")
```

Rules of thumb:

- **Safe labels:** bounded sets you can enumerate up front. Worst-case
  cardinality is a few dozen values.
- **Dangerous labels:** unbounded or per-step. Cardinality grows linearly
  with run length. Never put these on Prometheus metrics.

What to do with the dangerous values:

- Put them on **JSONL log lines** for Loki — Loki is indexed by *stream*
  labels (a few low-cardinality fields) but **content** can be arbitrary.
- Put them on **Tempo spans** as attributes — Tempo is built for span-level
  cardinality.

A small worked example. The compile event is a good case study:

```
metric in Prometheus:     cloud_tpu_compile_time_seconds
                          labels: tpu_version="v5e", framework="jax"

log line in Loki (same event):
                          { trace_id: TRACE-0001, hlo_op_id: HLO-0042,
                            executable_id: EXE-0007,
                            cloud_tpu_compile_time_seconds: 1.83 }

span in Tempo (same event):
                          name=xla.compile
                          attrs={ hlo_op_id: HLO-0042,
                                  executable_id: EXE-0007, ... }
```

Three stores; one event; one `trace_id` joins them.

If you find yourself wanting to label a Prometheus metric with `trace_id`,
stop and add a Loki query instead. The lab is opinionated about this; the
exporter will refuse to register a dangerous label.

---

## 5. Sample Grafana setup

The `observability/grafana/provisioning/` and
`observability/grafana/dashboards/` directories are auto-loaded by the
compose stack. The intended dashboards are:

- **TPU overview** — `cloud_tpu_step_time_seconds`,
  `cloud_tpu_tokens_per_second`, HBM util, cost-per-step.
- **Bottlenecks** — fractional breakdown by `operation_type`: input,
  compile, collective, host. Mirrors the categories in
  `src/profiling/bottleneck_report.py`.
- **Cost** — `cloud_tpu_cost_per_step` and a sum panel for total run
  USD. Cost is unit-aware via `--hourly-usd-per-chip`.

Add a new dashboard manually:

1. Grafana → Dashboards → New.
2. Pick the Prometheus data source.
3. Query the canonical metric name. Don't invent your own.
4. Save it as JSON to
   `observability/grafana/dashboards/<your-dashboard>.json`. The compose
   stack will re-provision it on the next restart.

---

## 6. Sample Loki queries

LogQL queries are run from Grafana → Explore → Loki. The Promtail config
labels each stream with at minimum `app="cloud_tpu_lab"`. Higher-cardinality
fields (`trace_id`, `hlo_op_id`, ...) live inside the JSON, not as stream
labels.

```logql
# All logs for one run.
{app="cloud_tpu_lab"} | json | trace_id="TRACE-0001"

# Only warnings or errors.
{app="cloud_tpu_lab"} | json | level=~"WARN|ERROR"

# Compile events with a high compile time.
{app="cloud_tpu_lab"} | json
  | event="compile"
  | metrics_cloud_tpu_compile_time_seconds > 1.0

# Group by hlo_op_id (cardinality is fine inside Loki).
sum by (hlo_op_id) (
  count_over_time(
    ({app="cloud_tpu_lab"} | json | event="compile")[1h]
  )
)
```

Tip: the JSONL writer emits `metrics` as a nested object. The `| json`
parser flattens it into fields like `metrics_cloud_tpu_compile_time_seconds`.

---

## 7. Sample Tempo / trace queries

Tempo accepts TraceQL. The lab produces spans with names that match the
pipeline layers in `src/`:

- `model.layer` — per-layer forward.
- `xla.lower` — HLO lowering.
- `xla.compile` — compile cache lookup + miss path.
- `pjrt.execute` — runtime dispatch.
- `device.execute` — simulated device-side execution.
- `collective.<op>` — `all_reduce`, `all_gather`, `reduce_scatter`.
- `hbm.alloc` / `hbm.free`.
- `input_pipeline.batch`.

Useful queries:

```traceql
# Every trace where a compile span was >2s.
{ name="xla.compile" && duration > 2s }

# Traces involving a specific tensor id.
{ .tensor_id = "TENSOR-0123" }

# Find collective-heavy traces.
{ name=~"collective.*" && duration > 100ms }
```

Service graph view in Grafana → Explore → Tempo gives a useful overview
of where time is spent across pipeline layers without picking a single
trace.

---

## 8. Joining the three stores by trace_id

The whole point of the OCT spine (`src/common/trace.py`) is that one ID
unlocks all three stores:

1. In Grafana Explore, find the suspicious data point on a **Prometheus**
   panel (e.g. a spike in `cloud_tpu_step_time_seconds`).
2. Click "View related logs" or paste the timestamp into Loki:
   `{app="cloud_tpu_lab"} | json | step_id="STEP-0007"`. Grab the
   `trace_id` from the result.
3. Paste that `trace_id` into Tempo to see the span tree for that step.

`src/traceability/` documents the join rules used by the offline equivalent
of this workflow (`make report` over a downloaded artefact tree).

---

## 9. Wiring a Cloud TPU VM into the local stack

If you are running on a real TPU VM (see
`10_cloud_tpu_setup_playbook.md`), there are two integration patterns:

- **Pull artefacts to laptop, replay locally.** Simpler. Use the section
  3b CSV-replay exporter. No additional cost beyond the existing
  `scp`.
- **Push from the VM to your laptop's Loki/Tempo over a tunnel.** More
  realistic but adds an SSH tunnel surface area. Use
  `gcloud compute tpus tpu-vm ssh ... -- -L 3100:localhost:3100`. The VM
  must run a Promtail/OTel collector pointed at `localhost`.

For most learning sessions, "pull artefacts, replay locally" is the right
default — it decouples cleanup of the (PAID) TPU VM from the analysis
phase.

---

## 10. Troubleshooting

| Symptom                                          | Likely cause                                         | Fix |
|--------------------------------------------------|------------------------------------------------------|-----|
| `/metrics` 404                                    | Exporter didn't start                                | Check `register_all_metrics()` and `start_http_server` were both called. |
| Prometheus says `target down`                    | Wrong host on Linux Docker                           | Use `localhost:9464` from the host network, not `host.docker.internal`. |
| Grafana panels are empty                         | No scrapes in the time range                         | Widen the range; confirm Prometheus has datapoints with a direct query. |
| Loki query returns nothing                       | Promtail not tailing your `artifacts/logs/`          | Check `promtail-config.yml`'s `__path__`; restart promtail. |
| Tempo trace view says "no spans"                 | The lab process didn't emit OTel spans               | Confirm the OTel exporter env vars are set and pointing at Tempo. |
| Prometheus index OOM                             | A label took on too many values                      | Re-read section 4. Remove the offending label. |

---

## 11. Exercises / TODOs

1. Boot the stack with `docker compose up -d`, run
   `python3 examples/run_cpu_simulation_demo.py`, and confirm
   `cloud_tpu_step_time_seconds` is plotted on the TPU overview dashboard.
2. Deliberately add `trace_id` as a label to a metric in a temporary copy
   of the exporter. Observe the cardinality explosion in
   `prometheus_tsdb_head_series`. Revert.
3. Write a LogQL query that finds the slowest 5 compile events across the
   last 10 simulator runs and outputs `trace_id`, `hlo_op_id`,
   `cloud_tpu_compile_time_seconds`.
4. Add a new Grafana panel showing `cloud_tpu_hbm_utilization_ratio` with
   a red threshold at 0.85 (matches the
   `bottleneck_report.py` rule). Save the dashboard JSON into
   `observability/grafana/dashboards/`.
5. From a real Cloud TPU VM run, `scp` the artefact tree and use the
   replay exporter to populate the same dashboards. Compare the shape of
   `cloud_tpu_step_time_seconds` to the simulator's prediction for the same
   SKU.

---

## 12. Cross-references

- `13_oct_metrics_dictionary.md` — definition of every metric and JSONL
  field.
- `14_benchmarking_playbook.md` — what you should be looking at on these
  dashboards.
- `src/observability/metrics.py` — `METRIC_NAMES`, `SAFE_LABELS`,
  `DANGEROUS_LABELS`.
- `src/observability/logger.py` — JSONL writer the Promtail config tails.
- `src/profiling/bottleneck_report.py` — the rules that map metric
  patterns to recommended fixes.
