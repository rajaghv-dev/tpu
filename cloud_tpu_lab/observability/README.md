# observability/

Optional local-observability stack for `cloud_tpu_lab`. **Cloud TPU only** —
nothing here targets Edge TPU, Coral, or mobile accelerators.

## Two modes

`cloud_tpu_lab` is deliberately usable with zero external dependencies. The
local stack here is purely additive: it consumes the artefacts the demo
already produces.

### Mode A — no-install (default)

Run the demo with stdlib only:

```bash
python3 examples/run_cpu_simulation_demo.py
```

Output lands in `cloud_tpu_lab/artifacts/`:

- `logs/run_<trace_id>.jsonl`     — structured events (Loki-compatible)
- `metrics/run_<trace_id>.csv`    — Prometheus-name metric stream
- `traces/run_<trace_id>.json`    — Chrome-trace JSON
- `reports/run_<trace_id>.md`     — Markdown bottleneck report

That is the whole pipeline. Nothing in this directory is required.

### Mode B — local stack (this directory)

If you want dashboards, alerts, and log search, bring up the stack:

```bash
cd cloud_tpu_lab/observability
docker compose up -d
```

That starts five services on a single bridge network:

| service     | image                       | port(s)            | URL                                      |
|-------------|-----------------------------|--------------------|------------------------------------------|
| Grafana     | grafana/grafana:10.4.0      | 3000               | http://localhost:3000  (admin / admin)   |
| Prometheus  | prom/prometheus:v2.51.0     | 9090               | http://localhost:9090                    |
| Loki        | grafana/loki:2.9.0          | 3100               | http://localhost:3100                    |
| Tempo       | grafana/tempo:2.4.0         | 3200, 4317, 4318   | http://localhost:3200                    |
| Promtail    | grafana/promtail:2.9.0      | 9080               | (sidecar — tails the JSONL logs)         |

Grafana auto-provisions:

- Datasources (Prometheus, Loki, Tempo) — from
  `grafana/provisioning/datasources/`.
- Dashboards — copied at boot from `grafana/dashboards/` via the provider
  defined in `grafana/provisioning/dashboards/dashboards.yml`.

Seven dashboards ship in this repo:

1. `cloud_tpu_overview.json`            — top-line panels
2. `cloud_tpu_compile_and_runtime.json` — XLA + PJRT
3. `cloud_tpu_hbm_memory.json`          — HBM + OOM
4. `cloud_tpu_sharding_and_collectives.json` — multi-chip
5. `cloud_tpu_input_pipeline.json`      — host + dataloader
6. `cloud_tpu_cost_performance.json`    — cost / $ panels
7. `cloud_tpu_debugging.json`           — errors, slow steps, recent log lines

Each dashboard exposes the template variables `workload_name`, `tpu_version`,
`framework`, `run_mode`.

## The Python exporter

Prometheus scrapes `host.docker.internal:9100`. That endpoint is served by:

```bash
python3 observability/exporters/cloud_tpu_metrics_exporter.py \
    --port 9100 \
    --log-path "cloud_tpu_lab/artifacts/logs/*.jsonl"
```

The exporter:

- Imports `prometheus_client` if installed; degrades to a stdlib HTTP
  endpoint if not.
- Defines exactly the metric names in
  `src/observability/metrics.py:METRIC_NAMES` — nothing else.
- Tails the JSONL log files and updates gauges / counters by `event`.
- Only attaches **SAFE_LABELS** (`workload_name`, `framework`, `tpu_version`,
  `run_mode`). Dangerous high-cardinality labels (`trace_id`, `step_id`,
  `hlo_op_id`, ...) stay in Loki where label cardinality is bounded
  per stream.

Cardinality discipline is enforced at the exporter level, not at the
scraper. If you fork the exporter and add a `trace_id` label, the
Prometheus index will blow up — please don't.

## Promtail

Promtail tails `cloud_tpu_lab/artifacts/logs/*.jsonl` (mounted into the
container at `/var/log/cloud_tpu_lab/`), parses each line as JSON, and
promotes a small set of low-cardinality fields to labels: `app`, `layer`,
`level`. `trace_id` is kept in the **log content**, not as a label — see the
comment block in `loki/promtail-config.yml`.

## GCP Cloud Monitoring (real TPU metrics)

The Python exporter above gives you **workload-level** metrics — what JAX sees
per step, sampled at whatever cadence your training loop emits events. That's
necessary but not sufficient: it can't tell you what the TPU silicon is
actually doing. For that we add a second source of truth.

| tier | source                          | how                              | port | cadence    | use for                                              |
|------|---------------------------------|----------------------------------|------|------------|------------------------------------------------------|
| 1    | JAX on the VM                   | JSONL -> Python exporter         | 9100 | per-step   | step time, compile time, HBM logical alloc, samples/s |
| 2    | GCP Cloud Monitoring            | `stackdriver-exporter` sidecar   | 9255 | per-minute | duty cycle, HBM physical usage, network IO, host CPU |

Tier 2 ships as the `stackdriver-exporter` service in `docker-compose.yml`. It
pulls from project `nellaiappar-001` and exposes Prometheus metrics named like
`stackdriver_tpu_worker_tpu_googleapis_com_tpu_duty_cycle`.

To enable tier 2 you need a Google service account with `roles/monitoring.viewer`
and its JSON key at `~/.config/gcloud/cloud_tpu_lab_sa.json`. The exact gcloud
commands are in [`stackdriver/README.md`](stackdriver/README.md).

The dashboard `cloud_tpu_gcp_metrics.json` (UID `cloud-tpu-gcp-metrics`,
title "Cloud TPU — GCP Cloud Monitoring") visualises tier-2 metrics:

- TPU Duty Cycle (%)
- HBM Used (GB) and HBM Utilization (%)
- TensorCore Idle (s)
- Network RX/TX (MB/s)
- Host CPU on TPU VM (%)

Both tiers feed the same Prometheus, so you can correlate workload events
(tier 1) with infrastructure reality (tier 2) on a single Grafana time axis.

## Cleanup

```bash
cd cloud_tpu_lab/observability
docker compose down -v
```

`-v` removes the named volumes, so the next `up` starts with an empty
Prometheus / Loki / Tempo. No state leaks across runs.

## What this stack will **not** do

- Talk to a real Cloud TPU API. The exporter only reads local JSONL files.
- Send your data anywhere. Everything is on `localhost`.
- Replace the no-install mode. Reports / CSVs / Markdown are still the
  canonical artefacts; this stack only adds interactive views.
