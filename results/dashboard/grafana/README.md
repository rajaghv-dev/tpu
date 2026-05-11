# Grafana Dashboards for the TPU Benchmark

Importable Grafana 10+ dashboard JSONs. Each file in this directory is a
self-contained dashboard spec.

## Dashboards in this directory

| File                  | What it shows                                                    | Datasources       |
| --------------------- | ---------------------------------------------------------------- | ----------------- |
| `roofline.json`       | Arithmetic intensity vs achieved TFLOPs scatter, hardware overlays | Prometheus / Mimir |
| `mxu_heatmap.json`    | TPU MXU / memory bandwidth / memory utilisation                  | Google Cloud Monitoring |
| `latency_violins.json`| p50 / p95 / p99 latency per model                                | Prometheus / Mimir |
| `failures.json`       | Error category pie + failure rate per phase + recent FAILED logs | Prometheus, Loki  |
| `cost.json`           | USD per 1k samples by model/device + cost over time              | Prometheus / Mimir |

## How to import

1. Grafana UI -> Dashboards -> New -> Import.
2. Click "Upload JSON file" and select one of the files in this directory.
3. On the importer page, pick the datasources for each `${DS_*}` variable:
   - `${DS_PROMETHEUS}` -> your Prometheus or Mimir datasource.
   - `${DS_LOKI}` -> your Loki datasource (only `failures.json` needs this).
   - `${DS_GCM}` -> a Google Cloud Monitoring datasource (only `mxu_heatmap.json` needs this).
4. Click "Import".

## Required datasources

Configure these once in Grafana (Configuration -> Data sources -> Add data source):

- **Prometheus / Mimir** - scrapes/ingests metrics from the OTel collector.
- **Loki** - receives structured log lines emitted by `logging.getLogger("benchmark")`.
- **Tempo** - receives `benchmark.run` and `phase.<name>` spans (used for trace links from the dashboards even though no panel queries it directly yet).
- **Google Cloud Monitoring** - service account with `roles/monitoring.viewer` on the project that owns the TPU.

## Pipeline expectations

The dashboards are wired against the metric names emitted by `observe/otel_probe.py`:

- `benchmark.phase.duration_ms` -> Prometheus name `benchmark_phase_duration_ms_bucket`
- `benchmark.latency_ms`        -> `benchmark_latency_ms_bucket`
- `benchmark.throughput_samples_per_sec` -> `benchmark_throughput_samples_per_sec_*`
- `benchmark.experiment_cost_usd` -> `benchmark_experiment_cost_usd_bucket`
- `benchmark.errors_total`      -> `benchmark_errors_total`

These populate **only after** an OTel collector is running and
`observe/otel_probe.py` is wired into the harness. Until then panels render
empty - that is expected and not a bug.

`roofline.json` additionally reads `achieved_tflops` and
`arithmetic_intensity_flops_per_byte`, which Stage 3 will add. The hardware
roofline overlays (v5e-1 / RTX 4090 / B200) render from constant `vector()`
queries so the panel is useful even without measured data.

## Adding new dashboards

Place new `*.json` files in this directory and update the table above. Use the
same `${DS_PROMETHEUS}`, `${DS_LOKI}`, `${DS_GCM}` datasource variable names so
imports stay uniform.

## Exporting changes back to JSON

After editing a dashboard in Grafana:

1. Open the dashboard, click the gear (Dashboard settings).
2. Choose "JSON Model".
3. Copy the JSON and overwrite the corresponding file in this directory.
4. Commit. Strip any inlined `__inputs` / `__requires` / numeric `id` fields
   that Grafana adds on export so re-imports stay clean.
