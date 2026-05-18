# stackdriver-exporter — real TPU metrics from GCP Cloud Monitoring

This is the bridge between **GCP Cloud Monitoring** (formerly Stackdriver) and
the local Prometheus in this stack. It runs as the `stackdriver-exporter`
service in `docker-compose.yml`, pulls TPU infrastructure metrics from the
project `nellaiappar-001` once a minute, and re-exposes them at
`http://stackdriver-exporter:9255/metrics` (Prometheus then scrapes that).

Upstream: <https://github.com/prometheus-community/stackdriver_exporter>.
Image: `prometheuscommunity/stackdriver-exporter:v0.16.0` (pinned).

## Metrics it pulls

All metric type prefixes live under `tpu.googleapis.com/tpu/`:

| GCP metric type                                     | meaning                                          |
|-----------------------------------------------------|--------------------------------------------------|
| `tpu.googleapis.com/tpu/duty_cycle`                 | Fraction of time the TPU is processing (0..1).   |
| `tpu.googleapis.com/tpu/memory/usage`               | HBM bytes used per chip.                         |
| `tpu.googleapis.com/tpu/memory/total`               | HBM bytes total per chip.                        |
| `tpu.googleapis.com/tpu/cpu/utilization`            | Host CPU utilization on the TPU VM (0..1).       |
| `tpu.googleapis.com/tpu/network/received_bytes_count` | Bytes received (counter).                      |
| `tpu.googleapis.com/tpu/network/sent_bytes_count`   | Bytes sent (counter).                            |
| `tpu.googleapis.com/tpu/tensorcore/idle_duration`   | TensorCore idle time per interval.               |

These names are best-effort matched against
<https://cloud.google.com/monitoring/api/metrics_gcp>. If Google renames any,
update the `--monitoring.metrics-type-prefixes` flag in `docker-compose.yml`
and the panel queries in `grafana/dashboards/cloud_tpu_gcp_metrics.json`.

## Metric-name mangling rule (Prometheus side)

stackdriver-exporter generates Prometheus metric names as:

```
stackdriver_<monitored_resource_type>_<metric_type_with_non_word_chars_as_underscores>
```

For TPU metrics the monitored resource is `tpu_worker`, so:

- `tpu.googleapis.com/tpu/duty_cycle`
  -> `stackdriver_tpu_worker_tpu_googleapis_com_tpu_duty_cycle`
- `tpu.googleapis.com/tpu/memory/usage`
  -> `stackdriver_tpu_worker_tpu_googleapis_com_tpu_memory_usage`

Labels exposed include `project_id`, `zone`, `node_id` (TPU name),
`tpu_worker_id`, `container_name` (where applicable).

## Service-account setup (run on your laptop, not in this repo)

Required role: `roles/monitoring.viewer` on the project. That is the
minimum — do **not** grant editor/admin.

```bash
# 1. Create the service account.
gcloud iam service-accounts create cloud-tpu-lab-monitoring \
    --project=nellaiappar-001 \
    --display-name="cloud_tpu_lab local stackdriver-exporter"

# 2. Grant monitoring.viewer at project scope.
gcloud projects add-iam-policy-binding nellaiappar-001 \
    --member="serviceAccount:cloud-tpu-lab-monitoring@nellaiappar-001.iam.gserviceaccount.com" \
    --role="roles/monitoring.viewer"

# 3. Download a JSON key to the location the compose file expects.
mkdir -p ~/.config/gcloud
gcloud iam service-accounts keys create \
    ~/.config/gcloud/cloud_tpu_lab_sa.json \
    --iam-account=cloud-tpu-lab-monitoring@nellaiappar-001.iam.gserviceaccount.com
chmod 600 ~/.config/gcloud/cloud_tpu_lab_sa.json
```

Rotate the key periodically and delete it (`gcloud iam service-accounts keys delete`)
when you're done with the lab.

## Rate limits — why scrape_interval is 60s

Cloud Monitoring's `monitoring.timeSeries.list` quota is approximately
**6000 read calls / minute / project**. Each scrape of the exporter triggers
one API call per metric prefix per resource. With 7 prefixes and even a few
TPU nodes you can blow that quota in seconds if you scrape every 15s.

Rules of thumb:

- `scrape_interval: 60s` minimum (already pinned in `prometheus.yml`).
- `--monitoring.metrics-interval=1m` — match Cloud Monitoring's native
  aggregation window.
- `--monitoring.metrics-offset=1m` — Cloud Monitoring data is not real-time;
  the most recent minute is often empty.

## Verifying it works

After `docker compose up -d stackdriver-exporter`:

```bash
# 1. The exporter responds and is publishing the duty-cycle metric.
curl -s localhost:9255/metrics | grep tpu_duty_cycle

# 2. Prometheus is scraping it (look for the gcp_cloud_monitoring_tpu job
#    with health=up).
curl -s localhost:9090/api/v1/targets | python3 -m json.tool | \
    grep -E '"job"|"health"' | grep -A1 gcp_cloud_monitoring_tpu

# 3. Grafana shows the new dashboard at:
#    http://localhost:3000/d/cloud-tpu-gcp-metrics
```

If `curl localhost:9255/metrics` returns nothing, check the container logs —
the most common failure is the SA JSON not being mounted (compose will refuse
to start the service rather than mount a missing host path).
