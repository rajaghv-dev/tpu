# cloud_tpu_lab

Hands-on Google Cloud TPU learning lab. Real TPU only — no simulation.

**Goal:** learn how a Cloud TPU actually behaves — XLA HLO, XProf,
HBM, ICI, duty cycle — by running a small instrumented workload on
real hardware and watching the metrics flow into Grafana.

## What this lab does

1. Provisions a Cloud TPU VM (default: v5e single chip).
2. Runs a small JAX workload (matmul) on the TPU under
   `jax.profiler.trace()` with HLO dumps enabled.
3. Emits per-step JSONL + CSV in a stable schema (the OCT model).
4. Pulls TPU infrastructure metrics from GCP Cloud Monitoring
   (duty cycle, HBM, network, host CPU).
5. Visualizes both streams in Grafana on `localhost:3000`.
6. Tears down the VM so it stops billing.

The workload is deliberately tiny — one jitted `N×N @ N×N` matmul — so
every byte of HLO and every microsecond of XProf is readable. To grow,
add a builder to `_BUILDERS` in `examples/run_jax_real_tpu.py`.

## Prerequisites

- A GCP project with **TPU API enabled** and quota for the chosen
  accelerator (default `v5litepod-1` in `us-central2-b`).
- `gcloud` CLI installed and authenticated (`gcloud auth login`).
- Docker + `docker compose` (for the local observability stack).
- `roles/iam.serviceAccountAdmin` on the project (for the Cloud
  Monitoring service account — one-time setup).

Defaults assume project `nellaiappar-001`. Override via
`export PROJECT_ID=...` before running any `gcp/` script.

## One-time setup

```bash
# 1) Service account for the local stackdriver-exporter (one-time)
gcloud iam service-accounts create cloud-tpu-lab-monitoring \
    --project=$PROJECT_ID \
    --display-name="cloud_tpu_lab local stackdriver-exporter"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cloud-tpu-lab-monitoring@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/monitoring.viewer"
mkdir -p ~/.config/gcloud
gcloud iam service-accounts keys create \
    ~/.config/gcloud/cloud_tpu_lab_sa.json \
    --iam-account=cloud-tpu-lab-monitoring@$PROJECT_ID.iam.gserviceaccount.com
chmod 600 ~/.config/gcloud/cloud_tpu_lab_sa.json

# 2) Bring up the local observability stack
cd observability
docker compose up -d
# Grafana: http://localhost:3000 (admin / admin)
```

See `observability/stackdriver/README.md` for the full SA runbook.

## Run a real TPU experiment

> Every `gcp/*.sh` script that creates a paid resource prints a PAID
> warning and prompts for confirmation. Pass `--yes` to skip the prompt
> when you're sure.

```bash
cd gcp

# Create the VM (paid; billing starts when state=READY)
./create_tpu_vm.sh

# Install JAX + tensorboard-plugin-profile on the VM
./install_jax_tpu.sh

# Run the matmul workload on the TPU; writes HLO + XProf + JSONL to the VM
./run_real_demo.sh --n-steps 10 --hidden-size 512

# Pull artifacts back to ./artifacts/from_vm/<RUN_TAG>/
./collect_artifacts.sh

# Stop billing
./delete_tpu_vm.sh
```

Knobs on the run:

```bash
./run_real_demo.sh --n-steps 50 --hidden-size 1024 --precision bf16
./run_real_demo.sh --n-steps 10 --hidden-size 4096 --precision fp32
```

## What lands in Grafana

Two metric tiers, both visible at `http://localhost:3000`:

| Tier              | Source                          | Cadence | Dashboards                                  |
|-------------------|---------------------------------|---------|---------------------------------------------|
| Workload-level    | JAX → JSONL → Python exporter @ `:9100` | per-step | `cloud_tpu_overview`, `compile_and_runtime`, `hbm_memory`, `cost_performance`, `debugging` |
| Infrastructure    | GCP Cloud Monitoring → `stackdriver-exporter` @ `:9255` | 60s | `cloud_tpu_gcp_metrics` |

The workload tier has fine-grained per-step events with the OCT
correlation spine (`trace_id → step_id → executable_id → ...`).
The infrastructure tier has the authoritative numbers GCP reports —
duty cycle, real HBM usage from the firmware, network IO.

## What lands locally (per run)

In `artifacts/from_vm/<RUN_TAG>/`:

- `run_<trace_id>.jsonl` — every OCT event with correlation IDs
- `run_<trace_id>.csv` — Prometheus-shape metric stream
- `run_<trace_id>.json` — Chrome/Perfetto trace (drop into [ui.perfetto.dev](https://ui.perfetto.dev))
- `run_<trace_id>.md` — human-readable run report (open this first)
- `hlo/` — XLA HLO dumps, every pass (set by `XLA_FLAGS`)
- `xprof/` — `jax.profiler` output (view in TensorBoard's profile plugin)

## Repo layout

```
cloud_tpu_lab/
├── docs/         Learning material (Cloud TPU architecture, XLA, sharding, profiling)
├── gcp/          Cloud TPU VM provisioning + run + cleanup scripts
│   ├── _env.sh                 PROJECT_ID, ZONE, TPU_NAME, ACCELERATOR_TYPE
│   ├── create_tpu_vm.sh        provision the VM (paid)
│   ├── install_jax_tpu.sh      JAX + tensorboard-plugin-profile on the VM
│   ├── run_real_demo.sh        the flagship run
│   ├── collect_artifacts.sh    pull artifacts back
│   ├── delete_tpu_vm.sh        stop billing
│   └── start_profiler.sh       live XProf server + port-forward
├── examples/
│   └── run_jax_real_tpu.py     real TPU JAX runner (matmul; add more via _BUILDERS)
├── src/
│   ├── common/                 trace IDs, configs, cost
│   ├── observability/          JSONL logger, METRIC_NAMES, Markdown report
│   ├── profiling/              trace analyzer, bottleneck report
│   └── tpu_versions/           catalog of v4/v5e/v5p/v6e for cost + roofline reference
├── observability/              Docker stack (Prometheus + Grafana + Loki + Tempo)
│   ├── docker-compose.yml
│   ├── grafana/dashboards/     7 prebuilt dashboards including cloud_tpu_gcp_metrics
│   ├── stackdriver/README.md   service-account runbook for GCP Cloud Monitoring
│   └── exporters/              host-side Python metric exporter
├── tests/                      Pytest smoke tests for surviving modules
└── artifacts/                  Per-run outputs (created at runtime)
```

## Cost safety

- Every `gcp/*.sh` script that creates a paid resource has a matching
  cleanup. Idle TPU VMs accrue cost — **always run `delete_tpu_vm.sh`
  when done**.
- Pricing is never hardcoded. Pass `--hourly-usd-per-chip` and look up
  the current rate at https://cloud.google.com/tpu/pricing.
- GCP Cloud Monitoring API is **read-only and free at this volume**
  (well under the 6000 reads/minute quota).

## Tests

```bash
cd /Users/AI-Test/raja/tpu
python3 -m pytest cloud_tpu_lab/tests/
```

13 tests, all surviving modules — no TPU needed.

## License

MIT — see `LICENSE` (or the repo metadata).
