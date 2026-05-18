# Runbook — real TPU experiment

This is the operational walkthrough. The README has the overview; this
file is what you run line-by-line.

## 0. Pre-flight

```bash
gcloud auth login
gcloud config set project nellaiappar-001
gcloud services enable tpu.googleapis.com monitoring.googleapis.com
gcloud compute tpus accelerator-types list --zone=us-central2-b | grep v5litepod-1
```

If quota is missing, request via the GCP console (TPU API → Quotas).
v5e quota usually approves in 1–2 days.

## 1. Service account for Cloud Monitoring (one-time)

```bash
PROJECT=nellaiappar-001
SA=cloud-tpu-lab-monitoring
gcloud iam service-accounts create $SA --project=$PROJECT \
    --display-name="cloud_tpu_lab local stackdriver-exporter"
gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:$SA@$PROJECT.iam.gserviceaccount.com" \
    --role="roles/monitoring.viewer"
mkdir -p ~/.config/gcloud
gcloud iam service-accounts keys create \
    ~/.config/gcloud/cloud_tpu_lab_sa.json \
    --iam-account=$SA@$PROJECT.iam.gserviceaccount.com
chmod 600 ~/.config/gcloud/cloud_tpu_lab_sa.json
```

## 2. Local observability stack

```bash
cd cloud_tpu_lab/observability
docker compose up -d
docker compose ps                   # all services should be 'running'
curl -s localhost:9255/metrics | grep -m1 tpu_duty_cycle    # exporter alive
open http://localhost:3000          # Grafana — admin / admin
```

If `tpu_duty_cycle` isn't there yet, that's expected — it only appears
after a TPU VM has been live in the project for a few minutes.

## 3. The TPU VM (paid)

```bash
cd cloud_tpu_lab/gcp
./create_tpu_vm.sh                  # ~2 min; billing STARTS at READY
./install_jax_tpu.sh                # JAX + tensorboard-plugin-profile
```

## 4. The experiment

```bash
./run_real_demo.sh --n-steps 10 --hidden-size 512
```

What this does on the VM:

1. Sets `XLA_FLAGS=--xla_dump_to=$OUT/hlo --xla_dump_hlo_pass_re=.*`
2. Sets `TPU_STDERR_LOG_LEVEL=0` and `TPU_MIN_LOG_LEVEL=0`
3. Wraps the loop in `jax.profiler.start_trace($OUT/xprof)`
4. Times each step with `block_until_ready()`
5. Snapshots `jax.devices()[0].memory_stats()` 3× (post-init, post-compile, post-final)
6. Emits OCT JSONL + Prometheus CSV + Chrome-trace JSON + Markdown report

Pull artifacts back:

```bash
./collect_artifacts.sh
# → artifacts/from_vm/<RUN_TAG>/
```

## 5. Look at the data

### Local files

```
artifacts/from_vm/<RUN_TAG>/
├── run_TRACE-XXXX.md      ← open this first (human report)
├── run_TRACE-XXXX.jsonl   ← every OCT event
├── run_TRACE-XXXX.csv     ← Prometheus-shape per-step metrics
├── run_TRACE-XXXX.json    ← Chrome trace; drag into ui.perfetto.dev
├── hlo/                   ← every HLO pass dump
└── xprof/                 ← jax.profiler output (load in TensorBoard)
```

### Grafana panels (http://localhost:3000)

| Dashboard | Use it for |
|---|---|
| `cloud_tpu_overview` | Top-line — step time, throughput, $/step |
| `cloud_tpu_compile_and_runtime` | XLA compile (cold step dominance) |
| `cloud_tpu_hbm_memory` | Workload-level HBM accounting |
| `cloud_tpu_gcp_metrics` | **Authoritative** GCP infrastructure metrics: duty cycle, HBM from firmware, network, host CPU |
| `cloud_tpu_debugging` | Errors, slow steps, recent Loki log lines |

The two HBM dashboards should **agree**. If they don't, trust the GCP
one — that's coming from libtpu, not from JAX.

### XProf in TensorBoard

```bash
pip install tensorboard tensorboard-plugin-profile
tensorboard --logdir artifacts/from_vm/<RUN_TAG>/xprof
open http://localhost:6006/#profile
```

### HLO dumps

Look at the largest `.txt` files first — those are the most-modified
modules. `module_*.after_optimizations.hlo.pb.txt` is the final IR.
Compare with `module_*.before_optimizations.hlo.pb.txt` to see what
XLA did.

## 6. Tear down (mandatory)

```bash
./delete_tpu_vm.sh                  # stops VM billing
cd ../observability
docker compose down -v              # optional; clears Prometheus/Loki/Tempo state
```

## Knobs that are interesting to vary

| Knob | What changes |
|---|---|
| `--hidden-size 4096` | matmul becomes HBM-bandwidth-bound; watch `duty_cycle` drop |
| `--precision fp32` | 2× memory, 2× bytes/op; useful contrast vs bf16 |
| `--n-steps 100` | smoother steady-state curve in Grafana |

To add a new workload (e.g. real MLP training), drop a builder into
`_BUILDERS` in `examples/run_jax_real_tpu.py`. The instrumentation is
already wired — just write the `step_fn` + `init_state_fn`.
