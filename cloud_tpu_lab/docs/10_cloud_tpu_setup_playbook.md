# 10 — Cloud TPU VM Setup Playbook

> **Learning goal:** stand up a single Cloud TPU VM, install one of the three
> supported frameworks (JAX / PyTorch-XLA / TensorFlow), run a tiny benchmark,
> pull the artefacts back to your laptop, and tear the VM down before it
> bills another hour. The point is to make every cost-bearing step
> deliberate and reversible.

This document is **Cloud TPU only**. There is no discussion of Edge / Coral /
mobile TPUs anywhere in `cloud_tpu_lab`. If you want pure simulation, use
`examples/run_cpu_simulation_demo.py` and skip this file entirely.

Every command that creates or runs a billable resource is prefixed with
`# PAID:` inside the code block. Treat those lines like land mines — read the
neighbouring lines before pressing Enter.

---

## 0. Pricing — do this before anything else

Never hardcode pricing in scripts or notes. Look up the rate per chip-hour
for the TPU version you intend to use:

- https://cloud.google.com/tpu/pricing

Then export it once at the top of your shell session so every cost
calculation in this lab uses it:

```bash
export TPU_HOURLY_USD_PER_CHIP="<lookup value>"   # e.g. for v5e
```

The simulator and the cost report both accept `--hourly-usd-per-chip`. If
you forget to pass it, you get a clearly-flagged "cost unknown" result, not a
silently wrong number.

Cross-reference: see `11_cleanup_and_cost_safety.md` for billing alerts,
egress traps, and orphan-resource patterns.

---

## 1. One-time GCP setup (free)

These steps do not create billable resources, but they unlock the API.

```bash
# Install / update the gcloud CLI if you don't have it.
# https://cloud.google.com/sdk/docs/install

gcloud --version
gcloud auth login
gcloud auth application-default login
```

Pick a project. Either use one you already pay for, or create a new one
named so you can spot it in the billing dashboard:

```bash
gcloud projects list
gcloud config set project <YOUR_PROJECT_ID>
```

Enable the TPU + Compute APIs (idempotent, free):

```bash
gcloud services enable tpu.googleapis.com compute.googleapis.com
```

If you have a billing budget alert configured, this is the moment to
double-check it. Open the Cloud Console → Billing → Budgets & Alerts.

---

## 2. Shell variables — define once, reuse everywhere

Put these at the top of every session. They make later `gcloud` commands
copy-paste-safe and they make cleanup trivial.

```bash
export PROJECT_ID="<your-gcp-project>"
export REGION="us-central1"          # pick a region that has the TPU SKU you want
export ZONE="us-central1-a"          # zone within the region
export TPU_NAME="cloud-tpu-lab-vm"   # the VM resource name (your handle for cleanup)
export TPU_TYPE="v5litepod-1"        # SKU + topology, e.g. v5litepod-1 / v5p-8 / v6e-1
export RUNTIME_VERSION="v2-alpha-tpuv5-lite"   # match your TPU_TYPE
export NETWORK="default"
export SUBNETWORK="default"
```

Notes on the variables:

- `REGION` / `ZONE`: not every zone has every TPU SKU. If the create command
  fails with "no allocation", try a different zone or a different
  `TPU_TYPE`.
- `TPU_TYPE`: the SKU + topology. For learning, prefer the smallest
  topology of the cheapest version. `v5litepod-1` is the canonical
  "single-chip-cost" starting point; `v6e-1` is its newer sibling.
- `RUNTIME_VERSION`: the disk image for the TPU VM. Each SKU has a small
  set of valid runtimes. See:
  https://cloud.google.com/tpu/docs/runtimes

If anything in this list is wrong, the create call in section 3 will fail
*before* any money is spent — that's by design.

---

## 3. Create the TPU VM — PAID

The moment this command returns success, billing starts. The matching
delete command is in section 9. Bind them together in your brain.

```bash
# PAID: starts billing the instant it returns success.
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --accelerator-type="$TPU_TYPE" \
  --version="$RUNTIME_VERSION" \
  --network="$NETWORK" \
  --subnetwork="$SUBNETWORK"
```

Immediately verify the VM exists and note its state:

```bash
# Read-only — free.
gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" \
  --zone="$ZONE"

gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

If `describe` shows the VM as `READY`, the chips are reserved and you are
paying for them whether or not you log in.

If you saw an error like "request was preempted" or a quota error, the VM
was *not* created — billing has not started. You can retry without deleting
anything. If you are unsure, run `tpu-vm list` to confirm.

### 3a. Reservations and preemptible / spot variants

For learning runs prefer the **on-demand** SKU (the default). Preemptible
("spot") TPU is cheaper but can be reclaimed mid-run, which destroys the
benchmark methodology in `14_benchmarking_playbook.md`. If you do need
spot, pass `--preemptible` (older SKUs) or `--spot` (newer) — but accept
that your run may be killed and re-billed in a new SKU class.

---

## 4. SSH in — free

SSH itself does not increment billing — but every minute the VM exists
*does*. So once you SSH in, work briskly.

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

From inside the VM, sanity-check that the TPU runtime sees the chips:

```bash
# On the TPU VM.
ls /dev/accel*       # PJRT plugin presents the chip(s) here on most runtimes
python3 -c "import os; print(os.uname())"
```

If `/dev/accel*` is empty *and* you intend to use a framework that uses
direct device access, you have the wrong `RUNTIME_VERSION` for your
`TPU_TYPE`. Most modern setups use PJRT and don't require this device file —
the framework check below is the real test.

---

## 5. Install ONE framework — free (CPU/disk usage only)

Pick exactly one for the first run. Mixing JAX and PyTorch-XLA on the same
VM is possible but the wheels and runtime envs conflict in subtle ways and
you'll spend the lab's whole budget debugging them. The point is to make
the TPU light up, not to integrate three stacks.

### 5a. JAX (recommended starting point)

```bash
pip install --upgrade pip
pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

python3 -c "
import jax
print('jax version:', jax.__version__)
print('devices:', jax.devices())
print('local_device_count:', jax.local_device_count())
"
```

A healthy result lists `TpuDevice(...)` entries. `[CpuDevice(id=0)]` means
the TPU runtime never attached — recheck `RUNTIME_VERSION` and `TPU_TYPE`.

### 5b. PyTorch-XLA

```bash
pip install --upgrade pip
pip install torch~=2.4.0 torch_xla[tpu]~=2.4.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html

python3 -c "
import torch_xla.core.xla_model as xm
print('xla device:', xm.xla_device())
print('world size:', xm.xrt_world_size())
"
```

### 5c. TensorFlow

```bash
pip install --upgrade pip
pip install tensorflow tensorflow-tpu

python3 -c "
import tensorflow as tf
print('TF', tf.__version__)
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print('TPU devices:', tf.config.list_logical_devices('TPU'))
"
```

---

## 6. Run a tiny benchmark — PAID time on the VM

Don't start with a transformer. Start with a matmul. If matmul is healthy
and a single-step training step is healthy, *then* scale up.

This snippet (JAX flavour) is roughly equivalent to the model in
`src/model_examples/` and intentionally mirrors what the simulator emits, so
you can cross-check the real run against the simulated one.

```python
# tiny_bench.py — run inside the TPU VM
import time, jax, jax.numpy as jnp

K = 4096
x = jnp.ones((K, K), dtype=jnp.bfloat16)
y = jnp.ones((K, K), dtype=jnp.bfloat16)

@jax.jit
def step(a, b):
    return (a @ b).sum()

# Warmup — discard the first step.  See 14_benchmarking_playbook.md.
step(x, y).block_until_ready()

N = 20
t0 = time.perf_counter()
for _ in range(N):
    out = step(x, y).block_until_ready()
dt = time.perf_counter() - t0
print(f"steady-state per-step: {dt/N*1000:.2f} ms  result={float(out):.3e}")
```

```bash
# PAID: each second this runs is chip-hours spent.
python3 tiny_bench.py
```

You should expect:

- First (discarded) step is slow (compile).
- Subsequent steps are stable to within a few percent.

If the per-step time is jumpy by >10× across iterations, suspect a
recompile loop — see `14_benchmarking_playbook.md` and the
`cloud_tpu_compile_time_seconds` metric in `13_oct_metrics_dictionary.md`.

---

## 7. Collect artefacts — pull them back to your laptop

The lab's whole point is producing artefacts you can analyse offline so you
can delete the TPU sooner. Save logs, traces, and any framework profile to
disk on the VM, then `scp` them down.

On the VM, point the lab's observability writer at a known directory:

```bash
mkdir -p ~/cloud_tpu_lab_artifacts/logs
mkdir -p ~/cloud_tpu_lab_artifacts/metrics
mkdir -p ~/cloud_tpu_lab_artifacts/traces
mkdir -p ~/cloud_tpu_lab_artifacts/reports
```

Run your real-TPU script with output redirected to that tree (the JSONL
schema is documented in `13_oct_metrics_dictionary.md`).

Back on your laptop:

```bash
gcloud compute tpus tpu-vm scp \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --recurse \
  "$TPU_NAME":~/cloud_tpu_lab_artifacts \
  ./real_tpu_run_$(date +%Y%m%d-%H%M%S)
```

Once the files are local, you can analyse them with the same tooling used
on simulator output (`src/profiling/trace_analyzer.py`,
`src/profiling/bottleneck_report.py`).

---

## 8. Intermediate cleanup — between runs

Do not let an idle VM "rest" overnight. The clock is always ticking. If you
need a break of more than a few minutes:

```bash
# PAID note: 'stop' on a TPU VM does NOT necessarily zero billing —
# check the pricing page for your SKU class. The safe option is delete.
gcloud compute tpus tpu-vm stop "$TPU_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE"
```

The honest cleanup is delete, in section 9. Treat "stop" as "I will resume
in 10 minutes and I will not close this terminal".

---

## 9. Final cleanup — STOP THE BILLING

This is the most important command in the whole document. Run it the
instant the run is done. Run it again. Verify with `list`.

```bash
# PAID: this command itself is free; the resource you are deleting was paid.
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --quiet
```

Verify the VM is actually gone — not just "DELETING":

```bash
gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

The expected output is either an empty list or a list that does **not**
contain `$TPU_NAME`. If you see `STATE: DELETING`, wait 30 seconds and run
`list` again.

Common gotchas covered in `11_cleanup_and_cost_safety.md`:

- Reserved static IPs survive VM delete (they still bill).
- A bucket you used for staging may now accrue storage / egress cost.
- Some quota holds (e.g. on a reservation block) are not paid resources but
  *can* prevent you from creating a new VM. Different problem.

---

## 10. Trouble-shooting flowchart

| Symptom                                         | Most likely cause                                  | Fix |
|-------------------------------------------------|----------------------------------------------------|-----|
| `create` returns `RESOURCE_EXHAUSTED`           | No capacity in this zone for this SKU              | Try another zone in the same region; try the previous TPU generation. |
| `create` returns `quota exceeded`               | Project quota for that TPU type is zero            | Request quota in the Cloud Console (free, but takes time). |
| `jax.devices()` returns CPU only                | `RUNTIME_VERSION` does not match `TPU_TYPE`        | Recreate VM with the matching pair from the runtime docs. |
| First step takes minutes, later steps are fine  | Normal compile / cache miss                        | Use `JAX_COMPILATION_CACHE_DIR`; see `14_benchmarking_playbook.md`. |
| Every step recompiles                           | Dynamic shapes (variable batch / seq)              | Pad to fixed shapes; pin batch size. |
| Steady-state step time jumps by 5×              | Recompile or input-pipeline starvation             | Inspect `cloud_tpu_input_wait_time_seconds`. |
| `scp` is slow                                   | Egress is going the long way                       | Run from a VM in the same region, not from your laptop. |

---

## 11. Exercises / TODOs

1. Run the section-6 benchmark on **two** different TPU SKUs (e.g. v5e-1 and
   v6e-1). Capture per-step time + cost-per-step. The cost calculation
   should come from `--hourly-usd-per-chip` flags you fed at the top of this
   doc, not from any hardcoded number.
2. Delete the VM in between, not just at the end. Confirm with `list` each
   time.
3. Inject one deliberate dynamic-shape op (e.g. variable last-batch size)
   and verify the simulator's `cloud_tpu_recompile_count_total` rises. Then
   confirm the real-TPU version exhibits the same pattern by checking the
   compile time across iterations.
4. Replicate the run with `JAX_COMPILATION_CACHE_DIR` pointing at a local
   directory. Show that the second invocation skips compile.
5. Write a one-page "what surprised me" note. Compare your real-TPU
   `cloud_tpu_step_time_seconds` to the simulator's prediction for the same
   `tpu_version` / `chip_count` / `batch_size`.

---

## 12. Cross-references

- `11_cleanup_and_cost_safety.md` — billing alerts, orphan-resource hunting.
- `12_observability_with_grafana_prometheus.md` — wiring this VM's output
  into a Grafana dashboard.
- `13_oct_metrics_dictionary.md` — definition of every metric name produced
  by your run.
- `14_benchmarking_playbook.md` — warmup, steady-state, confidence
  intervals, MFU.
- `src/profiling/bottleneck_report.py` — the rule set that turns a trace
  into prioritised findings.
