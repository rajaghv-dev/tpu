# `cloud_tpu_lab/gcp/` — optional Cloud TPU VM workflow

Shell scripts that provision a real Google **Cloud TPU VM**, install one of
three frameworks on it, run a tiny benchmark, pull artifacts back, and then
**tear the VM down** so billing stops.

This directory is entirely optional. The rest of `cloud_tpu_lab/` runs on a
laptop with no TPU and no GCP account. Only come here when you are ready to
spend real money on real silicon.

> Cloud TPU only. These scripts target Google Cloud TPU VMs.

---

## > **WARNING — paid resources**
>
> Every script tagged `# PAID:` creates or consumes billable Cloud TPU
> capacity. An idle TPU VM still costs money. **Always finish a session by
> running `./delete_tpu_vm.sh`** and verify with
> `gcloud compute tpus tpu-vm list --zone=$ZONE`.
>
> Pricing: https://cloud.google.com/tpu/pricing

---

## Setup assumptions

You have, on your local machine:

- `gcloud` installed and authenticated (`gcloud auth login`,
  `gcloud auth application-default login`).
- A GCP project selected (`gcloud config set project <id>`).
- The **Cloud TPU API enabled** on that project.
- **Billing enabled** on that project (TPUs will not provision without it).
- Enough quota in your chosen zone for the requested `ACCELERATOR_TYPE`.

Check capacity / zones: https://cloud.google.com/tpu/docs/regions-zones

## Required environment variables

All scripts source `_env.sh`, which reads these from the environment with
sensible (non-secret, non-project-specific) defaults where it can.

| Variable           | Required | Default                  | Notes                                       |
|--------------------|----------|--------------------------|---------------------------------------------|
| `PROJECT_ID`       | **yes**  | _(none — fails fast)_    | Your GCP project id.                        |
| `ZONE`             | yes      | `us-central2-b`          | Must have capacity for `ACCELERATOR_TYPE`.  |
| `TPU_NAME`         | yes      | `ctl-tpu-vm`             | Lowercase, hyphenated.                      |
| `ACCELERATOR_TYPE` | yes      | `v5litepod-1`            | e.g. `v5litepod-1`, `v5p-8`, `v6e-1`.       |
| `RUNTIME_VERSION`  | yes      | `tpu-ubuntu2204-base`    | Match to `ACCELERATOR_TYPE`.                |
| `NETWORK`          | yes      | `default`                | Override only for custom VPCs.              |
| `SUBNETWORK`       | yes      | `default`                | Override only for custom VPCs.              |

Typical usage:

```bash
export PROJECT_ID=my-gcp-project
export ZONE=us-central2-b
./create_tpu_vm.sh
```

## Scripts

| Script                          | One-liner                                                            |
|---------------------------------|----------------------------------------------------------------------|
| `_env.sh`                       | Shared env + one-line config summary. Sourced by every other script. |
| `create_tpu_vm.sh` **(PAID)**   | Create the TPU VM (idempotent). Prompts unless `--yes`.              |
| `ssh_tpu_vm.sh`                 | SSH wrapper; passes extra args through.                              |
| `install_jax_tpu.sh`            | Install JAX `[tpu]` + flax + optax + transformers on the VM.         |
| `install_pytorch_xla_tpu.sh`    | Install torch + torch_xla on the VM.                                 |
| `install_tensorflow_tpu.sh`     | Install TensorFlow + libtpu on the VM.                               |
| `run_jax_benchmark.sh` **(PAID)**          | Tiny JAX matmul benchmark on the VM.                      |
| `run_pytorch_xla_benchmark.sh` **(PAID)**  | Tiny torch_xla matmul benchmark on the VM.                |
| `run_tensorflow_benchmark.sh` **(PAID)**   | Tiny TF matmul benchmark on the VM.                       |
| `start_profiler.sh`             | Start `jax.profiler.start_server` on the VM + open local port-forward. |
| `collect_artifacts.sh`          | SCP `~/cloud_tpu_lab_artifacts/` back to `artifacts/from_vm/`.       |
| `delete_tpu_vm.sh` **(CLEANUP)**| Destroy the VM and stop billing. Idempotent.                         |

Every script that creates paid state has a matching cleanup path:
**`delete_tpu_vm.sh`** is the single owner of all teardown.

## Worked example — create → ssh → install → run → collect → delete

```bash
# 0. Configure (once per shell)
export PROJECT_ID=my-gcp-project
export ZONE=us-central2-b
export TPU_NAME=ctl-tpu-vm
export ACCELERATOR_TYPE=v5litepod-1
export RUNTIME_VERSION=tpu-ubuntu2204-base

cd cloud_tpu_lab/gcp

# 1. Create the TPU VM (paid; idempotent).
./create_tpu_vm.sh                # prompts for confirmation
# or: ./create_tpu_vm.sh --yes    # CI mode

# 2. (Optional) drop into a shell on the VM.
./ssh_tpu_vm.sh

# 3. Install a framework. Pick one — they share /usr/bin/python3 and will
#    interfere with each other if you install more than one in the same VM.
./install_jax_tpu.sh
# or ./install_pytorch_xla_tpu.sh
# or ./install_tensorflow_tpu.sh

# 4. Run a benchmark (paid — burns TPU-hours).
./run_jax_benchmark.sh --yes

# 5. (Optional) attach a profiler in another terminal.
./start_profiler.sh
# Then point TensorBoard's profile plugin at localhost:9012.

# 6. Pull results back to your laptop.
./collect_artifacts.sh
ls ../artifacts/from_vm/

# 7. **Stop billing.** Always.
./delete_tpu_vm.sh                # or --yes for CI

# 8. Verify.
gcloud compute tpus tpu-vm list --zone="$ZONE" --project="$PROJECT_ID"
```

## When something goes wrong

- **`create_tpu_vm.sh` fails with RESOURCE_EXHAUSTED** — the zone is out of
  capacity for that accelerator. Try a different `ZONE`, or wait 15–30 min.
- **install script picks the wrong wheels** — versions and wheels URLs are
  intentionally pinned at the top of each `install_*` script. Refresh them
  from the official guide linked in each script's header comment.
- **`delete_tpu_vm.sh` says "not found"** — that's the success message; the
  VM is gone. Confirm with the `gcloud ... list` command it prints.
- **Forgot to clean up?** Set a billing budget alert in your project so this
  never happens twice.
