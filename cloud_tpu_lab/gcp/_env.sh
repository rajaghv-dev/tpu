# shellcheck shell=bash
# ──────────────────────────────────────────────────────────────────────────────
# _env.sh — shared environment for cloud_tpu_lab/gcp/*.sh
#
# This file is SOURCED by the other scripts (not executed). It defines the
# variables they all need and prints a one-line summary so the caller can
# eyeball the config before anything paid happens.
#
# Override anything by exporting before sourcing, e.g.
#   PROJECT_ID=my-proj ZONE=us-central2-b ./create_tpu_vm.sh
#
# Cloud TPU only — this entire directory is about Google Cloud TPU VMs.
# ──────────────────────────────────────────────────────────────────────────────

# Required: GCP project id. No safe default — fail fast if unset.
PROJECT_ID="${PROJECT_ID:-nellaiappar-001}"

# Zone hosting the TPU VM. Pick one that has capacity for ACCELERATOR_TYPE.
# See https://cloud.google.com/tpu/docs/regions-zones.
ZONE="${ZONE:-us-central2-b}"

# Logical name of the TPU VM (lowercase, hyphens). Used by every other script.
TPU_NAME="${TPU_NAME:-ctl-tpu-vm}"

# Accelerator type. v5litepod-1 (a.k.a. v5e-1) is the cheapest single-chip
# slice and a sensible default for learning. Common alternatives:
#   v5litepod-1 / v5litepod-4 / v5litepod-8     (v5e family)
#   v5p-8       / v5p-16                        (v5p family)
#   v6e-1       / v6e-4                         (Trillium / v6e)
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-1}"

# TPU VM runtime image. Match this to ACCELERATOR_TYPE per
# https://cloud.google.com/tpu/docs/runtimes.
RUNTIME_VERSION="${RUNTIME_VERSION:-tpu-ubuntu2204-base}"

# Networking — leave as "default" unless you have a custom VPC.
NETWORK="${NETWORK:-default}"
SUBNETWORK="${SUBNETWORK:-default}"

# ── Fail fast on missing project id ──────────────────────────────────────────
if [[ -z "${PROJECT_ID:-}" ]]; then
    echo "[_env.sh] ERROR: PROJECT_ID is empty." >&2
    echo "[_env.sh] export PROJECT_ID=<your-gcp-project> and re-run." >&2
    return 1 2>/dev/null || exit 1
fi

# ── One-line config summary ──────────────────────────────────────────────────
echo "[_env.sh] config: project=$PROJECT_ID zone=$ZONE tpu=$TPU_NAME accel=$ACCELERATOR_TYPE runtime=$RUNTIME_VERSION net=$NETWORK/$SUBNETWORK"
