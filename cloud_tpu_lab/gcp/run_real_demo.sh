#!/usr/bin/env bash
# PAID: This script EXECUTES work on a running Cloud TPU VM and consumes
#       paid TPU-hours. It does not create or delete VMs. Cleanup of the
#       VM itself: ./delete_tpu_vm.sh
# ──────────────────────────────────────────────────────────────────────────────
# run_real_demo.sh — flagship real-TPU run.
#
# SCPs the runner + src/ tree onto the TPU VM, then runs the JAX matmul
# workload under `jax.profiler.trace(...)` with HLO dumps enabled and
# libtpu logging set to verbose. Artifacts land in
# `$HOME/cloud_tpu_lab_artifacts/<RUN_TAG>/` on the VM;
# `./collect_artifacts.sh` pulls them back locally.
#
# Usage:
#   ./run_real_demo.sh [--yes] [--n-steps N] [--batch-size B]
#                      [--hidden-size H] [--precision bf16|fp32]
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[run_real_demo] %s\n' "$*"; }

ASSUME_YES=0
N_STEPS=10
BATCH_SIZE=32
HIDDEN_SIZE=512
PRECISION="bf16"

usage() {
    cat <<USAGE
Usage: $0 [--yes] [options]
Options:
  --yes, -y            Skip the confirmation prompt
  --n-steps N          Steps including the compile step (default: 10)
  --batch-size B       Used for samples-per-step math (default: 32)
  --hidden-size H      Matmul dimension N (default: 512)
  --precision P        bf16 | fp32 (default: bf16)
  -h, --help           This message
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y) ASSUME_YES=1; shift ;;
        --n-steps) N_STEPS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) log "unknown arg: $1"; usage; exit 1 ;;
    esac
done

cat <<WARN
============================================================
  PAID RESOURCE WARNING
------------------------------------------------------------
  Running real JAX matmul on:
    project    = $PROJECT_ID
    zone       = $ZONE
    tpu        = $TPU_NAME ($ACCELERATOR_TYPE)
    steps      = $N_STEPS
    matrix N   = $HIDDEN_SIZE
    precision  = $PRECISION
  This burns TPU-hours. The VM keeps billing after this run
  exits — run ./delete_tpu_vm.sh when finished.
============================================================
WARN

if [[ "$ASSUME_YES" -ne 1 ]]; then
    read -r -p "Proceed? [y/N] " ans
    case "${ans:-}" in y|Y|yes|YES) ;; *) log "aborted."; exit 1 ;; esac
fi

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)-matmul"
REMOTE_OUT="\$HOME/cloud_tpu_lab_artifacts/${RUN_TAG}"

log "staging src/ + examples/run_jax_real_tpu.py to ${TPU_NAME}:~/cloud_tpu_lab/ ..."
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="mkdir -p ~/cloud_tpu_lab/examples ~/cloud_tpu_lab/src \
               && mkdir -p ${REMOTE_OUT}/hlo ${REMOTE_OUT}/xprof"

gcloud compute tpus tpu-vm scp \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --recurse \
    "$REPO_ROOT/src" \
    "$TPU_NAME:~/cloud_tpu_lab/"

gcloud compute tpus tpu-vm scp \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    "$REPO_ROOT/examples/run_jax_real_tpu.py" \
    "$TPU_NAME:~/cloud_tpu_lab/examples/"

# Make `cloud_tpu_lab.src.*` resolvable on the VM.
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="touch ~/cloud_tpu_lab/__init__.py ~/cloud_tpu_lab/examples/__init__.py"

log "running matmul on $TPU_NAME (out → $REMOTE_OUT) ..."

REMOTE_CMD=$(cat <<REMOTE
set -euo pipefail
export XLA_FLAGS="--xla_dump_to=${REMOTE_OUT}/hlo --xla_dump_hlo_pass_re=.*"
export TPU_STDERR_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0
cd ~/cloud_tpu_lab
python3 examples/run_jax_real_tpu.py \\
    --workload matmul \\
    --n-steps ${N_STEPS} \\
    --batch-size ${BATCH_SIZE} \\
    --hidden-size ${HIDDEN_SIZE} \\
    --precision "${PRECISION}" \\
    --output-dir "${REMOTE_OUT}"
echo "[remote] artifacts in ${REMOTE_OUT}"
ls -la "${REMOTE_OUT}"
REMOTE
)

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="$REMOTE_CMD"

log "done. To pull artifacts locally: ./collect_artifacts.sh"
log "Cleanup reminder: ./delete_tpu_vm.sh when finished with the VM."
