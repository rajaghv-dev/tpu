#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# collect_artifacts.sh — pull ~/cloud_tpu_lab_artifacts/ from the TPU VM back
# to the local cloud_tpu_lab/artifacts/from_vm/ directory.
#
# The real-TPU runner (run_jax_real_tpu.py invoked by run_real_demo.sh)
# writes one sub-directory per run, e.g.
#
#   ~/cloud_tpu_lab_artifacts/20260518T120000Z-matmul/
#       run_TRACE-0001.jsonl
#       run_TRACE-0001.csv
#       run_TRACE-0001.json     (chrome/perfetto trace)
#       run_TRACE-0001.md       (run report)
#       hlo/                    (XLA HLO dumps; from XLA_FLAGS)
#       xprof/                  (jax.profiler.trace output)
#
# This script recursively copies every per-run directory back to
# cloud_tpu_lab/artifacts/from_vm/, preserving the run-tag structure so
# multiple runs don't overwrite each other.
#
# Silent no-op if the remote directory does not exist (e.g. you haven't run
# a benchmark yet).
#
# Does not create paid resources. SCP traffic is metered by gcloud's normal
# egress rules but is generally negligible for benchmark artifacts.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[collect_artifacts] %s\n' "$*"; }

REMOTE_DIR="\$HOME/cloud_tpu_lab_artifacts"
LOCAL_DIR="$SCRIPT_DIR/../artifacts/from_vm"

# Check remote dir existence over SSH (quietly).
log "checking remote dir on $TPU_NAME ..."
if ! gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" \
        --command="test -d $REMOTE_DIR" >/dev/null 2>&1; then
    log "no artifacts directory on VM ($REMOTE_DIR) — nothing to collect."
    exit 0
fi

mkdir -p "$LOCAL_DIR"
log "copying $TPU_NAME:$REMOTE_DIR → $LOCAL_DIR/ ..."

# --recurse pulls everything under ~/cloud_tpu_lab_artifacts/ (including the
# per-run sub-directories with hlo/ and xprof/ inside). Trailing slash on
# both sides preserves layout.
gcloud compute tpus tpu-vm scp \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --recurse \
    "$TPU_NAME:cloud_tpu_lab_artifacts/*" \
    "$LOCAL_DIR/" || {
        log "scp finished with non-zero status (often means remote dir was empty)."
    }

log "local artifacts now under: $LOCAL_DIR"
log "expected per-run layout: $LOCAL_DIR/<run_tag>/{run_*.jsonl,csv,json,md,hlo/,xprof/}"
ls -la "$LOCAL_DIR" 2>/dev/null || true
