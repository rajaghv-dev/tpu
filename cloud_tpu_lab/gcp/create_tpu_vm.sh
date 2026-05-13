#!/usr/bin/env bash
# PAID: This script CREATES a Cloud TPU VM. Billing starts the moment the VM
#       reaches READY and continues until delete_tpu_vm.sh succeeds. Cleanup:
#       ./delete_tpu_vm.sh
# ──────────────────────────────────────────────────────────────────────────────
# create_tpu_vm.sh — provision a Cloud TPU VM (idempotent).
#
# Reuses the VM if one with the same name already exists in $ZONE.
# Otherwise prints a paid-resource warning, prompts for confirmation
# (skip with --yes), then calls `gcloud compute tpus tpu-vm create`.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[create_tpu_vm] %s\n' "$*"; }

ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help)
            echo "Usage: $0 [--yes]"
            echo "  --yes   skip the paid-resource confirmation prompt"
            exit 0
            ;;
        *) log "WARN: unknown arg '$arg' (ignored)";;
    esac
done

# ── Idempotency: does this TPU already exist? ────────────────────────────────
log "checking for existing TPU '$TPU_NAME' in $ZONE..."
existing_state=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" \
    --format='value(state)' 2>/dev/null || true)

if [[ -n "$existing_state" ]]; then
    log "TPU '$TPU_NAME' already exists in $ZONE (state=$existing_state) — nothing to do."
    log "SSH:    $SCRIPT_DIR/ssh_tpu_vm.sh"
    log "Delete: $SCRIPT_DIR/delete_tpu_vm.sh"
    exit 0
fi

# ── Paid-resource warning ────────────────────────────────────────────────────
cat <<WARN
============================================================
  PAID RESOURCE WARNING
------------------------------------------------------------
  About to create a Cloud TPU VM:
    project = $PROJECT_ID
    zone    = $ZONE
    name    = $TPU_NAME
    accel   = $ACCELERATOR_TYPE
    runtime = $RUNTIME_VERSION

  Billing STARTS when the VM reaches READY and continues
  until you delete it. Idle TPU VMs still accrue cost.

  Current rates: https://cloud.google.com/tpu/pricing
  Cleanup:       ./delete_tpu_vm.sh
============================================================
WARN

if [[ "$ASSUME_YES" -ne 1 ]]; then
    read -r -p "Proceed and create the TPU VM? [y/N] " ans
    case "${ans:-}" in
        y|Y|yes|YES) ;;
        *) log "aborted by user."; exit 1 ;;
    esac
fi

# ── Create ───────────────────────────────────────────────────────────────────
log "creating TPU VM (this can take 1–5 minutes)..."
gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --version="$RUNTIME_VERSION" \
    --network="$NETWORK" \
    --subnetwork="$SUBNETWORK" \
    --project="$PROJECT_ID"

log "TPU VM '$TPU_NAME' created."
log "SSH in with:"
log "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID"
log "or use the wrapper:"
log "  $SCRIPT_DIR/ssh_tpu_vm.sh"
log ""
log "REMEMBER: billing has started. Run ./delete_tpu_vm.sh when done."
