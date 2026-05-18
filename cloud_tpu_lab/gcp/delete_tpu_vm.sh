#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# delete_tpu_vm.sh — **CLEANUP** — destroy the Cloud TPU VM.
#
# This is the script that STOPS BILLING. Run it as soon as you are done.
# Idempotent: if the TPU VM does not exist, exits 0 with a message.
# Skip the confirmation prompt with --yes.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[delete_tpu_vm] %s\n' "$*"; }

ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) echo "Usage: $0 [--yes]"; exit 0 ;;
        *) log "WARN: unknown arg '$arg' (ignored)" ;;
    esac
done

# ── Idempotency: does it actually exist? ─────────────────────────────────────
log "checking for TPU '$TPU_NAME' in $ZONE ..."
state=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" \
    --format='value(state)' 2>/dev/null || true)

if [[ -z "$state" ]]; then
    log "TPU '$TPU_NAME' not found in $ZONE — nothing to delete."
    log "Verify with: gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT_ID"
    exit 0
fi
log "found TPU '$TPU_NAME' (state=$state)."

# ── Confirmation ─────────────────────────────────────────────────────────────
cat <<WARN
============================================================
  ** CLEANUP — IRREVERSIBLE **
------------------------------------------------------------
  About to DELETE the Cloud TPU VM:
    project = $PROJECT_ID
    zone    = $ZONE
    name    = $TPU_NAME
    state   = $state

  This stops billing. Anything stored only on the VM's local
  disk will be lost — pull results first with:
    ./collect_artifacts.sh
============================================================
WARN

if [[ "$ASSUME_YES" -ne 1 ]]; then
    read -r -p "Type the TPU name to confirm deletion: " ans
    if [[ "$ans" != "$TPU_NAME" ]]; then
        log "name did not match — aborted."
        exit 1
    fi
fi

# ── Delete ───────────────────────────────────────────────────────────────────
log "deleting ..."
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --quiet

log "delete request accepted."
log ""
log "BILLING REMINDER:"
log "  Verify deletion with:"
log "    gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT_ID"
log "  Also check the Cloud Console billing page — a still-listed VM means"
log "  the meter is still running."
