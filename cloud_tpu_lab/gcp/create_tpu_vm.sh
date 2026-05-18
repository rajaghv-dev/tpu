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

# ── Zone discovery: verify the configured ZONE actually offers our ──────────
# ACCELERATOR_TYPE; if not, probe known zones for that TPU generation and
# pick the first hit. Set `ZONE=...` explicitly to skip the probe.
if [[ -z "${ZONE_LOCKED:-}" ]]; then
    # shellcheck source=find_tpu_zone.sh
    source "$SCRIPT_DIR/find_tpu_zone.sh"
    log "verifying $ACCELERATOR_TYPE in $ZONE ..."
    if ! gcloud compute tpus accelerator-types list \
            --project="$PROJECT_ID" --zone="$ZONE" \
            --filter="type=${ACCELERATOR_TYPE}" \
            --format="value(type)" 2>/dev/null | grep -qx "$ACCELERATOR_TYPE"; then
        log "  $ACCELERATOR_TYPE not available in $ZONE — probing other zones..."
        if NEW_ZONE=$(find_tpu_zone); then
            log "  switching ZONE: $ZONE → $NEW_ZONE"
            ZONE="$NEW_ZONE"
        else
            log "ERROR: no probed zone offers '$ACCELERATOR_TYPE'."
            log "       Check quota at GCP Console → IAM & Admin → Quotas → Cloud TPU API."
            exit 1
        fi
    else
        log "  $ACCELERATOR_TYPE present in $ZONE — proceeding"
    fi
fi

# ── Network / subnetwork validation ──────────────────────────────────────────
# A TPU VM needs a subnet in the region of $ZONE. Many GCP projects no longer
# auto-create `default` subnets in every region, so we verify before calling
# create and switch to whatever subnet does exist in $NETWORK for that region.
REGION="${ZONE%-*}"
log "verifying subnet '$SUBNETWORK' in $REGION for network '$NETWORK' ..."
if ! gcloud compute networks subnets describe "$SUBNETWORK" \
        --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    log "  '$SUBNETWORK' not in $REGION — looking for any subnet of '$NETWORK' there..."
    ALT_SUBNET=$(gcloud compute networks subnets list \
        --project="$PROJECT_ID" \
        --filter="region:$REGION AND network:$NETWORK" \
        --format="value(name)" 2>/dev/null | head -n1)
    if [[ -n "$ALT_SUBNET" ]]; then
        log "  switching SUBNETWORK: $SUBNETWORK → $ALT_SUBNET"
        SUBNETWORK="$ALT_SUBNET"
    else
        log "  no subnet of '$NETWORK' exists in $REGION."
        log "  Two ways to fix:"
        log "    1) Create one (auto-mode default VPC behaviour):"
        log "         gcloud compute networks subnets create default \\"
        log "             --network=$NETWORK --region=$REGION \\"
        log "             --range=10.138.0.0/20 --project=$PROJECT_ID"
        log "    2) Enable auto-subnet on the existing VPC (preferred if it's the default):"
        log "         gcloud compute networks update $NETWORK \\"
        log "             --switch-to-custom-subnet-mode=false --project=$PROJECT_ID"
        log "       (this only works if $NETWORK is in legacy mode and has no conflicts)"
        exit 1
    fi
else
    log "  subnet '$SUBNETWORK' present in $REGION — proceeding"
fi

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
