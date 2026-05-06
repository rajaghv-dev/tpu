#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 70_teardown_tpu.sh — Stage 7a: delete the TPU VM to stop billing.
#
# Reads TPU_NAME / TPU_ZONE from .tpu-bench-state/state.env. Falls back to
# CLI args if state is missing.
#
# Usage:
#   ./scripts/70_teardown_tpu.sh                    # uses state file
#   ./scripts/70_teardown_tpu.sh tpu-bench-v5e us-east5-a   # explicit args
#
# Exit codes:
#   0 = TPU deleted (or didn't exist).
#   1 = delete failed (transient API error — retry).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="70_teardown_tpu"
setup_error_trap
banner "Stage 7a — Tear down TPU (stop billing)"

require_cmd gcloud

# Allow CLI override so this script works even if the state file was wiped.
TPU_NAME="${1:-$(state_get TPU_NAME "")}"
TPU_ZONE="${2:-$(state_get TPU_ZONE "")}"

if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "Need TPU name and zone."
    log_err "  Usage: $0 <TPU_NAME> <ZONE>"
    log_err "  Or run after 20_provision_tpu.sh which records state."
    exit 1
fi

log_info "Target: $TPU_NAME in $TPU_ZONE"

# ── 1. Existence check (idempotent) ───────────────────────────────────────────
state=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$TPU_ZONE" \
    --format='value(state)' 2>/dev/null || true)
if [[ -z "$state" ]]; then
    log_ok "TPU $TPU_NAME not found in $TPU_ZONE — nothing to delete."
    state_clear
    exit 0
fi
log_info "Current state: $state"

# ── 2. Delete ─────────────────────────────────────────────────────────────────
log_step "Deleting (this takes ~30-60s)..."
if gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$TPU_ZONE" --quiet; then
    log_ok "TPU deleted."
else
    rc=$?
    log_err "Delete failed (exit $rc). Retry with: $0 $TPU_NAME $TPU_ZONE"
    log_err "Or via console: https://console.cloud.google.com/compute/tpus?project=$GCP_PROJECT"
    exit 1
fi

# ── 3. Clear state file so the next session starts fresh ──────────────────────
state_clear
log_ok "State cleared. Next: ./scripts/71_verify_teardown.sh"
exit 0
