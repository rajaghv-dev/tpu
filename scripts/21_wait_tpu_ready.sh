#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 21_wait_tpu_ready.sh — Stage 2b: poll until the TPU is SSH-able.
#
# `gcloud compute tpus tpu-vm create` returns when the API has accepted the
# request and the resource exists, but SSH may take another 30-90 seconds to
# come up while the VM finishes booting and the gcloud-managed IAP tunnel
# stabilises. This script blocks until a no-op SSH command succeeds.
#
# Reads TPU_NAME / TPU_ZONE from .tpu-bench-state/state.env (set by Stage 2).
#
# Usage:
#   ./scripts/21_wait_tpu_ready.sh                  # default 5 min total wait
#   WAIT_SECONDS=600 ./scripts/21_wait_tpu_ready.sh # 10 min total wait
#
# Exit codes:
#   0 = SSH reached, TPU is usable.
#   1 = state file missing (run 20_provision_tpu.sh first).
#   2 = exceeded WAIT_SECONDS without success.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="21_wait_tpu_ready"
setup_error_trap
banner "Stage 2b — Wait for TPU SSH readiness"

require_cmd gcloud

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

WAIT_SECONDS="${WAIT_SECONDS:-300}"
INTERVAL=10
log_info "Polling SSH on $TPU_NAME ($TPU_ZONE) — up to ${WAIT_SECONDS}s, every ${INTERVAL}s."

start=$(date +%s)
attempt=0
while true; do
    attempt=$((attempt+1))
    elapsed=$(( $(date +%s) - start ))
    if (( elapsed > WAIT_SECONDS )); then
        log_err "Timed out after ${elapsed}s without SSH success."
        log_err "  Try: gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$TPU_ZONE"
        log_err "  Or:  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --command='echo ok'"
        exit 2
    fi
    # We use a trivial command (`true`) to test SSH, redirecting all output so
    # the polling loop is quiet. The `--quiet` flag silences the IAP tunnel
    # banner gcloud prints on first connection.
    if gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" \
            --command='true' --quiet </dev/null >/dev/null 2>&1; then
        log_ok "SSH ready after ${elapsed}s (attempt $attempt)"
        # Also verify the device is actually present, not just the VM.
        log_info "Verifying libtpu visibility..."
        if gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" \
                --command='ls /dev/accel* 2>/dev/null && echo "tpu_device_ok" || echo "tpu_device_missing"' \
                --quiet </dev/null 2>/dev/null | grep -q tpu_device_ok; then
            log_ok "/dev/accel* present"
        else
            log_warn "/dev/accel* not visible yet — JAX may still detect TPU via libtpu, but be ready to retry."
        fi
        exit 0
    fi
    printf '.' >&2
    sleep "$INTERVAL"
done
