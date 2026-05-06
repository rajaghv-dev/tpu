#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 41_run_pytests.sh — Stage 4b: run the harness unit tests on the TPU VM.
#
# The repo ships 97 unit tests in tests/ that DO NOT need a TPU/GPU. Running
# them on the VM (instead of locally) confirms the deployed environment is
# the same one the harness will use, catching version drift early.
#
# Usage:
#   ./scripts/41_run_pytests.sh
#
# Exit codes:
#   0 = all tests passed.
#   1 = state missing, or tests failed (output streamed to your terminal).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="41_run_pytests"
setup_error_trap
banner "Stage 4b — Run harness unit tests on TPU VM"

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

log_info "Running pytest tests/ on $TPU_NAME ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command='set -euo pipefail
        cd ~/tpu
        python3 -m pytest tests/ -q 2>&1 | tail -40
    '

log_ok "Tests passed. Next: ./scripts/42_dry_run.sh"
exit 0
