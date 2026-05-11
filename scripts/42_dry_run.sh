#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 42_dry_run.sh — Stage 4c: harness --dry-run on the TPU VM.
#
# Confirms the harness CLI loads the registry, builds configs, and prints what
# it would run — without downloading any models. Cheapest possible end-to-end
# wiring check before the smoke suite actually pulls weights.
#
# Usage:
#   ./scripts/42_dry_run.sh                  # default: smoke
#   ./scripts/42_dry_run.sh quick            # alternative: quick
#
# Exit codes:
#   0 = harness produced its dry-run plan.
#   1 = state missing, or harness errored before printing the plan.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="42_dry_run"
setup_error_trap
banner "Stage 4c — Harness dry-run"

SUITE="${1:-smoke}"

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

log_info "Running --suite $SUITE --device tpu --dry-run ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command="set -euo pipefail
        cd ~/tpu
        [ -f ~/.tpu-bench-env ] && source ~/.tpu-bench-env || true
        python3 -m benchmarks.harness --suite $SUITE --device tpu --dry-run
    "

log_ok "Dry-run plan printed. Next: ./scripts/50_run_smoke.sh"
exit 0
