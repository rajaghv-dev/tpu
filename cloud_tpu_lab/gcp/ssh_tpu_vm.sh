#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ssh_tpu_vm.sh — thin wrapper around `gcloud compute tpus tpu-vm ssh`.
#
# Extra args are passed through to gcloud (and then to ssh). Examples:
#   ./ssh_tpu_vm.sh                                    # interactive shell
#   ./ssh_tpu_vm.sh --command="nvidia-smi || true"     # one-off command
#   ./ssh_tpu_vm.sh -- -L 6006:localhost:6006          # local port forward
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[ssh_tpu_vm] %s\n' "$*"; }

log "ssh → $TPU_NAME ($ZONE, $PROJECT_ID)"
exec gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    "$@"
