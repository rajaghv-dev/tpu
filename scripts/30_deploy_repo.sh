#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 30_deploy_repo.sh — Stage 3a: tar the local repo and copy it to the TPU VM.
#
# We tar+scp a single archive rather than `--recurse` over many small files
# because gcloud SCP's per-file overhead for the ~150 files in this repo is
# noticeable (~30 s vs ~3 s for the tarball). We exclude .git, __pycache__,
# .pyc to keep the wire payload small (~150 KB vs ~10 MB with .git).
#
# Reads TPU_NAME / TPU_ZONE from .tpu-bench-state/state.env.
#
# Usage:
#   ./scripts/30_deploy_repo.sh
#
# Exit codes:
#   0 = repo deployed; remote ~/tpu populated.
#   1 = state missing or scp failed.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="30_deploy_repo"
setup_error_trap
banner "Stage 3a — Deploy repo to TPU VM"

require_cmd gcloud
require_cmd tar

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

# ── 1. Build tarball ──────────────────────────────────────────────────────────
section "tarball"
TARBALL="$(mktemp -t tpu-repo-XXXXXX.tar.gz)"
add_exit_handler "rm -f '$TARBALL'"
log_info "Tarring $REPO_ROOT (excluding .git, __pycache__, *.pyc, .tpu-bench-state)..."
tar --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.tpu-bench-state' \
    --exclude='results/runs.jsonl' \
    --exclude='results/run_logs' \
    -C "$REPO_ROOT" -czf "$TARBALL" .
sz=$(du -h "$TARBALL" | cut -f1)
log_ok "tarball: $TARBALL ($sz)"

# ── 2. SCP to TPU VM ──────────────────────────────────────────────────────────
section "scp"
log_info "Pushing tarball to $TPU_NAME:~/tpu-repo.tar.gz ..."
gcloud compute tpus tpu-vm scp "$TARBALL" "$TPU_NAME:~/tpu-repo.tar.gz" \
    --zone="$TPU_ZONE" --quiet
log_ok "scp complete"

# ── 3. Extract on remote ──────────────────────────────────────────────────────
section "extract on TPU"
# Single SSH session that:
#   - removes any prior extraction (idempotent);
#   - extracts into ~/tpu (creating it if needed);
#   - removes the tarball to free disk;
#   - reports the resulting path so the user can see what happened.
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command='set -euo pipefail
        rm -rf ~/tpu
        mkdir -p ~/tpu
        tar -xzf ~/tpu-repo.tar.gz -C ~/tpu
        rm -f ~/tpu-repo.tar.gz
        echo "Extracted to: $HOME/tpu"
        ls ~/tpu | head -20'

state_set REMOTE_DEPLOYED_AT "$(date -Iseconds)"
log_ok "Repo deployed. Next: ./scripts/31_install_deps.sh"
exit 0
