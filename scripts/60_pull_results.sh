#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 60_pull_results.sh — Stage 6: pull benchmark artefacts back from the TPU.
#
# Copies:
#   ~/tpu/results/runs.jsonl    → REPO_ROOT/results/runs.jsonl       (append)
#   ~/tpu/results/run_logs/     → REPO_ROOT/results/run_logs/         (rsync)
#   ~/tpu/results/*.log         → REPO_ROOT/results/                  (the tee logs)
#
# `runs.jsonl` is APPENDED (ADR-007: append-only). The remote file is renamed
# to `runs.<timestamp>.jsonl` after pulling, so re-running this script doesn't
# duplicate the same rows.
#
# Usage:
#   ./scripts/60_pull_results.sh
#
# Exit codes:
#   0 = pull succeeded.
#   1 = state missing or scp failed.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="60_pull_results"
setup_error_trap
banner "Stage 6 — Pull benchmark results back from TPU"

require_cmd gcloud

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

LOCAL_RESULTS="$REPO_ROOT/results"
mkdir -p "$LOCAL_RESULTS/run_logs"
TS="$(date +%Y%m%d-%H%M%S)"

# ── 1. Pull runs.jsonl as a fresh file we'll merge ────────────────────────────
section "runs.jsonl"
REMOTE_JSONL="\$HOME/tpu/results/runs.jsonl"
TMP_JSONL=$(mktemp -t runs-XXXXXX.jsonl)
add_exit_handler "rm -f '$TMP_JSONL'"

if gcloud compute tpus tpu-vm scp "$TPU_NAME:~/tpu/results/runs.jsonl" "$TMP_JSONL" \
        --zone="$TPU_ZONE" --quiet 2>/dev/null; then
    rows=$(wc -l < "$TMP_JSONL" | tr -d ' ')
    log_ok "fetched $rows rows"

    # Append (atomically — write a tmpfile then mv) to the local runs.jsonl.
    # A naive `cat >>` could mid-write get a partial line on signal; the
    # mv pattern is safer.
    if [[ -f "$LOCAL_RESULTS/runs.jsonl" ]]; then
        cat "$LOCAL_RESULTS/runs.jsonl" "$TMP_JSONL" > "$LOCAL_RESULTS/runs.jsonl.new"
    else
        cp "$TMP_JSONL" "$LOCAL_RESULTS/runs.jsonl.new"
    fi
    mv "$LOCAL_RESULTS/runs.jsonl.new" "$LOCAL_RESULTS/runs.jsonl"
    total=$(wc -l < "$LOCAL_RESULTS/runs.jsonl" | tr -d ' ')
    log_ok "local runs.jsonl now has $total rows"

    # Snapshot the remote file under a timestamp so re-runs don't re-pull
    # already-merged rows. The remote `runs.jsonl` is then truncated.
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
        --command="set -euo pipefail
            mv ~/tpu/results/runs.jsonl ~/tpu/results/runs.${TS}.jsonl
            : > ~/tpu/results/runs.jsonl
        " 2>/dev/null || log_warn "could not rotate remote runs.jsonl (non-fatal)"
else
    log_warn "no runs.jsonl on remote (or it was empty) — skipping merge"
fi

# ── 2. Pull run_logs/ in full ─────────────────────────────────────────────────
section "run_logs/"
# `gcloud compute tpus tpu-vm scp --recurse` is the supported way to pull a
# directory tree. We pull into a temp dir then move so re-runs are atomic.
TMP_LOGS=$(mktemp -d -t run-logs-XXXXXX)
add_exit_handler "rm -rf '$TMP_LOGS'"
if gcloud compute tpus tpu-vm scp --recurse \
        "$TPU_NAME:~/tpu/results/run_logs/" "$TMP_LOGS/" \
        --zone="$TPU_ZONE" --quiet 2>/dev/null; then
    if [[ -d "$TMP_LOGS/run_logs" ]]; then
        cp -r "$TMP_LOGS/run_logs/." "$LOCAL_RESULTS/run_logs/"
        n=$(find "$LOCAL_RESULTS/run_logs" -mindepth 1 -maxdepth 1 -type d | wc -l)
        log_ok "$n run_log directory(ies) merged into local results/run_logs/"
    fi
else
    log_warn "no run_logs/ on remote — skipping"
fi

# ── 3. Pull tmux session log files (smoke.*.log, quick.*.log) ────────────────
section "tee logs"
gcloud compute tpus tpu-vm scp --recurse \
    "$TPU_NAME:~/tpu/results/*.log" "$LOCAL_RESULTS/" \
    --zone="$TPU_ZONE" --quiet 2>/dev/null || log_warn "no .log files on remote"

state_set LAST_PULL_AT "$(date -Iseconds)"
log_ok "Pull complete. Next: ./scripts/70_teardown_tpu.sh"
exit 0
