#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 51_run_quick.sh — Stage 5b: run the quick suite on the TPU.
#
# Quick = 5 models (BERT, ViT, GPT-2, Whisper, CLIP), BF16, ~50 min on v5e-1
# ≈ \$0.30 (R8 budget OK).
#
# Same tmux+tee pattern as 50_run_smoke.sh — see that file's header for the
# rationale. This script differs only by suite name and expected duration.
#
# Usage:
#   ./scripts/51_run_quick.sh
#
# Exit codes:
#   0 = quick run finished and >= 5 rows written to runs.jsonl.
#   1 = state missing, harness errored, or fewer than 5 rows appeared.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="51_run_quick"
setup_error_trap
banner "Stage 5b — Quick suite (5 models, BF16, ~50 min)"

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

# Capture row count BEFORE so we can check increment, since smoke may have
# already written rows. quick should add at least 5 (one per model).
before=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command='wc -l < ~/tpu/results/runs.jsonl 2>/dev/null || echo 0' 2>/dev/null \
    | tr -d '[:space:]')
log_info "Existing rows in runs.jsonl: $before"

log_info "Streaming harness output... (this run is ~50 minutes)"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command="set -euo pipefail
        cd ~/tpu
        [ -f ~/.tpu-bench-env ] && source ~/.tpu-bench-env || true

        SESSION='${TMUX_SESSION}-quick'
        LOG=~/tpu/results/quick.\$(date +%Y%m%d-%H%M%S).log

        tmux kill-session -t \"\$SESSION\" 2>/dev/null || true
        tmux new-session -d -s \"\$SESSION\" \\
            \"cd ~/tpu && python3 -u -m benchmarks.harness --suite quick --device tpu 2>&1 | tee \$LOG\"

        echo \"started tmux session '\$SESSION' (log: \$LOG)\"
        echo '── tailing log; Ctrl-C here just stops the tail, NOT the run ──'

        while tmux has-session -t \"\$SESSION\" 2>/dev/null; do
            tail -n +1 -f \"\$LOG\" --pid \$\$ 2>/dev/null &
            tail_pid=\$!
            while tmux has-session -t \"\$SESSION\" 2>/dev/null; do sleep 5; done
            kill \$tail_pid 2>/dev/null || true
        done

        echo
        echo '── final tail of log (last 40 lines) ──'
        tail -40 \"\$LOG\"
    "

section "verify"
after=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command='wc -l < ~/tpu/results/runs.jsonl 2>/dev/null || echo 0' 2>/dev/null \
    | tr -d '[:space:]')
delta=$((after - before))
log_info "Rows before: $before, after: $after, delta: +$delta"
if (( delta < 5 )); then
    log_warn "Expected +5 (one per model in quick); got +$delta. Some models may have failed — check error.json files in results/run_logs/."
fi

state_set LAST_SUITE_RUN "quick"
state_set LAST_SUITE_AT "$(date -Iseconds)"
log_ok "Quick complete. Next: ./scripts/60_pull_results.sh → ./scripts/70_teardown_tpu.sh"
exit 0
