#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 50_run_smoke.sh — Stage 5a: run the smoke suite on the TPU.
#
# Smoke = 1 model (BERT-base), BF16, ~8 min on v5e-1 ≈ \$0.05 (R8 budget OK).
#
# Why we run inside tmux (R5):
#   The India ↔ us-east5 path drops SSH connections frequently. A bare
#   `gcloud ssh ... --command='python harness.py ...'` loses the run on every
#   blip. tmux + ADR-007 append-only JSONL means we can re-attach (or just
#   pull the JSONL) and not lose results.
#
# Usage:
#   ./scripts/50_run_smoke.sh
#
# Exit codes:
#   0 = smoke run finished and at least one row written to runs.jsonl.
#   1 = state missing, harness errored, or no row appeared.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="50_run_smoke"
setup_error_trap
banner "Stage 5a — Smoke suite (1 model, BF16, ~8 min)"

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

log_info "Streaming harness output from $TPU_NAME ($TPU_ZONE)"
log_info "Suite: smoke   Device: tpu   Output: ~/tpu/results/runs.jsonl"
log_info "Expected: ~8 min wall, ~\$0.05 cost on v5e-1 spot."

# We launch the harness inside tmux session $TMUX_SESSION (R5). The session
# survives SSH disconnects. We then attach a separate SSH that simply tails
# the run log so the user sees live progress; if that SSH drops, the tmux
# session keeps running and the JSONL keeps growing.
#
# The trick is: `tmux new-session -d -s NAME 'cmd'` starts the session
# detached, returning immediately. Then we tail the per-session output that
# tmux writes via `tmux pipe-pane`.
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command="set -euo pipefail
        cd ~/tpu
        [ -f ~/.tpu-bench-env ] && source ~/.tpu-bench-env || true

        SESSION='$TMUX_SESSION'
        LOG=~/tpu/results/smoke.\$(date +%Y%m%d-%H%M%S).log
        mkdir -p ~/tpu/results

        # Kill any leftover session of the same name so the new run is clean.
        tmux kill-session -t \"\$SESSION\" 2>/dev/null || true

        # Start the harness inside tmux, redirecting output to the log file.
        # The 'tee' duplicates output so we can also tail it from outside.
        tmux new-session -d -s \"\$SESSION\" \\
            \"cd ~/tpu && python3 -u -m benchmarks.harness --suite smoke --device tpu 2>&1 | tee \$LOG\"

        echo \"started tmux session '\$SESSION' (log: \$LOG)\"
        echo '── tailing log; Ctrl-C here just stops the tail, NOT the run ──'

        # Tail until tmux session ends. The wait-for trick: poll every 2 s and
        # break out when the session is gone.
        while tmux has-session -t \"\$SESSION\" 2>/dev/null; do
            tail -n +1 -f \"\$LOG\" --pid \$\$ 2>/dev/null &
            tail_pid=\$!
            # Watch for tmux session to end; when it does, kill the tail.
            while tmux has-session -t \"\$SESSION\" 2>/dev/null; do sleep 2; done
            kill \$tail_pid 2>/dev/null || true
        done

        echo
        echo '── final tail of log (last 30 lines) ──'
        tail -30 \"\$LOG\"
    "

# ── Verify a row was written to runs.jsonl ────────────────────────────────────
section "verify"
written=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command='wc -l < ~/tpu/results/runs.jsonl 2>/dev/null || echo 0' 2>/dev/null \
    | tr -d '[:space:]')
if (( written > 0 )); then
    log_ok "$written row(s) in results/runs.jsonl"
else
    log_err "results/runs.jsonl has 0 rows — smoke did not complete a single experiment"
    exit 1
fi

state_set LAST_SUITE_RUN "smoke"
state_set LAST_SUITE_AT "$(date -Iseconds)"
log_ok "Smoke complete. Next: ./scripts/51_run_quick.sh (5 models) OR ./scripts/60_pull_results.sh"
exit 0
