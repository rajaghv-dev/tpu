#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 92_idle_check.sh — flag long-running VMs/TPUs that may have been forgotten.
#
# Threshold: 2 hours since creation (override with IDLE_THRESHOLD_HOURS).
# Anything older than that is printed with "(possibly forgotten — running Xh)".
# This is a heuristic, not a guarantee — a legitimate long run will also be
# flagged. Triage and tear down what shouldn't be alive.
#
# RECOMMENDATIONS R8: "A forgotten v6e VM at \$0.75/hr × 168 hrs = \$126."
# Run this from cron daily for cheap insurance:
#   0 9 * * * /path/to/tpu/scripts/92_idle_check.sh > /tmp/idle.log 2>&1
#
# Usage:
#   ./scripts/92_idle_check.sh
#   IDLE_THRESHOLD_HOURS=6 ./scripts/92_idle_check.sh
#
# Exit codes:
#   0 = nothing flagged.
#   1 = at least one long-running resource flagged — review.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="92_idle_check"
setup_error_trap
banner "Idle-resource check (threshold: ${IDLE_THRESHOLD_HOURS:-2}h)"

require_cmd gcloud
THRESH_HRS="${IDLE_THRESHOLD_HOURS:-2}"
NOW_EPOCH=$(date +%s)
flagged=0

# Bash helper: epoch-of-iso-timestamp. Linux date supports `-d ISO`. macOS
# (BSD date) needs a different invocation; we detect and adapt.
to_epoch() {
    local iso="$1"
    if date -d "$iso" +%s >/dev/null 2>&1; then
        date -d "$iso" +%s
    else
        # BSD date fallback (macOS): strip the trailing Z and use -j -f
        date -j -f "%Y-%m-%dT%H:%M:%SZ" "${iso%Z}Z" +%s 2>/dev/null || echo 0
    fi
}

# ── 1. TPUs ───────────────────────────────────────────────────────────────────
section "TPUs"
for z in $TPU_ZONES_PRIMARY $TPU_ZONES_FALLBACK; do
    while IFS=$'\t' read -r name state accel created; do
        [[ -z "$name" ]] && continue
        ce=$(to_epoch "$created")
        (( ce == 0 )) && continue
        age_hrs=$(( (NOW_EPOCH - ce) / 3600 ))
        if (( age_hrs > THRESH_HRS )); then
            log_warn "$z/$name ($accel $state) — running ${age_hrs}h (possibly forgotten)"
            flagged=$((flagged+1))
        fi
    done < <(gcloud compute tpus tpu-vm list --zone="$z" \
        --format="value(name,state,acceleratorType,createTime)" 2>/dev/null || true)
done

# ── 2. Compute Engine instances ───────────────────────────────────────────────
section "Compute Engine instances"
while IFS=$'\t' read -r name zone status mtype created; do
    [[ -z "$name" ]] && continue
    [[ "$status" != "RUNNING" ]] && continue
    ce=$(to_epoch "$created")
    (( ce == 0 )) && continue
    age_hrs=$(( (NOW_EPOCH - ce) / 3600 ))
    if (( age_hrs > THRESH_HRS )); then
        log_warn "$zone/$name ($mtype) — running ${age_hrs}h (possibly forgotten)"
        flagged=$((flagged+1))
    fi
done < <(gcloud compute instances list \
    --format="value(name,zone.basename(),status,machineType.basename(),creationTimestamp)" 2>/dev/null || true)

# ── Summary ───────────────────────────────────────────────────────────────────
section "summary"
if (( flagged == 0 )); then
    log_ok "No resources running longer than ${THRESH_HRS}h."
    exit 0
fi
log_warn "$flagged resource(s) flagged. Tear them down with:"
log_warn "  ./scripts/70_teardown_tpu.sh <NAME> <ZONE>          (for TPUs)"
log_warn "  gcloud compute instances delete <NAME> --zone=<Z>   (for VMs)"
exit 1
