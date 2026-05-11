#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 91_predict_cost.sh — predict the cost of a benchmark suite on a given device.
#
# Pulls baseline minutes (per suite) and speedup factors (per device) from
# lib/config.sh, multiplies by the per-device hourly rate from PRICE_USD_PER_HR,
# and prints a one-line forecast. Useful as a guardrail before
# `20_provision_tpu.sh` ($0.36/hr starts immediately).
#
# Usage:
#   ./scripts/91_predict_cost.sh <SUITE> <DEVICE_KEY>
#
# Examples:
#   ./scripts/91_predict_cost.sh smoke tpu_v5litepod_1
#   ./scripts/91_predict_cost.sh quick gpu_t4
#   ./scripts/91_predict_cost.sh full  tpu_v6e_1
#
# Valid SUITE values:    smoke, quick, domain, arch, llm, full
# Valid DEVICE_KEY:      tpu_v5litepod_1, tpu_v6e_1, gpu_t4, gpu_l4
#
# Exit codes:
#   0 = forecast printed.
#   2 = invalid suite name or device key.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="91_predict_cost"
setup_error_trap

usage() {
    grep -E '^#' "$0" | sed 's/^# \{0,1\}//' | head -25
    echo "Available suites:    ${!SUITE_BASELINE_MINUTES[*]}"
    echo "Available devices:   ${!PRICE_USD_PER_HR[*]}"
    exit 2
}

(( $# == 2 )) || usage
SUITE="$1"; DEVICE="$2"

# ── Validate inputs ───────────────────────────────────────────────────────────
if [[ -z "${SUITE_BASELINE_MINUTES[$SUITE]:-}" ]]; then
    log_err "Unknown suite: $SUITE"
    log_err "Valid: ${!SUITE_BASELINE_MINUTES[*]}"
    exit 2
fi
if [[ -z "${PRICE_USD_PER_HR[$DEVICE]:-}" ]]; then
    log_err "Unknown device: $DEVICE"
    log_err "Valid: ${!PRICE_USD_PER_HR[*]}"
    exit 2
fi

# ── Compute ──────────────────────────────────────────────────────────────────
baseline_min="${SUITE_BASELINE_MINUTES[$SUITE]}"
speedup="${DEVICE_SPEEDUP_FACTOR[$DEVICE]:-1.0}"   # 1.0 = same as baseline
hourly="${PRICE_USD_PER_HR[$DEVICE]}"

est_min=$(awk -v b="$baseline_min" -v s="$speedup" 'BEGIN{printf "%.0f", b*s}')
est_hr=$(awk -v m="$est_min" 'BEGIN{printf "%.3f", m/60}')
cost=$(awk -v h="$est_hr" -v p="$hourly" 'BEGIN{printf "%.2f", h*p}')

# ── Print ─────────────────────────────────────────────────────────────────────
banner "Cost forecast — $SUITE on $DEVICE"
printf '  Baseline (v5e-1):   %s min\n' "$baseline_min"
printf '  Speedup factor:     %sx\n' "$speedup"
printf '  Est. wall time:     %s min\n' "$est_min"
printf '  Hourly rate:        \$%s/hr (preemptible/spot)\n' "$hourly"
printf '  Est. total cost:    \$%s\n' "$cost"
printf '\n'
printf '  Note: speedup factors are approximate (educated guesses based on TFLOPs ratios).\n'
printf '  Refit after the first real %s run for better accuracy.\n' "$SUITE"

# Budget warning
if awk -v c="$cost" -v b="$SESSION_BUDGET_USD" 'BEGIN{exit !(c>b)}'; then
    log_warn "FORECAST EXCEEDS SESSION_BUDGET_USD=\$$SESSION_BUDGET_USD"
fi
exit 0
