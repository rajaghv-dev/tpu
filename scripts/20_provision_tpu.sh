#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 20_provision_tpu.sh — Stage 2: create a preemptible v5e-1 TPU VM.
#
# Tries each zone in TPU_ZONES_PRIMARY (then TPU_ZONES_FALLBACK) until one
# accepts the create request. Spot capacity for v5e is regionally uneven —
# this is the script's biggest job.
#
# Idempotent within a session: if a TPU named $TPU_NAME already exists in any
# configured zone, this script reuses it (records the zone) and exits 0.
#
# Records winning zone to .tpu-bench-state/state.env so later stages know
# where to ssh without re-querying gcloud.
#
# Usage:
#   ./scripts/20_provision_tpu.sh                # default name + zone list
#   TPU_NAME=mybench ./scripts/20_provision_tpu.sh
#   TPU_PROVISIONING_MODEL=STANDARD ./scripts/20_provision_tpu.sh   # on-demand
#
# Exit codes:
#   0 = TPU created or reused.
#   1 = all zones returned RESOURCE_EXHAUSTED or other error.
#   2 = configuration problem (bad accelerator type, missing project, etc.).
#
# Cost reminder:
#   v5e-1 spot: ~\$0.36/hr. Once this returns success, billing has STARTED.
#   Tear down promptly with ./scripts/70_teardown_tpu.sh when done.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="20_provision_tpu"
setup_error_trap
banner "Stage 2 — Provision $TPU_ACCEL preemptible TPU"

require_cmd gcloud
require_env GCP_PROJECT "run gcloud config set project <ID>"

log_info "Name      : $TPU_NAME"
log_info "Type      : $TPU_ACCEL"
log_info "Runtime   : $TPU_RUNTIME"
log_info "Mode      : $TPU_PROVISIONING_MODEL  (SPOT = preemptible / cheaper / can be reclaimed)"
log_info "Try zones : $TPU_ZONES_PRIMARY → $TPU_ZONES_FALLBACK"

# ── 1. Reuse existing TPU if same name already exists ────────────────────────
section "checking for existing TPU"
ALL_ZONES="$TPU_ZONES_PRIMARY $TPU_ZONES_FALLBACK"
for z in $ALL_ZONES; do
    state=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$z" \
        --format='value(state)' 2>/dev/null || true)
    if [[ -n "$state" ]]; then
        log_ok "Found existing TPU $TPU_NAME in $z (state=$state) — reusing."
        state_set TPU_NAME "$TPU_NAME"
        state_set TPU_ZONE "$z"
        state_set TPU_ACCEL "$TPU_ACCEL"
        if [[ "$state" != "READY" ]]; then
            log_warn "State is $state, not READY — wait for it to come up before deploying."
            log_warn "  Watch:  gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$z"
        fi
        exit 0
    fi
done
log_ok "no collision — proceeding with create"

# ── 2. Try each zone ──────────────────────────────────────────────────────────
# We attach a small helper to translate `--spot`/`--on-demand` from the config
# variable, since the gcloud CLI doesn't accept the value directly.
case "$TPU_PROVISIONING_MODEL" in
    SPOT)     mode_flags=(--spot) ;;
    STANDARD) mode_flags=()       ;;
    *) log_err "Bad TPU_PROVISIONING_MODEL=$TPU_PROVISIONING_MODEL (use SPOT or STANDARD)"; exit 2 ;;
esac

create_in_zone() {
    local zone="$1"
    log_step "Trying $zone..."
    # Capture both stdout+stderr so we can scrape the error if it fails.
    local out rc
    out=$(gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --zone="$zone" \
        --accelerator-type="$TPU_ACCEL" \
        --version="$TPU_RUNTIME" \
        "${mode_flags[@]}" 2>&1) && rc=0 || rc=$?
    if (( rc == 0 )); then
        log_ok "Created in $zone"
        printf '%s' "$zone"
        return 0
    fi
    # Classify the failure so the caller knows whether to try another zone or
    # bail out completely (config error).
    if echo "$out" | grep -qiE "RESOURCE_EXHAUSTED|Reservation not found|insufficient quota|capacity"; then
        log_warn "$zone: capacity exhausted — trying next zone"
        return 11   # capacity error → keep trying
    fi
    if echo "$out" | grep -qiE "ALREADY_EXISTS"; then
        log_warn "$zone: name collision (race?) — trying next zone"
        return 11
    fi
    if echo "$out" | grep -qiE "PERMISSION_DENIED|FORBIDDEN"; then
        log_err "$zone: permission denied"
        log_err "$(echo "$out" | tail -3)"
        return 2  # config error → don't keep trying other zones
    fi
    log_err "$zone: unrecognised failure (exit $rc):"
    log_err "$(echo "$out" | tail -5)"
    return 11    # unknown → still try other zones (safer than bailing)
}

section "creating TPU"
WINNING_ZONE=""
for z in $TPU_ZONES_PRIMARY $TPU_ZONES_FALLBACK; do
    if WINNING_ZONE=$(create_in_zone "$z" 2>>"$(state_dir)/provision.log"); then
        break
    else
        rc=$?
        if (( rc == 2 )); then
            log_err "Hard config failure — not retrying other zones."
            exit 2
        fi
        # capacity / unknown → continue
    fi
done

if [[ -z "$WINNING_ZONE" ]]; then
    log_err "No zone accepted the create request. Spot capacity for $TPU_ACCEL is exhausted right now."
    log_err "Options:"
    log_err "  1. Wait 15-30 min and re-run (capacity rotates)."
    log_err "  2. Try on-demand: TPU_PROVISIONING_MODEL=STANDARD ./scripts/20_provision_tpu.sh"
    log_err "     (~3× the price; check 91_predict_cost.sh first)"
    log_err "  3. Add more zones to TPU_ZONES_FALLBACK in lib/config.sh."
    exit 1
fi

# ── 3. Persist state ──────────────────────────────────────────────────────────
section "persisting state"
state_set TPU_NAME  "$TPU_NAME"
state_set TPU_ZONE  "$WINNING_ZONE"
state_set TPU_ACCEL "$TPU_ACCEL"
state_set TPU_PROVISIONING_MODEL "$TPU_PROVISIONING_MODEL"
state_set TPU_PROVISIONED_AT "$(date -Iseconds)"
log_ok "State recorded → $(state_dir)/state.env"

# ── 4. Cost reminder ──────────────────────────────────────────────────────────
hourly="${PRICE_USD_PER_HR[tpu_v5litepod_1]:-0.36}"
log_warn "BILLING HAS STARTED — \$$hourly/hr"
log_warn "Don't forget: ./scripts/70_teardown_tpu.sh when done."
log_ok "Next: ./scripts/21_wait_tpu_ready.sh"
exit 0
