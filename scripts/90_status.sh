#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 90_status.sh — show active GCP resources + estimated hourly burn rate.
#
# Honest scope. The Cloud Billing API requires either (a) a BigQuery billing
# export or (b) special billing-account roles to read historical totals. This
# script doesn't pretend to know what you spent yesterday — it just totals the
# CURRENT BURN RATE from resources it can list with `compute.viewer`-level
# perms.
#
# If you DO have billing export configured, set BILLING_BQ_TABLE in the env
# and the script will additionally print month-to-date spend via `bq query`.
#
# Usage:
#   ./scripts/90_status.sh
#   BILLING_BQ_TABLE=my-proj.billing.gcp_billing_export_v1_xxx ./scripts/90_status.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="90_status"
setup_error_trap
banner "Status — current GCP burn rate"

require_cmd gcloud
require_env GCP_PROJECT

burn=0   # accumulated hourly USD as integer milli-cents (avoid floating point)

# Helper: add `dollars-per-hr × 1000` to burn (so 0.36 → 360 milli-cents).
add_burn() {
    local price="$1"; local count="${2:-1}"
    # Use bash arithmetic for integer math at the milli-cent scale.
    local mc
    mc=$(awk -v p="$price" -v c="$count" 'BEGIN{ printf "%d", p*c*1000 }')
    burn=$((burn + mc))
}

# ── 1. TPUs ───────────────────────────────────────────────────────────────────
section "TPUs"
for z in $TPU_ZONES_PRIMARY $TPU_ZONES_FALLBACK; do
    while IFS=$'\t' read -r name state accel; do
        [[ -z "$name" ]] && continue
        # We don't price each TPU type individually — assume v5e-1 unless we can
        # tell. v6e-1 is also possible; both are in the price table.
        local_key="tpu_v5litepod_1"
        [[ "$accel" == "v6e-1" ]] && local_key="tpu_v6e_1"
        price="${PRICE_USD_PER_HR[$local_key]:-0}"
        log_warn "$z/$name (type=$accel state=$state): \$$price/hr"
        add_burn "$price"
    done < <(gcloud compute tpus tpu-vm list --zone="$z" \
        --format="value(name,state,acceleratorType)" 2>/dev/null || true)
done

# ── 2. Compute Engine instances ───────────────────────────────────────────────
# Only RUNNING instances bill at full rate. We attribute a flat n1-standard-4
# rate plus the GPU price if a GPU is attached. Imperfect but close.
section "Compute Engine instances"
while IFS=$'\t' read -r name zone status mtype gpu; do
    [[ -z "$name" ]] && continue
    [[ "$status" != "RUNNING" ]] && continue
    # GPU price (if any). Map gpu type → key in PRICE_USD_PER_HR.
    gpu_price=0
    case "$gpu" in
        nvidia-tesla-t4) gpu_price="${PRICE_USD_PER_HR[gpu_t4]:-0}" ;;
        nvidia-l4)       gpu_price="${PRICE_USD_PER_HR[gpu_l4]:-0}" ;;
        "") : ;;  # CPU-only
        *) log_warn "Unknown GPU type $gpu; price not estimated" ;;
    esac
    # Base CPU/RAM cost for the n1-standard-4 host (we don't try to price every
    # machine type — keep this honest as an estimate).
    base_price="${PRICE_USD_PER_HR[n1_standard_4]:-0}"
    inst_price=$(awk -v a="$base_price" -v b="$gpu_price" 'BEGIN{print a+b}')
    log_warn "$zone/$name (mtype=$mtype, gpu=${gpu:-none}): ~\$$inst_price/hr"
    add_burn "$inst_price"
done < <(gcloud compute instances list \
    --format="value(name,zone.basename(),status,machineType.basename(),guestAccelerators[0].acceleratorType.basename())" 2>/dev/null || true)

# ── 3. Reserved-but-idle static IPs ───────────────────────────────────────────
section "Idle reserved IPs"
n_ips=0
while IFS=$'\t' read -r name addr region; do
    [[ -z "$name" ]] && continue
    n_ips=$((n_ips+1))
    log_warn "$region/$name ($addr): ~\$0.005/hr"
done < <(gcloud compute addresses list --filter="status=RESERVED" \
    --format="value(name,address,region.basename())" 2>/dev/null || true)
if (( n_ips > 0 )); then
    add_burn "${PRICE_USD_PER_HR[external_ip_in_use]:-0.005}" "$n_ips"
fi

# ── 4. Unattached persistent disks (storage cost, not hourly compute) ────────
section "Unattached disks (storage cost only)"
total_gb=0
while IFS=$'\t' read -r name size type zone; do
    [[ -z "$name" ]] && continue
    log_warn "$zone/$name: ${size}GB ($type)"
    total_gb=$((total_gb + size))
done < <(gcloud compute disks list --filter="-users:*" \
    --format="value(name,sizeGb,type,zone.basename())" 2>/dev/null || true)
if (( total_gb > 0 )); then
    monthly=$(awk -v gb="$total_gb" -v p="${PRICE_USD_PER_HR[pd_balanced_gb_mo]:-0.10}" 'BEGIN{printf "%.2f", gb*p}')
    log_warn "Orphan disks total: ${total_gb}GB ≈ \$$monthly/month"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "summary"
hourly_usd=$(awk -v mc="$burn" 'BEGIN{printf "%.3f", mc/1000}')
daily_usd=$(awk -v h="$hourly_usd" 'BEGIN{printf "%.2f", h*24}')
log_info "Hourly burn (compute, IPs): \$${hourly_usd}/hr"
log_info "Projected if left running: \$${daily_usd}/day"

# ── Optional MTD spend via BigQuery billing export ────────────────────────────
if [[ -n "${BILLING_BQ_TABLE:-}" ]] && command -v bq >/dev/null 2>&1; then
    section "Month-to-date spend (BigQuery billing export)"
    bq query --use_legacy_sql=false --format=prettyjson \
        "SELECT SUM(cost) AS mtd_usd FROM \`$BILLING_BQ_TABLE\`
         WHERE TIMESTAMP_TRUNC(usage_start_time, MONTH) =
               TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)
           AND project.id = '$GCP_PROJECT'" 2>&1 | tail -10 || \
        log_warn "BQ query failed; check BILLING_BQ_TABLE is correct."
fi

if [[ -z "${BILLING_BQ_TABLE:-}" ]]; then
    log_info "Set BILLING_BQ_TABLE=<dataset.table> + install bq for month-to-date totals."
fi
exit 0
