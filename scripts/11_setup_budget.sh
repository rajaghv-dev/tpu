#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 11_setup_budget.sh — Stage 1b: configure a GCP billing budget alert.
#
# RECOMMENDATIONS R8 — "A forgotten v6e VM at \$0.75/hr × 168 hrs = \$126."
# Without a budget alert you only learn about the spend at month-end. This
# script creates a soft alert so you get an email when burn crosses 50/90/100%
# of $SESSION_BUDGET_USD (default \$5/month).
#
# IAM caveat:
#   Creating a budget requires `roles/billing.user` on the BILLING ACCOUNT,
#   not just on the project. Personal accounts usually have this for their
#   own billing account; org-managed projects often do not. If creation fails
#   with permission-denied, ask your billing-account admin (or use the
#   console: Billing → Budgets & alerts → Create).
#
# Idempotent: re-running only creates the budget if no budget with the
# expected display name already exists.
#
# Usage:
#   ./scripts/11_setup_budget.sh
#   SESSION_BUDGET_USD=20 ./scripts/11_setup_budget.sh    # override default
#
# Exit codes:
#   0 = budget created OR already existed.
#   1 = creation failed; manual setup required (URL printed).
#   2 = skipped (caller declined or no billing account permission).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="11_setup_budget"
setup_error_trap
banner "Stage 1b — Budget alert (R8)"

require_cmd gcloud

DISPLAY_NAME="${BUDGET_DISPLAY_NAME:-tpu-bench-budget}"
AMOUNT="$SESSION_BUDGET_USD"
log_info "Project : $GCP_PROJECT"
log_info "Budget  : \$$AMOUNT/month (display name: $DISPLAY_NAME)"

# ── 1. Resolve billing account ────────────────────────────────────────────────
billing_account=$(gcloud beta billing projects describe "$GCP_PROJECT" \
    --format='value(billingAccountName)' 2>/dev/null || echo "")
if [[ -z "$billing_account" ]]; then
    log_err "No billing account linked to $GCP_PROJECT. Link one first."
    exit 1
fi
billing_account_id="${billing_account#billingAccounts/}"
log_ok "Billing account: $billing_account_id"

# ── 2. Already exists? ────────────────────────────────────────────────────────
existing=$(gcloud billing budgets list --billing-account="$billing_account_id" \
    --format="value(displayName)" 2>/dev/null | grep -Fx "$DISPLAY_NAME" || true)
if [[ -n "$existing" ]]; then
    log_ok "Budget '$DISPLAY_NAME' already exists — nothing to do."
    log_info "Manage at: https://console.cloud.google.com/billing/$billing_account_id/budgets"
    exit 0
fi

# ── 3. Confirm with user ──────────────────────────────────────────────────────
log_info "Thresholds: 50%, 90%, 100% (email to project billing owners)"
if ! confirm "Create budget '$DISPLAY_NAME' for \$$AMOUNT/month?" Y; then
    log_warn "Skipped. Create manually at:"
    log_warn "  https://console.cloud.google.com/billing/$billing_account_id/budgets/create"
    exit 2
fi

# ── 4. Create ─────────────────────────────────────────────────────────────────
# Note: scopes the budget to the single project so cross-project spend doesn't
# trip alerts for an unrelated workload.
if gcloud billing budgets create \
        --billing-account="$billing_account_id" \
        --display-name="$DISPLAY_NAME" \
        --budget-amount="$AMOUNT" \
        --threshold-rule=percent=0.5 \
        --threshold-rule=percent=0.9 \
        --threshold-rule=percent=1.0 \
        --filter-projects="projects/$GCP_PROJECT" 2>&1 | tee /tmp/budget_out.$$; then
    log_ok "Budget created."
    rm -f /tmp/budget_out.$$
else
    rc=$?
    err=$(cat /tmp/budget_out.$$ 2>/dev/null || true)
    rm -f /tmp/budget_out.$$
    log_err "Budget creation failed (exit $rc):"
    log_err "  $err"
    log_err "Manual setup: https://console.cloud.google.com/billing/$billing_account_id/budgets/create"
    exit 1
fi

log_ok "Budget alert active. Next: ./scripts/20_provision_tpu.sh"
exit 0
