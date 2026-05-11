#!/usr/bin/env bash
# Create a Cloud Billing budget with email alerts via the API.
# Idempotent — skips if a budget with the same display name already exists.
#
# Usage:
#   ./scripts/gcp_set_budget.sh                # $20/mo, alerts at 50/90/100%
#   AMOUNT=50 ./scripts/gcp_set_budget.sh      # custom amount in USD
#   NAME="My Cap" ./scripts/gcp_set_budget.sh  # custom display name
#
# What this does:
#   - Looks up the billing account linked to the current project.
#   - Creates a budget scoped ONLY to this project (won't affect other projects).
#   - Alerts at 50% / 90% / 100% of $AMOUNT (current-spend basis).
#   - Email goes to billing account admins by default (i.e. you).
#
# What this does NOT do:
#   - Hard-cap spend. GCP budgets ALERT, they do not STOP services.
#   - For a true hard cap, you need Budget → Pub/Sub → Cloud Function that
#     calls `gcloud billing projects unlink`. That's a separate setup; see
#     comment block at the bottom of this file.

set -euo pipefail

AMOUNT="${AMOUNT:-20}"
NAME="${NAME:-TPU Bench Budget}"
CURRENCY="${CURRENCY:-USD}"

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "✗ No project configured. Run: gcloud config set project <id>" >&2
  exit 1
fi

echo "▶ Project: $PROJECT_ID"

# `gcloud billing` was promoted from beta; fall back if needed.
BILLING="gcloud billing"
$BILLING projects --help >/dev/null 2>&1 || BILLING="gcloud beta billing"

BUDGETS="gcloud billing budgets"
$BUDGETS --help >/dev/null 2>&1 || BUDGETS="gcloud beta billing budgets"

# 1) Find the billing account linked to this project.
BILLING_NAME=$($BILLING projects describe "$PROJECT_ID" \
  --format="value(billingAccountName)" 2>/dev/null || true)
if [[ -z "$BILLING_NAME" || "$BILLING_NAME" == "billingAccounts/" ]]; then
  echo "✗ No billing account linked to $PROJECT_ID." >&2
  echo "  Link one: https://console.cloud.google.com/billing/projects" >&2
  exit 1
fi
BILLING_ID="${BILLING_NAME#billingAccounts/}"
echo "  Billing account: $BILLING_ID"

# 2) Resolve project number (budgets filter on project number, not id).
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" \
  --format="value(projectNumber)" 2>/dev/null)
if [[ -z "$PROJECT_NUMBER" ]]; then
  echo "✗ Could not resolve project number for $PROJECT_ID" >&2
  exit 1
fi
echo "  Project number:  $PROJECT_NUMBER"

# 3) Idempotency: skip if a budget with this display name already exists.
EXISTING=$($BUDGETS list --billing-account="$BILLING_ID" \
  --filter="displayName=\"$NAME\"" \
  --format="value(name)" 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" ]]; then
  echo "✓ Budget '$NAME' already exists: $EXISTING"
  echo "  To re-create, delete it first:"
  echo "    $BUDGETS delete $EXISTING --billing-account=$BILLING_ID"
  exit 0
fi

# 4) Create the budget.
echo "▶ Creating budget '$NAME' = \$${AMOUNT} $CURRENCY/month ..."
$BUDGETS create \
  --billing-account="$BILLING_ID" \
  --display-name="$NAME" \
  --budget-amount="${AMOUNT}${CURRENCY}" \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0 \
  --filter-projects="projects/$PROJECT_NUMBER" \
  --calendar-period=month

echo
echo "✓ Budget created. Email alerts will go to billing-account admins"
echo "  ($BILLING_ID) at 50% / 90% / 100% of \$${AMOUNT}/month."
echo
echo "Inspect / edit:"
echo "  https://console.cloud.google.com/billing/$BILLING_ID/budgets"

# ─────────────────────────────────────────────────────────────────────────
# HARD CAP — additional setup (not done by this script)
#
# Budgets only ALERT. To actually STOP billing when the cap is hit:
#
#   1. Create a Pub/Sub topic:
#        gcloud pubsub topics create budget-alerts
#
#   2. Edit the budget (in console or via `gcloud billing budgets update`)
#      and add the Pub/Sub topic under "Manage notifications".
#
#   3. Deploy a Cloud Function subscribed to that topic. On message receipt,
#      check `costAmount/budgetAmount` from the payload; if ≥ 1.0, call:
#        gcloud billing projects unlink $PROJECT_ID
#      This severs billing and stops all chargeable resources immediately.
#      Restoring requires manual re-link.
#
#   See: https://cloud.google.com/billing/docs/how-to/notify
#
# Worth it if you'll genuinely leave VMs running unattended. For attended
# benchmark runs with explicit teardown, alert-only is usually enough.
# ─────────────────────────────────────────────────────────────────────────
