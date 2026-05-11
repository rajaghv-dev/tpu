#!/usr/bin/env bash
# One-time GCP setup for the TPU benchmark project.
# Idempotent — re-run safely; each phase skips work already done.
#
# Usage:
#   ./scripts/gcp_bootstrap.sh           # interactive bootstrap
#   ./scripts/gcp_bootstrap.sh --check   # verify-only, no changes
#
# Env overrides:
#   REGION=us-west4  ZONE=us-west4-a  PROJECT_ID=<existing>  BUCKET_SUFFIX=models
#
# Covers Phases 1–7 of the GCP setup playbook. Phases 8 (quota) and 9 (budget)
# are web-UI only — this script prints the links and pauses for those.

set -euo pipefail

REGION="${REGION:-us-west4}"
ZONE="${ZONE:-us-west4-a}"
BUCKET_SUFFIX="${BUCKET_SUFFIX:-models}"
PROJECT_ID="${PROJECT_ID:-}"

REQUIRED_APIS=(
  compute.googleapis.com
  tpu.googleapis.com
  storage.googleapis.com
  logging.googleapis.com
  monitoring.googleapis.com
  billingbudgets.googleapis.com
)

CHECK_ONLY=false
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=true

log()   { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
ok()    { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn()  { printf "  \033[33m!\033[0m %s\n" "$*"; }
err()   { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; }
pause() {
  printf "\n\033[33mACTION REQUIRED:\033[0m %s\n" "$1"
  printf "Press Enter when done (Ctrl+C to abort) ... "
  read -r
}

FAILED=0
fail_check() { err "$1"; FAILED=$((FAILED+1)); }

# ── Phase 1: gcloud CLI ─────────────────────────────────────────────────
log "Phase 1 — gcloud CLI"
if command -v gcloud >/dev/null 2>&1; then
  ok "gcloud present: $(gcloud --version 2>/dev/null | head -1)"
else
  $CHECK_ONLY && { fail_check "gcloud not installed"; }
  if ! $CHECK_ONLY; then
    warn "gcloud not installed — installing via apt (needs sudo) ..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq apt-transport-https ca-certificates gnupg curl
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq google-cloud-cli
    ok "gcloud installed"
  fi
fi

if ! command -v gcloud >/dev/null 2>&1; then
  err "gcloud unavailable — cannot continue"
  exit 1
fi

# ── Phase 2: Authentication ─────────────────────────────────────────────
log "Phase 2 — authentication"
ACTIVE_ACCT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
if [[ -n "$ACTIVE_ACCT" ]]; then
  ok "Active account: $ACTIVE_ACCT"
else
  if $CHECK_ONLY; then
    fail_check "no active gcloud account"
  else
    warn "No active account — opening browser for gcloud auth login ..."
    gcloud auth login
    ACTIVE_ACCT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)
    ok "Authenticated as: $ACTIVE_ACCT"
  fi
fi

if gcloud auth application-default print-access-token >/dev/null 2>&1; then
  ok "Application Default Credentials present"
else
  if $CHECK_ONLY; then
    fail_check "no Application Default Credentials"
  else
    warn "Setting up ADC ..."
    gcloud auth application-default login
    ok "ADC configured"
  fi
fi

# ── Phase 3: Project ────────────────────────────────────────────────────
log "Phase 3 — project"
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
if [[ -n "${CURRENT_PROJECT:-}" && "$CURRENT_PROJECT" != "(unset)" ]]; then
  PROJECT_ID="$CURRENT_PROJECT"
  ok "Project: $PROJECT_ID"
elif $CHECK_ONLY; then
  fail_check "no project configured"
  PROJECT_ID=""
else
  if [[ -z "$PROJECT_ID" ]]; then
    echo "  Existing projects:"
    gcloud projects list --format="value(projectId,name)" 2>/dev/null | sed 's/^/    /' || true
    echo
    read -r -p "  Project ID to use (blank to create new): " PROJECT_ID
  fi
  if [[ -z "$PROJECT_ID" ]]; then
    PROJECT_ID="tpu-bench-$(date +%s)"
    warn "Creating new project: $PROJECT_ID"
    gcloud projects create "$PROJECT_ID" --name="TPU Benchmark"
  fi
  gcloud config set project "$PROJECT_ID" >/dev/null
  ok "Project set: $PROJECT_ID"
fi

# ── Phase 4: Billing ────────────────────────────────────────────────────
log "Phase 4 — billing"
if [[ -z "${PROJECT_ID:-}" ]]; then
  fail_check "skipping (no project)"
else
  # `gcloud billing` is GA; fall back to beta if missing.
  BILLING_CMD="gcloud billing"
  if ! gcloud billing projects --help >/dev/null 2>&1; then
    BILLING_CMD="gcloud beta billing"
  fi
  BILLING_ENABLED=$($BILLING_CMD projects describe "$PROJECT_ID" \
    --format="value(billingEnabled)" 2>/dev/null || echo "False")
  if [[ "$BILLING_ENABLED" == "True" ]]; then
    ok "Billing linked"
  elif $CHECK_ONLY; then
    fail_check "billing not enabled for $PROJECT_ID"
  else
    warn "Billing not linked for $PROJECT_ID"
    echo "  Open https://console.cloud.google.com/billing/projects"
    echo "  Link $PROJECT_ID to a billing account (claim \$300 free credit if first time)"
    pause "Link billing in the web console"
    BILLING_ENABLED=$($BILLING_CMD projects describe "$PROJECT_ID" \
      --format="value(billingEnabled)" 2>/dev/null || echo "False")
    [[ "$BILLING_ENABLED" == "True" ]] || { err "Still not linked — aborting"; exit 1; }
    ok "Billing linked"
  fi
fi

# ── Phase 5: APIs ───────────────────────────────────────────────────────
log "Phase 5 — APIs"
ENABLED=$(gcloud services list --enabled --format="value(config.name)" 2>/dev/null || true)
for api in "${REQUIRED_APIS[@]}"; do
  if grep -qx "$api" <<<"$ENABLED"; then
    ok "$api"
  elif $CHECK_ONLY; then
    fail_check "$api not enabled"
  else
    warn "Enabling $api ..."
    gcloud services enable "$api"
    ok "$api enabled"
  fi
done

# ── Phase 6: Defaults ───────────────────────────────────────────────────
log "Phase 6 — defaults"
if $CHECK_ONLY; then
  CUR_REGION=$(gcloud config get-value compute/region 2>/dev/null || echo "(unset)")
  CUR_ZONE=$(gcloud config get-value compute/zone 2>/dev/null || echo "(unset)")
  [[ "$CUR_REGION" == "$REGION" ]] && ok "region=$REGION" || fail_check "region is $CUR_REGION (want $REGION)"
  [[ "$CUR_ZONE" == "$ZONE" ]] && ok "zone=$ZONE" || fail_check "zone is $CUR_ZONE (want $ZONE)"
else
  gcloud config set compute/region "$REGION" >/dev/null
  gcloud config set compute/zone "$ZONE" >/dev/null
  ok "region=$REGION zone=$ZONE"
fi

# ── Phase 7: GCS bucket ─────────────────────────────────────────────────
log "Phase 7 — GCS bucket for model cache (ADR-006)"
if [[ -z "${PROJECT_ID:-}" ]]; then
  fail_check "skipping (no project)"
else
  BUCKET="gs://${PROJECT_ID}-${BUCKET_SUFFIX}"
  if gsutil ls -b "$BUCKET" >/dev/null 2>&1; then
    ok "Bucket exists: $BUCKET"
  elif $CHECK_ONLY; then
    fail_check "bucket missing: $BUCKET"
  else
    warn "Creating bucket: $BUCKET (region=$REGION) ..."
    gsutil mb -l "$REGION" -c STANDARD "$BUCKET"
    LIFECYCLE=$(mktemp --suffix=.json)
    cat > "$LIFECYCLE" <<'JSON'
{"lifecycle": {"rule": [{"action": {"type": "Delete"}, "condition": {"age": 90}}]}}
JSON
    gsutil lifecycle set "$LIFECYCLE" "$BUCKET"
    rm -f "$LIFECYCLE"
    ok "Bucket created with 90-day weight-cache lifecycle"
  fi
fi

# ── Phase 8: TPU quota (informational) ──────────────────────────────────
log "Phase 8 — TPU quota (web-UI only; can't auto-request)"
echo "  Current TPU-related quotas in $REGION:"
gcloud compute regions describe "$REGION" --format=json 2>/dev/null \
  | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    rows = [q for q in d.get('quotas', []) if 'TPU' in q.get('metric','')]
    if not rows:
        print('    (none visible — quota request may be needed)')
    for q in rows:
        print(f\"    {q['metric']:48s} {q.get('usage',0)}/{q.get('limit',0)}\")
except Exception as e:
    print(f'    (could not parse: {e})')
" 2>/dev/null || warn "Quota inspection failed — check console manually"

cat <<EOF

  To request quota (one-time, 24–72h approval):
    1. https://console.cloud.google.com/iam-admin/quotas
    2. Filter: Service=Cloud TPU API,
               Metric=Preemptible TPU v5 Lite chips per region,
               Region=$REGION
    3. Edit Quotas → request 1 → submit
    4. Justification: "ML inference benchmarking — single-host v5e-1 preemptible,
                       ≤\$50/month, 75-model registry"
EOF

# ── Phase 9: Budget alert (programmatic) ────────────────────────────────
log "Phase 9 — budget alert"
echo "  Highly recommended before your first provision (a forgotten v5e-1"
echo "  preemptible = \$0.36/hr × 24 × 30 ≈ \$259/month)."
echo
echo "  Programmatic (alerts at 50/90/100%, scoped to this project only):"
echo "    ./scripts/gcp_set_budget.sh                  # \$20/mo default"
echo "    AMOUNT=50 ./scripts/gcp_set_budget.sh        # custom amount"
echo
echo "  Note: budgets ALERT only — they do not hard-stop services."
echo "  See gcp_set_budget.sh footer for the Pub/Sub + Cloud Function"
echo "  pattern that hard-caps by unlinking billing automatically."
echo
echo "  Panic button if you need to nuke all VMs right now:"
echo "    ./scripts/kill_all_tpus.sh                   # list + confirm + delete"
echo "    ./scripts/kill_all_tpus.sh --dry-run         # list only"

# ── Summary ─────────────────────────────────────────────────────────────
log "Summary"
printf "  Account:   %s\n" "${ACTIVE_ACCT:-(unset)}"
printf "  Project:   %s\n" "${PROJECT_ID:-(unset)}"
printf "  Region:    %s\n" "$REGION"
printf "  Zone:      %s\n" "$ZONE"
printf "  Bucket:    %s\n" "${BUCKET:-(unset)}"

if $CHECK_ONLY; then
  echo
  if [[ "$FAILED" -eq 0 ]]; then
    ok "All checks passed — ready for ./scripts/provision_tpu.sh (after quota approval)"
    exit 0
  else
    err "$FAILED check(s) failed — re-run without --check to fix"
    exit 1
  fi
fi

cat <<EOF

Next:
  1. Request TPU quota (Phase 8 above) — wait for approval email.
  2. Set budget alert (Phase 9 above).
  3. Verify anytime:  ./scripts/gcp_bootstrap.sh --check
  4. When quota approves:  ./scripts/provision_tpu.sh
EOF
