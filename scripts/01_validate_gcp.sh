#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 01_validate_gcp.sh — Stage 0b: GCP-side preflight.
#
# Confirms the project is funded, APIs are on, IAM is sufficient, and we have
# v5e quota in at least one configured zone. Read-only; never mutates state.
#
# Usage:
#   ./scripts/01_validate_gcp.sh
#
# Inputs:
#   $GCP_PROJECT, $TPU_ZONES_PRIMARY, $TPU_ZONES_FALLBACK from lib/config.sh
#   (override by exporting them before running, or edit lib/config.sh).
#
# Exit codes:
#   0 = all critical checks pass; safe to provision.
#   1 = critical fail (auth/billing/API/no quota in any zone).
#   2 = warnings only (e.g. caller missing one optional IAM perm).
#
# Why this exists:
#   R-T07: forgetting to enable billing on a project is the most expensive
#   mistake — the next gcloud call returns 403 30 minutes into a run.
#   R-C02: spot capacity for v5e is regionally uneven; we want to know NOW
#   if every configured zone has zero quota, not at provision time.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="01_validate_gcp"
setup_error_trap
banner "Stage 0b — GCP preflight"

require_cmd gcloud "https://cloud.google.com/sdk/docs/install"

PASS=0; WARN=0; FAIL=0
log_info "Project: ${GCP_PROJECT:-<unset>}"

# Hard fail early if no project — every subsequent gcloud call would fail too.
if [[ -z "$GCP_PROJECT" ]]; then
    log_err "No GCP project set. Run: gcloud config set project <PROJECT_ID>"
    exit 1
fi

# ── 1. Billing enabled ────────────────────────────────────────────────────────
section "billing"
billing_enabled=$(gcloud beta billing projects describe "$GCP_PROJECT" \
    --format='value(billingEnabled)' 2>/dev/null || echo "False")
if [[ "$billing_enabled" == "True" ]]; then
    log_ok "billing enabled"
    PASS=$((PASS+1))
else
    log_err "billing NOT enabled on $GCP_PROJECT"
    log_err "  → Console: https://console.cloud.google.com/billing/linkedaccount?project=$GCP_PROJECT"
    FAIL=$((FAIL+1))
fi

# ── 2. APIs enabled ───────────────────────────────────────────────────────────
section "APIs"
required_apis=(compute.googleapis.com tpu.googleapis.com)
enabled_apis=$(gcloud services list --enabled --format='value(config.name)' 2>/dev/null)
for api in "${required_apis[@]}"; do
    if grep -qx "$api" <<<"$enabled_apis"; then
        log_ok "$api"
        PASS=$((PASS+1))
    else
        log_err "$api NOT enabled"
        log_err "  → Run: gcloud services enable $api"
        FAIL=$((FAIL+1))
    fi
done

# ── 3. IAM permissions (heuristic) ────────────────────────────────────────────
# Older gcloud versions don't ship `gcloud projects test-iam-permissions`, and
# even where it exists it requires `resourcemanager.projects.getIamPolicy`
# which a non-owner often lacks. We use a softer probe:
#   - `gcloud compute instances list` exercises compute.instances.list
#   - `gcloud compute tpus tpu-vm list` exercises tpu.nodes.list
# If both succeed we assume the caller has enough IAM for provisioning. If
# either fails with PERMISSION_DENIED we surface the missing role. This
# handles the common case (project Owner) and the failure mode (read-only
# Viewer) without needing test-iam-permissions.
section "IAM (read probe — actual create perms checked at provision time)"

# Probe Compute Engine read access by listing in a known zone (us-central1-a
# always exists; an empty list still returns 0 if the caller has the perm).
if gcloud compute instances list --zones=us-central1-a --limit=1 \
        --format='value(name)' >/dev/null 2>&1; then
    log_ok "compute.instances.list — read access confirmed"
    PASS=$((PASS+1))
else
    log_warn "compute.instances.list denied — caller likely lacks roles/compute.viewer"
    log_warn "  → ask project owner to grant roles/compute.instanceAdmin.v1"
    WARN=$((WARN+1))
fi

# Probe Cloud TPU read access. `tpus tpu-vm list` may return ALREADY_EXISTS
# variants depending on perms; we just check exit code.
if gcloud compute tpus tpu-vm list --zone=us-central1-a \
        --format='value(name)' >/dev/null 2>&1; then
    log_ok "tpu.nodes.list — read access confirmed"
    PASS=$((PASS+1))
else
    log_warn "tpu.nodes.list denied — caller likely lacks roles/tpu.viewer"
    log_warn "  → ask project owner to grant roles/tpu.admin"
    WARN=$((WARN+1))
fi

log_info "Note: actual create permissions (compute.instances.create, tpu.nodes.create,"
log_info "      iam.serviceAccounts.actAs) only fail at provision time. Stage 20 will"
log_info "      surface specific missing perms with the gcloud error message."

# ── 4. v5e quota in any configured zone ───────────────────────────────────────
# Quota for v5e single-chip preemptible is regional, named
# PREEMPTIBLE_TPU_LITE_PODSLICE_V5 and counted in chips. We need >= 1.
section "v5e preemptible quota (per region of zone list)"
declare -A region_seen
zones_to_check="$TPU_ZONES_PRIMARY $TPU_ZONES_FALLBACK"
quota_found_anywhere=0
for z in $zones_to_check; do
    region="${z%-*}"   # strip zone suffix → region (us-east5-a → us-east5)
    if [[ -n "${region_seen[$region]:-}" ]]; then continue; fi
    region_seen[$region]=1
    limit=$(gcloud compute regions describe "$region" --format=json 2>/dev/null \
        | python3 -c "
import json, sys
d = json.load(sys.stdin)
for q in d.get('quotas', []):
    if q['metric'] == 'PREEMPTIBLE_TPU_LITE_PODSLICE_V5':
        print(int(q['limit'])); break
else:
    print(0)
" 2>/dev/null || echo "0")
    if (( limit >= 1 )); then
        log_ok "$region: PREEMPTIBLE_TPU_LITE_PODSLICE_V5 limit=$limit"
        PASS=$((PASS+1)); quota_found_anywhere=1
    else
        log_warn "$region: PREEMPTIBLE_TPU_LITE_PODSLICE_V5 limit=$limit (no v5e capacity here)"
        WARN=$((WARN+1))
    fi
done
if (( quota_found_anywhere == 0 )); then
    log_err "No v5e preemptible quota in any region from TPU_ZONES_PRIMARY/FALLBACK"
    log_err "  → Request quota: https://console.cloud.google.com/iam-admin/quotas?project=$GCP_PROJECT"
    FAIL=$((FAIL+1))
fi

# ── 5. Existing TPU with the chosen name? ─────────────────────────────────────
# If the user re-runs after a partial provision, we want to surface the existing
# VM rather than try to create another with the same name (which fails noisily).
section "TPU name collision check"
collision=0
for z in $zones_to_check; do
    state=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$z" \
        --format='value(state)' 2>/dev/null || true)
    if [[ -n "$state" ]]; then
        log_warn "TPU $TPU_NAME already exists in $z (state=$state) — 20_provision_tpu.sh will reuse it"
        WARN=$((WARN+1))
        collision=1
        break
    fi
done
if (( collision == 0 )); then
    log_ok "no existing TPU named '$TPU_NAME' in configured zones"
    PASS=$((PASS+1))
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "summary"
log_info "Pass=$PASS  Warn=$WARN  Fail=$FAIL"
if (( FAIL > 0 )); then
    log_err "GCP preflight FAILED. Resolve the issues above and re-run."
    exit 1
fi
if (( WARN > 0 )); then
    log_warn "GCP preflight passed with warnings — review them before provisioning."
    exit 2
fi
log_ok "GCP preflight passed. Next: ./scripts/02_validate_bucket.sh"
exit 0
