#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 02_validate_bucket.sh — Stage 0c: GCS bucket preflight.
#
# Confirms the model-cache bucket exists, is in the expected region, and the
# caller can read+write it. The bucket itself is created by 10_setup_bucket.sh
# — this script only VERIFIES, never creates.
#
# Why bucket region matters:
#   ADR-006 specifies single-region GCS in us-central1 to match v5e-1 in the
#   same region (free intra-region read). v5e moved away from us-central1 in
#   2025; current default keeps the bucket in us-central1 per ADR but this is
#   a legitimate revisit point — see lib/config.sh "IMPORTANT REGION NOTE".
#
# Usage:
#   ./scripts/02_validate_bucket.sh
#
# Exit codes:
#   0 = bucket exists and is accessible.
#   1 = bucket missing → run 10_setup_bucket.sh.
#   2 = bucket exists but in unexpected region or perms missing → see message.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="02_validate_bucket"
setup_error_trap
banner "Stage 0c — GCS bucket preflight"

require_cmd gcloud
log_info "Checking bucket: $GCS_BUCKET (expected region: $GCS_BUCKET_REGION)"

# ── 1. Bucket exists? ─────────────────────────────────────────────────────────
# `gcloud storage buckets describe` returns non-zero if the bucket doesn't
# exist OR if the caller lacks `storage.buckets.get`. The error string
# differentiates them.
section "existence"
if ! describe_out=$(gcloud storage buckets describe "$GCS_BUCKET" --format=json 2>&1); then
    if echo "$describe_out" | grep -q "404\|not found\|does not exist"; then
        log_err "Bucket $GCS_BUCKET does not exist."
        log_err "  → Create it: ./scripts/10_setup_bucket.sh"
        exit 1
    elif echo "$describe_out" | grep -q "403\|permission"; then
        log_err "Bucket exists but you cannot describe it (storage.buckets.get denied)."
        log_err "  → Ask the project owner for roles/storage.admin on $GCS_BUCKET."
        exit 2
    else
        log_err "Unexpected error: $describe_out"
        exit 1
    fi
fi
log_ok "bucket exists"

# ── 2. Region ─────────────────────────────────────────────────────────────────
section "region"
actual_region=$(echo "$describe_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print((d.get('location') or d.get('Location') or '').lower())
")
if [[ "$actual_region" == "$GCS_BUCKET_REGION" ]]; then
    log_ok "region=$actual_region matches GCS_BUCKET_REGION"
elif [[ -z "$actual_region" ]]; then
    log_warn "could not parse region from describe output"
else
    log_warn "region mismatch: bucket is in '$actual_region', expected '$GCS_BUCKET_REGION'"
    log_warn "  Cross-region GCS reads incur egress charges (~\$0.02/GB)."
    log_warn "  Either: (a) export GCS_BUCKET_REGION=$actual_region (accept egress)"
    log_warn "       or (b) recreate bucket in $GCS_BUCKET_REGION"
fi

# ── 3. Uniform bucket-level access (recommended by R1) ───────────────────────
section "access policy"
ubla=$(echo "$describe_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ubla = d.get('iamConfiguration', {}).get('uniformBucketLevelAccess', {})
print('enabled' if ubla.get('enabled') else 'disabled')
")
if [[ "$ubla" == "enabled" ]]; then
    log_ok "uniform bucket-level access: enabled (matches R1)"
else
    log_warn "uniform bucket-level access: $ubla (R1 recommends enabled)"
fi

# ── 4. Read + write probe ─────────────────────────────────────────────────────
# Atomic write of a small file via `gcloud storage cp` (NOT through gcsfuse —
# R-I01: gcsfuse partial writes can corrupt). Then read it back, then delete.
section "read/write probe"
probe_local=$(mktemp)
probe_remote="$GCS_BUCKET/.tpu-bench-probe-$$"
echo "tpu-bench probe $(date -Iseconds)" > "$probe_local"
add_exit_handler "rm -f '$probe_local'; gcloud storage rm '$probe_remote' >/dev/null 2>&1"

if gcloud storage cp "$probe_local" "$probe_remote" >/dev/null 2>&1; then
    log_ok "write succeeded ($probe_remote)"
else
    log_err "write FAILED — caller likely missing storage.objects.create"
    exit 2
fi
if gcloud storage cp "$probe_remote" "$probe_local.read" >/dev/null 2>&1 \
        && diff -q "$probe_local" "$probe_local.read" >/dev/null; then
    log_ok "read + roundtrip integrity ok"
    rm -f "$probe_local.read"
else
    log_err "read FAILED or content mismatch"
    exit 2
fi
if gcloud storage rm "$probe_remote" >/dev/null 2>&1; then
    log_ok "delete succeeded"
else
    log_warn "delete failed — probe object may linger; harmless but tidy up via console"
fi

section "summary"
log_ok "Bucket preflight passed. Next: ./scripts/03_validate_hf.sh (or skip if not running gated models)"
exit 0
