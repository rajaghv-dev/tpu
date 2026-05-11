#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 10_setup_bucket.sh — Stage 1: create the GCS model+compile cache bucket.
#
# Idempotent. If the bucket already exists this exits 0 with a friendly note;
# it never overwrites existing buckets or modifies their policy.
#
# Why this exists:
#   ADR-006 + RECOMMENDATIONS R1 — every TPU/GPU VM mounts this bucket via
#   gcsfuse and uses it for both the HuggingFace cache (HF_HOME) and the JAX
#   compilation cache (JAX_COMPILATION_CACHE_DIR). Without it the harness
#   re-downloads weights on every preemptible-VM lifetime (R-C01).
#
# What it creates:
#   - A single-region bucket at $GCS_BUCKET in $GCS_BUCKET_REGION.
#   - Uniform bucket-level access enabled (R1 — IAM-only, no per-object ACLs).
#   - Standard storage class.
#
# Usage:
#   ./scripts/10_setup_bucket.sh
#
# Exit codes:
#   0 = bucket created OR already existed.
#   1 = creation failed (perms, name collision, etc.).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="10_setup_bucket"
setup_error_trap
banner "Stage 1 — Create GCS model cache bucket (idempotent)"

require_cmd gcloud

# ── Already there? ────────────────────────────────────────────────────────────
# A successful describe means we're done; no action needed.
if gcloud storage buckets describe "$GCS_BUCKET" >/dev/null 2>&1; then
    log_ok "Bucket $GCS_BUCKET already exists — nothing to do."
    log_info "If you want to verify region/policy, run: ./scripts/02_validate_bucket.sh"
    exit 0
fi

# ── Confirm intent ────────────────────────────────────────────────────────────
# Bucket names are globally unique. If $GCS_BUCKET is taken by someone else,
# `buckets create` returns 409. Surface that instead of a stack trace.
log_info "Will create:"
log_info "  Bucket : $GCS_BUCKET"
log_info "  Region : $GCS_BUCKET_REGION (single-region, standard storage)"
log_info "  Policy : uniform bucket-level access (IAM-only)"
log_info "  Cost   : ~\$1.60/month for ~80 GB of model weights (README §Cost Reference)"

if ! confirm "Create the bucket now?" Y; then
    log_warn "Skipped at user request."
    exit 0
fi

# ── Create ────────────────────────────────────────────────────────────────────
section "creating"
if gcloud storage buckets create "$GCS_BUCKET" \
        --location="$GCS_BUCKET_REGION" \
        --uniform-bucket-level-access \
        --default-storage-class=STANDARD; then
    log_ok "Bucket created."
else
    rc=$?
    log_err "Bucket creation failed (exit $rc)."
    log_err "  Common causes:"
    log_err "    - Name '$GCS_BUCKET' already taken globally → set GCS_BUCKET=gs://your-unique-name"
    log_err "    - Caller lacks roles/storage.admin → ask project owner to grant it"
    log_err "    - Region $GCS_BUCKET_REGION typo → must be a single-region location code"
    exit 1
fi

section "verification"
"$SCRIPT_DIR/02_validate_bucket.sh"

log_ok "Bucket setup complete. Next: ./scripts/20_provision_tpu.sh"
exit 0
