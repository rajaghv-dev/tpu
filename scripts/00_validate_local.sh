#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 00_validate_local.sh — Stage 0a: local environment preflight.
#
# Verifies that the LOCAL machine has everything it needs to drive the TPU
# benchmark. Read-only; never mutates state. Run first, before everything else.
#
# Checks:
#   1. bash 4+ (lib/common.sh assumes associative arrays).
#   2. gcloud installed and on PATH.
#   3. gcloud authenticated (at least one ACTIVE account).
#   4. gcloud project set (defaults to that account's primary).
#   5. python3 + tar present (we tar+scp the repo to the TPU VM).
#   6. ssh (gcloud delegates to system ssh).
#
# Usage:
#   ./scripts/00_validate_local.sh
#
# Exit codes:
#   0  = everything passes — safe to proceed to 01_validate_gcp.sh.
#   1  = one or more critical checks failed.
#
# Idempotent. Run as often as you like. Re-running is the right move after you
# install a missing tool.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="00_validate_local"
setup_error_trap
banner "Stage 0a — Local environment preflight"

# Tally counters so the summary at the end is honest.
PASS=0; WARN=0; FAIL=0

# Helper: run a check, classify the result, count it. Each check function below
# echoes one of OK/WARN/FAIL on stdout; this wrapper does the bookkeeping.
check() {
    local name="$1"; shift
    local detail
    detail="$("$@" 2>&1)" || true
    case "$detail" in
        OK*)   log_ok   "$name — ${detail#OK }";   PASS=$((PASS+1)) ;;
        WARN*) log_warn "$name — ${detail#WARN }"; WARN=$((WARN+1)) ;;
        FAIL*) log_err  "$name — ${detail#FAIL }"; FAIL=$((FAIL+1)) ;;
        *)     log_err  "$name — unexpected check output: $detail"; FAIL=$((FAIL+1)) ;;
    esac
}

# ── Individual checks ─────────────────────────────────────────────────────────
# Each prints one of:  "OK <message>", "WARN <message>", or "FAIL <message>".
# This contract keeps the summary code dumb.

check_bash_version() {
    if (( BASH_VERSINFO[0] >= 4 )); then
        echo "OK bash $BASH_VERSION"
    else
        echo "FAIL bash $BASH_VERSION; need 4+ (macOS users: brew install bash)"
    fi
}

check_gcloud() {
    if command -v gcloud >/dev/null 2>&1; then
        echo "OK $(gcloud --version 2>/dev/null | head -1)"
    else
        echo "FAIL gcloud not on PATH (install: https://cloud.google.com/sdk/docs/install)"
    fi
}

check_gcloud_auth() {
    local account
    account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)
    if [[ -n "$account" ]]; then
        echo "OK active account: $account"
    else
        echo "FAIL no ACTIVE gcloud account; run: gcloud auth login"
    fi
}

check_gcloud_project() {
    if [[ -z "$GCP_PROJECT" ]]; then
        echo "FAIL no project set; run: gcloud config set project <PROJECT_ID>"
    else
        echo "OK project=$GCP_PROJECT"
    fi
}

check_python3() {
    if command -v python3 >/dev/null 2>&1; then
        echo "OK $(python3 --version 2>&1)"
    else
        echo "FAIL python3 not on PATH"
    fi
}

check_tar() {
    if command -v tar >/dev/null 2>&1; then
        echo "OK $(tar --version 2>&1 | head -1)"
    else
        echo "FAIL tar not on PATH"
    fi
}

check_ssh() {
    if command -v ssh >/dev/null 2>&1; then
        echo "OK $(ssh -V 2>&1)"
    else
        echo "WARN ssh not on PATH; gcloud uses bundled ssh on some platforms — usually fine"
    fi
}

# Optional: gsutil (only needed for 02_validate_bucket.sh and 60_pull_results)
check_gsutil() {
    if command -v gsutil >/dev/null 2>&1; then
        echo "OK $(gsutil version 2>&1 | head -1)"
    else
        echo "WARN gsutil not on PATH; needed for GCS bucket interactions (install: gcloud components install gsutil)"
    fi
}

# ── Run them ──────────────────────────────────────────────────────────────────
section "checks"
check "bash version"      check_bash_version
check "gcloud installed"  check_gcloud
check "gcloud auth"       check_gcloud_auth
check "gcloud project"    check_gcloud_project
check "python3"           check_python3
check "tar"               check_tar
check "ssh"               check_ssh
check "gsutil"            check_gsutil

# ── Summary ───────────────────────────────────────────────────────────────────
section "summary"
log_info "Pass=$PASS  Warn=$WARN  Fail=$FAIL"
if (( FAIL > 0 )); then
    log_err "Local preflight FAILED. Address the issues above and re-run."
    exit 1
fi
log_ok "Local preflight passed. Next: ./scripts/01_validate_gcp.sh"
exit 0
