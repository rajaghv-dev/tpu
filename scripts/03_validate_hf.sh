#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 03_validate_hf.sh — Stage 0d: HuggingFace credential preflight.
#
# Verifies HF_TOKEN is set and valid BEFORE we burn TPU minutes downloading
# weights that turn out to be gated. Mitigates R-T07.
#
# Stage-1 gated-model check:
#   The default Stage 1 registry (BERT-base, ViT-B/16, GPT-2, Whisper-base,
#   CLIP-ViT-B/32) has NO gated models, so this script is OPTIONAL for the
#   smoke and quick suites. It becomes mandatory once Gemma/PaliGemma/Llama
#   join the registry (Stages 5+).
#
# Usage:
#   ./scripts/03_validate_hf.sh                # check token if HF_TOKEN set
#   ./scripts/03_validate_hf.sh --required     # fail if HF_TOKEN unset
#
# Exit codes:
#   0 = token valid (or absent and not required).
#   1 = token required but unset, or token invalid.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="03_validate_hf"
setup_error_trap
banner "Stage 0d — HuggingFace credential preflight"

REQUIRED=0
[[ "${1:-}" == "--required" ]] && REQUIRED=1

# ── 1. HF_TOKEN env var ───────────────────────────────────────────────────────
section "HF_TOKEN env var"
if [[ -z "${HF_TOKEN:-}" ]]; then
    if (( REQUIRED == 1 )); then
        log_err "HF_TOKEN unset but --required passed"
        log_err "  → Get a token: https://huggingface.co/settings/tokens (read scope)"
        log_err "  → Then: export HF_TOKEN=hf_xxxx; ./scripts/03_validate_hf.sh"
        exit 1
    else
        log_warn "HF_TOKEN unset — skipping validation."
        log_warn "  Stage 1 (BERT, ViT, GPT-2, Whisper, CLIP) doesn't need it; later stages will."
        exit 0
    fi
fi
# Mask token in logs — never echo full bytes. Format check (HF tokens look like
# `hf_<chars>` or `hf-<chars>` with at least 30 chars).
masked="${HF_TOKEN:0:6}…${HF_TOKEN: -4}"
log_ok "HF_TOKEN present (masked: $masked, length=${#HF_TOKEN})"

# ── 2. Token validity probe ───────────────────────────────────────────────────
# The /api/whoami-v2 endpoint returns 200 + JSON for a valid token, 401 for
# invalid. We use curl to keep this script Python-free (no HF SDK install).
section "token validity (huggingface.co/api/whoami-v2)"
if ! command -v curl >/dev/null 2>&1; then
    log_warn "curl not installed; cannot validate token remotely. Trusting it."
    exit 0
fi
http_code=$(curl -s -o /tmp/hf_whoami.$$.json -w "%{http_code}" \
    -H "Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/api/whoami-v2 || echo "000")

case "$http_code" in
    200)
        user=$(python3 -c "import json,sys; print(json.load(open('/tmp/hf_whoami.$$.json')).get('name',''))" 2>/dev/null || echo "?")
        plan=$(python3 -c "
import json
d = json.load(open('/tmp/hf_whoami.$$.json'))
print(d.get('type','?'), '| paid:', d.get('isPro', d.get('canPay', False)))
" 2>/dev/null || echo "?")
        log_ok "authenticated as: $user ($plan)"
        ;;
    401)
        log_err "token rejected (401 unauthorized)"
        log_err "  → Token may be expired or revoked. Generate a new one: https://huggingface.co/settings/tokens"
        rm -f /tmp/hf_whoami.$$.json
        exit 1
        ;;
    000)
        log_err "could not reach huggingface.co — network issue?"
        exit 1
        ;;
    *)
        log_warn "unexpected HTTP $http_code from whoami; token MAY be valid but couldn't confirm"
        ;;
esac
rm -f /tmp/hf_whoami.$$.json

# ── 3. Optional: probe the first gated model in the registry ──────────────────
# If the user has a gated model registered (e.g. Gemma) we surface a 403 here
# rather than 30 minutes into the run.
section "gated-model probe"
GATED_PROBE_HF_ID="${GATED_PROBE_HF_ID:-google/gemma-2b}"
http_code=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/models/$GATED_PROBE_HF_ID" || echo "000")
case "$http_code" in
    200) log_ok "$GATED_PROBE_HF_ID accessible (gated check passes)" ;;
    403) log_warn "$GATED_PROBE_HF_ID returned 403 — accept the model's licence on its HF page first" ;;
    404) log_warn "$GATED_PROBE_HF_ID not found — repo renamed or removed; not a token problem" ;;
    *)   log_warn "$GATED_PROBE_HF_ID probe returned HTTP $http_code" ;;
esac

section "summary"
log_ok "HuggingFace preflight passed. Next: ./scripts/10_setup_bucket.sh (or 20_provision_tpu.sh if bucket exists)"
exit 0
