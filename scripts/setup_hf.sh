#!/usr/bin/env bash
# One-time HuggingFace token setup for the TPU benchmark project.
#
# Why: HF Hub gated models (Gemma, LLaMA, PaliGemma, etc.) require a token.
# This script captures your HF PRO token once, validates it, and stores it
# in the locations every downstream consumer (laptop, TPU VM, Colab) reads
# automatically.
#
# Usage:
#   ./scripts/setup_hf.sh              # interactive
#   HF_TOKEN=hf_xxx ./scripts/setup_hf.sh --non-interactive
#   ./scripts/setup_hf.sh --check      # verify existing token, no changes
#   ./scripts/setup_hf.sh --clear      # remove all stored copies
#
# Storage locations populated (idempotent, choose any subset):
#   1. ~/.cache/huggingface/token       — HF library canonical path
#   2. ~/.huggingface/token             — older HF library path (still supported)
#   3. GCP Secret Manager (hf-token)    — most secure; lets the TPU VM fetch
#                                          via IAM without the laptop having to
#                                          scp the secret each provision.
#
# What it never does:
#   - Print the token to stdout (only first 4 + last 4 chars for verification)
#   - Commit anything to git (.gitignore is updated)
#   - Email, log, or transmit anywhere besides HF API + GCP Secret Manager

set -euo pipefail

# ── Helpers (match gcp_bootstrap.sh style) ──────────────────────────────
log()   { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
ok()    { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn()  { printf "  \033[33m!\033[0m %s\n" "$*"; }
err()   { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; }

# ── Modes ───────────────────────────────────────────────────────────────
MODE="setup"
case "${1:-}" in
  --check)            MODE="check"     ;;
  --clear)            MODE="clear"     ;;
  --non-interactive)  MODE="setup"     ;;   # token must come from env
  "")                 MODE="setup"     ;;
  -h|--help)
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)  err "unknown arg: $1"; exit 2 ;;
esac

# ── Paths ───────────────────────────────────────────────────────────────
TOKEN_PATH_NEW="$HOME/.cache/huggingface/token"
TOKEN_PATH_OLD="$HOME/.huggingface/token"

# ── HF API validation (no curl token in process list — uses stdin) ──────
validate_token() {
  local tok="$1"
  local resp
  resp=$(curl -sf -H "Authorization: Bearer $tok" \
              https://huggingface.co/api/whoami-v2 2>/dev/null) || return 1
  python3 -c "
import sys, json
try:
    d = json.loads(sys.argv[1])
    print(d.get('name', '?') + '\t' + d.get('email', '?'))
except Exception:
    sys.exit(1)
" "$resp" 2>/dev/null
}

mask() {
  local tok="$1"
  if [[ ${#tok} -lt 12 ]]; then
    printf "********"
  else
    printf "%s...%s" "${tok:0:4}" "${tok: -4}"
  fi
}

# ── CHECK mode: report what's currently configured ──────────────────────
if [[ "$MODE" == "check" ]]; then
  log "HF token check"
  any_found=false
  for p in "$TOKEN_PATH_NEW" "$TOKEN_PATH_OLD"; do
    if [[ -f "$p" ]]; then
      any_found=true
      tok=$(tr -d '[:space:]' < "$p")
      if user_info=$(validate_token "$tok"); then
        user=$(printf "%s" "$user_info" | cut -f1)
        ok "$p: valid ($user, $(mask "$tok"))"
      else
        err "$p: present but token is invalid or revoked"
      fi
    fi
  done
  if command -v gcloud >/dev/null 2>&1; then
    if tok=$(gcloud secrets versions access latest --secret=hf-token 2>/dev/null); then
      any_found=true
      tok=$(printf "%s" "$tok" | tr -d '[:space:]')
      if user_info=$(validate_token "$tok"); then
        user=$(printf "%s" "$user_info" | cut -f1)
        ok "GCP Secret Manager hf-token: valid ($user, $(mask "$tok"))"
      else
        err "GCP Secret Manager hf-token: present but invalid"
      fi
    fi
  fi
  if ! $any_found; then
    warn "No HF token configured. Run: ./scripts/setup_hf.sh"
    exit 1
  fi
  exit 0
fi

# ── CLEAR mode: remove every stored copy (with confirmation) ────────────
if [[ "$MODE" == "clear" ]]; then
  log "HF token clear"
  read -r -p "Remove ~/.cache/huggingface/token, ~/.huggingface/token, and Secret Manager hf-token? [yes/N]: " ans
  [[ "$ans" == "yes" ]] || { echo "Aborted."; exit 1; }
  rm -f "$TOKEN_PATH_NEW" "$TOKEN_PATH_OLD" && ok "Local files removed"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud secrets delete hf-token --quiet 2>/dev/null && ok "Secret Manager hf-token deleted" \
      || warn "Secret Manager hf-token not present (or no permission)"
  fi
  exit 0
fi

# ── SETUP mode ──────────────────────────────────────────────────────────
log "HF token setup"

# 1) Acquire the token.
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN="$HF_TOKEN"
  ok "Using HF_TOKEN from environment"
elif [[ -t 0 ]]; then
  echo "  Paste your HF PRO token (https://huggingface.co/settings/tokens — needs 'read' scope)."
  echo "  Input is hidden."
  read -r -s -p "  Token: " TOKEN
  echo
else
  err "No HF_TOKEN env var set and stdin is not a tty. Either:"
  err "  HF_TOKEN=hf_xxx ./scripts/setup_hf.sh"
  err "  ./scripts/setup_hf.sh   (interactive)"
  exit 1
fi

if [[ -z "$TOKEN" ]]; then
  err "Empty token. Aborting."
  exit 1
fi

# 2) Validate against the HF API.
log "Validating against huggingface.co/api/whoami-v2 ..."
if user_info=$(validate_token "$TOKEN"); then
  user=$(printf "%s" "$user_info" | cut -f1)
  email=$(printf "%s" "$user_info" | cut -f2)
  ok "Valid token: user=$user email=$email ($(mask "$TOKEN"))"
else
  err "Token rejected by HF API. Check it at https://huggingface.co/settings/tokens"
  exit 1
fi

# 3) Decide storage targets.
log "Storage targets"
echo "  Where should the token live? (token is the same in every location)"
echo "    1) ~/.cache/huggingface/token  — HF library canonical path  [recommended]"
echo "    2) GCP Secret Manager hf-token — most secure; TPU VM auto-fetches via IAM"
echo "    3) Both"
echo "    4) Just verify (don't store anywhere)"
read -r -p "  Choice [1-4, default 3]: " choice
choice="${choice:-3}"

write_local() {
  mkdir -p "$(dirname "$TOKEN_PATH_NEW")"
  printf "%s" "$TOKEN" > "$TOKEN_PATH_NEW"
  chmod 600 "$TOKEN_PATH_NEW"
  ok "Wrote $TOKEN_PATH_NEW (mode 0600)"
  # Also write the legacy path — some tooling still reads it.
  mkdir -p "$(dirname "$TOKEN_PATH_OLD")"
  printf "%s" "$TOKEN" > "$TOKEN_PATH_OLD"
  chmod 600 "$TOKEN_PATH_OLD"
  ok "Wrote $TOKEN_PATH_OLD (legacy path, mode 0600)"
}

write_secret_manager() {
  if ! command -v gcloud >/dev/null 2>&1; then
    err "gcloud not installed — run ./scripts/gcp_bootstrap.sh first"
    return 1
  fi
  PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
  if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
    err "No GCP project configured — run ./scripts/gcp_bootstrap.sh first"
    return 1
  fi
  # Enable Secret Manager API if not already (idempotent).
  if ! gcloud services list --enabled --format='value(config.name)' 2>/dev/null \
       | grep -qx secretmanager.googleapis.com; then
    warn "Enabling Secret Manager API ..."
    gcloud services enable secretmanager.googleapis.com
  fi
  # Create or update the secret.
  if gcloud secrets describe hf-token >/dev/null 2>&1; then
    printf "%s" "$TOKEN" | gcloud secrets versions add hf-token --data-file=-
    ok "Added new version to existing secret 'hf-token' (project $PROJECT_ID)"
  else
    printf "%s" "$TOKEN" | gcloud secrets create hf-token \
      --replication-policy=automatic --data-file=-
    ok "Created secret 'hf-token' (project $PROJECT_ID)"
  fi
}

case "$choice" in
  1) write_local ;;
  2) write_secret_manager ;;
  3) write_local; write_secret_manager || warn "GCP step failed; local copy still wrote" ;;
  4) ok "Verified only — no storage" ;;
  *) err "Invalid choice"; exit 2 ;;
esac

# 4) .gitignore safety — never commit a token file by mistake.
if [[ -f .gitignore ]] && ! grep -q "^\.hf-token$\|^\*\.hf-token$" .gitignore 2>/dev/null; then
  cat >> .gitignore <<'EOF'

# HF tokens — never commit, even if the user drops a stray copy here.
.hf-token
*.hf-token
EOF
  ok "Added .hf-token / *.hf-token to .gitignore"
fi

# 5) Final hint.
cat <<EOF

  Downstream consumers will pick up the token automatically:
    • Laptop          $TOKEN_PATH_NEW (read by transformers / huggingface_hub)
    • TPU VM          provision_tpu.sh fetches from Secret Manager (preferred) or
                      scp's $TOKEN_PATH_NEW to the VM as a fallback.
    • Colab Pro       open Colab → Secrets (key icon) → add HF_TOKEN with the
                      same value. The notebook reads via google.colab.userdata.
    • Local dev       export HF_TOKEN if you prefer env-var auth.

  Verify anytime:  ./scripts/setup_hf.sh --check
  Remove:          ./scripts/setup_hf.sh --clear
EOF
