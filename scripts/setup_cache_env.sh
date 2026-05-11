#!/usr/bin/env bash
# Source me (or run me) to populate WHEEL_CACHE_URL / HF_MODEL_CACHE_URL /
# XLA_CACHE_URL from the current gcloud project. Matches gcp_bootstrap.sh
# Phase 7 convention: bucket = gs://$PROJECT-models.
#
# Usage:
#   . ./scripts/setup_cache_env.sh        # source — exports the three vars
#   ./scripts/setup_cache_env.sh          # run direct — prints a summary
#
# Env overrides (preserved if already set):
#   BUCKET_SUFFIX=models   PROJECT_ID=<override>
#   WHEEL_CACHE_URL / HF_MODEL_CACHE_URL / XLA_CACHE_URL — pre-set wins

set -eu  # not -o pipefail: this file is sourced

BUCKET_SUFFIX="${BUCKET_SUFFIX:-models}"

# Resolve project. Pre-set PROJECT_ID wins; else ask gcloud.
if [[ -z "${PROJECT_ID:-}" ]]; then
  if command -v gcloud >/dev/null 2>&1; then
    PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
  else
    PROJECT_ID=""
  fi
fi

if [[ -z "${PROJECT_ID:-}" || "$PROJECT_ID" == "(unset)" ]]; then
  printf "\033[31m✗\033[0m setup_cache_env: no project (set PROJECT_ID or run gcloud config set project ...)\n" >&2
  # When sourced, do NOT exit the parent shell; just leave vars unset.
  return 1 2>/dev/null || exit 1
fi

BUCKET="gs://${PROJECT_ID}-${BUCKET_SUFFIX}"

export WHEEL_CACHE_URL="${WHEEL_CACHE_URL:-${BUCKET}/wheels/}"
export HF_MODEL_CACHE_URL="${HF_MODEL_CACHE_URL:-${BUCKET}/hf-cache/}"
export XLA_CACHE_URL="${XLA_CACHE_URL:-${BUCKET}/xla-cache/}"

# If executed (not sourced), print summary. Detect via BASH_SOURCE != $0.
if [[ "${BASH_SOURCE[0]:-$0}" == "${0}" ]]; then
  printf "  Project:           %s\n" "$PROJECT_ID"
  printf "  WHEEL_CACHE_URL=   %s\n" "$WHEEL_CACHE_URL"
  printf "  HF_MODEL_CACHE_URL=%s\n" "$HF_MODEL_CACHE_URL"
  printf "  XLA_CACHE_URL=     %s\n" "$XLA_CACHE_URL"
  echo
  echo "  Source this script to export them into your shell:"
  echo "    . ./scripts/setup_cache_env.sh"
fi
