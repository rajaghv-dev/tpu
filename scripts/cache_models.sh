#!/usr/bin/env bash
# Sync HuggingFace model cache (~/.cache/huggingface/hub/) to/from GCS so
# model weights don't redownload every provision. ADR-006; Tier 2 #5.
#
# Usage:
#   ./scripts/cache_models.sh --upload          # local hub/ → GCS
#   ./scripts/cache_models.sh --download        # GCS → local hub/ (used by provision)
#   ./scripts/cache_models.sh --check           # size + count, no transfer
#   ./scripts/cache_models.sh --list-models     # best-effort HF IDs in cache
#   ./scripts/cache_models.sh -h | --help
#
# Env:
#   HF_MODEL_CACHE_URL   gs://...; defaults via setup_cache_env.sh logic
#   HF_HOME              if set, uses $HF_HOME/hub; else ~/.cache/huggingface/hub

set -euo pipefail

log()  { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
err()  { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; }

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

MODE=""
for arg in "$@"; do
  case "$arg" in
    --upload)      MODE="upload" ;;
    --download)    MODE="download" ;;
    --check)       MODE="check" ;;
    --list-models) MODE="list" ;;
    -h|--help)     usage; exit 0 ;;
    *) err "Unknown arg: $arg"; usage; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { err "Pass one of --upload/--download/--check/--list-models"; usage; exit 2; }

if [[ -z "${HF_MODEL_CACHE_URL:-}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
    err "HF_MODEL_CACHE_URL not set and no gcloud project — source scripts/setup_cache_env.sh first."
    exit 1
  fi
  HF_MODEL_CACHE_URL="gs://${PROJECT}-models/hf-cache/"
fi

HUB_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
ok "HF_MODEL_CACHE_URL = $HF_MODEL_CACHE_URL"
ok "local hub dir     = $HUB_DIR"

case "$MODE" in
  upload)
    log "Uploading $HUB_DIR → $HF_MODEL_CACHE_URL"
    [[ -d "$HUB_DIR" ]] || { err "Local hub dir not present: $HUB_DIR (no models cached yet?)"; exit 1; }
    COUNT="$(find "$HUB_DIR" -maxdepth 1 -type d -name 'models--*' | wc -l)"
    SIZE="$(du -sh "$HUB_DIR" 2>/dev/null | cut -f1)"
    ok "local: $COUNT model dirs, $SIZE total"
    gsutil -m rsync -r "$HUB_DIR" "$HF_MODEL_CACHE_URL"
    ok "uploaded"
    ;;

  download)
    log "Downloading $HF_MODEL_CACHE_URL → $HUB_DIR"
    mkdir -p "$HUB_DIR"
    gsutil -m rsync -r "$HF_MODEL_CACHE_URL" "$HUB_DIR"
    COUNT="$(find "$HUB_DIR" -maxdepth 1 -type d -name 'models--*' | wc -l)"
    SIZE="$(du -sh "$HUB_DIR" 2>/dev/null | cut -f1)"
    ok "restored: $COUNT model dirs, $SIZE total"
    ;;

  check)
    log "Inspecting $HF_MODEL_CACHE_URL"
    SIZE="$(gsutil du -sh "$HF_MODEL_CACHE_URL" 2>/dev/null | awk '{print $1, $2}')"
    COUNT="$(gsutil ls "$HF_MODEL_CACHE_URL" 2>/dev/null | grep -c 'models--' || true)"
    ok "${COUNT:-0} model directories, ${SIZE:-unknown} total"
    ;;

  list)
    log "Listing models under $HF_MODEL_CACHE_URL"
    # HF cache layout: models--$ORG--$MODEL — best-effort split on '--'
    gsutil ls "$HF_MODEL_CACHE_URL" 2>/dev/null \
      | sed -n 's#.*/models--\([^/]*\)/*$#\1#p' \
      | awk -F'--' '
          NF >= 2 { org=$1; $1=""; sub(/^--/,""); model=$0; gsub(/--/, "/", model); printf "  %s/%s\n", org, model }
          NF == 1 { printf "  %s\n", $1 }' \
      | sort -u \
      || warn "No models found or cache empty"
    ;;
esac
