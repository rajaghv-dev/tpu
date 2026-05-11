#!/usr/bin/env bash
# Persist the XLA JIT compilation cache (JAX_COMPILATION_CACHE_DIR) to GCS so
# subsequent provisions skip the cold compile. Tier 2 #6 in todo.md.
#
# Usage:
#   ./scripts/cache_xla.sh --upload     # local cache → GCS
#   ./scripts/cache_xla.sh --download   # GCS → local cache (used by provision)
#   ./scripts/cache_xla.sh --check      # size + count, no transfer
#   ./scripts/cache_xla.sh -h | --help
#
# Env:
#   XLA_CACHE_URL                gs://...; defaults via setup_cache_env.sh logic
#   JAX_COMPILATION_CACHE_DIR    local dir (default: /tmp/xla-cache)

set -euo pipefail

log()  { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
err()  { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; }

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

MODE=""
for arg in "$@"; do
  case "$arg" in
    --upload)   MODE="upload" ;;
    --download) MODE="download" ;;
    --check)    MODE="check" ;;
    -h|--help)  usage; exit 0 ;;
    *) err "Unknown arg: $arg"; usage; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { err "Pass one of --upload/--download/--check"; usage; exit 2; }

if [[ -z "${XLA_CACHE_URL:-}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
    err "XLA_CACHE_URL not set and no gcloud project — source scripts/setup_cache_env.sh first."
    exit 1
  fi
  XLA_CACHE_URL="gs://${PROJECT}-models/xla-cache/"
fi

XLA_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/xla-cache}"
ok "XLA_CACHE_URL            = $XLA_CACHE_URL"
ok "JAX_COMPILATION_CACHE_DIR= $XLA_DIR"

case "$MODE" in
  upload)
    log "Uploading $XLA_DIR → $XLA_CACHE_URL"
    [[ -d "$XLA_DIR" ]] || { err "Local XLA cache dir missing: $XLA_DIR"; exit 1; }
    SIZE="$(du -sh "$XLA_DIR" 2>/dev/null | cut -f1)"
    ok "local size: $SIZE"
    gsutil -m rsync -r "$XLA_DIR" "$XLA_CACHE_URL"
    ok "uploaded"
    ;;

  download)
    log "Downloading $XLA_CACHE_URL → $XLA_DIR"
    warn "XLA cache is sensitive to JAX + libtpu versions. Cross-version reuse is"
    warn "NOT guaranteed; runner.clear_xla_cache() exists for a reason. Treat this"
    warn "as a speedup opportunity, not a correctness guarantee."
    mkdir -p "$XLA_DIR"
    gsutil -m rsync -r "$XLA_CACHE_URL" "$XLA_DIR"
    SIZE="$(du -sh "$XLA_DIR" 2>/dev/null | cut -f1)"
    ok "restored: $SIZE"
    ;;

  check)
    log "Inspecting $XLA_CACHE_URL"
    SIZE="$(gsutil du -sh "$XLA_CACHE_URL" 2>/dev/null | awk '{print $1, $2}')"
    COUNT="$(gsutil ls -r "$XLA_CACHE_URL" 2>/dev/null | grep -cv '/$' || true)"
    ok "${COUNT:-0} cache entries, ${SIZE:-unknown} total"
    ;;
esac
