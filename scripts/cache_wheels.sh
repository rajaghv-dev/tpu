#!/usr/bin/env bash
# Download requirements.txt wheels (linux x86_64 / py310) into a local dir,
# then rsync to GCS. Amortizes ~5–10 min of pip install across provisions.
# See Tier 2 #4 in todo.md and ADR-006 in DECISIONS.md.
#
# Usage:
#   ./scripts/cache_wheels.sh --build           # pip download + gsutil rsync up
#   ./scripts/cache_wheels.sh --check           # list contents + size, no upload
#   ./scripts/cache_wheels.sh --clear [--force] # rm -r the cache (confirms)
#   ./scripts/cache_wheels.sh -h | --help
#
# Env:
#   WHEEL_CACHE_URL   gs://... path; defaults via setup_cache_env.sh logic
#   REQUIREMENTS      path to requirements.txt (default: ~/tpu-examples/requirements.txt
#                     if it exists, else ./requirements.txt)
#   WHEEL_DIR         local download dir (default: /tmp/wheels)

set -euo pipefail

log()  { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
err()  { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; }

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
}

MODE=""
FORCE=false
for arg in "$@"; do
  case "$arg" in
    --build) MODE="build" ;;
    --check) MODE="check" ;;
    --clear) MODE="clear" ;;
    --force) FORCE=true ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $arg"; usage; exit 2 ;;
  esac
done
MODE="${MODE:-build}"

# Resolve WHEEL_CACHE_URL — env wins, else derive from gcloud project.
if [[ -z "${WHEEL_CACHE_URL:-}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
    err "WHEEL_CACHE_URL not set and no gcloud project — source scripts/setup_cache_env.sh first."
    exit 1
  fi
  WHEEL_CACHE_URL="gs://${PROJECT}-models/wheels/"
fi
ok "WHEEL_CACHE_URL = $WHEEL_CACHE_URL"

# Pick a requirements.txt — TPU VM path first, then cwd.
if [[ -z "${REQUIREMENTS:-}" ]]; then
  if [[ -f "$HOME/tpu-examples/requirements.txt" ]]; then
    REQUIREMENTS="$HOME/tpu-examples/requirements.txt"
  else
    REQUIREMENTS="./requirements.txt"
  fi
fi
WHEEL_DIR="${WHEEL_DIR:-/tmp/wheels}"

case "$MODE" in
  build)
    log "Building wheel cache"
    [[ -f "$REQUIREMENTS" ]] || { err "Requirements file not found: $REQUIREMENTS"; exit 1; }
    ok "requirements: $REQUIREMENTS"
    ok "local dir:    $WHEEL_DIR"

    # Cross-platform warning: TPU VMs are linux x86_64 / py3.10. If we're
    # not on that target, wheels resolved here will be platform-wrong.
    HOST_OS="$(uname -s)"; HOST_ARCH="$(uname -m)"
    if [[ "$HOST_OS" != "Linux" || "$HOST_ARCH" != "x86_64" ]]; then
      warn "Host is $HOST_OS/$HOST_ARCH — resolved wheels may not match the TPU VM"
      warn "(linux x86_64 / py3.10). Prefer running this on a TPU VM after first install."
    fi

    mkdir -p "$WHEEL_DIR"
    log "pip download -r $REQUIREMENTS -d $WHEEL_DIR"
    pip download -r "$REQUIREMENTS" -d "$WHEEL_DIR"
    COUNT="$(find "$WHEEL_DIR" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' \) | wc -l)"
    SIZE="$(du -sh "$WHEEL_DIR" 2>/dev/null | cut -f1)"
    ok "downloaded: $COUNT files, $SIZE total"

    log "gsutil -m rsync -r $WHEEL_DIR $WHEEL_CACHE_URL"
    gsutil -m rsync -r "$WHEEL_DIR" "$WHEEL_CACHE_URL"
    ok "uploaded to $WHEEL_CACHE_URL"
    ;;

  check)
    log "Checking wheel cache at $WHEEL_CACHE_URL"
    COUNT="$(gsutil ls "$WHEEL_CACHE_URL" 2>/dev/null | grep -c -E '\.(whl|tar\.gz)$' || true)"
    SIZE="$(gsutil du -sh "$WHEEL_CACHE_URL" 2>/dev/null | awk '{print $1, $2}')"
    ok "${COUNT:-0} wheel files, ${SIZE:-unknown} total"
    ;;

  clear)
    log "Clearing wheel cache at $WHEEL_CACHE_URL"
    if ! $FORCE; then
      read -r -p "  Delete all objects under $WHEEL_CACHE_URL ? [type 'yes']: " A
      [[ "$A" == "yes" ]] || { warn "Aborted."; exit 1; }
    fi
    gsutil -m rm -r "${WHEEL_CACHE_URL}**" || warn "Nothing to delete (or already empty)"
    ok "cleared"
    ;;
esac
