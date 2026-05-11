#!/usr/bin/env bash
# One-shot inventory of all three GCS caches: size, item count, last sync.
# Pure reporter — no uploads, no deletes.
#
# Usage:
#   ./scripts/cache_status.sh
#   ./scripts/cache_status.sh -h | --help
#
# Env:
#   WHEEL_CACHE_URL / HF_MODEL_CACHE_URL / XLA_CACHE_URL  (else derived from
#   gcloud project, matching setup_cache_env.sh)

set -euo pipefail

log()  { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
err()  { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; }

case "${1:-}" in
  -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) err "Unknown arg: $1"; exit 2 ;;
esac

# Resolve URLs if not in env.
if [[ -z "${WHEEL_CACHE_URL:-}" || -z "${HF_MODEL_CACHE_URL:-}" || -z "${XLA_CACHE_URL:-}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
    err "No cache URLs in env and no gcloud project — source scripts/setup_cache_env.sh first."
    exit 1
  fi
  WHEEL_CACHE_URL="${WHEEL_CACHE_URL:-gs://${PROJECT}-models/wheels/}"
  HF_MODEL_CACHE_URL="${HF_MODEL_CACHE_URL:-gs://${PROJECT}-models/hf-cache/}"
  XLA_CACHE_URL="${XLA_CACHE_URL:-gs://${PROJECT}-models/xla-cache/}"
fi

# Inspect one URL. Echoes a tab-separated line: SIZE\tCOUNT\tLAST
inspect() {
  local url="$1"
  local size count last
  size="$(gsutil du -sh "$url" 2>/dev/null | awk '{print $1, $2}')"
  count="$(gsutil ls -r "$url" 2>/dev/null | grep -cv '/$' || true)"
  # Last update time: latest "Update time" across the listing
  last="$(gsutil ls -L -r "$url" 2>/dev/null \
            | awk -F': +' '/Update time/ {print $2}' \
            | sort | tail -1)"
  [[ -z "$size"  ]] && size="-"
  [[ -z "$count" ]] && count="0"
  [[ -z "$last"  ]] && last="-"
  printf "%s\t%s\t%s\n" "$size" "$count" "$last"
}

log "GCS cache inventory"

ROW_DIV="────────────────────────────────────────────────────────────────────────────────"
printf "  %-12s %-38s %-12s %-7s %s\n" "Cache" "Location" "Size" "Items" "Last sync"
printf "  %s\n" "$ROW_DIV"

for entry in "Wheels|$WHEEL_CACHE_URL" "HF models|$HF_MODEL_CACHE_URL" "XLA compile|$XLA_CACHE_URL"; do
  label="${entry%%|*}"
  url="${entry#*|}"
  IFS=$'\t' read -r size count last <<<"$(inspect "$url")"
  printf "  %-12s %-38s %-12s %-7s %s\n" "$label" "$url" "$size" "$count" "$last"
done

printf "  %s\n" "$ROW_DIV"
echo "  Note: storage cost ≈ \$0.02/GB/mo (us-west4 STANDARD). Five GB total ≈ \$0.10/mo."
