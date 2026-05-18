#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# find_tpu_zone.sh — discover zones that offer $ACCELERATOR_TYPE.
#
# Asks gcloud for the full list of TPU-API locations in $PROJECT_ID, then
# probes each one for $ACCELERATOR_TYPE. Probes are done in parallel and
# zones with the right preference order (us-west / us-east / europe-west
# first, then everything else) are tried first.
#
# Usage:
#   PROJECT_ID=foo ACCELERATOR_TYPE=v5litepod-1 ./find_tpu_zone.sh         # first hit
#   PROJECT_ID=foo ACCELERATOR_TYPE=v5litepod-1 ./find_tpu_zone.sh --all   # all hits
#   source ./find_tpu_zone.sh && find_tpu_zone                              # function call
#
# Read-only: only calls `gcloud compute tpus locations list` and
# `gcloud compute tpus accelerator-types list`. Both are free.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Preferred-region prefixes, in priority order. A zone whose name starts with
# one of these (after stripping the trailing -X) is tried before the rest.
_PREFERRED_PREFIXES="us-west us-east europe-west asia-northeast"

# Probe a single zone. Prints the zone name if it offers $ACCELERATOR_TYPE,
# nothing otherwise. Designed to be safe to background.
_probe_zone() {
    local proj="$1" accel="$2" zone="$3"
    if gcloud compute tpus accelerator-types list \
            --project="$proj" --zone="$zone" \
            --filter="type=${accel}" \
            --format="value(type)" 2>/dev/null | grep -qx "$accel"; then
        echo "$zone"
    fi
}

# Order zones: preferred-prefix matches first (in prefix order), then the rest
# alphabetically. Reads zones from stdin, prints reordered to stdout.
_order_zones() {
    local pref_re=""
    local p
    for p in $_PREFERRED_PREFIXES; do
        pref_re+="^${p}"$'|'
    done
    pref_re="${pref_re%|}"

    local all
    all="$(cat)"
    # First: preferred zones, in input order (already sorted by gcloud).
    echo "$all" | grep -E "$pref_re" || true
    # Then: the rest.
    echo "$all" | grep -Ev "$pref_re" || true
}

# Public function: returns 0 + prints zone(s); returns 1 if none.
find_tpu_zone() {
    local proj="${PROJECT_ID:?PROJECT_ID must be set}"
    local accel="${ACCELERATOR_TYPE:?ACCELERATOR_TYPE must be set}"
    local mode="${1:-first}"   # "first" | "all"

    # 1. Fetch the universe of TPU-API locations for this project.
    local zones
    if ! zones=$(gcloud compute tpus locations list \
            --project="$proj" \
            --format="value(locationId)" 2>/dev/null); then
        echo "[find_tpu_zone] failed to list TPU locations for project '$proj'" >&2
        return 1
    fi
    if [[ -z "$zones" ]]; then
        echo "[find_tpu_zone] gcloud returned no TPU locations for '$proj'" >&2
        return 1
    fi
    zones=$(echo "$zones" | _order_zones)

    # 2. Probe in parallel. Each match prints its zone name on its own line.
    #    `wait` blocks until all background probes finish.
    local zone hits=""
    local tmp
    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' RETURN

    local pids=()
    for zone in $zones; do
        ( _probe_zone "$proj" "$accel" "$zone" >> "$tmp" ) &
        pids+=("$!")
        # Cap concurrency at 8 to be friendly to the gcloud API.
        if (( ${#pids[@]} >= 8 )); then
            wait "${pids[0]}" 2>/dev/null || true
            pids=("${pids[@]:1}")
        fi
    done
    wait 2>/dev/null || true

    hits=$(sort -u "$tmp" | _order_zones)
    if [[ -z "$hits" ]]; then
        return 1
    fi

    if [[ "$mode" == "all" ]]; then
        echo "$hits"
    else
        echo "$hits" | head -n1
    fi
    return 0
}

# Standalone CLI.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    MODE="first"
    for arg in "$@"; do
        case "$arg" in
            --all)  MODE="all" ;;
            -h|--help)
                echo "Usage: PROJECT_ID=p ACCELERATOR_TYPE=t $0 [--all]"; exit 0 ;;
        esac
    done
    if find_tpu_zone "$MODE"; then
        exit 0
    else
        echo "[find_tpu_zone] no zone offers '$ACCELERATOR_TYPE' in project '$PROJECT_ID'" >&2
        echo "[find_tpu_zone] you may need to request quota for this TPU generation." >&2
        exit 1
    fi
fi
