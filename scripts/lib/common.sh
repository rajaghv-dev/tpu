# shellcheck shell=bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/lib/common.sh — shared helpers for the staged TPU benchmark scripts.
#
# This file is SOURCED, not executed. It has no shebang and no exec bit.
# Every script in scripts/ that follows the staged convention should source it
# as its first action so that:
#   - logging output looks identical across scripts (ts | stage | level | msg);
#   - errors auto-trap with the failing line number + command;
#   - cleanup/teardown handlers can be registered with one call;
#   - colour usage is consistent and TTY-aware (no escapes when piping to file).
#
# Idempotency. Sourcing this file multiple times in the same shell is safe — all
# function definitions overwrite previous ones; all variable assignments use
# `: "${VAR:=default}"` so an outer caller's value wins.
#
# Conventions other scripts in this directory follow:
#   - Filenames are prefix-numbered (00_*, 10_*, …) so a `ls` lexically matches
#     the run order. Stages are documented in scripts/README.md.
#   - All scripts are bash, `set -euo pipefail`, gcloud/gsutil + python3 only.
#   - No script writes outside the repo root or its configured GCS bucket.
# ──────────────────────────────────────────────────────────────────────────────

# Guard against double-source clobbering caller env (idempotent re-source is fine).
if [[ "${_TPU_BENCH_COMMON_SOURCED:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi
_TPU_BENCH_COMMON_SOURCED=1

# ── Colour ────────────────────────────────────────────────────────────────────
# Only emit colour when stdout is a TTY. Piping to file/CI gives plain text.
if [[ -t 1 ]]; then
    _C_RED=$'\033[0;31m'
    _C_GREEN=$'\033[0;32m'
    _C_YELLOW=$'\033[0;33m'
    _C_BLUE=$'\033[0;34m'
    _C_DIM=$'\033[2m'
    _C_BOLD=$'\033[1m'
    _C_RESET=$'\033[0m'
else
    _C_RED='' _C_GREEN='' _C_YELLOW='' _C_BLUE='' _C_DIM='' _C_BOLD='' _C_RESET=''
fi

# Stage label (for prefix). Each script sets this near top:
#   STAGE="20_provision_tpu"
: "${STAGE:=common}"

# ── Logging primitives ────────────────────────────────────────────────────────
# Format:  HH:MM:SS  STAGE  LEVEL  message
# LEVEL is colour-coded on a TTY. All output goes to stderr so that scripts can
# still pipe meaningful values on stdout (e.g. `echo "$VM_IP"` for capture).
_log() {
    local level="$1"; shift
    local colour="$1"; shift
    printf '%s%s%s  %s%-22s%s  %s%-5s%s  %s\n' \
        "$_C_DIM" "$(date +%H:%M:%S)" "$_C_RESET" \
        "$_C_BOLD" "$STAGE" "$_C_RESET" \
        "$colour" "$level" "$_C_RESET" \
        "$*" >&2
}

log_info()  { _log "INFO"  "$_C_BLUE"   "$@"; }
log_ok()    { _log "OK"    "$_C_GREEN"  "$@"; }
log_warn()  { _log "WARN"  "$_C_YELLOW" "$@"; }
log_err()   { _log "ERROR" "$_C_RED"    "$@"; }
log_step()  { _log "STEP"  "$_C_BOLD"   "$@"; }

# Big banner for stage entry points. One per script start.
banner() {
    local title="$*"
    local line
    line=$(printf '%*s' 72 '' | tr ' ' '─')
    printf '\n%s%s%s\n%s%s%s\n%s%s%s\n\n' \
        "$_C_BOLD" "$line" "$_C_RESET" \
        "$_C_BOLD" "  $title" "$_C_RESET" \
        "$_C_BOLD" "$line" "$_C_RESET" >&2
}

# ── Error trap ────────────────────────────────────────────────────────────────
# Registered by each stage script via `setup_error_trap`. On any unhandled
# failure prints the script, line, and the offending command — much more useful
# than the default bash diagnostic which just shows the line number.
_on_err() {
    local exit_code=$?
    local line_no=$1
    local last_cmd=$2
    log_err "Failed at ${BASH_SOURCE[1]:-?}:${line_no} (exit=$exit_code)"
    log_err "Last command: ${last_cmd}"
    # Run any exit handlers registered with `add_exit_handler` so we still tear
    # down state (e.g. delete a half-created TPU) even on hard failure.
    _run_exit_handlers "fail"
    exit "$exit_code"
}

setup_error_trap() {
    set -Eeuo pipefail
    trap '_on_err "$LINENO" "$BASH_COMMAND"' ERR
    trap '_run_exit_handlers ok' EXIT
    trap '_run_exit_handlers interrupt; exit 130' INT
    trap '_run_exit_handlers terminate; exit 143' TERM
}

# ── Exit handlers ─────────────────────────────────────────────────────────────
# Scripts can register cleanup with `add_exit_handler "command string"`. Each
# handler is called once at shell exit, with the exit reason ("ok" | "fail" |
# "interrupt" | "terminate") available as $EXIT_REASON inside the handler.
_EXIT_HANDLERS=()
add_exit_handler() { _EXIT_HANDLERS+=("$1"); }
_run_exit_handlers() {
    local reason="$1"
    [[ ${#_EXIT_HANDLERS[@]} -eq 0 ]] && return 0
    local h
    for h in "${_EXIT_HANDLERS[@]}"; do
        EXIT_REASON="$reason" bash -c "$h" || true
    done
    _EXIT_HANDLERS=()
}

# ── Required-tool guard ───────────────────────────────────────────────────────
# Use at the top of any script that depends on a CLI tool. Errors out with a
# helpful install hint rather than a cryptic "command not found".
require_cmd() {
    local cmd="$1"; local hint="${2:-}"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_err "Required command not on PATH: $cmd"
        [[ -n "$hint" ]] && log_err "Hint: $hint"
        exit 127
    fi
}

# ── Env-var guard ─────────────────────────────────────────────────────────────
# Use for variables that MUST be set (e.g. HF_TOKEN). Prefer config defaults
# (lib/config.sh) over hard requirements when there's a sensible fallback.
require_env() {
    local var="$1"; local hint="${2:-}"
    if [[ -z "${!var:-}" ]]; then
        log_err "Required environment variable not set: \$$var"
        [[ -n "$hint" ]] && log_err "Hint: $hint"
        exit 64
    fi
}

# ── Pretty confirmation prompt ────────────────────────────────────────────────
# Skipped automatically when stdin is not a TTY (CI, pipelines) — caller must
# pass YES_TO_ALL=1 to bypass interactively too. Returns 0 = yes, 1 = no.
confirm() {
    local prompt="$1"; local default="${2:-N}"
    if [[ "${YES_TO_ALL:-0}" == "1" ]] || [[ ! -t 0 ]]; then
        log_info "Auto-confirming (YES_TO_ALL=1 or non-interactive): $prompt"
        return 0
    fi
    local hint="[y/N]"
    [[ "$default" == "Y" ]] && hint="[Y/n]"
    local reply
    read -r -p "$(printf '%s%s%s %s ' "$_C_YELLOW" "?" "$_C_RESET" "$prompt $hint")" reply
    reply="${reply:-$default}"
    [[ "$reply" =~ ^[Yy]$ ]]
}

# ── Section divider for terminal output ──────────────────────────────────────
section() { printf '\n%s── %s ──%s\n' "$_C_DIM" "$*" "$_C_RESET" >&2; }

# ── State file helpers ───────────────────────────────────────────────────────
# Provisioning hands off context (zone chosen, TPU name, etc.) to later stages
# via a small JSON file. Avoids re-querying gcloud and keeps stages decoupled.
state_dir() {
    local d="${TPU_STATE_DIR:-$_REPO_ROOT/.tpu-bench-state}"
    mkdir -p "$d"
    echo "$d"
}

state_set() {
    local key="$1"; local value="$2"
    local f
    f="$(state_dir)/state.env"
    # Remove any prior assignment then append. Atomic via mv.
    if [[ -f "$f" ]]; then
        grep -v "^${key}=" "$f" > "$f.tmp" || true
        mv "$f.tmp" "$f"
    fi
    printf '%s=%q\n' "$key" "$value" >> "$f"
}

state_get() {
    local key="$1"; local default="${2:-}"
    local f
    f="$(state_dir)/state.env"
    [[ -f "$f" ]] || { echo "$default"; return; }
    # shellcheck disable=SC1090
    ( source "$f" 2>/dev/null; printf '%s\n' "${!key:-$default}" )
}

state_clear() {
    local f
    f="$(state_dir)/state.env"
    rm -f "$f"
}

# ── Repo root detection ───────────────────────────────────────────────────────
# Resolves the repo root from this file's location (scripts/lib/common.sh) so
# scripts can reference repo paths without depending on the user's PWD.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT="$_REPO_ROOT"

# ── Final guard: bash version ────────────────────────────────────────────────
# Associative arrays + ${!var} indirection require bash 4+. macOS default is
# bash 3.2 — surface that early.
if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: bash 4+ required (you have $BASH_VERSION). On macOS:" >&2
    echo "       brew install bash; then run scripts via /opt/homebrew/bin/bash" >&2
    exit 65
fi
