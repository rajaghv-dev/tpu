#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_all.sh — master orchestrator for the staged TPU benchmark workflow.
#
# Walks through stages 00 → 71 in lexical order, stopping if any stage fails.
# Designed for safe resumption: if you Ctrl-C halfway, re-running with the
# same flags re-enters the pipeline at (or just before) the stage that failed.
#
# Pipeline stages:
#   00_validate_local.sh    — local tooling (bash 4+, gcloud, python3)
#   01_validate_gcp.sh      — billing, APIs, IAM, quota
#   02_validate_bucket.sh   — GCS bucket exists + accessible
#   03_validate_hf.sh       — optional, only if HF_TOKEN set or --hf-required
#   10_setup_bucket.sh      — create bucket if missing (idempotent)
#   11_setup_budget.sh      — create budget alert (idempotent)
#   20_provision_tpu.sh     — create TPU VM (multi-zone fallback)
#   21_wait_tpu_ready.sh    — wait for SSH readiness
#   30_deploy_repo.sh       — tar+scp repo to VM
#   31_install_deps.sh      — pip install requirements.txt
#   32_mount_gcs.sh         — gcsfuse mount + env vars (skipped if --no-gcs)
#   40_verify_jax.sh        — confirm jax.devices() shows TPU
#   41_run_pytests.sh       — pytest tests/ on VM
#   42_dry_run.sh           — harness --dry-run
#   50_run_smoke.sh         — smoke suite
#   51_run_quick.sh         — quick suite (skipped unless --suite quick)
#   60_pull_results.sh      — pull results back
#   70_teardown_tpu.sh      — delete VM (skipped if --keep-tpu)
#   71_verify_teardown.sh   — confirm zero billable resources
#
# Usage:
#   ./scripts/run_all.sh                       # smoke suite, full pipeline
#   ./scripts/run_all.sh --suite quick         # quick suite (50 min, ~\$0.30)
#   ./scripts/run_all.sh --from 30             # resume from stage 30
#   ./scripts/run_all.sh --to 42               # stop after stage 42 (no run)
#   ./scripts/run_all.sh --keep-tpu            # don't tear down at end
#   ./scripts/run_all.sh --no-gcs              # skip 32_mount_gcs.sh
#   ./scripts/run_all.sh --dry-run             # print plan, don't execute
#   YES_TO_ALL=1 ./scripts/run_all.sh          # auto-confirm all prompts
#
# Exit codes:
#   0 = pipeline finished.
#   N = the stage that failed; check the log above for details.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="run_all"
setup_error_trap

# ── Argument parsing ──────────────────────────────────────────────────────────
SUITE="smoke"
FROM_STAGE=0
TO_STAGE=99
KEEP_TPU=0
NO_GCS=0
DRY_RUN=0
HF_REQUIRED=0

while (( $# > 0 )); do
    case "$1" in
        --suite)        SUITE="$2"; shift 2 ;;
        --from)         FROM_STAGE="$2"; shift 2 ;;
        --to)           TO_STAGE="$2"; shift 2 ;;
        --keep-tpu)     KEEP_TPU=1; shift ;;
        --no-gcs)       NO_GCS=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --hf-required)  HF_REQUIRED=1; shift ;;
        -h|--help)
            grep -E '^#' "$0" | sed 's/^# \{0,1\}//' | head -55
            exit 0 ;;
        *) log_err "Unknown flag: $1"; exit 64 ;;
    esac
done

# ── Stage definitions ─────────────────────────────────────────────────────────
# Each stage is (number, script, optional args, optional skip-condition expr).
# Skip condition is a bash expression evaluated at run time; if it returns 0
# the stage is skipped.
#
# NOTE on stage 02 (validate_bucket):
#   The standalone script `02_validate_bucket.sh` errors if the bucket is
#   missing — correct behaviour when run on its own. In the pipeline, stage
#   10 (`10_setup_bucket.sh`) creates the bucket when missing AND calls 02
#   internally for verification. So 02 is intentionally absent from this
#   array — including it before 10 would always fail on first run, and
#   after 10 would be redundant.
#
declare -a STAGES=(
    "00:00_validate_local.sh"
    "01:01_validate_gcp.sh"
    "03:03_validate_hf.sh"
    "10:10_setup_bucket.sh"
    "11:11_setup_budget.sh"
    "20:20_provision_tpu.sh"
    "21:21_wait_tpu_ready.sh"
    "30:30_deploy_repo.sh"
    "31:31_install_deps.sh"
    "32:32_mount_gcs.sh"
    "40:40_verify_jax.sh"
    "41:41_run_pytests.sh"
    "42:42_dry_run.sh ${SUITE}"
    "50:50_run_smoke.sh"
    "51:51_run_quick.sh"
    "60:60_pull_results.sh"
    "70:70_teardown_tpu.sh"
    "71:71_verify_teardown.sh"
)

# ── Skip rules ────────────────────────────────────────────────────────────────
should_skip() {
    local stage_num="$1"; local script="$2"
    # Range filter from --from / --to
    if (( stage_num < FROM_STAGE )) || (( stage_num > TO_STAGE )); then
        return 0
    fi
    # HF validation only required when caller demands it OR HF_TOKEN is set
    if [[ "$script" == "03_validate_hf.sh" ]]; then
        if (( HF_REQUIRED == 0 )) && [[ -z "${HF_TOKEN:-}" ]]; then
            log_info "skip 03 (HF_TOKEN unset and --hf-required not passed)"
            return 0
        fi
    fi
    # Budget setup is best-effort; skip if user declined or no perms
    # (script returns 2 in that case — handled by main loop)
    # GCS mount can be skipped with --no-gcs
    if (( NO_GCS == 1 )) && [[ "$script" == "32_mount_gcs.sh" ]]; then
        log_info "skip 32 (--no-gcs)"
        return 0
    fi
    # Smoke vs quick: only run the requested one
    if [[ "$SUITE" == "quick" ]] && [[ "$script" == "50_run_smoke.sh" ]]; then
        log_info "skip 50 (--suite quick selected, skipping smoke)"
        return 0
    fi
    if [[ "$SUITE" == "smoke" ]] && [[ "$script" == "51_run_quick.sh" ]]; then
        log_info "skip 51 (--suite smoke selected, skipping quick)"
        return 0
    fi
    # Teardown: skip with --keep-tpu
    if (( KEEP_TPU == 1 )) && { [[ "$script" == "70_teardown_tpu.sh" ]] || [[ "$script" == "71_verify_teardown.sh" ]]; }; then
        log_info "skip $(printf '%02d' "$stage_num") (--keep-tpu)"
        return 0
    fi
    return 1
}

# ── Cost upfront ──────────────────────────────────────────────────────────────
banner "Plan"
log_info "Suite        : $SUITE"
log_info "From stage   : $FROM_STAGE"
log_info "To stage     : $TO_STAGE"
log_info "Keep TPU     : $KEEP_TPU"
log_info "Mount GCS    : $((1 - NO_GCS))"
log_info "Dry-run      : $DRY_RUN"

# Show forecast so the user knows what they're about to spend.
if [[ -x "$SCRIPT_DIR/91_predict_cost.sh" ]]; then
    "$SCRIPT_DIR/91_predict_cost.sh" "$SUITE" tpu_v5litepod_1 || true
fi

if (( DRY_RUN == 1 )); then
    banner "Dry-run plan (would execute)"
    for entry in "${STAGES[@]}"; do
        num="${entry%%:*}"
        rest="${entry#*:}"
        if should_skip "$((10#$num))" "${rest%% *}" >/dev/null 2>&1; then
            printf '  [skip ] %s\n' "$rest"
        else
            printf '  [run  ] %s\n' "$rest"
        fi
    done
    exit 0
fi

# Confirmation gate before any cost-incurring stage runs.
if (( FROM_STAGE <= 20 )) && (( TO_STAGE >= 20 )); then
    log_warn "Stage 20 will provision a TPU and START BILLING (~\$0.36/hr)."
    if ! confirm "Proceed?" Y; then
        log_warn "Aborted at user request."
        exit 0
    fi
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
banner "Executing pipeline"
for entry in "${STAGES[@]}"; do
    num="${entry%%:*}"
    rest="${entry#*:}"
    n=$((10#$num))    # base-10 to avoid octal-leading-zero gotcha
    if should_skip "$n" "${rest%% *}"; then
        continue
    fi
    log_step "── stage $num — $rest ──"
    set +e   # disable -e for the call so we can inspect rc and decide
    "$SCRIPT_DIR"/$rest
    rc=$?
    set -e
    if (( rc == 0 )); then
        continue
    fi
    # Soft-fail stages: 11_setup_budget.sh exits 2 when the caller lacks
    # roles/billing.user — the rest of the pipeline doesn't need a budget
    # alert to proceed, so we warn and continue. Hard-fail otherwise.
    case "${rest%% *}" in
        11_setup_budget.sh)
            if (( rc == 2 )); then
                log_warn "Stage $num declined or insufficient billing perms; continuing without budget alert."
                continue
            fi ;;
        01_validate_gcp.sh)
            # Stage 01 returns 2 for "passed with warnings" — that's fine to continue.
            if (( rc == 2 )); then
                log_warn "Stage $num passed with warnings; continuing."
                continue
            fi ;;
    esac
    log_err "Stage $num failed (exit $rc). Pipeline stopped."
    log_err "Resume after fixing with: ./scripts/run_all.sh --from $n"
    exit "$rc"
done

banner "Pipeline complete"
log_ok "All stages succeeded."
[[ "$KEEP_TPU" == "1" ]] && log_warn "TPU still running — tear down with ./scripts/70_teardown_tpu.sh"
exit 0
