#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 55_repro_validation.sh — R23 from RECOMMENDATIONS.md.
#
# Cross-validate one Stage 1 result by re-running it on a FRESH VM in a
# DIFFERENT zone, with a FRESH checkout. Confirms the entire pipeline (registry
# → harness → runner → JSONL → dashboard) is reproducible end-to-end.
#
# What we re-run:
#   - same model: bert_base
#   - same precision: bf16
#   - same suite: smoke
# Reference run: results/run_logs/6f049c5d-d1fb-4f1b-aa9a-998c34d2e894/
#   p50=0.6407 ms, throughput=5261.2 ± 7.8 samp/s
#
# Acceptance: new mean throughput within ±10% of reference (5261 ± 526) AND
# new p50 within ±10% of reference (0.6407 ± 0.064 ms). If outside the band,
# something in the pipeline is non-reproducible — find it before Stage 2.
#
# Method:
#   - Provision a v5e-1 in a DIFFERENT zone than the original (us-west4-a was
#     original; we try us-east5-a here).
#   - Fresh deploy from main + fresh dependency install.
#   - Run smoke; pull JSONL; compare against reference numbers.
#
# Cost: ~15 min on v5e-1 spot ≈ \$0.09.
#
# Usage:
#   ./scripts/55_repro_validation.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="55_repro_validation"
setup_error_trap
banner "R23 — Cross-zone reproducibility check (fresh VM, ~15 min, \$0.09)"

# Capture original zone before we change anything.
ORIG_ZONE=$(state_get TPU_ZONE "")
ALT_ZONE_DEFAULT="us-east5-a"
[[ "$ORIG_ZONE" == "$ALT_ZONE_DEFAULT" ]] && ALT_ZONE_DEFAULT="us-west1-a"
ALT_ZONE="${ALT_ZONE:-$ALT_ZONE_DEFAULT}"

REFERENCE_TP=5261.2
REFERENCE_P50=0.6407

log_info "Original zone: ${ORIG_ZONE:-(none recorded)}"
log_info "Alt zone for repro: $ALT_ZONE"
log_info "Reference: bert_base bf16  tp=$REFERENCE_TP smp/s  p50=$REFERENCE_P50 ms"

confirm "Tear down any current TPU VM and provision a fresh one in $ALT_ZONE?" Y || {
    log_warn "Aborted by user."
    exit 0
}

# Tear down any active TPU first so we get a truly fresh environment.
if [[ -n "${ORIG_ZONE}" ]]; then
    log_step "Tearing down current TPU…"
    bash "$SCRIPT_DIR/70_teardown_tpu.sh" || true
fi

# Override TPU_ZONE for the subsequent provisioning step. config.sh reads
# from env first, then state, so exporting wins.
export TPU_ZONE="$ALT_ZONE"
log_step "Provisioning in $ALT_ZONE…"
bash "$SCRIPT_DIR/20_provision_tpu.sh"
bash "$SCRIPT_DIR/21_wait_tpu_ready.sh"
bash "$SCRIPT_DIR/30_deploy_repo.sh"
bash "$SCRIPT_DIR/31_install_deps.sh"
bash "$SCRIPT_DIR/40_verify_jax.sh"

log_step "Running smoke on the fresh VM…"
bash "$SCRIPT_DIR/50_run_smoke.sh"

# Pull and compare.
log_step "Pulling results and comparing to reference…"
bash "$SCRIPT_DIR/60_pull_results.sh"

python3 - <<PY
import json
from pathlib import Path

rows = [json.loads(l) for l in Path("results/runs.jsonl").read_text().splitlines() if l.strip()]
recent = next(
    (r for r in reversed(rows)
     if r.get("model") == "bert_base"
     and r.get("precision") == "bf16"
     and r.get("status") != "failed"),
    None,
)
if not recent:
    print("VERDICT: incomplete — no fresh bert_base bf16 row found")
    raise SystemExit(2)

ref_tp, ref_p50 = $REFERENCE_TP, $REFERENCE_P50
new_tp = recent["throughput_mean_samples_sec"]
new_p50 = recent["latency_p50_ms"]
tp_drift = 100.0 * (new_tp - ref_tp) / ref_tp
p50_drift = 100.0 * (new_p50 - ref_p50) / ref_p50
print(f"new run_id: {recent.get('run_id')}")
print(f"throughput: ref={ref_tp:.1f}  new={new_tp:.1f}  drift={tp_drift:+.1f}%")
print(f"p50 ms:     ref={ref_p50:.4f} new={new_p50:.4f} drift={p50_drift:+.1f}%")

ok = abs(tp_drift) <= 10.0 and abs(p50_drift) <= 10.0
print(f"VERDICT: {'REPRODUCIBLE' if ok else 'DRIFT — investigate before Stage 2'}")
raise SystemExit(0 if ok else 3)
PY

state_set R23_COMPLETED_AT "$(date -Iseconds)"
log_ok "R23 done."
