#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 53_run_bf16_validation.sh — R20 from RECOMMENDATIONS.md.
#
# Validates the "BF16 is free on TPU" claim that load-bears the rest of the
# project (ADRs assume BF16 throughput ≈ FP32 throughput on v5e). If it isn't
# true, Stage 2 dtype defaults are wrong.
#
# Method:
#   - Run the same model (vit_b16 — Stage 1 vision proxy for ResNet-50 since
#     the registry doesn't yet ship a ResNet-50 entry; both are vision-cls
#     workloads with similar arithmetic intensity at bs=32) twice on TPU:
#       run 1 — precision=bf16
#       run 2 — precision=fp32
#   - Both runs use the exact same harness (3 blocks × 100 passes, MAD outlier
#     removal, CV<10% gate); only --precision changes.
#   - Results land in results/runs.jsonl as ordinary rows; compare manually.
#
# What "BF16 is free" means concretely:
#   throughput_mean_samples_sec(bf16) within ±5% of throughput_mean_samples_sec(fp32),
#   AND latency_p50_ms(bf16) within ±5% of latency_p50_ms(fp32),
#   on v5e-1.
#
# Cost: ~10 min × 2 runs ≈ 20 min on v5e-1 spot ≈ \$0.12.
# Run inside tmux (R5) — same flow as 50_run_smoke.sh.
#
# Usage:
#   ./scripts/53_run_bf16_validation.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="53_bf16_validation"
setup_error_trap
banner "R20 — BF16-vs-FP32 validation (vit_b16 on v5e-1, ~20 min, \$0.12)"

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No TPU state. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

MODEL_ID="${MODEL_ID:-vit_b16}"
log_info "Model: $MODEL_ID  TPU: $TPU_NAME ($TPU_ZONE)"
log_info "This runs --model $MODEL_ID --precision bf16 then --precision fp32."

gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command="set -euo pipefail
        cd ~/tpu
        [ -f ~/.tpu-bench-env ] && source ~/.tpu-bench-env || true

        SESSION='${TMUX_SESSION}-r20'
        LOG=~/tpu/results/r20_bf16_val.\$(date +%Y%m%d-%H%M%S).log
        mkdir -p ~/tpu/results

        tmux kill-session -t \"\$SESSION\" 2>/dev/null || true

        tmux new-session -d -s \"\$SESSION\" \\
            \"cd ~/tpu && (
                echo '── BF16 ──' && \\
                python3 -u -m benchmarks.harness --model $MODEL_ID --precision bf16 --device tpu && \\
                echo '── FP32 ──' && \\
                python3 -u -m benchmarks.harness --model $MODEL_ID --precision fp32 --device tpu
            ) 2>&1 | tee \$LOG\"

        echo \"started tmux session '\$SESSION' (log: \$LOG)\"
        echo '── tailing log; Ctrl-C here just stops the tail, NOT the run ──'

        while tmux has-session -t \"\$SESSION\" 2>/dev/null; do
            tail -n +1 -f \"\$LOG\" --pid \$\$ 2>/dev/null &
            tail_pid=\$!
            while tmux has-session -t \"\$SESSION\" 2>/dev/null; do sleep 2; done
            kill \$tail_pid 2>/dev/null || true
        done

        echo
        echo '── tail (last 60 lines) ──'
        tail -60 \"\$LOG\"
    "

# ── Verify both rows landed and compare ────────────────────────────────────────
section "verify"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command="cd ~/tpu && python3 - <<'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path('results/runs.jsonl').read_text().splitlines() if l.strip()]
recent = [r for r in rows[-10:] if r.get('model') == '$MODEL_ID' and r.get('status') != 'failed']
bf16 = next((r for r in reversed(recent) if r['precision'] == 'bf16'), None)
fp32 = next((r for r in reversed(recent) if r['precision'] == 'fp32'), None)
if not bf16 or not fp32:
    print('VERDICT: incomplete — need both bf16 and fp32 rows for $MODEL_ID')
    raise SystemExit(2)
ratio_tp = bf16['throughput_mean_samples_sec'] / fp32['throughput_mean_samples_sec']
ratio_lat = bf16['latency_p50_ms'] / fp32['latency_p50_ms']
print(f'bf16: tp={bf16[\"throughput_mean_samples_sec\"]:.1f}, p50={bf16[\"latency_p50_ms\"]:.4f} ms')
print(f'fp32: tp={fp32[\"throughput_mean_samples_sec\"]:.1f}, p50={fp32[\"latency_p50_ms\"]:.4f} ms')
print(f'tp ratio bf16/fp32: {ratio_tp:.3f}')
print(f'p50 ratio bf16/fp32: {ratio_lat:.3f}')
within = (0.95 <= ratio_tp <= 1.05) and (0.95 <= ratio_lat <= 1.05)
print(f'VERDICT: BF16 is {\"FREE\" if within else \"NOT FREE\"} on v5e-1 (±5% gate)')
PY
"

state_set R20_COMPLETED_AT "$(date -Iseconds)"
log_ok "R20 done. Pull rows with ./scripts/60_pull_results.sh and update context.md §19."
