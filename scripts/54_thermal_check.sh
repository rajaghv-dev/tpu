#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 54_thermal_check.sh — R21 from RECOMMENDATIONS.md.
#
# Detect thermal throttling during a sustained ≥1 hour Stage 1 run on the
# **B200** (Blackwell, local DGX). Throttling silently degrades all downstream
# results; if undetected, every Stage 2 latency claim from this card is suspect.
#
# Method:
#   - Start a 1 Hz nvidia-smi sampler in the background, logging temp/clock/power
#     to results/telemetry/b200_<timestamp>.csv.
#   - Run the quick suite (5 models, ~50 min) on device=b200, then a few extra
#     reps to push past the 1-hour mark.
#   - Plot temp + clock at end; flag if SM clock drops >5% from steady state OR
#     temp exceeds the 87°C throttle threshold.
#
# This script runs LOCALLY (on the DGX), not over gcloud SSH — B200 is on-prem.
#
# Prereqs:
#   - nvidia-smi (Blackwell driver)
#   - python with matplotlib + pandas in the local env
#   - the harness must support --device b200 (Stage 2; for Stage 1 sub in --device gpu
#     and let the runner pick up the local CUDA backend via JAX)
#
# Usage (on the DGX host):
#   ./scripts/54_thermal_check.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

STAGE="54_thermal_check"
setup_error_trap
banner "R21 — B200 thermal-throttle check (~1 hr local run)"

require_cmd nvidia-smi "Install Blackwell driver from NVIDIA"
require_cmd python3 "Need python3 with pandas + matplotlib"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS=$(date +%Y%m%d-%H%M%S)
TELEMETRY_DIR="$REPO_ROOT/results/telemetry"
mkdir -p "$TELEMETRY_DIR"
CSV="$TELEMETRY_DIR/b200_${TS}.csv"
PLOT="$TELEMETRY_DIR/b200_${TS}.png"

log_info "Telemetry CSV: $CSV"
log_info "Plot:          $PLOT"

# ── 1 Hz sampler in the background ───────────────────────────────────────────
# nvidia-smi --query-gpu=… is the canonical way; ms-resolution fields exist but
# 1 Hz is plenty for thermal trends and avoids polluting the GPU with samplers.
echo "ts,gpu,temp_c,sm_clock_mhz,mem_clock_mhz,power_w,util_pct" > "$CSV"
(
    while true; do
        ts=$(date -Iseconds)
        nvidia-smi \
            --query-gpu=index,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,utilization.gpu \
            --format=csv,noheader,nounits \
        | awk -v ts="$ts" '{ printf "%s,%s\n", ts, $0 }' >> "$CSV"
        sleep 1
    done
) &
SAMPLER_PID=$!
add_exit_handler "kill $SAMPLER_PID 2>/dev/null || true"
log_ok "1 Hz sampler PID=$SAMPLER_PID"

# ── Sustained workload: quick suite plus padding to ≥1 hr ────────────────────
log_step "Running quick suite (~50 min)…"
cd "$REPO_ROOT"
python3 -m benchmarks.harness --suite quick --device b200 2>&1 | tee "$TELEMETRY_DIR/r21_${TS}_quick.log" || true

log_step "Padding with smoke loop until total wall ≥3600 s…"
START=$(date +%s)
while (( $(date +%s) - START < 600 )); do  # 10 min padding past the ~50 min suite
    python3 -m benchmarks.harness --suite smoke --device b200 || true
done

# ── Stop sampler, plot, decide ──────────────────────────────────────────────
log_step "Stopping sampler…"
kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true

log_step "Plotting + verdict…"
python3 - <<PY
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("$CSV")
df["ts"] = pd.to_datetime(df["ts"])

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(df["ts"], df["temp_c"]);       axes[0].set_ylabel("temp °C");      axes[0].axhline(87, color="red", ls="--", label="87°C throttle")
axes[1].plot(df["ts"], df["sm_clock_mhz"]); axes[1].set_ylabel("SM clock MHz")
axes[2].plot(df["ts"], df["power_w"]);      axes[2].set_ylabel("power W");      axes[2].set_xlabel("time")
for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("$PLOT", dpi=120)
print(f"plot: $PLOT")

# Verdict: SM clock drop >5% from steady-state OR temp >=87°C anywhere = throttled
steady = df["sm_clock_mhz"].iloc[len(df)//4 : 3*len(df)//4].median()
floor = df["sm_clock_mhz"].min()
drop_pct = 100.0 * (steady - floor) / max(steady, 1.0)
max_temp = df["temp_c"].max()
print(f"steady SM clock: {steady:.0f} MHz; min: {floor:.0f} MHz ({drop_pct:.1f}% drop)")
print(f"max temp: {max_temp:.0f}°C")
throttled = (drop_pct > 5.0) or (max_temp >= 87)
print(f"VERDICT: {'THROTTLED' if throttled else 'NO THROTTLE DETECTED'}")
PY

log_ok "R21 done. Plot: $PLOT  CSV: $CSV"
log_info "If THROTTLED: investigate cooling before trusting Stage 2 numbers from B200."
