#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# load_run.sh — turn a collected real-TPU run into live Grafana data.
#
# Given a run directory under artifacts/from_vm/ (or the latest one if no arg),
# this script:
#   1. Normalises the JSONL to the exporter's event schema (renames
#      train.step → runtime.step, flattens hbm.snapshot fields, injects
#      workload_name / framework / tpu_version / run_mode labels).
#   2. Copies the JSONL + CSV into artifacts/{logs,metrics}/ where Promtail
#      and the Python exporter watch.
#   3. Restarts the host-side metrics exporter and the Promtail container
#      so they re-read the file from the beginning.
#   4. Verifies the data is actually queryable in Prometheus + Loki.
#   5. Prints a deep-link to Grafana with the right time range bracketing
#      the run.
#
# Usage:
#   ./observability/load_run.sh                                 # newest run
#   ./observability/load_run.sh artifacts/from_vm/<RUN_TAG>     # specific
#   ./observability/load_run.sh --tpu-version v5e               # override label
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { printf '[load_run] %s\n' "$*"; }
err() { printf '[load_run] ERROR: %s\n' "$*" >&2; exit 1; }

RUN_DIR=""
TPU_VERSION="v5e"
FRAMEWORK="jax"
RUN_MODE="cloud_tpu_vm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu-version) TPU_VERSION="$2"; shift 2 ;;
        --framework)   FRAMEWORK="$2";   shift 2 ;;
        --run-mode)    RUN_MODE="$2";    shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | head -25 | sed 's/^# //'
            exit 0
            ;;
        -*) err "unknown flag: $1" ;;
        *)  RUN_DIR="$1"; shift ;;
    esac
done

# ── 1. Resolve the run dir ──────────────────────────────────────────────────
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -1dt "$REPO/artifacts/from_vm/"*/ 2>/dev/null | head -1 || true)
    [[ -z "$RUN_DIR" ]] && err "no runs found under $REPO/artifacts/from_vm/"
fi
RUN_DIR="${RUN_DIR%/}"   # strip trailing slash
[[ -d "$RUN_DIR" ]] || err "not a directory: $RUN_DIR"

SRC_JSONL=$(ls "$RUN_DIR"/run_*.jsonl 2>/dev/null | head -1 || true)
SRC_CSV=$(ls "$RUN_DIR"/run_*.csv 2>/dev/null | head -1 || true)
[[ -f "$SRC_JSONL" ]] || err "no run_*.jsonl in $RUN_DIR"
[[ -f "$SRC_CSV"   ]] || err "no run_*.csv in $RUN_DIR"

log "run dir   : $RUN_DIR"
log "src jsonl : $(basename "$SRC_JSONL")"
log "labels    : framework=$FRAMEWORK tpu_version=$TPU_VERSION run_mode=$RUN_MODE"

# ── 2. Normalise + copy ─────────────────────────────────────────────────────
DST_LOGS="$REPO/artifacts/logs"
DST_METRICS="$REPO/artifacts/metrics"
mkdir -p "$DST_LOGS" "$DST_METRICS"
DST_JSONL="$DST_LOGS/$(basename "$SRC_JSONL")"
DST_CSV="$DST_METRICS/$(basename "$SRC_CSV")"

python3 - "$SRC_JSONL" "$DST_JSONL" "$FRAMEWORK" "$TPU_VERSION" "$RUN_MODE" <<'PY'
import json, sys, pathlib
src, dst, framework, tpu_version, run_mode = sys.argv[1:]
identity = {"framework": framework, "tpu_version": tpu_version, "run_mode": run_mode}
out, ts_first, ts_last = [], None, None
for line in pathlib.Path(src).read_text().splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    # Identity labels (for Prometheus safe-label set)
    for k, v in identity.items():
        d.setdefault(k, v)
    d.setdefault("workload_name", "real_" + str(d.get("workload", "matmul")))
    # Event renames + field flattening
    ev = d.get("event")
    if ev == "train.step":
        d["event"] = "runtime.step"
        st, sps = d.get("step_time_s"), d.get("samples_per_step")
        if st and sps:
            d["samples_per_second"] = sps / st
        d["device_execution_time_s"] = d.get("step_time_s", 0)
        # Emit a separate xla.compile event implicit in the first compile_step
        if d.get("compile_step"):
            comp = dict(d, event="xla.compile", layer="xla",
                        compile_time_s=d.get("step_time_s", 0), cache_hit=False)
            out.append(json.dumps(comp))
    elif ev == "hbm.snapshot":
        m = d.get("metrics") or {}
        for ks, kd in (("used_bytes","hbm_used_bytes"),
                       ("capacity_bytes","hbm_capacity_bytes"),
                       ("utilization","hbm_utilization_ratio"),
                       ("peak_bytes","hbm_peak_bytes")):
            if ks in m:
                d[kd] = m[ks]
    ts = d.get("timestamp")
    if ts:
        ts_first = ts_first or ts
        ts_last = ts
    out.append(json.dumps(d))
pathlib.Path(dst).write_text("\n".join(out) + "\n")
print(f"normalised {len(out)} events")
print(f"first_ts={ts_first}")
print(f"last_ts={ts_last}")
PY

cp "$SRC_CSV" "$DST_CSV"
log "wrote     : $DST_JSONL"
log "wrote     : $DST_CSV"

# ── 3. Restart exporter (host) ──────────────────────────────────────────────
log "restarting host metrics exporter ..."
pkill -f cloud_tpu_metrics_exporter 2>/dev/null || true
sleep 1
nohup python3 "$REPO/observability/exporters/cloud_tpu_metrics_exporter.py" \
    --port 9100 \
    --log-path "$DST_LOGS/*.jsonl" > /tmp/cloud_tpu_exporter.log 2>&1 &
EXPORTER_PID=$!
log "exporter pid=$EXPORTER_PID (logs in /tmp/cloud_tpu_exporter.log)"

# ── 4. Restart promtail (container) so it re-reads the new file ────────────
log "restarting promtail container + clearing its position cache ..."
docker exec cloud_tpu_lab_promtail rm -f /tmp/positions.yaml 2>/dev/null || true
docker restart cloud_tpu_lab_promtail >/dev/null 2>&1 \
    || log "WARN: docker restart cloud_tpu_lab_promtail failed (is the stack up?)"

sleep 5

# ── 5. Verify ───────────────────────────────────────────────────────────────
log "─── verifying ──────────────────────────────────────────────────────────"

EXP_VAL=$(curl -s localhost:9100/metrics | grep -E "^cloud_tpu_step_time_seconds\{.*framework=\"${FRAMEWORK}\"" | head -1 || true)
if [[ -n "$EXP_VAL" ]]; then
    log "  exporter   : OK  $EXP_VAL"
else
    log "  exporter   : EMPTY — check /tmp/cloud_tpu_exporter.log"
fi

PROM_VAL=$(curl -s -G 'http://localhost:9090/api/v1/query' --data-urlencode "query=cloud_tpu_step_time_seconds{framework=\"${FRAMEWORK}\"}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data']['result'][0]['value'][1] if d['data']['result'] else 'empty')")
log "  prometheus : value = $PROM_VAL"

LOKI_N=$(curl -G -s "http://localhost:3100/loki/api/v1/query_range" \
    --data-urlencode "query={app=\"cloud_tpu_lab\",framework=\"${FRAMEWORK}\"}" \
    --data-urlencode "start=$(($(date +%s) - 86400))000000000" \
    --data-urlencode "end=$(date +%s)000000000" \
    --data-urlencode "limit=1000" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(len(s.get('values',[])) for s in d.get('data',{}).get('result',[])))")
log "  loki       : $LOKI_N log lines in last 24h"

# ── 6. Build a Grafana time-range URL ───────────────────────────────────────
# Bracket the run with ±10 min for visual context.
log "─── done ──────────────────────────────────────────────────────────────"
echo ""
echo "Open Grafana:"
echo "  http://localhost:3000/d/cloud_tpu_overview?orgId=1&from=now-24h&to=now"
echo ""
echo "Or pin to this specific run:"
echo "  http://localhost:3000/d/cloud_tpu_overview?orgId=1&from=now-2h&to=now&var-framework=${FRAMEWORK}&var-tpu_version=${TPU_VERSION}"
echo ""
echo "Quick Explore queries:"
echo "  Prometheus:  cloud_tpu_step_time_seconds"
echo "  Prometheus:  cloud_tpu_samples_per_second"
echo "  Prometheus:  cloud_tpu_hbm_used_bytes / 1e6     # MB"
echo "  Loki:        {app=\"cloud_tpu_lab\"} | json | event=\"runtime.step\""
