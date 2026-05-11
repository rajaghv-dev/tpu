#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# colab_bootstrap.sh — one-cell setup for the Colab Pro TPU path (todo.md Tier 3 #9).
#
# Clones the repo to /content/tpu, installs deps, runs a suite, prints summary.
# Idempotent: re-running skips the clone and reuses installed wheels.
#
# Usage (inside any Colab cell):
#   !curl -sL https://raw.githubusercontent.com/rajaghv-dev/tpu/main/scripts/colab_bootstrap.sh \
#       | bash -s -- [suite] [probes_set]
#
# Args (positional, both optional):
#   suite        smoke | quick                  (default: smoke)
#   probes_set   none  | default | full         (default: default)
#
# Requires: Colab runtime with TPU selected (Runtime → Change runtime type → TPU).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SUITE="${1:-smoke}"
PROBES="${2:-default}"
REPO_URL="${REPO_URL:-https://github.com/rajaghv-dev/tpu.git}"
REPO_DIR="${REPO_DIR:-/content/tpu}"

echo "[colab_bootstrap] suite=$SUITE  probes=$PROBES  repo=$REPO_URL"

# 1. Clone (idempotent)
if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "[colab_bootstrap] cloning into $REPO_DIR ..."
    git clone --depth=1 "$REPO_URL" "$REPO_DIR"
else
    echo "[colab_bootstrap] repo present, pulling latest ..."
    git -C "$REPO_DIR" pull --ff-only || echo "[colab_bootstrap] pull skipped (continuing)"
fi
cd "$REPO_DIR"

# 2. Install deps (Colab has jax[tpu] pre-installed; only the gaps matter)
echo "[colab_bootstrap] installing deps ..."
pip install -q -r requirements.txt
pip install -q 'transformers>=4.40,<4.45' 'flax>=0.8.3' \
              'opentelemetry-sdk>=1.27' 'opentelemetry-exporter-otlp-proto-grpc>=1.27'

# 3. Configure env (OTel → local file; XLA cache in /content)
export PYTHONPATH="$REPO_DIR"
export TPU_BENCH_OTEL=file
export TPU_BENCH_OTEL_DIR="$REPO_DIR/results/otel"
export JAX_COMPILATION_CACHE_DIR=/content/xla-cache
mkdir -p "$TPU_BENCH_OTEL_DIR" "$JAX_COMPILATION_CACHE_DIR"

# 4. Verify TPU visible
python - <<'PY'
import jax
devs = jax.devices()
print(f"[colab_bootstrap] jax {jax.__version__} | devices: {devs}")
assert any("TPU" in str(d) for d in devs), "No TPU found — Runtime → Change runtime type → TPU"
PY

# 5. Run the suite
echo "[colab_bootstrap] running: harness --suite $SUITE --device tpu --probes $PROBES"
python benchmarks/harness.py --suite "$SUITE" --device tpu --probes "$PROBES"

# 6. Summary — last 5 rows of runs.jsonl
echo "[colab_bootstrap] last 5 rows of results/runs.jsonl:"
python - <<'PY'
import json
from pathlib import Path
p = Path("results/runs.jsonl")
if not p.exists():
    print("  (no runs.jsonl yet)")
else:
    for line in p.read_text().strip().splitlines()[-5:]:
        r = json.loads(line)
        print(f"  {r.get('model','?')} | {r.get('device','?')} | "
              f"p50={r.get('latency_p50_ms','?')}ms | "
              f"tp={r.get('throughput_mean_samples_sec','?')} smp/s")
PY

echo "[colab_bootstrap] done. Download results/ for local Grafana replay (ADR-016)."
