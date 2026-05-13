#!/usr/bin/env bash
# PAID: This script EXECUTES work on a running Cloud TPU VM and so consumes
#       paid TPU-hours. Cleanup of the VM itself: ./delete_tpu_vm.sh
# ──────────────────────────────────────────────────────────────────────────────
# run_tensorflow_benchmark.sh — tiny TF matmul benchmark on the TPU VM.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[run_tensorflow_benchmark] %s\n' "$*"; }

ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) echo "Usage: $0 [--yes]"; exit 0 ;;
    esac
done

cat <<WARN
============================================================
  PAID RESOURCE WARNING
------------------------------------------------------------
  Running a TensorFlow benchmark on TPU VM:
    project = $PROJECT_ID
    zone    = $ZONE
    tpu     = $TPU_NAME ($ACCELERATOR_TYPE)
  This burns TPU-hours. Run ./delete_tpu_vm.sh when finished.
============================================================
WARN

if [[ "$ASSUME_YES" -ne 1 ]]; then
    read -r -p "Proceed? [y/N] " ans
    case "${ans:-}" in y|Y|yes|YES) ;; *) log "aborted."; exit 1 ;; esac
fi

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail
mkdir -p ~/cloud_tpu_lab_artifacts
python3 - <<'PY'
import time, json, os
import tensorflow as tf

print("tf:", tf.__version__)
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
print("tpu devices:", tf.config.list_logical_devices("TPU"))

N = 4096
with strategy.scope():
    a = tf.random.normal((N, N), dtype=tf.float32)
    b = tf.random.normal((N, N), dtype=tf.float32)

@tf.function
def matmul(x, y):
    return tf.linalg.matmul(x, y)

# Warmup (trace + compile).
_ = strategy.run(matmul, args=(a, b))

iters = 10
t0 = time.perf_counter()
for _ in range(iters):
    out = strategy.run(matmul, args=(a, b))
# Force sync by reading a scalar.
_ = float(tf.reduce_mean(out).numpy()) if hasattr(out, "numpy") else 0.0
t1 = time.perf_counter()

avg_ms = (t1 - t0) / iters * 1000.0
flops = 2 * N**3
tflops = flops / ((t1 - t0) / iters) / 1e12

result = {
    "framework": "tensorflow",
    "matrix_size": N,
    "iters": iters,
    "avg_ms": avg_ms,
    "tflops_est": tflops,
    "tpu_devices": [d.name for d in tf.config.list_logical_devices("TPU")],
}
print("RESULT:", json.dumps(result, indent=2))
out_path = os.path.expanduser("~/cloud_tpu_lab_artifacts/tensorflow_benchmark.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("saved →", out_path)
PY
EOF
)

log "executing benchmark on $TPU_NAME ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="$REMOTE_CMD"

log "done. To pull the JSON result locally: ./collect_artifacts.sh"
log "Cleanup reminder: ./delete_tpu_vm.sh"
