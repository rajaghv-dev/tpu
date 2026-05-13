#!/usr/bin/env bash
# PAID: This script EXECUTES work on a running Cloud TPU VM and so consumes
#       paid TPU-hours. It does not create or delete VMs. Cleanup of the VM
#       itself: ./delete_tpu_vm.sh
# ──────────────────────────────────────────────────────────────────────────────
# run_jax_benchmark.sh — tiny JAX matmul benchmark on the TPU VM.
#
# Prints device info, runs a few jit'd matmul forward passes, times them.
# Output from the remote run is echoed back here.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[run_jax_benchmark] %s\n' "$*"; }

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
  Running a JAX benchmark on TPU VM:
    project = $PROJECT_ID
    zone    = $ZONE
    tpu     = $TPU_NAME ($ACCELERATOR_TYPE)
  This burns TPU-hours. The VM keeps billing after the
  benchmark exits — run ./delete_tpu_vm.sh when finished.
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
import jax
import jax.numpy as jnp

print("jax", jax.__version__)
print("devices:", jax.devices())
print("backend:", jax.default_backend())

N = 4096
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (N, N), dtype=jnp.float32)
b = jax.random.normal(key, (N, N), dtype=jnp.float32)

@jax.jit
def matmul(x, y):
    return x @ y

# Warmup (compile).
out = matmul(a, b).block_until_ready()

iters = 10
t0 = time.perf_counter()
for _ in range(iters):
    out = matmul(a, b).block_until_ready()
t1 = time.perf_counter()

avg_ms = (t1 - t0) / iters * 1000.0
flops = 2 * N**3
tflops = flops / ((t1 - t0) / iters) / 1e12

result = {
    "framework": "jax",
    "matrix_size": N,
    "iters": iters,
    "avg_ms": avg_ms,
    "tflops_est": tflops,
    "devices": [str(d) for d in jax.devices()],
}
print("RESULT:", json.dumps(result, indent=2))
out_path = os.path.expanduser("~/cloud_tpu_lab_artifacts/jax_benchmark.json")
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
