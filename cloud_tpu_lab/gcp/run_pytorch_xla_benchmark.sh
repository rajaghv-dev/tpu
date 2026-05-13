#!/usr/bin/env bash
# PAID: This script EXECUTES work on a running Cloud TPU VM and so consumes
#       paid TPU-hours. Cleanup of the VM itself: ./delete_tpu_vm.sh
# ──────────────────────────────────────────────────────────────────────────────
# run_pytorch_xla_benchmark.sh — tiny torch_xla matmul benchmark on the TPU VM.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[run_pytorch_xla_benchmark] %s\n' "$*"; }

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
  Running a PyTorch/XLA benchmark on TPU VM:
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
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print("xla_device:", device)
print("torch:", torch.__version__)

N = 4096
a = torch.randn(N, N, device=device, dtype=torch.float32)
b = torch.randn(N, N, device=device, dtype=torch.float32)

# Warmup + force compile/exec.
c = a @ b
xm.mark_step()
_ = c.cpu()

iters = 10
t0 = time.perf_counter()
for _ in range(iters):
    c = a @ b
    xm.mark_step()
_ = c.cpu()  # sync
t1 = time.perf_counter()

avg_ms = (t1 - t0) / iters * 1000.0
flops = 2 * N**3
tflops = flops / ((t1 - t0) / iters) / 1e12

result = {
    "framework": "torch_xla",
    "matrix_size": N,
    "iters": iters,
    "avg_ms": avg_ms,
    "tflops_est": tflops,
    "device": str(device),
}
print("RESULT:", json.dumps(result, indent=2))
out_path = os.path.expanduser("~/cloud_tpu_lab_artifacts/pytorch_xla_benchmark.json")
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
