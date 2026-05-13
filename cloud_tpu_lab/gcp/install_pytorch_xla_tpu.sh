#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# install_pytorch_xla_tpu.sh — install torch + torch_xla on the TPU VM.
#
# Versions and the wheels URL below should be refreshed from the official
# PyTorch/XLA installation page:
#   https://github.com/pytorch/xla#available-images-and-wheels
#   https://pytorch.org/xla/release/2.4/index.html
#
# Does NOT create paid resources. The TPU VM it targets is already running.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[install_pytorch_xla_tpu] %s\n' "$*"; }

# update from https://github.com/pytorch/xla#available-images-and-wheels
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.4.0}"
TORCH_XLA_WHEELS_URL="${TORCH_XLA_WHEELS_URL:-https://storage.googleapis.com/libtpu-releases/index.html}"

log "installing torch==$TORCH_VERSION + torch_xla==$TORCH_XLA_VERSION on $TPU_NAME ..."

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
echo "[remote] python: \$(python3 --version)"
python3 -m pip install --upgrade pip
# CPU wheel for torch is fine — torch_xla provides the TPU runtime.
python3 -m pip install "torch==${TORCH_VERSION}" \\
    --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install "torch_xla[tpu]==${TORCH_XLA_VERSION}" \\
    -f "${TORCH_XLA_WHEELS_URL}"
echo "[remote] verifying ..."
python3 -c "import torch_xla.core.xla_model as xm; print('xla_device:', xm.xla_device())"
EOF
)

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="$REMOTE_CMD"

log "PyTorch/XLA install complete."
log "Cleanup reminder: TPU VM '$TPU_NAME' is still running — ./delete_tpu_vm.sh when done."
