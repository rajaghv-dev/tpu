#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# install_tensorflow_tpu.sh — install TensorFlow + libtpu on the TPU VM.
#
# Refresh versions and the libtpu wheels URL from the official guide:
#   https://cloud.google.com/tpu/docs/run-calculation-tensorflow
#   https://www.tensorflow.org/install
#
# Does NOT create paid resources.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[install_tensorflow_tpu] %s\n' "$*"; }

# update from https://cloud.google.com/tpu/docs/run-calculation-tensorflow
TF_VERSION="${TF_VERSION:-2.17.0}"
LIBTPU_WHEELS_URL="${LIBTPU_WHEELS_URL:-https://storage.googleapis.com/libtpu-releases/index.html}"

log "installing tensorflow==$TF_VERSION + libtpu on $TPU_NAME ..."

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
echo "[remote] python: \$(python3 --version)"
python3 -m pip install --upgrade pip
python3 -m pip install "tensorflow==${TF_VERSION}"
# libtpu is the TPU plugin TF loads at runtime.
python3 -m pip install libtpu-nightly -f "${LIBTPU_WHEELS_URL}" || \\
    python3 -m pip install libtpu -f "${LIBTPU_WHEELS_URL}"
echo "[remote] verifying TPUClusterResolver ..."
python3 - <<'PY'
import tensorflow as tf
print("tf", tf.__version__)
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("tpu devices:", tf.config.list_logical_devices("TPU"))
PY
EOF
)

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="$REMOTE_CMD"

log "TensorFlow TPU install complete."
log "Cleanup reminder: TPU VM '$TPU_NAME' is still running — ./delete_tpu_vm.sh when done."
