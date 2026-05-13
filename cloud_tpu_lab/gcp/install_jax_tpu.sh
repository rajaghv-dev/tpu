#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# install_jax_tpu.sh — install JAX + flax + optax + transformers on the TPU VM.
#
# Runs the installer remotely over SSH. The VM itself is already paid for
# (created by create_tpu_vm.sh) — this script does not create new resources,
# but it does consume VM time, so don't leave it running idle.
#
# Pinned versions below are a reasonable starting point. Update them from
# https://docs.jax.dev/en/latest/installation.html (especially the TPU wheels
# URL and the supported jaxlib build for your RUNTIME_VERSION).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[install_jax_tpu] %s\n' "$*"; }

# update from https://docs.jax.dev/en/latest/installation.html
JAX_TPU_WHEELS_URL="${JAX_TPU_WHEELS_URL:-https://storage.googleapis.com/jax-releases/libtpu_releases.html}"
JAX_VERSION="${JAX_VERSION:-0.4.34}"
FLAX_VERSION="${FLAX_VERSION:-0.9.0}"
OPTAX_VERSION="${OPTAX_VERSION:-0.2.3}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.44.2}"

log "installing JAX[tpu]==$JAX_VERSION + flax + optax + transformers on $TPU_NAME ..."

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
echo "[remote] python: \$(python3 --version)"
python3 -m pip install --upgrade pip
python3 -m pip install \\
    "jax[tpu]==${JAX_VERSION}" \\
    -f "${JAX_TPU_WHEELS_URL}"
python3 -m pip install \\
    "flax==${FLAX_VERSION}" \\
    "optax==${OPTAX_VERSION}" \\
    "transformers==${TRANSFORMERS_VERSION}"
echo "[remote] verifying ..."
python3 -c "import jax; print('jax', jax.__version__); print('devices:', jax.devices())"
EOF
)

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="$REMOTE_CMD"

log "JAX TPU install complete."
log "Cleanup reminder: TPU VM '$TPU_NAME' is still running — ./delete_tpu_vm.sh when done."
