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

# By default install LATEST stable JAX[tpu] + matching bundled libtpu (since
# JAX 0.4.35 libtpu is on PyPI, so the legacy `-f libtpu_releases.html` flag
# is no longer required). Override below if you need a specific version.
JAX_VERSION="${JAX_VERSION:-}"   # empty = latest

log "installing JAX[tpu]${JAX_VERSION:+==$JAX_VERSION} + flax + optax + transformers on $TPU_NAME ..."

# The remote pip install must be aggressive — TPU VMs often ship with a
# pre-installed JAX in the system / user site that pip won't otherwise
# downgrade or replace. We uninstall jax/jaxlib/libtpu explicitly, then
# pip install --force-reinstall to get a clean matched set.
JAX_SPEC="jax[tpu]"
[[ -n "$JAX_VERSION" ]] && JAX_SPEC="jax[tpu]==${JAX_VERSION}"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
echo "[remote] python: \$(python3 --version)"
python3 -m pip install --upgrade pip
echo "[remote] uninstalling any pre-existing jax / libtpu ..."
python3 -m pip uninstall -y jax jaxlib libtpu libtpu-nightly 2>/dev/null || true
echo "[remote] installing ${JAX_SPEC} (latest matched libtpu via PyPI) ..."
python3 -m pip install --upgrade --force-reinstall "${JAX_SPEC}"
python3 -m pip install --upgrade flax optax transformers
python3 -m pip install --upgrade tensorboard tensorboard-plugin-profile
echo "[remote] verifying ..."
python3 -c "import jax; print('jax', jax.__version__)"
python3 -c "import jaxlib; print('jaxlib', jaxlib.__version__)"
python3 -c "import libtpu; print('libtpu', libtpu.__version__)" 2>/dev/null || echo "(libtpu version probe skipped)"
python3 -c "import jax; print('devices:', jax.devices()); print('backend:', jax.default_backend())"
python3 -c "import tensorboard_plugin_profile; print('tb-profile OK')"
EOF
)

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="$REMOTE_CMD"

log "JAX TPU install complete."
log "Cleanup reminder: TPU VM '$TPU_NAME' is still running — ./delete_tpu_vm.sh when done."
