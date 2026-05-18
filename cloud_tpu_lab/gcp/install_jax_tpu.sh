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

# Pin JAX to match the libtpu BAKED into the tpu-ubuntu2204-base image.
# That image ships libtpu_nightly_20241002 (Oct 2024) → JAX 0.4.34 era.
# Newer JAX (0.6.x) emits StableHLO 1.9.x which the image libtpu (1.7.x)
# can't parse — fails at first jit'd op. Pip can't replace the image
# libtpu without root, so we downgrade JAX instead.
#
# Override JAX_VERSION when you switch to a newer TPU image runtime.
JAX_VERSION="${JAX_VERSION:-0.4.34}"

log "installing JAX[tpu]${JAX_VERSION:+==$JAX_VERSION} + flax + optax + transformers on $TPU_NAME ..."

# The remote pip install must be aggressive — TPU VMs often ship with a
# pre-installed JAX in the system / user site that pip won't otherwise
# downgrade or replace. We uninstall jax/jaxlib/libtpu explicitly, then
# pip install --force-reinstall to get a clean matched set.
JAX_PIN="${JAX_VERSION}"   # always set (default 0.4.34 above)

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
echo "[remote] python: \$(python3 --version)"
python3 -m pip install --upgrade pip
# IMPORTANT: do NOT uninstall the system libtpu — the tpu-ubuntu2204-base image
# ships it and pip can't reinstall it without root. We only touch jax/jaxlib
# in user-site so they match the system libtpu.
echo "[remote] uninstalling user-site jax / jaxlib (leaving system libtpu alone) ..."
python3 -m pip uninstall -y jax jaxlib 2>/dev/null || true
echo "[remote] installing jax==${JAX_PIN} jaxlib==${JAX_PIN} from PyPI ..."
python3 -m pip install --upgrade --force-reinstall \\
    "jax==${JAX_PIN}" "jaxlib==${JAX_PIN}"
# Companion packages pinned to versions compatible with jax ${JAX_PIN}.
python3 -m pip install --upgrade --force-reinstall \\
    "flax==0.9.0" "optax==0.2.3" "transformers==4.44.2"
python3 -m pip install --upgrade tensorboard tensorboard-plugin-profile
echo "[remote] ───── verifying ───────────────────────────────────────────────"
python3 -c "import sys; print('python   :', sys.executable)"
python3 -c "import jax, jaxlib; print('jax      :', jax.__version__); print('jaxlib   :', jaxlib.__version__)"
python3 -c "import jax; print('devices  :', jax.devices()); print('backend  :', jax.default_backend())" || {
    echo "[remote] jax.devices() failed — printing diagnostics:"
    python3 -m pip show jax jaxlib 2>&1 | grep -E "Name|Version|Location" || true
    find /usr -name libtpu.so 2>/dev/null | head -3 || true
    find ~/.local -name libtpu.so 2>/dev/null | head -3 || true
    exit 1
}
python3 -c "import tensorboard_plugin_profile; print('tb-profile OK')"
echo "[remote] ───── install OK ──────────────────────────────────────────────"
EOF
)

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="$REMOTE_CMD"

log "JAX TPU install complete."
log "Cleanup reminder: TPU VM '$TPU_NAME' is still running — ./delete_tpu_vm.sh when done."
