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

# Install latest jax[tpu] including the matching libtpu wheel via the
# legacy libtpu_releases.html URL (still authoritative for TPU libtpu wheels
# as of 2026). Pinning is intentionally NOT used by default — user-site
# libtpu wins Python's import path, so a fresh wheel + fresh jax pair
# overrides whatever the image baked in.
#
# Override JAX_VERSION (e.g. "==0.4.34") to pin if you know what you're doing.
JAX_VERSION="${JAX_VERSION:-}"

log "installing JAX[tpu]${JAX_VERSION:+==$JAX_VERSION} + flax + optax + transformers on $TPU_NAME ..."

# The remote pip install must be aggressive — TPU VMs often ship with a
# pre-installed JAX in the system / user site that pip won't otherwise
# downgrade or replace. We uninstall jax/jaxlib/libtpu explicitly, then
# pip install --force-reinstall to get a clean matched set.
JAX_SPEC="jax[tpu]${JAX_VERSION:+==${JAX_VERSION}}"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
echo "[remote] python: \$(python3 --version)"
python3 -m pip install --upgrade pip
echo "[remote] uninstalling jax/jaxlib/libtpu (force a clean reinstall) ..."
python3 -m pip uninstall -y jax jaxlib libtpu libtpu-nightly 2>/dev/null || true
echo "[remote] installing ${JAX_SPEC} via libtpu_releases.html ..."
python3 -m pip install --upgrade --force-reinstall "${JAX_SPEC}" \\
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install --upgrade tensorboard tensorboard-plugin-profile
echo "[remote] ───── verifying ───────────────────────────────────────────────"
python3 -c "import sys; print('python   :', sys.executable)"
python3 -c "import jax, jaxlib; print('jax      :', jax.__version__); print('jaxlib   :', jaxlib.__version__)"
python3 -c "
import sys, jax
backend = jax.default_backend()
devs = jax.devices()
print('devices  :', devs)
print('backend  :', backend)
if backend != 'tpu':
    print('[remote] ERROR: backend is', backend, '— expected tpu. libtpu not loadable by this jax.', file=sys.stderr)
    sys.exit(2)
" || {
    rc=\$?
    echo "[remote] jax verification failed (rc=\$rc) — printing diagnostics:"
    python3 -m pip show jax jaxlib libtpu libtpu-nightly 2>&1 | grep -E "Name|Version|Location" || true
    echo "[remote] system libtpu.so candidates:"
    find /usr -name "libtpu.so*" 2>/dev/null | head -3 || true
    find ~/.local -name "libtpu.so*" 2>/dev/null | head -3 || true
    exit \$rc
}
python3 -c "import tensorboard_plugin_profile; print('tb-profile OK')"
echo "[remote] ───── install OK — jax sees TPU ─────────────────────────────"
EOF
)

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="$REMOTE_CMD"

log "JAX TPU install complete."
log "Cleanup reminder: TPU VM '$TPU_NAME' is still running — ./delete_tpu_vm.sh when done."
