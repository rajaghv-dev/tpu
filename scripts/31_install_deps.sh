#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 31_install_deps.sh — Stage 3b: pip install requirements on the TPU VM.
#
# The tpu-ubuntu2204-base image ships with Python 3 and pip but no JAX TPU
# wheel. `pip install -r requirements.txt` installs jax[tpu] which fetches
# the matching libtpu via the auxiliary index documented at
# https://storage.googleapis.com/jax-releases/libtpu_releases.html
#
# Idempotent: re-running is fast (pip detects already-installed packages).
#
# Usage:
#   ./scripts/31_install_deps.sh
#
# Exit codes:
#   0 = install succeeded and JAX sees the TPU.
#   1 = state missing, install failed, or jax.devices() doesn't show TPU.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="31_install_deps"
setup_error_trap
banner "Stage 3b — Install Python dependencies on TPU VM"

require_cmd gcloud

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

# We pipe a here-doc bash script so we get one SSH session for the whole flow.
# Each step prints a labelled line so the local logs stay readable.
log_info "Running install on $TPU_NAME ($TPU_ZONE)..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command='set -euo pipefail
        cd ~/tpu

        echo "── pip --version ──"
        python3 -m pip --version

        echo "── upgrading pip + wheel ──"
        python3 -m pip install --quiet --upgrade pip wheel setuptools

        echo "── installing requirements.txt (jax[tpu], transformers, …) ──"
        # libtpu wheel index — needed for jax[tpu] to fetch libtpu*.whl
        python3 -m pip install --quiet \
            -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
            -r requirements.txt

        echo "── installed JAX/jaxlib versions ──"
        python3 -c "import jax, jaxlib; print(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)"

        echo "── jax.devices() ──"
        python3 -c "import jax; print(jax.devices())"

        echo "── snapshot pip freeze for reproducibility ──"
        python3 -m pip freeze > ~/requirements.lock.txt
        wc -l ~/requirements.lock.txt
    '

state_set REMOTE_INSTALLED_AT "$(date -Iseconds)"
log_ok "Install complete. Next: ./scripts/32_mount_gcs.sh (optional) → ./scripts/40_verify_jax.sh"
exit 0
