#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 32_mount_gcs.sh — Stage 3c: mount the GCS cache bucket on the TPU VM.
#
# Mounts $GCS_BUCKET at $GCS_MOUNT_POINT via gcsfuse so the HuggingFace cache
# (HF_HOME) and the JAX compile cache (JAX_COMPILATION_CACHE_DIR) survive
# preemptible-VM lifetimes. Implements ADR-006 + RECOMMENDATIONS R4.
#
# IMPORTANT — gcsfuse usage policy (R-I01):
#   - READ from the gcsfuse mount: fine.
#   - WRITE to the gcsfuse mount: AVOID for large files (atomicity not
#     guaranteed; partial-write corruption observed). Use `gcloud storage cp`
#     for bulk seed/upload instead. The harness writes only small JSONL
#     records and per-run log JSON, which gcsfuse handles correctly.
#
# Idempotent: if /mnt/gcs-cache is already a mountpoint we skip the mount.
#
# Usage:
#   ./scripts/32_mount_gcs.sh
#
# Exit codes:
#   0 = mounted (or already mounted).
#   1 = mount failed (gcsfuse not installed, perms, etc.).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="32_mount_gcs"
setup_error_trap
banner "Stage 3c — Mount GCS bucket via gcsfuse"

require_cmd gcloud

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

# Strip the gs:// prefix once because gcsfuse takes the bare bucket name.
BUCKET_NAME="${GCS_BUCKET#gs://}"

# We push a small bash heredoc to the VM that:
#   1. Installs gcsfuse if missing (one-time per VM lifetime).
#   2. Creates the mount point and mounts.
#   3. Sets HF_HOME/JAX_COMPILATION_CACHE_DIR/TRANSFORMERS_CACHE in ~/.bashrc
#      so future shells have them; also writes to a sourced file we can
#      `source` from later scripts without modifying interactive shells.
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command="set -euo pipefail
        BUCKET='$BUCKET_NAME'
        MOUNT='$GCS_MOUNT_POINT'

        echo '── gcsfuse install check ──'
        if ! command -v gcsfuse >/dev/null 2>&1; then
            echo 'installing gcsfuse...'
            export GCSFUSE_REPO=gcsfuse-\$(lsb_release -c -s)
            echo \"deb https://packages.cloud.google.com/apt \$GCSFUSE_REPO main\" \
              | sudo tee /etc/apt/sources.list.d/gcsfuse.list
            curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
              | sudo apt-key add -
            sudo apt-get update -q
            sudo apt-get install -y gcsfuse
        fi
        gcsfuse --version

        echo '── mount point ──'
        sudo mkdir -p \"\$MOUNT\"
        sudo chown \$(whoami) \"\$MOUNT\"

        echo '── mount status ──'
        if mountpoint -q \"\$MOUNT\"; then
            echo \"already mounted at \$MOUNT\"
        else
            # --implicit-dirs lets us see directories created via console UI.
            # No --foreground; we want it in the background.
            gcsfuse --implicit-dirs \"\$BUCKET\" \"\$MOUNT\"
            echo \"mounted \$BUCKET → \$MOUNT\"
        fi

        echo '── env vars ──'
        cat > ~/.tpu-bench-env <<EOF
export HF_HOME=$HF_HOME_REMOTE
export TRANSFORMERS_CACHE=\$HF_HOME/transformers
export JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR_REMOTE
export HF_TOKEN=\${HF_TOKEN:-}
EOF
        # Append-once to .bashrc so interactive shells inherit too.
        if ! grep -q 'tpu-bench-env' ~/.bashrc 2>/dev/null; then
            echo '[ -f ~/.tpu-bench-env ] && source ~/.tpu-bench-env' >> ~/.bashrc
        fi
        # Create the cache subdirectories now so first-run is faster.
        mkdir -p \"$HF_HOME_REMOTE\" \"$JAX_COMPILATION_CACHE_DIR_REMOTE\"

        echo '── write probe through gcsfuse (small file only — see R-I01) ──'
        probe=\"\$MOUNT/.bench-probe-\$(date +%s)\"
        echo 'mount-probe' > \"\$probe\"
        cat \"\$probe\" >/dev/null
        rm \"\$probe\"
        echo 'mount probe ok'
    "

state_set REMOTE_GCS_MOUNTED_AT "$(date -Iseconds)"
log_ok "GCS mounted. Next: ./scripts/40_verify_jax.sh"
exit 0
