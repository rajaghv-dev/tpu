#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# start_profiler.sh — start jax.profiler.start_server on the TPU VM and open
# a local SSH port-forward so you can attach the JAX/TensorBoard profiler from
# your laptop.
#
# The remote process is a foreground python that holds the profiler server
# open. Ctrl+C locally tears down the SSH tunnel and (because we use
# `--command` over a single SSH session) also kills the remote process.
#
# Does not create paid resources itself, but the TPU VM is billed for as long
# as it is up. Run ./delete_tpu_vm.sh when finished.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[start_profiler] %s\n' "$*"; }

PROFILER_PORT="${PROFILER_PORT:-9012}"
LOCAL_PORT="${LOCAL_PORT:-$PROFILER_PORT}"

log "starting JAX profiler server on $TPU_NAME:$PROFILER_PORT"
log "local port-forward: localhost:$LOCAL_PORT → $TPU_NAME:$PROFILER_PORT"
log "attach with TensorBoard's profile plugin: 'localhost:$LOCAL_PORT'"
log "Ctrl+C to stop."

REMOTE_CMD="python3 -c \"import jax, jax.profiler, time; \
jax.profiler.start_server($PROFILER_PORT); \
print('profiler server listening on :$PROFILER_PORT'); \
print('devices:', jax.devices()); \
[time.sleep(3600) for _ in iter(int, 1)]\""

exec gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --ssh-flag="-L ${LOCAL_PORT}:localhost:${PROFILER_PORT}" \
    --command="$REMOTE_CMD"
