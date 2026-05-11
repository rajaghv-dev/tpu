#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 40_verify_jax.sh — Stage 4a: confirm JAX sees the TPU on the remote VM.
#
# A no-op `import jax; jax.devices()` test. If this fails, the install is
# broken (libtpu missing, mismatched jaxlib, no driver) and there's no point
# running pytest or the harness.
#
# Reads TPU_NAME / TPU_ZONE from .tpu-bench-state/state.env.
#
# Usage:
#   ./scripts/40_verify_jax.sh
#
# Exit codes:
#   0 = `jax.devices()` returns at least one TpuDevice.
#   1 = state missing, JAX import failed, or no TPU device visible.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="40_verify_jax"
setup_error_trap
banner "Stage 4a — Verify JAX sees the TPU"

TPU_NAME=$(state_get TPU_NAME "")
TPU_ZONE=$(state_get TPU_ZONE "")
if [[ -z "$TPU_NAME" || -z "$TPU_ZONE" ]]; then
    log_err "No state recorded. Run ./scripts/20_provision_tpu.sh first."
    exit 1
fi

# We send a small Python probe that:
#   - imports JAX (catches install errors),
#   - lists devices,
#   - asserts at least one device is a TPU,
#   - exits non-zero on mismatch so this script's exit code reflects the truth.
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --quiet \
    --command='set -euo pipefail
        cd ~/tpu
        python3 - <<PY
import sys
try:
    import jax
except Exception as e:
    print(f"FAIL import jax: {e!r}", file=sys.stderr)
    sys.exit(2)

devs = jax.devices()
print("jax", jax.__version__)
print("devices:", devs)
ok = any(("Tpu" in type(d).__name__) or ("tpu" in str(d.platform).lower()) for d in devs)
if not ok:
    print(f"FAIL no TPU devices in {devs!r}", file=sys.stderr)
    sys.exit(3)

# A quick matmul confirms compute actually works on the device, not just the
# device-listing API.
import jax.numpy as jnp, time
x = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
@jax.jit
def f(a): return a @ a
y = f(x); y.block_until_ready()
t0 = time.perf_counter()
for _ in range(50): y = f(x)
y.block_until_ready()
dt = (time.perf_counter() - t0)/50*1000
print(f"1024x1024 bf16 matmul: {dt:.2f} ms/iter (50-iter avg)")
print("OK jax+tpu working")
PY
    '

log_ok "JAX/TPU verification passed. Next: ./scripts/41_run_pytests.sh"
exit 0
