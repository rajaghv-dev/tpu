#!/usr/bin/env python3
"""
JAX Cloud TPU demo — CPU-safe.

If jax is installed: runs a tiny jit'd matmul on whatever the default
backend is (CPU locally, TPU on a Cloud TPU VM).
If jax is not installed: prints a clear message and exits 0.

This is the JAX entry point referenced by docs/04_jax_on_tpu.md.
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # noqa: BLE001
        print(f"[jax-demo] JAX not installed ({type(exc).__name__}). "
              f"Install jax[tpu] on a Cloud TPU VM with:")
        print("  pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html")
        return 0

    print(f"[jax-demo] jax version: {jax.__version__}")
    print(f"[jax-demo] default backend: {jax.default_backend()}")
    print(f"[jax-demo] devices: {jax.devices()}")

    @jax.jit
    def matmul(a, b):
        return a @ b

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (1024, 1024))
    b = jax.random.normal(key, (1024, 1024))

    # Warmup (compile).
    _ = matmul(a, b).block_until_ready()
    # Time 10 calls.
    t0 = time.perf_counter()
    for _ in range(10):
        c = matmul(a, b)
    c.block_until_ready()
    dt = time.perf_counter() - t0
    print(f"[jax-demo] 10× 1024×1024 matmul: {dt*1000:.2f} ms total, "
          f"{dt*100:.2f} ms/iter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
