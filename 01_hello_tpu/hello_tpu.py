"""
01 – Hello TPU
Verifies the TPU is reachable and runs a few basic operations.
Run: python hello_tpu.py
"""

import jax
import jax.numpy as jnp


def device_info() -> None:
    devices = jax.devices()
    print(f"Backend  : {jax.default_backend()}")
    print(f"Devices  : {len(devices)}")
    for d in devices:
        print(f"  {d}")


def basic_ops() -> None:
    # Create arrays directly on TPU
    x = jnp.arange(1.0, 9.0).reshape(2, 4)
    y = jnp.ones_like(x)

    print("\nMatrix multiply (2x4) @ (4x2):")
    print(jnp.dot(x, y.T))

    print("\nSoftmax over last axis:")
    print(jax.nn.softmax(x, axis=-1))


@jax.jit
def jit_fn(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2  # should be all-ones


def pmap_example() -> None:
    n = jax.device_count()
    x = jnp.ones((n, 128))
    result = jax.pmap(jit_fn)(x)
    assert jnp.allclose(result, jnp.ones_like(result)), "pmap sanity check failed"
    print(f"\npmap across {n} device(s): OK (all-ones confirmed)")


if __name__ == "__main__":
    device_info()
    basic_ops()
    pmap_example()
    print("\nAll checks passed!")
