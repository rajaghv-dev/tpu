"""
08 – Multi-Host TPU Pod Training
Initialises jax.distributed so all hosts in a TPU pod see a single logical
device mesh, then trains a model with fully-sharded data parallelism.

Run on each host:
  python train.py --coordinator_address=<host0-ip>:8476 \
                  --num_processes=<N>          \
                  --process_id=<this-host-id>
"""

import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state


# ---------------------------------------------------------------------------
# Model (same as example 02 for simplicity)
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(10)(x)
        return x


# ---------------------------------------------------------------------------
# Distributed training step
# ---------------------------------------------------------------------------

@partial(jax.pmap, axis_name="batch")
def train_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x, train=True)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # pmean across ALL devices on ALL hosts
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    acc = jax.lax.pmean(jnp.mean(jnp.argmax(logits, -1) == y), axis_name="batch")
    return new_state, {"loss": loss, "accuracy": acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator_address", default=None,
                        help="host:port of the coordinator (host 0). "
                             "Omit to run single-process (for testing).")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Initialise distributed JAX
    # -----------------------------------------------------------------------
    if args.coordinator_address:
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
        )

    local_devices  = jax.local_devices()
    global_devices = jax.devices()
    print(f"[process {jax.process_index()}]  "
          f"local={len(local_devices)} device(s)  "
          f"global={len(global_devices)} device(s)")

    # Only process 0 should print epoch-level metrics
    is_host0 = jax.process_index() == 0

    n_global = jax.device_count()
    n_local  = jax.local_device_count()
    batch_size_per_device = 32

    # -----------------------------------------------------------------------
    # Model + state
    # -----------------------------------------------------------------------
    model = CNN()
    dummy = jnp.ones((1, 14, 14, 1))  # half-size for demo
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    # Replicate across local devices; pmap+pmean will sync across hosts
    state = jax.device_put_replicated(state, local_devices)

    rng = np.random.default_rng(jax.process_index())

    print(f"Starting training on {n_global} total device(s) ...")
    for step in range(1, 201):
        # Each host generates its own shard of the global batch
        local_batch = batch_size_per_device * n_local
        x = rng.standard_normal((local_batch, 14, 14, 1), dtype=np.float32)
        y = rng.integers(0, 10, size=(local_batch,))
        x_s = x.reshape(n_local, batch_size_per_device, 14, 14, 1)
        y_s = y.reshape(n_local, batch_size_per_device)

        state, m = train_step(state, x_s, y_s)

        if is_host0 and step % 50 == 0:
            print(f"step {step:3d}  loss={float(m['loss'][0]):.4f}  "
                  f"acc={float(m['accuracy'][0])*100:.1f}%")


if __name__ == "__main__":
    main()
