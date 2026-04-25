"""
07 – Custom Training Loop
Demonstrates gradient accumulation, mixed-precision (bfloat16), and
per-layer learning-rate scaling — all without a high-level trainer.
Run: python train.py
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, traverse_util
from flax.training import train_state


# ---------------------------------------------------------------------------
# A simple two-tower model for demonstration
# ---------------------------------------------------------------------------

class TwoLayerMLP(nn.Module):
    hidden: int = 512
    out: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden, name="layer1")(x)
        x = nn.LayerNorm(name="norm1")(x)
        x = nn.relu(x)
        x = nn.Dense(self.out, name="layer2")(x)
        return x


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

def make_grad_accumulator(n_accum: int):
    """
    Returns (init_fn, update_fn, finalize_fn) that accumulate gradients
    over n_accum micro-batches before applying an optimizer update.
    """

    def init(params):
        return jax.tree_util.tree_map(jnp.zeros_like, params)

    def accumulate(acc_grads, new_grads):
        return jax.tree_util.tree_map(lambda a, g: a + g / n_accum, acc_grads, new_grads)

    return init, accumulate


# ---------------------------------------------------------------------------
# Mixed-precision helpers
# ---------------------------------------------------------------------------

def cast_to_bf16(batch: dict) -> dict:
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, batch
    )


def loss_scale_and_unscale(loss: jnp.ndarray, grads, scale: float = 2.0 ** 15):
    """Simple static loss scaling for bfloat16."""
    scaled_grads = jax.tree_util.tree_map(lambda g: g / scale, grads)
    return loss / scale, scaled_grads


# ---------------------------------------------------------------------------
# Per-layer LR multipliers
# ---------------------------------------------------------------------------

def create_layerwise_tx(base_lr: float) -> optax.GradientTransformation:
    """Applies 0.1× LR to layer1 params, 1.0× to everything else."""

    def label_fn(params):
        flat = traverse_util.flatten_dict(params)
        return {
            path: "slow" if path[0] == "layer1" else "normal"
            for path in flat
        }

    return optax.multi_transform(
        {
            "slow":   optax.adam(base_lr * 0.1),
            "normal": optax.adam(base_lr),
        },
        label_fn,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

@partial(jax.pmap, axis_name="batch")
def train_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
    x_bf16 = x.astype(jnp.bfloat16)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x_bf16).astype(jnp.float32)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    acc = jax.lax.pmean(jnp.mean(jnp.argmax(logits, -1) == y), axis_name="batch")
    return new_state, {"loss": loss, "accuracy": acc}


# ---------------------------------------------------------------------------
# Gradient-accumulation loop (runs on host, not pmapped)
# ---------------------------------------------------------------------------

def train_step_with_accumulation(
    state: train_state.TrainState,
    micro_batches: list[tuple],
    accum_init,
    accum_fn,
) -> tuple:
    acc_grads = accum_init(state.params)

    total_loss = 0.0
    for x, y in micro_batches:
        def loss_fn(params, _x=x, _y=y):
            logits = state.apply_fn({"params": params}, _x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, _y))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        acc_grads = accum_fn(acc_grads, grads)
        total_loss += float(loss)

    new_state = state.apply_gradients(grads=acc_grads)
    return new_state, total_loss / len(micro_batches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n = jax.device_count()
    input_dim  = 784
    batch_size = 128
    n_accum    = 4
    iters      = 200
    rng = np.random.default_rng(0)

    model = TwoLayerMLP()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, input_dim)))["params"]

    # Layerwise LR
    tx = create_layerwise_tx(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # --- pmap demo ---
    state_p = jax.device_put_replicated(state, jax.devices())
    print("=== pmap with bfloat16 ===")
    for i in range(1, 51):
        x = rng.standard_normal((n * batch_size, input_dim), dtype=np.float32)
        y = rng.integers(0, 10, size=(n * batch_size,))
        x_s = x.reshape(n, batch_size, input_dim)
        y_s = y.reshape(n, batch_size)
        state_p, m = train_step(state_p, x_s, y_s)
        if i % 10 == 0:
            print(f"  step {i:3d}  loss={float(m['loss'][0]):.4f}  acc={float(m['accuracy'][0])*100:.1f}%")

    # --- gradient accumulation demo (single device) ---
    accum_init, accum_fn = make_grad_accumulator(n_accum)
    print(f"\n=== Gradient accumulation (n_accum={n_accum}) ===")
    for i in range(1, 51):
        micro_batches = [
            (
                jnp.array(rng.standard_normal((batch_size, input_dim), dtype=np.float32)),
                jnp.array(rng.integers(0, 10, size=(batch_size,))),
            )
            for _ in range(n_accum)
        ]
        state, loss = train_step_with_accumulation(state, micro_batches, accum_init, accum_fn)
        if i % 10 == 0:
            print(f"  step {i:3d}  loss={loss:.4f}")


if __name__ == "__main__":
    main()
