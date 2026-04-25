"""
02 – MNIST Classification
Simple CNN trained on MNIST using JAX + Flax with pmap data-parallelism.
Run: python train.py
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import train_state


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.25, deterministic=not train)(x)
        x = nn.Dense(10)(x)
        return x


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, 10)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> dict[str, Any]:
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Training step (pmap-friendly)
# ---------------------------------------------------------------------------

@partial(jax.pmap, axis_name="batch")
def train_step(state: train_state.TrainState, batch: dict) -> tuple:
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"], train=True,
                                rngs={"dropout": jax.random.PRNGKey(0)})
        return cross_entropy_loss(logits, batch["label"]), logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch["label"])
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return new_state, metrics


@partial(jax.pmap, axis_name="batch")
def eval_step(state: train_state.TrainState, batch: dict) -> dict:
    logits = state.apply_fn({"params": state.params}, batch["image"], train=False)
    metrics = compute_metrics(logits, batch["label"])
    return jax.lax.pmean(metrics, axis_name="batch")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(split: str, batch_size: int):
    ds = tfds.load("mnist", split=split, as_supervised=True)
    ds = ds.map(lambda img, lbl: (tf_preprocess(img), lbl))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(4)
    return tfds.as_numpy(ds)


def tf_preprocess(img):
    import tensorflow as tf
    img = tf.cast(img, tf.float32) / 255.0
    return img  # (28, 28, 1)


def shard(batch: dict, n_devices: int) -> dict:
    return jax.tree_util.tree_map(
        lambda x: x.reshape((n_devices, -1) + x.shape[1:]), batch
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_devices = jax.device_count()
    batch_size = 256
    epochs = 5
    lr = 1e-3

    # Init model
    model = CNN()
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, jnp.ones((1, 28, 28, 1)))["params"]
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    # Replicate across devices
    state = jax.device_put_replicated(state, jax.devices())

    train_ds = load_dataset("train", batch_size)
    test_ds = load_dataset("test", batch_size)

    for epoch in range(1, epochs + 1):
        # Train
        for batch in train_ds:
            images, labels = batch
            b = shard({"image": images, "label": labels}, n_devices)
            state, metrics = train_step(state, b)

        # Eval
        eval_metrics = []
        for batch in test_ds:
            images, labels = batch
            b = shard({"image": images, "label": labels}, n_devices)
            m = eval_step(state, b)
            eval_metrics.append(jax.tree_util.tree_map(lambda x: float(x[0]), m))

        avg = {k: np.mean([m[k] for m in eval_metrics]) for k in eval_metrics[0]}
        print(f"Epoch {epoch:2d}  loss={avg['loss']:.4f}  acc={avg['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
