"""
03 – ResNet-50 on ImageNet
Data-parallel training with pmap, cosine LR schedule, and checkpoint saving.
Run: python train.py --data_dir=/path/to/imagenet
"""

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import train_state
from flax import struct

from model import ResNet50


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@struct.dataclass
class Config:
    batch_size: int = 1024
    epochs: int = 90
    base_lr: float = 0.1        # scaled linearly with batch size
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    num_classes: int = 1000
    image_size: int = 224
    checkpoint_dir: str = "checkpoints/resnet50"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def preprocess_train(image, label, image_size: int):
    image = tf.image.random_crop(image, (image_size, image_size, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std  = tf.constant(IMAGENET_STD,  dtype=tf.float32)
    image = (image - mean) / std
    return image, label


def preprocess_eval(image, label, image_size: int):
    image = tf.image.resize(image, (image_size + 32, image_size + 32))
    image = tf.image.central_crop(image, image_size / (image_size + 32))
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std  = tf.constant(IMAGENET_STD,  dtype=tf.float32)
    image = (image - mean) / std
    return image, label


def build_dataset(split: str, config: Config, data_dir: str | None = None):
    ds_kwargs = {"data_dir": data_dir} if data_dir else {}
    ds = tfds.load("imagenet2012", split=split, as_supervised=True, **ds_kwargs)
    if split == "train":
        ds = ds.shuffle(10_000).map(
            lambda img, lbl: preprocess_train(img, lbl, config.image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda img, lbl: preprocess_eval(img, lbl, config.image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(config.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def create_train_state(config: Config, rng):
    model = ResNet50(num_classes=config.num_classes)
    dummy = jnp.ones((1, config.image_size, config.image_size, 3))
    variables = model.init(rng, dummy, train=False)

    steps_per_epoch = 1_281_167 // config.batch_size
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    lr = config.base_lr * config.batch_size / 256  # linear scaling rule
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    tx = optax.chain(
        optax.add_decayed_weights(config.weight_decay),
        optax.sgd(schedule, momentum=0.9, nesterov=True),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    ), variables["batch_stats"]


@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def train_step(state, batch_stats, images, labels):
    def loss_fn(params):
        logits, updates = ResNet50().apply(
            {"params": params, "batch_stats": batch_stats},
            images,
            train=True,
            mutable=["batch_stats"],
        )
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        return loss, (logits, updates["batch_stats"])

    (loss, (logits, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    acc = jax.lax.pmean(acc, axis_name="batch")
    return new_state, new_batch_stats, {"loss": loss, "accuracy": acc}


@partial(jax.pmap, axis_name="batch")
def eval_step(state, batch_stats, images, labels):
    logits = ResNet50().apply(
        {"params": state.params, "batch_stats": batch_stats},
        images,
        train=False,
    )
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    acc  = jnp.mean(jnp.argmax(logits, -1) == labels)
    return jax.lax.pmean({"loss": loss, "accuracy": acc}, axis_name="batch")


def shard(x, n):
    return x.reshape((n, -1) + x.shape[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=90)
    args = parser.parse_args()

    config = Config(epochs=args.epochs)
    n = jax.device_count()
    rng = jax.random.PRNGKey(0)

    state, batch_stats = create_train_state(config, rng)
    state       = jax.device_put_replicated(state, jax.devices())
    batch_stats = jax.device_put_replicated(batch_stats, jax.devices())

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    train_ds = build_dataset("train", config, args.data_dir)
    val_ds   = build_dataset("validation", config, args.data_dir)

    for epoch in range(1, config.epochs + 1):
        # Train
        for images, labels in train_ds:
            images = shard(images, n)
            labels = shard(labels, n)
            state, batch_stats, _ = train_step(state, batch_stats, images, labels)

        # Eval
        val_metrics = []
        for images, labels in val_ds:
            images = shard(images, n)
            labels = shard(labels, n)
            m = eval_step(state, batch_stats, images, labels)
            val_metrics.append({k: float(v[0]) for k, v in m.items()})

        avg = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
        print(f"Epoch {epoch:3d}/{config.epochs}  "
              f"val_loss={avg['loss']:.4f}  val_acc={avg['accuracy']*100:.2f}%")

        # Save checkpoint (unreplicate first)
        ckpt_state = jax.tree_util.tree_map(lambda x: x[0], state)
        checkpointer.save(ckpt_dir / f"epoch_{epoch:03d}", ckpt_state)


if __name__ == "__main__":
    main()
