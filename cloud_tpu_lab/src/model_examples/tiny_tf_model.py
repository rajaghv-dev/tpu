"""
TensorFlow-ready tiny MLP.

CPU-safe: only builds the model if tensorflow is installed. See notebook
`06_tensorflow_cpu_to_tpu_ready.ipynb` for TPUStrategy instructions.
"""
from __future__ import annotations

from typing import Tuple


def build_tiny_tf_mlp(
    input_dim: int = 64, hidden: int = 128, output_dim: int = 10,
) -> Tuple["tf.keras.Model", str]:  # noqa: F821
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "tensorflow not installed — install tensorflow for the TPU-ready "
            "path."
        ) from exc

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(output_dim),
    ])
    note = (
        "Wrap your training step under `with strategy.scope():` where "
        "strategy = tf.distribute.TPUStrategy(tf.distribute.cluster_resolver."
        "TPUClusterResolver())."
    )
    return model, note
