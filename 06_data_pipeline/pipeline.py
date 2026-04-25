"""
06 – Efficient Data Pipeline for TPUs
Demonstrates tf.data best-practices: prefetching, interleaving, sharding,
and converting to numpy for JAX consumption without GPU/TPU data starvation.
Run: python pipeline.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Synthetic dataset (replace with real data source in production)
# ---------------------------------------------------------------------------

def make_fake_imagenet(n_samples: int = 10_000, image_size: int = 224) -> tf.data.Dataset:
    """Creates a synthetic dataset that mimics ImageNet batch shapes."""
    images = tf.data.Dataset.from_tensor_slices(
        tf.random.uniform((n_samples, image_size, image_size, 3), dtype=tf.float32)
    )
    labels = tf.data.Dataset.from_tensor_slices(
        tf.random.uniform((n_samples,), maxval=1000, dtype=tf.int32)
    )
    return tf.data.Dataset.zip((images, labels))


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_train(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def normalize(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.constant([0.485, 0.456, 0.406])
    std  = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image, label


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_train_pipeline(
    ds: tf.data.Dataset,
    batch_size: int,
    n_devices: int,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """
    Key TPU pipeline principles applied here:
    1. shuffle before batching
    2. map with AUTOTUNE parallelism
    3. drop_remainder=True so XLA sees static shapes
    4. prefetch to overlap GPU/host compute with device training
    """
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size * n_devices, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_eval_pipeline(
    ds: tf.data.Dataset,
    batch_size: int,
    n_devices: int,
) -> tf.data.Dataset:
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size * n_devices, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# GCS-backed pipeline example
# ---------------------------------------------------------------------------

def build_gcs_pipeline(
    gcs_pattern: str,
    batch_size: int,
    n_devices: int,
    image_size: int = 224,
) -> tf.data.Dataset:
    """
    Example of reading TFRecords from Google Cloud Storage.
    Replace gcs_pattern with e.g. 'gs://my-bucket/train-*.tfrecord'.
    """
    feature_desc = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_desc)
        image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(features["image/class/label"], tf.int32)
        return image, label

    files = tf.data.Dataset.list_files(gcs_pattern, shuffle=True)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=32,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10_000)
    ds = ds.batch(batch_size * n_devices, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_throughput(ds: tf.data.Dataset, n_steps: int = 50) -> float:
    """Returns images/sec throughput."""
    total_images = 0
    t0 = time.perf_counter()
    for images, _ in ds.take(n_steps):
        total_images += images.shape[0]
    elapsed = time.perf_counter() - t0
    return total_images / elapsed


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    n_devices = jax.device_count()
    batch_size = 64  # per device

    print(f"Devices : {n_devices}")
    print(f"Total batch size : {batch_size * n_devices}")

    raw_ds = make_fake_imagenet()
    train_ds = build_train_pipeline(raw_ds, batch_size, n_devices)

    print("\nBenchmarking pipeline throughput ...")
    imgs_per_sec = benchmark_throughput(train_ds)
    print(f"Throughput: {imgs_per_sec:,.0f} images/sec (fake data; real GCS will vary)")

    print("\nFirst batch shapes:")
    for images, labels in train_ds.take(1):
        print(f"  images : {tuple(images.shape)}  dtype={images.dtype}")
        print(f"  labels : {tuple(labels.shape)}  dtype={labels.dtype}")

    # Show how to consume in a JAX training loop
    @jax.pmap
    def dummy_forward(x):
        return jnp.mean(x)

    print("\nRunning one pmap step ...")
    for images, labels in train_ds.take(1):
        images_np = images.numpy().reshape(n_devices, batch_size, *images.shape[1:])
        out = dummy_forward(images_np)
        print(f"  pmap output shape: {out.shape}  (one scalar per device)")


if __name__ == "__main__":
    main()
