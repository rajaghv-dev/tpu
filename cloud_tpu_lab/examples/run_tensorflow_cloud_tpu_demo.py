#!/usr/bin/env python3
"""
TensorFlow Cloud TPU demo — CPU-safe.

If tensorflow is installed: tries TPUClusterResolver, falls back to CPU.
Else: prints install instructions and exits 0.

Referenced by docs/06_tensorflow_on_tpu.md.
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        print(f"[tf-demo] tensorflow not installed ({type(exc).__name__}).")
        print("  Install on a Cloud TPU VM per https://cloud.google.com/tpu/docs/run-calculation-tensorflow")
        return 0

    print(f"[tf-demo] tf version: {tf.__version__}")
    strategy = None
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"[tf-demo] TPUStrategy with {strategy.num_replicas_in_sync} replicas")
    except Exception as exc:  # noqa: BLE001
        print(f"[tf-demo] No TPU detected ({type(exc).__name__}). Using CPU.")
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        a = tf.random.normal((1024, 1024))
        b = tf.random.normal((1024, 1024))

        @tf.function
        def mm(x, y):
            return x @ y

        # Warmup compile.
        _ = mm(a, b).numpy()
        t0 = time.perf_counter()
        for _ in range(10):
            c = mm(a, b)
        _ = c.numpy()
        dt = time.perf_counter() - t0
        print(f"[tf-demo] 10× 1024×1024 mm: {dt*1000:.2f} ms total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
