# 02 – MNIST Classification

Trains a small CNN on MNIST using `pmap` data-parallelism across all TPU cores. This is the smallest end-to-end Flax training loop — the goal is to exercise the full pattern (model → state → step → pmap → pmean) on a problem that converges in seconds.

## Concepts

- `flax.linen.Module` definition (Conv, Dense, Dropout)
- `flax.training.train_state.TrainState` — the canonical (params, opt_state, step) container
- `jax.pmap` + `jax.lax.pmean` for synchronous gradient averaging
- `jax.device_put_replicated` to broadcast state across devices
- `tensorflow_datasets` as data source (TFDS download cached locally)

## Hardware context

MNIST at 28×28 with batch=256 is far below the arithmetic-intensity ridge of every accelerator we benchmark:

| Accelerator | Ridge (FLOPs/byte) | This workload | Verdict |
|-------------|-------------------|---------------|---------|
| RTX 3080 | 156 | ~5–10 | Memory-bound |
| RTX 4090 | 327 | ~5–10 | Memory-bound |
| B200 | 562 | ~5–10 | Memory-bound |
| v5e-1 | 480 | ~5–10 | Memory-bound |
| v6e-1 | 560 | ~5–10 | Memory-bound |

The model is small enough that HBM bandwidth, not FLOPs, decides throughput. Don't read percent-of-peak from this example — it will look terrible everywhere. The value is correctness and pipeline shape, not utilization.

## Run

```bash
python train.py
```

Expected wall time: 30–60 s on v5e-1 (5 epochs, including first-compile).

## Expected output

```
Epoch  1  loss=0.1423  acc=95.78%
Epoch  2  loss=0.0612  acc=98.12%
Epoch  3  loss=0.0489  acc=98.51%
Epoch  4  loss=0.0445  acc=98.74%
Epoch  5  loss=0.0421  acc=98.87%
```

Hardware-specific notes:
- All accelerators converge to ~98.8% — the model is the bottleneck, not the silicon.
- TPU v5e-1: ~6 s/epoch after first-compile.
- RTX 4090 (PyTorch path equivalent): ~4 s/epoch.
- B200: ~2 s/epoch (overkill for this task).

## What to observe

- **First-compile vs steady-state.** Epoch 1 wall time includes 12–25 s of XLA compile; epochs 2–5 are pure execution. Time them separately.
- **`pmean` synchronization.** With 1 device the all-reduce is a no-op; on a v3-8 (8 cores) you'll see identical loss across all replicas every step — that's the pmean working.
- **Memory headroom.** Model is ~1 MB; activations at batch=256 are ~50 MB. You're using <1% of HBM on every accelerator. Useful sanity check that OOMs in later examples are real, not config errors.
- **Step-time vs accuracy.** Loss drops fast in epoch 1 because the problem is easy relative to the LR. Bottleneck is TFDS data loading, not compute.

## Connection to the benchmark

This is the canonical "trainer skeleton" that examples 03–05 elaborate on. The benchmark harness runs inference (not training), but it reuses the same Flax/Orbax patterns to load checkpoints. If you can't run `02_mnist_classification` cleanly, you won't be able to load a ResNet-50 or BERT checkpoint either.

`observe/stats.py` (n=3 runs, Grubbs outlier test, CV<10%) will be applied to inference latency, but the same statistical discipline applies if you ever want to time step throughput here.
