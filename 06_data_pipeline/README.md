# 06 – Efficient Data Pipelines for TPUs

`tf.data` best-practices that prevent the accelerator from starving for data. With a 394–2250 TFLOP device, you can easily spend most of your wall time waiting on the host pipeline if it's wrong.

## Key principles

1. **`drop_remainder=True`** — XLA requires **static shapes**. A variable-sized last batch forces XLA to recompile (or refuse entirely). This is non-negotiable on TPU and strongly recommended for `torch.compile` paths too.
2. **`num_parallel_calls=tf.data.AUTOTUNE`** — let `tf.data` saturate host CPU cores for decode and augment.
3. **`prefetch(AUTOTUNE)`** — overlap host preprocessing with device step. Without this, the device idles every step waiting for the next batch.
4. **`interleave`** for GCS TFRecords — concurrent reads mask single-stream I/O latency. **Critical for India → us-central1 GCS reads** where single-stream RTT is 200–300 ms; 32-way interleave hides it entirely.
5. **Shuffle before batching, buffer ≥ 10× batch size** — otherwise you're shuffling within batches only.

## Why static shapes matter (XLA-specific)

Every `jit`-compiled function is keyed on input shape. If your pipeline emits `[256, 224, 224, 3]` for 9999 steps and `[123, 224, 224, 3]` on step 10000, XLA triggers a **second 12–45 s compile** for the new shape — and again at the next epoch start.

`drop_remainder=True` makes the cost visible (≤1 batch lost per epoch) instead of hidden (a full recompile injected mid-epoch). Always pay the visible cost.

## Hardware context — pipeline throughput needed

To keep these accelerators fed at ImageNet 224×224, batch=1024:

| Accelerator | Approx ResNet-50 step rate | Required pipeline images/sec |
|-------------|---------------------------|------------------------------|
| RTX 3080 | ~3 steps/s | ~3,000 |
| RTX 4090 | ~6 steps/s | ~6,000 |
| B200 | ~20+ steps/s | ~20,000+ |
| v5e-1 | ~4 steps/s | ~4,000 |
| v6e-1 | ~8 steps/s | ~8,000 |

This example's `make_fake_imagenet` benchmark achieves **~2,000–5,000 images/sec** on local CPU (fake, in-memory data). That's sufficient for a 3080 or v5e-1 but already a bottleneck for B200 — which is why production pipelines need GCS + interleave + parallel decode, not local fake data.

## Run

```bash
python pipeline.py
```

Expected output (representative):
```
Devices : 1
Total batch size : 64

Benchmarking pipeline throughput ...
Throughput: 3,847 images/sec (fake data; real GCS will vary)

First batch shapes:
  images : (64, 224, 224, 3)  dtype=float32
  labels : (64,)  dtype=int32

Running one pmap step ...
  pmap output shape: (1,)  (one scalar per device)
```

## What to observe

- **Throughput plateaus at host-CPU saturation.** On a 4-core VM, expect ~2,000 images/sec; on a 16-core, ~5,000+. The pipeline is CPU-bound, not device-bound.
- **`drop_remainder` toggle.** Flip it to `False` and watch a second compile fire on the last batch of an epoch. This is the most common cause of "my training got 10× slower at epoch boundaries."
- **`prefetch(AUTOTUNE)` removed.** Step time becomes additive (`host_time + device_time`) instead of `max(host_time, device_time)`. On v6e-1 (fast device, slow data) this can halve throughput.
- **GCS interleave from India.** With a real GCS bucket in us-central1, single-stream read suffers 200–300 ms RTT. `interleave(cycle_length=32)` opens 32 concurrent reads → throughput improves 20–30×. Test with `cycle_length=1` first to observe the pathology, then `cycle_length=32` to see the fix.

## Connection to the benchmark

The benchmark harness is inference-focused but still needs reproducible static-shape inputs:

- `harness.py` generates fixed-shape synthetic batches (NumPy seed=42) to avoid the recompile trap
- `runner.py` warm-starts each model with a fixed-shape dummy batch before timing
- `observe/compile_controller.py` measures `first_compile_ms` vs `subsequent_compile_ms` — a misconfigured pipeline that emits variable shapes would corrupt the "subsequent compile" timing with silent recompiles

For Path 5 (HF Inference API) the equivalent issue is **request batching** and India → us-central1 latency — the HTTP-layer analog of GCS single-stream reads. The `warm/cold split` in the HF API path (Stage 7) mirrors the interleave story: you need to account for cold-start separately.
