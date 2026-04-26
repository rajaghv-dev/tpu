# 01 – Hello TPU

Verifies that JAX can see your TPU/GPU and runs a few sanity-check operations. This is the smallest possible test that the entire JAX → XLA → accelerator stack is wired up correctly.

## What it covers

- `jax.devices()` — enumerate accelerators (TPU cores, GPUs)
- `jnp` ops compiled to XLA HLO → device code
- `jax.jit` tracing and the first-compile cost
- `jax.pmap` across all cores (data-parallel SPMD)
- Identity check: `sin²(x) + cos²(x) == 1` for every device

## Hardware context

What this example reveals about the stack you're sitting on:

| Backend | What `jax.devices()` reports | First-compile cost | Cache behavior |
|---------|------------------------------|-------------------|----------------|
| TPU v5e-1 | 1 chip, 16 GB HBM, 820 GB/s, 394 BF16 TFLOPs | ~12–20 s for trivial kernels | Persistent across same process |
| TPU v6e-1 | 1 chip, 32 GB HBM, 1640 GB/s, 918 BF16 TFLOPs | ~12–20 s | Persistent |
| RTX 3080 (XLA-CUDA) | 1 GPU, 16 GB GDDR6X, 760 GB/s, 119 BF16 TFLOPs | ~5–15 s | Cached on disk |
| RTX 4090 (XLA-CUDA) | 1 GPU, 24 GB GDDR6X, 1008 GB/s, 330 BF16 TFLOPs | ~5–15 s | Cached on disk |
| B200 SXM (XLA-CUDA) | 1 GPU, 192 GB HBM3e, 4000 GB/s, 2250 BF16 TFLOPs | ~5–15 s | Cached on disk |

The `sin²+cos²` kernel is too small to expose any compute difference between these — that's the point. It's a wiring test, not a benchmark.

## Run

```bash
python hello_tpu.py
```

Expected wall time: 15–30 s (almost all of which is the first XLA compile).

## Expected output

### v3-8 (multi-chip)
```
Backend  : tpu
Devices  : 8
  TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)
  ...
pmap across 8 device(s): OK (all-ones confirmed)
All checks passed!
```

### v5e-1 or v6e-1 (single-chip VM)
```
Backend  : tpu
Devices  : 1
  TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)
pmap across 1 device(s): OK (all-ones confirmed)
All checks passed!
```

### GPU (RTX 3080 / 4090 / B200, XLA backend)
```
Backend  : gpu
Devices  : 1
  CudaDevice(id=0)
pmap across 1 device(s): OK (all-ones confirmed)
All checks passed!
```

## What to observe

- **First-compile latency.** The very first `jit` call takes 12–20 s on TPU because XLA has to lower HLO → LLO → TPU machine code. Subsequent identical calls are near-zero (cache hit). This is the same behavior you'll see throughout the benchmark harness.
- **`pmap` replicates compilation.** The compiled program is sent to every device; the all-ones result confirms SPMD execution worked end-to-end.
- **No precision choice yet.** This script uses default FP32. On TPU, switching to BF16 would be free — MXU runs both at the same speed. On GPU, BF16 would be ~2× faster via Tensor Cores — that's an active optimization decision, not a free upgrade. Examples 04, 05, 07 will exercise this.

## Connection to the benchmark

This is the precondition for everything else. Before `harness.py` runs any model, it confirms:
1. `jax.devices()` returns the expected accelerator type
2. A trivial `jit`+`pmap` round-trip works
3. `observe/lineage.py` can stamp a git SHA and seed onto the run

If `01_hello_tpu` fails on a given path (Path 1 JAX+TPU, Path 2 JAX+GPU), nothing downstream will work. Treat it as the smoke test for the environment itself.
