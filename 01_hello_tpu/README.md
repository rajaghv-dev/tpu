# 01 – Hello TPU

Verifies that JAX can see your TPU and runs a few sanity-check operations.

## What it covers
- `jax.devices()` — enumerate TPU cores
- `jnp` ops compiled to XLA
- `jax.jit` tracing
- `jax.pmap` across all cores

## Run
```bash
python hello_tpu.py
```

## Expected output (v3-8)
```
Backend  : tpu
Devices  : 8
  TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)
  ...
pmap across 8 device(s): OK (all-ones confirmed)
All checks passed!
```
