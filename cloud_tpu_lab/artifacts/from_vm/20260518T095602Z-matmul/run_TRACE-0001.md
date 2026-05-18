# Cloud TPU Lab — Run Report

**trace_id**: `TRACE-0001`

## Workload
- `name`: real_matmul
- `framework`: jax
- `model_kind`: matmul
- `batch_size`: 32
- `seq_len`: 32
- `hidden_size`: 512
- `num_layers`: 2
- `vocab_size`: 1024
- `input_dim`: 64
- `output_dim`: 10
- `n_steps`: 10
- `precision`: bf16
- `tpu_version`: v5e
- `chip_count`: 1
- `mesh_shape`: [1]
- `hourly_usd_per_chip`: 1.2

## XLA compile
- compile_time_s: 0.1730
- recompile_count: 0

## Time breakdown
- compile              0.1730s ( 49.8%)
- device               0.1746s ( 50.2%)
- collective           0.0000s (  0.0%)
- input_pipeline       0.0000s (  0.0%)
- host                 0.0000s (  0.0%)
- checkpoint           0.0000s (  0.0%)

## Step time
- n_steps: 10
- first step: 173.03 ms
- median:     0.17 ms
- p95:        0.20 ms

## HBM
- capacity: 16.909 GB
- used:     0.002 GB (0.0%)
- peak:     0.002 GB
- OOM events: 0

## Cost (placeholder rate; update from cloud.google.com/tpu/pricing)
- total wall time: 0.002 s
- total USD:       $0.000001
- per step:        $0.000000
- per sample:      $0.000000

# Bottleneck Report

## WARN

- **[xla]** Compile = 49.8% — likely recompiles
  - Fix: Stabilise shapes (avoid dynamic batch / seq); set JAX_COMPILATION_CACHE_DIR to persist compiled executables across runs; check `jax.config.jax_log_compiles`.
