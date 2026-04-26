# 07 – Custom Training Loop

Advanced training techniques without a high-level trainer abstraction. Each technique here surfaces a hardware-level concern that the benchmark harness will care about.

## Concepts

- **Gradient accumulation** — simulate a large effective batch by summing gradients over `n_accum=4` micro-batches. Essential when peak HBM forces a small per-step batch.
- **Mixed precision (bfloat16)** — cast activations to BF16, keep params and optimizer state in FP32.
- **Per-layer learning-rate scaling** — `optax.multi_transform` with a label function (`layer1` → 0.1×, `layer2` → 1.0×). Mirrors the discriminative fine-tuning pattern.
- **Static loss scaling** — manual scale/unscale to preserve gradient magnitude in low-precision regimes.

## Hardware context

### Gradient accumulation — about HBM pressure

When model + activations + optimizer state don't fit, you reduce per-step batch and accumulate. Concrete HBM envelopes:

| Accelerator | HBM | What forces accumulation |
|-------------|-----|--------------------------|
| RTX 3080 | 16 GB | GPT-2-small at seq=1024, batch>8 |
| v5e-1 | 16 GB | Same envelope as 3080 |
| RTX 4090 | 24 GB | GPT-2-XL training still tight |
| v6e-1 | 32 GB | More headroom, but large LLMs still need it |
| B200 | 192 GB | Almost never needed — this is why B200 is uniquely good for large batches |

### BF16 — the precision-decision asymmetry

This is the most important hardware-aware insight in the whole benchmark:

| Backend | FP32 vs BF16 speed | BF16 verdict |
|---------|-------------------|--------------|
| TPU MXU (v5e, v6e) | **Identical** — MXU runs both at the same TFLOPs | **Free upgrade — always on** |
| GPU Tensor Cores (3080, 4090, B200) | BF16 is **~2× faster** | **Active optimization decision — opt in** |

In this example, the pmap step casts inputs to BF16 and outputs back to FP32 before loss. On TPU you'll see no wall-clock change. On GPU you should see roughly a 2× speedup. Run on both to feel the asymmetry directly.

### Per-layer LR — optimizer state cost is unchanged

`optax.multi_transform` partitions params across two Adam chains. Each param lives in exactly one chain, so total optimizer state size is the same as a single Adam — it does not double.

### Static loss scaling — BF16 vs FP16

BF16 has the **same exponent range as FP32** (8-bit exponent), so loss scaling is usually unnecessary. The pattern is included for completeness because FP16 (5-bit exponent) does need it. The benchmark uses BF16 throughout for exactly this reason — one less footgun.

The related story in the benchmark: **2:4 structured sparsity** gives ~2× speedup on Ampere/Ada/Blackwell GPUs via Sparse Tensor Cores. TPU has no equivalent hardware primitive. Same shape as the BF16 decision (one-sided optimization) but in the opposite direction.

## Run

```bash
python train.py
```

Expected wall time: ~30 s on v5e-1 (two 50-step demo loops on a small MLP).

## Expected output

```
=== pmap with bfloat16 ===
  step  10  loss=2.1834  acc=12.5%
  step  20  loss=2.0891  acc=15.2%
  step  30  loss=1.9643  acc=20.8%
  step  40  loss=1.8211  acc=27.3%
  step  50  loss=1.7102  acc=32.1%

=== Gradient accumulation (n_accum=4) ===
  step  10  loss=2.1701
  step  20  loss=2.0344
  step  30  loss=1.9012
  step  40  loss=1.7843
  step  50  loss=1.6892
```

## What to observe

- **BF16 cast is free on TPU.** Time the BF16 pmap loop and compare to FP32 — they should match within noise (CV<10%). On a 3080/4090, the BF16 loop should be roughly half the FP32 loop wall time.
- **Gradient accumulation correctness.** With `n_accum=4`, the optimizer is updated every 4 micro-batches; the effective learning rate is equivalent to running batch × 4. Loss curve should match a true batch=4× run.
- **Per-layer LR.** After a few steps, print the LR scale for each label: `layer1` should be 1/10th of `layer2`. If they're equal, the `label_fn` isn't being applied.
- **No NaNs from BF16.** If NaN appears, the issue is almost always loss being computed in BF16. The cast-back-to-FP32-before-loss pattern is the fix.

## Connection to the benchmark

These four techniques are exactly the knobs the benchmark harness toggles:

- **BF16-by-default on TPU paths** (Paths 1, 4): codified because it's free
- **BF16 explicit on GPU paths** (Paths 2, 3): measured against FP32 baseline to quantify the ~2×
- **Gradient accumulation** maps to inference batch sweeping — the same HBM pressure logic determines `max_batch_before_oom` in `observe/memory_profiler.py`
- **Mixed precision in inference** is the same casting pattern but simpler (no optimizer state), and gets extended to INT8/FP8 in Stage 6
