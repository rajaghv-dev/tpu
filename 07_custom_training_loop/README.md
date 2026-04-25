# 07 – Custom Training Loop

Advanced training techniques without a high-level trainer abstraction.

## Concepts
- **Gradient accumulation** — simulate large batches by summing gradients over micro-batches
- **Mixed precision (bfloat16)** — cast activations to bf16 while keeping params in float32
- **Per-layer learning-rate scaling** — `optax.multi_transform` with a label function
- **Static loss scaling** — manual scale/unscale pattern for bf16 stability

## Run
```bash
python train.py
```
