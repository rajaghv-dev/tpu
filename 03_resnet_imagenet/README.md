# 03 – ResNet-50 on ImageNet

Full ResNet-50 training on ImageNet-1k with linear-scaling LR, warmup + cosine decay, and Orbax checkpointing. ResNet-50 is the canonical conv-net baseline and a primary entry in the benchmark registry.

## Model in the benchmark registry

ResNet-50 is one of the 75 models benchmarked across all 5 execution paths.

| Property | Value |
|----------|-------|
| Parameters | 86 M |
| FLOPs/image (224×224) | ~4 G |
| Architecture | Bottleneck residual blocks (1×1 → BN → 3×3 → BN → 1×1 → BN) |
| Why it's in the registry | Canonical CNN baseline; cuDNN heuristics (Winograd / FFT / direct) vs XLA's static padding to 128-multiples produce sharply different per-layer costs |

## Concepts

- Bottleneck residual blocks with BatchNorm and identity / projection shortcuts
- `pmap` + `donate_argnums` for zero-copy gradient updates (critical at batch=1024)
- Linear scaling rule: `base_lr = 0.1 × (batch_size / 256)`
- Warmup (5 epochs) + cosine decay via `optax.join_schedules`
- `orbax-checkpoint` for atomic, async-safe saves per epoch

## Hardware context

ResNet-50 forward at batch=1024 has ~4 TFLOPs of compute and ~12 GB of activation traffic — well past the ridge on TPU and roughly at the ridge on RTX 4090.

| Accelerator | BF16 TFLOPs | Ridge (FLOPs/byte) | Expected step shape |
|-------------|-------------|---------------------|---------------------|
| RTX 3080 | 119 | 156 | Compute-bound; cuDNN Winograd makes 3×3 conv much faster than naive |
| RTX 4090 | 330 | 327 | Right at ridge — both compute and bandwidth matter |
| B200 | 2250 | 562 | Compute-bound; Tensor Core utilization should be high |
| v5e-1 | 394 | 480 | MXU well-fed by 1×1 and 3×3 conv (no depthwise here) |
| v6e-1 | 918 | 560 | Same, ~2× faster than v5e-1 |

**Key contrast — ResNet-50 vs EfficientNet on TPU:** EfficientNet uses depthwise-separable conv whose arithmetic intensity is far below the ridge. On TPU, EfficientNet throughput < ResNet-50 throughput despite EfficientNet having *fewer* FLOPs — depthwise conv starves the MXU (<20% utilization). ResNet's plain 3×3 + 1×1 keeps MXU utilization >70%. This example is your reference for "what good MXU utilization looks like."

## Run

```bash
# With TFDS auto-download (slow first run; ~150 GB ImageNet download)
python train.py

# With pre-downloaded ImageNet
python train.py --data_dir=/path/to/imagenet
```

Expected wall time:
- First-compile: 25–45 s (largest of any example so far)
- Per-epoch on TPU v3-8: ~30 min
- Full 90-epoch training: ~45 h on v3-8

## Files

- `model.py` — ResNet-50 Flax module (`BottleneckBlock` × {3, 4, 6, 3})
- `train.py` — training loop with Orbax checkpoint per epoch

## Expected output

After 90 epochs: top-1 ~76.0% (standard ResNet-50 / ImageNet result, matches He et al.).

Per-epoch printout:
```
Epoch   1/90  val_loss=4.81  val_acc=12.3%
Epoch  10/90  val_loss=2.65  val_acc=51.2%
Epoch  90/90  val_loss=1.32  val_acc=76.1%
```

## What to observe

- **First-compile is real (25–45 s).** XLA pads all conv shapes to 128-multiples and selects fusion patterns. cuDNN's equivalent is per-layer algorithm selection (Winograd vs FFT vs direct), which on `torch.compile(mode="max-autotune")` takes up to 60 s. The compile cost is one-time per unique shape.
- **MXU utilization.** With pure 1×1 and 3×3 conv and batch=1024, expect >70% MXU on TPU. This is the upper-bound reference for conv models.
- **`donate_argnums` matters here.** At batch=1024 the optimizer state is several GB; without buffer donation you risk OOM on v5e-1 (16 GB HBM).
- **BatchNorm sync.** `flax.linen.BatchNorm` with `use_running_average=False` cross-replica syncs through `pmean` — confirm this in the trace, otherwise per-replica statistics drift.
- **Linear scaling rule.** At batch=1024 → LR=0.4. Without 5-epoch warmup, the model diverges in the first ~500 steps.

## Connection to the benchmark

ResNet-50 is exercised on **all 5 paths** in Stage 2+:

| Path | Framework | What it measures |
|------|-----------|-----------------|
| 1 | JAX+TPU (XLA) | MXU utilization, XLA conv fusion |
| 2 | JAX+GPU (XLA-CUDA) | XLA vs cuDNN conv algorithm selection |
| 3 | PyTorch+GPU | cuDNN Winograd + `torch.compile` |
| 4 | PyTorch+torch_xla+TPU | PyTorch API vs JAX API on same XLA backend |
| 5 | HF Inference API | End-to-end managed serving overhead |

The cross-path comparison reveals the Winograd vs XLA-padding tradeoff. This example provides the trained checkpoint all 5 paths consume.

`observe/lineage.py` stamps the git SHA + Orbax checkpoint hash onto every inference run. `observe/flops_counter.py` will confirm the 4 GFLOPs/image independently from the architectural spec.
