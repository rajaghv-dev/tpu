# 04 – BERT Fine-tuning (GLUE / SST-2)

Loads `bert-base-uncased` from HuggingFace and fine-tunes for binary sentiment classification on SST-2. BERT-base is the **primary benchmark reference model** in the registry.

## Model in the benchmark registry

| Property | Value |
|----------|-------|
| Parameters | 110 M |
| Architecture | 12-layer bidirectional encoder, MHA (no GQA), absolute position embeddings |
| Sequence length | 128 in this example; benchmark sweeps 128 → 512 to expose O(n²) attention scaling |
| Why it's the reference | Mature, well-understood, fits all 5 paths cleanly; attention cost scales as `seq_len²`, making it the cleanest probe for attention-kernel quality |

## Concepts

- `FlaxAutoModelForSequenceClassification` from `transformers` — pulls weights from HF Hub
- Linear LR warmup + linear decay to 0 via `optax.linear_schedule`
- `optax.adamw` with weight_decay=0.01 (decoupled weight decay)
- `pmap` across all TPU cores with synchronous `pmean` gradient averaging

## Hardware context

BERT-base at seq_len=128, batch=32 per device:

| Accelerator | BF16 TFLOPs | Notes |
|-------------|-------------|-------|
| RTX 3080 | 119 | Tight on 16 GB; activation memory at seq=128 is ~3 GB |
| RTX 4090 | 330 | Comfortable; BF16 gives ~2× over FP32 via Tensor Cores |
| B200 | 2250 | Overkill for seq=128; interesting comparison is at seq=512 |
| v5e-1 | 394 | BF16 is **free** — same speed as FP32 |
| v6e-1 | 918 | ~2× faster than v5e-1 on this workload |

**The seq_len knob is the headline finding.** At seq_len=128, BERT is roughly compute-bound (FFN matmuls dominate). At seq_len=512, the O(n²) attention cost grows 16× while parameter FLOPs only grow 4×, so attention dominates and the comparison shifts toward whichever stack has the better fused-attention kernel. This example uses seq_len=128 (training only); the benchmark harness sweeps 128 → 256 → 512.

## Run

```bash
python train.py
```

Expected wall time:
- First-compile: 20–35 s
- Per-epoch on v5e-1: ~3 min (67k samples, batch=32)
- Total (3 epochs): ~10 min

## Expected output

```
Epoch 1/3  val_loss=0.2841  val_acc=91.74%
Epoch 2/3  val_loss=0.2103  val_acc=93.12%
Epoch 3/3  val_loss=0.2218  val_acc=93.46%
```

93.46% matches the BERT-base SST-2 paper number (Devlin et al. report ~93%). The slight epoch-3 val_loss bump is expected with LR=2e-5 and no early stopping.

## What to observe

- **First-compile (20–35 s).** Includes the full `FlaxBertForSequenceClassification` graph. Subsequent steps are near-zero.
- **BF16 on TPU is free.** Cast the inputs to `bfloat16` and the wall time will not change — MXU runs both at identical speed. Cast on a 3080/4090 and you should see ~2× speedup. This is the single clearest illustration of why "BF16 by default" is the right TPU policy and an opt-in decision on GPU.
- **AdamW state size.** Adam's `m` and `v` tensors double the param memory. 110M × 4 bytes × 3 (params + m + v) ≈ 1.3 GB just for optimizer state — comfortable on all hardware.
- **Attention kernel utilization.** BERT's MHA at seq=128 with large batch is roughly compute-bound; MXU should be >60%. At seq=512 in the benchmark sweep, watch for the MXU curve to stay flat (good fused attention) or droop (inefficient attention path).

## Connection to the benchmark

BERT-base is the **primary reference model** for Stage 1. Every path runs inference on a fine-tuned checkpoint, and the seq_len 128/256/512 sweep is a headline result.

| Path | Framework | What it measures |
|------|-----------|-----------------|
| 1 | JAX+TPU (XLA) | XLA attention fusion, MXU at seq=128→512 |
| 2 | JAX+GPU (XLA-CUDA) | Same fusion on CUDA; XLA-CUDA vs cuDNN |
| 3 | PyTorch+GPU | PyTorch `sdpa` / FlashAttention 2 vs XLA |
| 4 | PyTorch+torch_xla+TPU | XLA backend, PyTorch API |
| 5 | HF Inference API | Network overhead India → us-central1 |

`observe/compile_controller.py` explicitly clears the XLA cache before the first run, then re-runs with cache hot — so both the 20–35 s first-compile and the near-zero subsequent compile appear as separate measurements. CV<10% across n=3 (Grubbs outlier test) is the acceptance bar for every latency claim.
