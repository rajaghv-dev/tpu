# 03 – ResNet-50 on ImageNet

Full ResNet-50 training on ImageNet-1k with cosine LR schedule and Orbax checkpointing.

## Concepts
- Bottleneck residual blocks with BatchNorm
- `pmap` + `donate_argnums` for zero-copy gradient updates
- Linear scaling rule for learning rate
- Warmup + cosine decay schedule via `optax.join_schedules`
- `orbax-checkpoint` for saving/restoring state

## Run
```bash
# With TFDS auto-download (slow first run)
python train.py

# With pre-downloaded ImageNet
python train.py --data_dir=/path/to/imagenet
```

## Files
- `model.py` — ResNet-50 Flax module
- `train.py` — training loop
