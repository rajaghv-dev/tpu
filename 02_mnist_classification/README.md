# 02 – MNIST Classification

Trains a small CNN on MNIST using `pmap` data-parallelism across all TPU cores.

## Concepts
- `flax.linen.Module` definition
- `flax.training.train_state.TrainState`
- `jax.pmap` + `jax.lax.pmean` for gradient averaging
- `jax.device_put_replicated` to shard state
- `tensorflow_datasets` as data source

## Run
```bash
python train.py
```

## Expected output
```
Epoch  1  loss=0.1423  acc=95.78%
...
Epoch  5  loss=0.0421  acc=98.87%
```
