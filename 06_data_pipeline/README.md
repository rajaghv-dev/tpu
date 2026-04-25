# 06 – Efficient Data Pipelines for TPUs

`tf.data` best-practices that prevent the TPU from starving for data.

## Key principles
1. **`drop_remainder=True`** — XLA requires static shapes; variable batch size breaks compilation
2. **`num_parallel_calls=tf.data.AUTOTUNE`** — maximise CPU utilisation for decoding/augmentation
3. **`prefetch(AUTOTUNE)`** — overlap host preprocessing with device training
4. **`interleave`** for GCS TFRecords — concurrent reads across many files mask I/O latency
5. Shuffle *before* batching, with a buffer ≥ 10× the batch size

## Run
```bash
python pipeline.py
```

## GCS pipeline
Replace the `gcs_pattern` argument in `build_gcs_pipeline` with your bucket path:
```python
ds = build_gcs_pipeline(
    gcs_pattern="gs://my-bucket/imagenet/train-*.tfrecord",
    batch_size=64,
    n_devices=jax.device_count(),
)
```
