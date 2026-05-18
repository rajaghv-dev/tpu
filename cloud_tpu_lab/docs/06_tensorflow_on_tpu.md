> **Note:** this doc predates the real-TPU pivot. References to `src/xla_sim/`, `src/pjrt_sim/`, `src/sharding/`, `src/memory/`, `src/input_pipeline/`, and `examples/run_cpu_simulation_demo.py` are historical — those modules were removed. The TPU architecture / XLA / observability concepts below are still accurate. Current run flow lives in [README.md](../README.md) and [16_runbook_real_tpu.md](16_runbook_real_tpu.md).

# 06 - TensorFlow on Cloud TPU

> **Learning goal:** understand the `TPUClusterResolver` / `TPUStrategy` pattern that puts a TensorFlow program on Cloud TPU, the `tf.data` discipline that keeps the input pipeline fed (`AUTOTUNE`, `cache`, `prefetch`), and the profiler workflow for diagnosing problems.

TF on TPU is the oldest of the three framework paths. _[PUBLIC]_ It is mature, but the surface area is large; this doc focuses on the parts you'll actually touch.

Reference: <https://cloud.google.com/tpu/docs/tensorflow-quickstart-tpu-vm>

---

## 1. The shape of a TF/TPU program

Almost every TF-on-TPU program has the same skeleton:

1. Resolve the TPU cluster.
2. Initialize the TPU system.
3. Create a `TPUStrategy`.
4. Build the model and training step **inside** `strategy.scope()`.
5. Wrap the dataset with `strategy.experimental_distribute_dataset(...)`.
6. Train.

You do not need to invent any of this from scratch. Google's quickstart at <https://cloud.google.com/tpu/docs/tensorflow-quickstart-tpu-vm> contains a runnable template.

---

## 2. `TPUClusterResolver` — finding the TPU

```python
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
print("num_replicas:", strategy.num_replicas_in_sync)
```

Two key flags:

- `tpu="local"` — for TPU VM, the TPU is on the same host as the Python process. _[PUBLIC]_
- `tpu="<name>"` — for the legacy TPU Node architecture, you pass the node's name.

> If you are running on a real TPU VM, `"local"` is almost always what you want.

`initialize_tpu_system()` resets the TPU's state. **It is destructive** for any model currently running. _[PUBLIC]_ Run it once at process startup, never inside the training loop.

---

## 3. `TPUStrategy` — replication across cores

`tf.distribute.TPUStrategy` is the SPMD wrapper. Inside `strategy.scope()`:

- Variables are created sharded/replicated across TPU cores.
- The training step you wrap with `@tf.function` is compiled for the TPU device and run on every core in parallel.

```python
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(10),
    ])
    optimizer = tf.keras.optimizers.SGD(1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

`strategy.num_replicas_in_sync` tells you the global batch is split this many ways. _[PUBLIC]_

### Per-replica vs global batch

When you say `batch_size=64` to a `tf.data.Dataset`, that is the **per-replica** batch size after sharding. To express **global** batch size, divide:

```python
global_batch = 1024
per_replica = global_batch // strategy.num_replicas_in_sync
dataset = make_dataset().batch(per_replica)
dist_dataset = strategy.experimental_distribute_dataset(dataset)
```

Getting this wrong is the most common silent-correctness bug in TF on TPU: people set "batch size = 64" thinking it's global, but it ends up being 64×N.

---

## 4. The `@tf.function` training step

```python
@tf.function
def train_step(batch):
    x, y = batch
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        per_example = loss_fn(y, logits)
        loss = tf.nn.compute_average_loss(per_example, global_batch_size=global_batch)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def distributed_step(batch):
    per_replica_loss = strategy.run(train_step, args=(batch,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
```

What's happening:

- `@tf.function` traces the Python into a `ConcreteFunction` — TF's equivalent of `jax.jit`. _[PUBLIC]_
- `strategy.run(...)` dispatches the function on every replica.
- `strategy.reduce(...)` reduces per-replica outputs to a single global value via an all-reduce.

The same recompile rules apply: stable shapes are essential. Varying batch size, sequence length, or any tensor dim re-traces. _[PUBLIC]_

---

## 5. The `tf.data` discipline

`tf.data` is the input pipeline. The three operators you will use constantly:

| Operator | What it does |
| --- | --- |
| `.map(fn, num_parallel_calls=tf.data.AUTOTUNE)` | Parallel preprocessing |
| `.cache()` | After this point, results are cached (in RAM by default, optionally to disk) |
| `.prefetch(tf.data.AUTOTUNE)` | Overlap producer with consumer |

A skeleton:

```python
AUTOTUNE = tf.data.AUTOTUNE

ds = tf.data.Dataset.from_tensor_slices(filenames)
ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=AUTOTUNE,
                   num_parallel_calls=AUTOTUNE)
ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
ds = ds.cache()                       # only OK if it fits in RAM
ds = ds.shuffle(10_000)
ds = ds.batch(per_replica, drop_remainder=True)
ds = ds.prefetch(AUTOTUNE)
```

Why each operator matters on TPU:

- `interleave` reads multiple shards in parallel — essential for high-throughput TPUs that consume batches faster than a single file can supply them.
- `map(..., num_parallel_calls=AUTOTUNE)` runs preprocessing across host CPU cores.
- `cache()` lets you re-use a preprocessed dataset for multiple epochs without recomputing. **Only use it when the cached form fits in RAM**, or specify a cache file path for disk-backed caching.
- `shuffle(buffer)` is local; for true global shuffle on big datasets, shuffle file order first.
- `batch(..., drop_remainder=True)` is critical on TPU — uneven last batches cause shape variation → recompiles.
- `prefetch(AUTOTUNE)` overlaps the input pipeline with TPU compute. Without it, the TPU idles waiting for data.

> Reference: <https://www.tensorflow.org/guide/data_performance>

---

## 6. Profiling — TensorBoard's profile plugin

Cloud TPU integrates with the **TensorBoard profile plugin** out of the box. _[PUBLIC]_ Two main capture modes:

1. **Programmatic.**
   ```python
   tf.profiler.experimental.start(logdir="./tb_logs")
   for step in range(100):
       distributed_step(next(it))
   tf.profiler.experimental.stop()
   ```
2. **On-demand capture.** Start a profiler server, then trigger captures from the TensorBoard UI's "Profile" tab.

Once captured, the key views:

- **Overview Page** — single-page summary including average step time, dominant op categories, and "TPU duty cycle" (utilization).
- **TraceViewer** — Chrome-trace style timeline. Look for big gaps; they're input-pipeline starvation.
- **Input Pipeline Analyzer** — explicitly diagnoses tf.data bottlenecks.
- **Op Profile** — per-op cost, sorted.

The simulator emits a Chrome-trace JSON (`artifacts/traces/run_<trace_id>.json`) that loads in `chrome://tracing` or in TensorBoard's TraceViewer. The schema is intentionally compatible.

---

## 7. Mixed precision

On TPU, the standard recommendation is the **`mixed_bfloat16`** policy:

```python
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_bfloat16")
```

Effects:

- Activations are bf16 inside the model.
- Variables (and accumulators) remain fp32 for numerical stability.
- The MXU operates on bf16 → near-peak throughput. _[PUBLIC]_

Don't pick `mixed_float16` on TPU — that's the GPU path and runs through different code paths. _[PUBLIC]_

---

## 8. Checkpointing

Checkpointing on TPU should use `tf.train.Checkpoint` or Keras `Model.save_weights` + a **GCS** path:

```python
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, "gs://my-bucket/ckpts", max_to_keep=3)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()

for step in range(n_steps):
    distributed_step(next(it))
    if step % 1000 == 0:
        manager.save()
```

- Always save to **GCS** when training on TPU. Local-disk-only checkpoints disappear when the VM is deleted (and your `delete_tpu_vm.sh` should always run!).
- `expect_partial()` silences spurious "missing variable" warnings on restore.

---

## 9. Common TF-on-TPU pitfalls

A short list:

1. **Implicit Python control flow inside `@tf.function`.** Use `tf.cond`, `tf.while_loop`. _[PUBLIC]_
2. **Variable batch sizes.** Use `drop_remainder=True` everywhere.
3. **Forgetting to wrap variable creation in `strategy.scope()`.** Variables created outside scope can't be replicated; you get cryptic errors.
4. **Calling Python `print(tensor.numpy())` inside the step.** Forces materialisation.
5. **`tf.data` without `prefetch(AUTOTUNE)`.** Input starvation. The simulator's bottleneck report fires on this above 10 %.
6. **Caching to RAM a dataset bigger than RAM.** Process dies after first epoch — looks mysterious, is in fact an OOM.
7. **Using `tf.distribute.MirroredStrategy` instead of `TPUStrategy`.** Wrong strategy → CPU/GPU code path.

---

## 10. Where this maps onto the lab's simulator

The simulator does not run TensorFlow. But the same costs show up:

- `@tf.function` ≈ `src/xla_sim/lowering.py` + `compile_cache.py` (trace → HLO → cached).
- `TPUStrategy.run(...)` ≈ a replicated call into `src/pjrt_sim/runtime.py`.
- `tf.data` prefetch ≈ `src/input_pipeline/prefetch_sim.py`.
- TF profiler trace ≈ `src/profiling/profiler_trace.py` output.

If you can predict the bottleneck on the simulator, the production diagnosis flow on TF is nearly identical: open the profile, find the dominant bar, fix the corresponding layer.

---

## 11. Cross-references

- [`docs/03_xla_pjrt_runtime.md`](03_xla_pjrt_runtime.md) — `@tf.function` and `jax.jit` both go through XLA.
- [`docs/04_jax_on_tpu.md`](04_jax_on_tpu.md) — same problem, JAX surface.
- [`docs/05_pytorch_xla_on_tpu.md`](05_pytorch_xla_on_tpu.md) — PyTorch surface.
- [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md) — profiler workflow in depth.

Code:

- `src/input_pipeline/prefetch_sim.py` — the prefetch model that mirrors tf.data semantics.
- `src/profiling/profiler_trace.py` — Chrome-trace JSON output.
- `src/profiling/bottleneck_report.py` — input-pipeline rule fires the same way.

Official:

- TF on TPU VM quickstart: <https://cloud.google.com/tpu/docs/tensorflow-quickstart-tpu-vm>
- `tf.data` performance: <https://www.tensorflow.org/guide/data_performance>
- TPUStrategy: <https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy>
- Profiler: <https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras>

---

## 12. Exercises

1. **TPUStrategy boilerplate.** Write (don't run) the minimal Python that resolves a local TPU, creates a `TPUStrategy`, prints `num_replicas_in_sync`, and shuts down cleanly. Confirm every step matches the §2 / §3 template above.

2. **Global vs per-replica batch.** Take a desired global batch size of 256. For TPU slices of `v5e-4`, `v5e-8`, `v5e-16`, write the per-replica batch you'd pass to `.batch(...)`. What happens to throughput if you mistakenly pass the global batch directly?

3. **Diagnose input starvation (paper exercise).** You see TPU duty cycle = 35 % in the TF profiler overview. List four pipeline changes you'd try, ranked by expected impact. Cross-reference the `tf.data` performance guide.

4. **Mixed precision audit.** For a Keras model with `Dense(...)` layers, what changes after `mixed_precision.set_global_policy("mixed_bfloat16")`? Which tensors stay fp32? Why? Read the TF mixed-precision guide and confirm.
