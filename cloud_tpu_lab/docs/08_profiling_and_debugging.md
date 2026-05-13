# 08 - Profiling and Debugging Cloud TPU Workloads

> **Learning goal:** know which tools to reach for when a Cloud TPU run is slower, more expensive, or less correct than expected. Master HLO dumps, the JAX profiler, the TensorBoard Profile plugin, Cloud TPU on-demand profiling, and a checklist of the most common performance pathologies — with the lab's `src/profiling/bottleneck_report.py` as a guide.

This doc is your single-page diagnostic playbook. It should be the first thing you read when something "looks wrong".

---

## 1. The diagnostic triangle

Every TPU performance problem reduces to one of three questions:

1. **What is the TPU spending its time on?** → profiler tools.
2. **What did XLA compile?** → HLO dumps.
3. **Is the host keeping up?** → input pipeline metrics + host CPU profile.

Pick the right tool per question. If you start with the wrong one (e.g. reading HLO when the actual issue is the dataloader), you'll burn hours.

---

## 2. HLO dumps — see what XLA built

When you suspect XLA is generating something surprising — too many ops, missing fusion, unwanted layout transposes — dump the HLO and read it.

### JAX

```python
# Two options:
# 1) Environment variable, system-wide:
#    XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
# 2) jax.config in code (works for some flags):
import os
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text",
)
```

After a run, `/tmp/xla_dump/` contains text HLO modules per JIT compilation. You can grep for op kinds, layer names (if you've passed them through), or AllReduce patterns.

### PyTorch / XLA

```python
import torch_xla.debug.metrics as met

print(met.metrics_report())   # counters per op, compile counts, transfer counts
```

For HLO text, set environment variables before importing torch_xla:

```bash
export PT_XLA_DEBUG=1
export XLA_DUMP_HLO_GRAPH=1
```

This will print HLO for each compiled graph. _[PUBLIC]_ See <https://github.com/pytorch/xla/blob/master/docs/troubleshoot.md>.

### TensorFlow

`tf.config.experimental.tensor_float_32_execution_enabled()` and `tf.summary` are useful for general debugging, but for HLO specifically, set `XLA_FLAGS=--xla_dump_to=...` exactly as for JAX — they share the compiler.

### What to look for in an HLO dump

- **Many small ops where you expected fusion.** If your softmax becomes 6 separate ops with HBM round-trips, you've defeated fusion. Look for unusual dtypes, dynamic shapes, or transposes between the ops.
- **`copy` / `transpose` ops between matmuls.** Layout fights — the compiler is converting between layouts mid-graph.
- **`all-reduce` placement.** Is it where you expected? Are there more than you expected (e.g. one per layer when you only meant one per step)?

---

## 3. JAX profiler

JAX has a built-in profiler that emits Chrome-trace JSON or TensorBoard event files. _[PUBLIC]_

### Programmatic capture

```python
import jax

jax.profiler.start_trace("/tmp/jax_trace")
# ... run a few steps ...
for _ in range(5):
    loss = train_step(params, batch).block_until_ready()
jax.profiler.stop_trace()
```

Then open the resulting `.json.gz` in `chrome://tracing` or load into TensorBoard.

### On-demand capture via server

```python
jax.profiler.start_server(9012)
```

This starts an in-process server. From TensorBoard's "Profile" tab, you point it at `localhost:9012` and click "Capture Profile". This is the same pattern PyTorch/XLA uses (see [`docs/05_pytorch_xla_on_tpu.md`](05_pytorch_xla_on_tpu.md), §8).

### Useful flags

- `jax.config.update("jax_log_compiles", True)` — print a line every time JAX compiles. Catches recompile loops immediately.
- `jax.config.update("jax_debug_nans", True)` — fail when a NaN appears. Slow, but invaluable for correctness debugging.
- `jax.config.update("jax_disable_jit", True)` — turn off jit for quick eager-mode debugging. **Never run benchmarks with this.**

---

## 4. TensorBoard Profile plugin and the Cloud TPU Profiler

The TensorBoard Profile plugin is the standard UI for reading TPU profiles. _[PUBLIC]_

Install and use:

```bash
pip install tensorboard tensorboard-plugin-profile
tensorboard --logdir ./tb_logs --bind_all --port 6006
```

Then open `http://<host>:6006` and click the **Profile** tab.

The most useful views, in order of how often I open them:

1. **Overview page.** Step time histogram, "TPU duty cycle" (utilization %), top op categories.
2. **Trace viewer.** Chrome-trace timeline. Big gaps = input starvation. Long single bars = compile or collective.
3. **Input pipeline analyzer.** Diagnoses dataloader bottlenecks specifically.
4. **Op profile.** Top ops by self time. Goes from "where is time spent" to "which line of code is responsible" via layer names.
5. **Memory profile.** HBM usage by allocation, peak vs steady state.

### Cloud TPU profiler (on-demand)

For TPU VM, you can capture a profile from outside the training process:

```bash
gcloud compute tpus tpu-vm ssh <vm-name> -- \
  "python3 -c 'from cloud_tpu_profiler import capture_profile; capture_profile(...)'"
```

Or via the framework's profile server (the `start_server(9012)` pattern). The on-demand path is convenient when you don't want to modify the running script.

Reference: <https://cloud.google.com/tpu/docs/cloud-tpu-tools>

---

## 5. The lab's `bottleneck_report.py` walkthrough

The lab encodes the diagnostic heuristics in `src/profiling/bottleneck_report.py`. The rules:

| Finding | Threshold | Severity | Fix hint (from code) |
| --- | --- | --- | --- |
| Input pipeline | > 10 % of step time (high if > 25 %) | warn / high | "Increase prefetch depth, use tf.data autotune / torch DataLoader num_workers..." |
| Compile / recompile | > 20 % of step time | warn | "Stabilise shapes; set `JAX_COMPILATION_CACHE_DIR`..." |
| Collective communication | n_chips > 1 and collectives > 30 % | high | "Larger batch; shard model instead of data; move to higher-bandwidth topology..." |
| HBM utilization | > 85 % | high | "Gradient checkpointing; smaller batch; shard params..." |
| HBM OOM events | > 0 | high | Same as above. |
| Host overhead | > 15 % | warn | "Move loss / metric reduction into the JIT'd step..." |
| Cost sanity | total run > $10 | info | "Use --dry-run; enable Spot; reduce n_steps for dev..." |

These thresholds are heuristic, not laws. Tune them for your environment. _[INFER]_

You can run the simulator end-to-end and see the report in `artifacts/reports/run_<trace_id>.md`:

```bash
python3 examples/run_cpu_simulation_demo.py
```

---

## 6. The seven recurring problems

A canonical list. For each, the **symptom**, the **root cause**, and the **fix**.

### 6.1 Slow first step

- **Symptom:** the first iteration takes seconds-to-minutes; subsequent iterations are fast.
- **Root cause:** XLA compilation. Normal.
- **Fix:** ignore the first few steps in your throughput averages, **and** enable a persistent compile cache (`JAX_COMPILATION_CACHE_DIR=gs://...`).

### 6.2 Recompile loops (the "every step is slow" disaster)

- **Symptom:** every step is slow, or step times are wildly variable. `jax_log_compiles` prints constantly.
- **Root cause:** input shapes vary per step. Dynamic batch size. Sequence length not padded. Last-batch-smaller.
- **Fix:**
  - Pad inputs to a fixed shape.
  - Use `drop_remainder=True` (TF) or filter the last partial batch.
  - For variable seq lengths, bucket and pad per bucket.

### 6.3 Unsupported ops

- **Symptom:** an exception at compile time naming an op XLA does not lower for the TPU backend.
- **Root cause:** some PyTorch/TF ops have no XLA lowering for TPU yet. Or the op has a fallback that materialises to host CPU, defeating performance.
- **Fix:**
  - Replace with a supported equivalent (e.g. write your own attention instead of a CUDA-only kernel).
  - Check the framework's TPU op coverage. _[PUBLIC]_ See PyTorch/XLA's "Known issues" doc.

### 6.4 Dynamic-shape recompile

- **Symptom:** as 6.2 but the dynamism is hidden — e.g. a `tf.unique` or `jnp.where` followed by `[mask]` produces a dynamic shape that triggers recompile.
- **Root cause:** any op whose output shape depends on runtime values.
- **Fix:** rewrite to use fixed-shape masking (`* mask` then reduce) instead of shape-changing operations.

### 6.5 Input pipeline starvation

- **Symptom:** TPU duty cycle low; large idle bars in the trace; throughput grows with `num_workers` but never saturates.
- **Root cause:** the CPU-side dataloader can't produce batches fast enough.
- **Fix:**
  - tf.data: `prefetch(AUTOTUNE)`, `interleave(... cycle_length=AUTOTUNE)`, `map(... num_parallel_calls=AUTOTUNE)`.
  - PyTorch/XLA: `num_workers > 0`, `MpDeviceLoader`.
  - Pre-process offline if the on-the-fly preprocessing is heavy.
  - Lab analogue: `src/input_pipeline/prefetch_sim.py` and the bottleneck "Input pipeline" rule.

### 6.6 Poor sharding / collective bottleneck

- **Symptom:** scaling efficiency drops sharply when you add chips; collectives dominate the timeline.
- **Root cause:** model parallel dim is too small relative to communication cost, or `data` axis is on a slow tier.
- **Fix:**
  - Increase batch size (amortises DP all-reduce).
  - Pick a higher-ICI generation (v5p / v6e vs v5e).
  - Re-layout your mesh — sometimes flipping `(data, model)` matters.

### 6.7 HBM OOM

- **Symptom:** OOM at compile or first execution.
- **Root cause:** activation + parameter + optimizer state exceeds HBM.
- **Fix:**
  - Gradient checkpointing (recompute activations on backward).
  - bf16 / fp8 optimizer state (where supported).
  - Shard params across more chips.
  - Smaller batch / shorter sequence.
  - Lab analogue: `src/memory/{activation_memory,checkpoint_memory,hbm_sim}.py`.

### 6.8 Checkpoint stalls

- **Symptom:** step time spikes every N steps; the spike is exactly when you save.
- **Root cause:** checkpoint write is synchronous and on the critical path.
- **Fix:**
  - Async checkpointing (Orbax for JAX, async save for TF / Torch).
  - Write to GCS in parallel from multiple hosts.
  - Avoid saving on every step.

---

## 7. A debugging routine, in order

When a real run looks slow, do these in order:

1. **Confirm JAX/Torch/TF is on TPU.** `jax.devices()`, `xm.xla_device()`, `strategy.num_replicas_in_sync`.
2. **Check for recompiles.** `jax_log_compiles=True` or `PT_XLA_DEBUG=1`. If it logs on every step, fix shapes first.
3. **Capture a profile.** Even 5 seconds is enough.
4. **Open the trace viewer.** Are step bars adjacent (good) or separated by gaps (input starvation)? Are bars dominated by one op color?
5. **Check input pipeline analyzer.** If it complains about producer being slow, your data path is the bottleneck.
6. **Check op profile.** Top 5 ops — is there an op you didn't expect (e.g. a `copy` or `transpose`)?
7. **Only then read HLO.** If profile is suggestive of layout / fusion issues, dump the HLO.

This order is roughly cost-ordered: each step is cheap to do and rules out a class of problems.

---

## 8. Cross-references

- [`docs/03_xla_pjrt_runtime.md`](03_xla_pjrt_runtime.md) — what HLO is and the cache-key model.
- [`docs/04_jax_on_tpu.md`](04_jax_on_tpu.md) — `jax.profiler` API surface.
- [`docs/05_pytorch_xla_on_tpu.md`](05_pytorch_xla_on_tpu.md) — `torch_xla.debug.profiler`.
- [`docs/06_tensorflow_on_tpu.md`](06_tensorflow_on_tpu.md) — TF profiler / TensorBoard.
- [`docs/07_sharding_and_spmd.md`](07_sharding_and_spmd.md) — collective bottleneck math.

Code:

- `src/profiling/profiler_trace.py` — Chrome-trace JSON emitter.
- `src/profiling/trace_analyzer.py` — `Breakdown` of step time fractions.
- `src/profiling/bottleneck_report.py` — the rules driving the report.
- `src/memory/hbm_sim.py` — HBM utilization tracking, OOM events.
- `src/input_pipeline/prefetch_sim.py` — input pipeline cost.

Official:

- Cloud TPU profiling tools: <https://cloud.google.com/tpu/docs/cloud-tpu-tools>
- TF Profile guide: <https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras>
- JAX profiling: <https://jax.readthedocs.io/en/latest/profiling.html>
- PyTorch/XLA troubleshooting: <https://github.com/pytorch/xla/blob/master/docs/troubleshoot.md>

---

## 9. Exercises

1. **Force a recompile and confirm it shows up.** Enable `jax.config.update("jax_log_compiles", True)`. Run a JIT'd function with two input shapes alternating. Confirm the log shows two compiles, not one.

2. **Read a Chrome trace.** Run `python3 examples/run_cpu_simulation_demo.py` and open `artifacts/traces/run_<trace_id>.json` in `chrome://tracing`. Identify the longest single op. Cross-check with the bottleneck report.

3. **Synthesise an input-starvation finding.** Find a knob in the simulator (or wrap it) that artificially slows the input pipeline so the bottleneck report fires "Input pipeline > 25 %". Verify the finding appears with severity `"high"`.

4. **Plan an HLO investigation.** You see one suspicious `transpose` per layer in HLO. Write the next three things you'd check, in order, before changing any code.
