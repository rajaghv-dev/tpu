# 03 - XLA + PJRT: From Python Op to Compiled Executable

> **Learning goal:** trace, step by step, what happens between writing `y = jnp.dot(a, b)` and the bytes flowing through HBM on a Cloud TPU. Learn the compile pipeline (tracing → lowering → fusion → layout → scheduling → execution), how the XLA cache is keyed, and the most common "silent recompile" footgun.

The mental model in this doc is grounded in this repo's simulator:

- `src/xla_sim/lowering.py` does the model → HLO step.
- `src/xla_sim/fake_hlo.py` defines the HLO op types.
- `src/xla_sim/compile_cache.py` mirrors the cache-key concept.
- `src/pjrt_sim/runtime.py` is the synchronous PJRT loop.

---

## 1. The pipeline at a glance

```
+-----------+    +---------+    +-----+    +--------+    +------------+    +-------+
|  Python   |--->| Tracing |--->| HLO |--->|  XLA   |--->| Compiled   |--->| PJRT  |
|  (model)  |    | (jaxpr) |    |  IR |    | compile|    | executable |    |  run  |
+-----------+    +---------+    +-----+    +--------+    +------------+    +-------+
                                              |
                                              +-> fusion
                                              +-> layout assignment
                                              +-> scheduling
                                              +-> codegen (TPU LLO)
```

Each box is a real, named phase, with documented behaviour in OpenXLA and JAX/PyTorch-XLA. _[PUBLIC]_

> Reference: <https://openxla.org/xla> and <https://jax.readthedocs.io/en/latest/jit-compilation.html>

---

## 2. Tracing — Python becomes a graph

When you call `jax.jit(f)(x)`, JAX does **not** run your Python with real numbers. It runs your Python with **tracer objects** that record every op into an intermediate representation called **jaxpr**. _[PUBLIC]_ See <https://jax.readthedocs.io/en/latest/key-concepts.html>.

PyTorch/XLA does the same conceptually via lazy tensors: ops record into a graph until you `mark_step()`. _[PUBLIC]_ See <https://github.com/pytorch/xla>.

TensorFlow's `@tf.function` traces too, into a `ConcreteFunction`. _[PUBLIC]_

In this lab the analogue is the **`ModelGraph`** in `src/xla_sim/lowering.py`. It's a list of `Layer` objects — `Linear`, `Conv`, `Attention`, `LayerNorm`. That's what a tracer would produce if it were as simple as possible.

```python
from cloud_tpu_lab.src.xla_sim.lowering import Layer, ModelGraph

g = ModelGraph(
    name="tiny",
    layers=[
        Layer(name="fc1", kind="linear", shape_in=(8, 64), shape_out=(8, 128)),
        Layer(name="fc2", kind="linear", shape_in=(8, 128), shape_out=(8, 10)),
    ],
)
```

That's the "what should run" object. It's still framework-agnostic; nothing has been compiled yet.

### What tracing captures

- Shapes (the simulator uses concrete `Tuple[int, ...]`; real JAX uses `ShapedArray`).
- Dtypes (the simulator uses bf16 by default).
- The op kind and its inputs.

### What tracing does NOT capture

- Python control flow that depends on runtime values. `if x[0] > 0: ...` will either trace one branch only (and bake it in), or — if the tracer can detect the issue — raise. This is **the** single biggest mental shift when moving from PyTorch eager to TPU.

The fix patterns are framework-specific:

- JAX: `jax.lax.cond`, `jax.lax.scan`, `jax.lax.while_loop`. _[PUBLIC]_
- PyTorch/XLA: rewrite into static-shape ops, or call out to host with `xm.mark_step()` boundaries.
- TF: `tf.cond`, `tf.while_loop`.

---

## 3. Lowering — graph to HLO

After tracing, the framework lowers the graph to **HLO** (High-Level Operations), XLA's IR. _[PUBLIC]_

In this lab the lowering rules in `src/xla_sim/lowering.py` are:

| User-facing layer | HLO ops emitted |
| --- | --- |
| `Linear(in_dim → out_dim)` | `DotGeneral` + `Add` (bias) |
| `Conv` (single conv) | `Convolution` |
| `Attention(Q,K,V)` | `DotGeneral(QK^T)` + `Softmax` + `DotGeneral((softmax)V)` |
| `LayerNorm` | `ReduceMean` + `Variance` + `Normalize` |

Real XLA HLO has dozens of op kinds (DotGeneral, Convolution, Reduce, Broadcast, Reshape, Transpose, Slice, DynamicSlice, While, Conditional, AllReduce, ...). _[PUBLIC]_ See <https://openxla.org/xla/operation_semantics> for the full reference.

The lab keeps the set small intentionally so the lowering is auditable in 50 lines of Python.

### Example: Linear lowered

```python
from cloud_tpu_lab.src.xla_sim.lowering import Layer, ModelGraph, lower_to_hlo

g = ModelGraph("toy", [Layer("fc", "linear", (32, 256), (32, 512))])
hlo = lower_to_hlo(g)
for op in hlo.ops:
    print(op.kind.value, op.shape, "flops=", op.flops, "bytes=", op.bytes_in + op.bytes_out)
```

Output (truncated):

```
DotGeneral (32, 512) flops=8388608 bytes=...
Add        (32, 512) flops=16384   bytes=...
```

Each op carries:

- `op_id` — unique HLO op id (correlation key in the OCT model).
- `shape`, `dtype`.
- `model_layer_id` and `layer_name` — pointer back to the user-facing layer that produced it. Critical for profiling: it lets you say "this DotGeneral came from `fc1`".
- `flops`, `bytes_in`, `bytes_out` — roofline inputs.

---

## 4. Fusion — the most important XLA transform

A naive lowering of `Attention` is 3 ops, each reading and writing HBM. **Fusion** is XLA combining adjacent ops into a single kernel so the intermediate tensors never touch HBM — they live in registers / scratchpad. _[PUBLIC]_

Why fusion matters:

- HBM bandwidth is the single biggest constraint on most ops.
- Cutting HBM round-trips in half can roughly double throughput on memory-bound ops.

XLA's fusion strategy is rule-based + cost-driven _[PUBLIC]_. Practical implications for you:

1. **Elementwise ops fuse aggressively.** A chain of `add -> relu -> mul` typically becomes one kernel.
2. **Matmul + elementwise epilogue fuses.** A `DotGeneral` followed by `Add` (bias) and `Relu` can be one fused op _[INFER per XLA docs]_.
3. **Reductions sometimes block fusion.** Softmax is the classic example; XLA has specific patterns for it, but custom variants can defeat fusion.
4. **Reshape / transpose may or may not be free.** Layout assignment (next step) tries to keep them free, but you can't assume.

This lab's simulator does **not** perform fusion. Every op is independent. That makes the simulator pessimistic vs reality — which is what we want for teaching: students see "this op cost X" and then learn fusion as an optimisation.

> Reference: <https://openxla.org/xla/operation_semantics> and the XLA design docs.

---

## 5. Layout assignment

A tensor's **layout** says which logical dim is innermost in memory. _[PUBLIC]_

XLA picks layouts that:

1. Match the MXU's expected operand layout for matmul throughput.
2. Avoid emitting `transpose` ops unless necessary.
3. Keep elementwise op fusion intact.

You generally do not interact with layout directly. The visible signals when layout goes wrong:

- HLO dumps show many `transpose` or `copy` ops between matmuls.
- Profile traces show "data movement" time exceeding "matmul" time.

When that happens, the fix is usually reshaping your model so that contracting dimensions are at the end of the shape, not the middle. _[INFER]_

---

## 6. Scheduling

After fusion + layout, XLA schedules the ops — picking the order in which they'll be executed, and where to overlap compute with collectives / memory ops. _[PUBLIC]_

For the user, this means:

- A collective op (`all-reduce`) **may** be overlapped with subsequent compute, hiding its latency.
- Long-dependency chains cannot be overlapped — you'll see them serially in the profile.

The simulator's `PjrtRuntime.execute` (`src/pjrt_sim/runtime.py`) is intentionally **fully serial**: every op runs after the previous one finishes. This overestimates step time vs reality. We document the limitation in code:

> "The runtime is single-threaded and synchronous — perfectly fine for a teaching simulator. Async / streams / overlap can be a later module."

---

## 7. Codegen and the compiled executable

The final phase is codegen — XLA produces a binary that the TPU runtime can load. _[PUBLIC]_

The deliverable is a **`CompiledExecutable`** (the term used by PJRT, and by this lab in `src/pjrt_sim/executable.py`). It:

- Holds a list of ops to execute (in scheduled order).
- Has an `executable_id` — another correlation key in the OCT model.
- Can be re-used across many invocations on the same shapes.

This is where the **compile cache** matters.

---

## 8. The XLA cache and "silent recompiles"

The compile cache key (this is the footgun) is roughly:

- The HLO module text (including shapes, dtypes, sharding).
- The target backend (TPU version, slice).
- The compiler flags / env vars.

The lab models this as `src/xla_sim/compile_cache.py:cache_key()`.

If anything in that key changes between calls, you recompile. Compile takes seconds to minutes _[INFER, varies wildly]_. In a training loop, this is catastrophic — every step looks slow, and you can't tell why.

### The classic "silent recompile" patterns

1. **Dynamic shapes.** Sequence length varies per batch → every new length recompiles. Fix: pad to a fixed length, or use a bucketed approach. _[PUBLIC]_ See <https://jax.readthedocs.io/en/latest/aot.html>.
2. **Batch-size variation.** Last batch of an epoch is smaller → recompile. Fix: drop the last partial batch, or pad it.
3. **Dtype variation.** Sometimes a stray fp32 constant flips a sub-expression's dtype → recompile. Fix: force-cast at the input boundary.
4. **PRNG keys.** Treating a key as a tracer can change the graph between calls. Fix: pass keys as standard `jax.random.PRNGKey` values.
5. **Sharding spec change.** Same shapes, different `PartitionSpec` → recompile.
6. **JAX env vars.** Toggling `JAX_COMPILATION_CACHE_DIR` or compiler flags mid-run → recompile.

The bottleneck report in this lab fires a "Compile = X %" finding when compile time exceeds 20 % of step time, with the suggested fix being exactly the above:

> _"Stabilise shapes (avoid dynamic batch / seq); set JAX_COMPILATION_CACHE_DIR to persist compiled executables across runs; check jax.config.jax_log_compiles."_

(See `src/profiling/bottleneck_report.py`, "Compile / recompile" block.)

### Persistent compile cache

Both JAX and PyTorch/XLA support a persistent on-disk compile cache.

- JAX: set `JAX_COMPILATION_CACHE_DIR` to a directory (a GCS bucket works). _[PUBLIC]_ See <https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html>.
- PyTorch/XLA: the framework caches per-process; cross-process caches are an evolving feature — check release notes.

A GCS-backed JAX cache is one of the highest-leverage optimisations you can apply for repeated experiments. The first run compiles; subsequent runs with the same shapes/dtypes hit the cache and start near-instantly.

---

## 9. PJRT — the runtime under the framework

**PJRT** (Pretty/Portable JAX Runtime, but pronounced as "PJRT") is the C++ runtime layer that owns devices, buffers, and executable execution. _[PUBLIC]_ See <https://openxla.org/xla/pjrt> and <https://github.com/openxla/xla/tree/main/xla/pjrt>.

From the framework's perspective, PJRT exposes:

- `Client` — the entry point. Has devices.
- `Device` — one TPU chip (or virtual device).
- `Buffer` — a chunk of device memory.
- `Executable` — a compiled program.

The lab models this in `src/pjrt_sim/runtime.py:PjrtRuntime`. The execute loop is:

```python
for op in executable.ops():
    ev_id = new_device_event_id()
    op_dur = primary.roofline_op_time_s(op.flops, op.bytes_in + op.bytes_out)
    # emit one record per op, with the full correlation bundle
    ...
```

Every op execution emits an `OpExecutionRecord` carrying:

- `device_event_id` — unique per execution.
- `device_id` — which chip.
- `op_id` — back-link to HLO.
- `model_layer_id` — back-link to the user model.
- `sim_duration_s`, `flops`, `bytes_moved` — the per-op cost.

This is the OCT (Observability / Controllability / Traceability) bundle the repo's README describes. Any event can be joined back through `device_event_id → op_id → model_layer_id → step_id → trace_id`.

---

## 10. Mental model: the four stable views of one step

Once you have lived inside XLA + PJRT for a while, every step has four views you should be able to pull up at will:

1. **User-facing graph view.** The Python code. `ModelGraph` here.
2. **HLO view.** The fused, shaped, scheduled IR. `HloModule` here, real `XLA HLO` in production.
3. **Executable view.** The compiled program, identified by `executable_id`. The cache lives at this layer.
4. **Trace view.** The per-op execution record, with timing. The simulator emits `OpExecutionRecord`; real TPU profiles emit Chrome-trace events.

When you debug a slow run, you ask: at which view does the surprise appear? If it appears at view 4, the fix is in views 1–3.

---

## 11. Cross-references

- [`docs/04_jax_on_tpu.md`](04_jax_on_tpu.md) — JAX-specific surface (`jax.jit`, sharding).
- [`docs/05_pytorch_xla_on_tpu.md`](05_pytorch_xla_on_tpu.md) — PyTorch/XLA's lazy-tensor view.
- [`docs/06_tensorflow_on_tpu.md`](06_tensorflow_on_tpu.md) — TF `@tf.function` tracing.
- [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md) — where these phases show up in profile traces.

Code:

- `src/xla_sim/lowering.py` — model → HLO.
- `src/xla_sim/fake_hlo.py` — HLO op definitions.
- `src/xla_sim/compile_cache.py` — cache-key model.
- `src/pjrt_sim/runtime.py` — execute loop with OCT records.
- `src/pjrt_sim/executable.py` — `CompiledExecutable`.

Official docs:

- XLA: <https://openxla.org/xla>
- HLO op semantics: <https://openxla.org/xla/operation_semantics>
- PJRT: <https://openxla.org/xla/pjrt>
- JAX jit: <https://jax.readthedocs.io/en/latest/jit-compilation.html>
- JAX persistent compile cache: <https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html>

---

## 12. Exercises

1. **Trigger a recompile.** Write a small JAX function and call it twice with two different input shapes (e.g. `(8, 64)` and `(16, 64)`). Enable `jax.config.update("jax_log_compiles", True)` and confirm both calls trigger compile. Then add `jax.numpy.zeros((16, 64))`-style padding to keep shapes constant and verify only one compile happens.

2. **Run the lab lowering.** Build a 4-layer `ModelGraph` with one Linear, one Attention, one LayerNorm, one Linear. Print the HLO ops with their `model_layer_id`. Verify the join works: every HLO op should point back to exactly one layer.

3. **Predict step time.** For the same model, sum `op.flops` across ops and divide by the catalog's `peak_bf16_tflops * 1e12`. Compare that floor with what `PjrtRuntime.execute` reports. Why is the simulator slower than the FLOPS floor?

4. **Cache miss design.** Sketch (in prose) a strategy to keep the JAX persistent cache warm across multiple jobs running on the same TPU VM. Where should `JAX_COMPILATION_CACHE_DIR` point — local disk, GCS, or a shared FUSE mount? Trade-offs?
