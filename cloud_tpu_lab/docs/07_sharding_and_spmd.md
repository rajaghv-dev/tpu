# 07 - Sharding and SPMD on Cloud TPU

> **Learning goal:** build a working model of how a tensor program scales across many TPU chips. Understand data parallel, tensor parallel, pipeline parallel, the SPMD substrate, and the three collectives (all-reduce, all-gather, reduce-scatter) that underlie all of it. Read and write `PartitionSpec`s with confidence, and reason about scaling efficiency before running anything.

This doc grounds itself in the lab's simulator at `src/sharding/`:

- `src/sharding/mesh.py` — `Mesh`, `PartitionSpec`.
- `src/sharding/partitioner.py` — naive op placement.
- `src/sharding/all_reduce.py` — collective cost.

---

## 1. The four parallelism strategies you actually use

There are four orthogonal parallelism dimensions you can compose. Mix and match — for large models, you typically use all four at once.

| Name | What's split across devices | Communication pattern |
| --- | --- | --- |
| **Data parallel** | The batch dimension | All-reduce of gradients per step |
| **Tensor parallel** | A model dim (often hidden, sometimes attention heads) | All-reduce / all-gather / reduce-scatter per layer |
| **Pipeline parallel** | Layers (stages) | Send/receive between adjacent stages |
| **Sequence parallel** | The sequence dim (variants: context parallel, sequence-sharded LayerNorm) | All-gather + reduce-scatter |

All four are expressible in JAX or PyTorch/XLA today, via the same primitive: declare a **mesh**, then attach a **PartitionSpec** to each tensor saying which mesh axis its dims live on. _[PUBLIC]_

> Reference: <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>, <https://github.com/pytorch/xla/blob/master/docs/spmd.md>.

---

## 2. Mesh and PartitionSpec, from the lab

The lab's mesh API is intentionally a one-screen mirror of JAX's:

```python
@dataclass(frozen=True)
class Mesh:
    shape: Tuple[int, ...]
    axis_names: Tuple[str, ...]
    @property
    def n_devices(self) -> int: ...
    def axis_size(self, name: str) -> int: ...

@dataclass(frozen=True)
class PartitionSpec:
    spec: Tuple[Optional[str], ...]
    def is_replicated(self) -> bool: ...
```

(See `src/sharding/mesh.py`.)

A 2D mesh:

```python
from cloud_tpu_lab.src.sharding.mesh import Mesh, PartitionSpec

mesh = Mesh(shape=(2, 4), axis_names=("data", "model"))
print(mesh.n_devices)                  # 8
print(mesh.axis_size("model"))         # 4
```

A `PartitionSpec` for a 3D tensor `(B, S, D)`:

```python
ps = PartitionSpec(spec=("data", None, "model"))
print(ps.is_replicated())  # False
```

Interpretation: the **batch** dim is sharded across the `"data"` axis; the **sequence** dim is replicated; the **hidden** dim is sharded across the `"model"` axis. That is the canonical "data × tensor parallel" sharding for a transformer's main activations.

---

## 3. Data parallel

The simplest strategy. Every device holds a full copy of the model; each device processes a fraction of the batch; gradients are all-reduced across devices.

Mesh: `Mesh(shape=(N,), axis_names=("data",))`.

PartitionSpecs:

- Batched activations: `PartitionSpec("data", None, ...)` — shard along batch dim.
- Model parameters: `PartitionSpec(None, None)` — replicated.
- Gradients: same shape as parameters, but the gradient computation produces a sharded result that XLA all-reduces to a replicated form.

**Communication cost per step:** one all-reduce of size `|grad|` per parameter (in practice batched, but conceptually that).

When data parallel works:

- Small enough model to fit on one chip's HBM.
- You want a bigger global batch.

When it fails:

- Model too big for one chip. You need tensor or pipeline parallel.

---

## 4. Tensor parallel

Split one or more model dimensions across devices. Two famous patterns:

### Megatron-style tensor parallel for an MLP block

For an MLP `Y = (X W1) W2` where `W1: [D, 4D]` and `W2: [4D, D]`, on a model-parallel axis of size `M`:

- Shard `W1` along its **output** dim. Each device gets `[D, 4D/M]`.
- Compute `X W1` locally. Output is `[B, S, 4D/M]` (sharded).
- Apply the activation locally (still sharded).
- Shard `W2` along its **input** dim. Each device gets `[4D/M, D]`.
- Multiply `(X W1) W2` locally. Output is `[B, S, D]` but **partial** — each device has a partial sum.
- **All-reduce** across the `model` axis to combine partial sums into the final `[B, S, D]`. _[DOC]_

PartitionSpecs:

- `W1`: `PartitionSpec(None, "model")`.
- `W2`: `PartitionSpec("model", None)`.
- Activations between W1 and W2: `PartitionSpec(None, None, "model")`.
- Final activations after W2: `PartitionSpec(None, None, None)` after the all-reduce.

This pattern is what `src/sharding/partitioner.py` simulates at a coarse level.

### Why tensor parallel is bounded by ICI

Every layer's all-reduce traverses ICI. If `model` axis = 8 and the activation is 4 GB, you move 4 GB through ICI **per layer**. At v5e's `[DOC]` 200 GB/s ICI, that's 20 ms of pure communication per layer. With 32 layers, that's 640 ms / step of communication alone. _[Computation; uses [DOC] catalog values.]_

So tensor parallel **does not scale infinitely**. There's a sweet spot per generation. v5p (`[DOC]` ICI = 600 GB/s) tolerates more tensor parallel than v5e.

---

## 5. Pipeline parallel

Split the model into **stages** (groups of layers), one stage per device-group. Each batch is broken into **micro-batches** that flow through the pipeline.

```
batch -> [stage 0]  fwd
                    fwd -> [stage 1]  fwd
                                       fwd -> [stage 2]  fwd
                                                          fwd -> [stage 3] fwd
                                                                            bwd
                                                          bwd <- ...
```

Communication: a `send`/`recv` between adjacent stages per micro-batch (forward and backward). _[DOC]_

Pros:

- Reduces HBM pressure per device (only a slice of layers needed).
- Communication is point-to-point, not collective — cheaper per byte than all-reduce.

Cons:

- **Bubble overhead** — the first and last micro-batches see idle stages. Scales as `(n_stages - 1) / n_microbatches`. You need many micro-batches to amortise.
- Stage balancing is non-trivial.

This lab does not simulate pipeline parallel directly — it's the most complex of the four to model fairly. But the partitioner is structured to leave room for it (`src/sharding/partitioner.py`).

---

## 6. SPMD — the unifying model

**SPMD** (Single Program, Multiple Data) is the execution model: every device runs the same compiled program. Differences in behaviour come from each device knowing its own coordinate in the mesh.

In JAX:

```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(jax.devices(), ("data", "model"))

@jax.jit
def f(x, w):
    return x @ w

# Place x sharded on (data, None), w sharded on (None, model), and the
# output is partial-sum on the model axis — XLA will insert the all-reduce.
```

In PyTorch/XLA:

```python
import torch_xla.distributed.spmd as xs
xs.mark_sharding(x, mesh, ("data", None))
xs.mark_sharding(w, mesh, (None, "model"))
```

The XLA compiler takes the user's intent (per-tensor sharding) and **propagates** it through the graph, inserting collectives where necessary. _[PUBLIC]_

This is qualitatively different from MPI / NCCL programming where you write the collectives by hand. With SPMD on XLA you declare end-state and the compiler figures out the comms.

---

## 7. The three collectives you must know

Three primitives underlie almost every SPMD pattern. _[PUBLIC]_

| Collective | Input (per device) | Output (per device) | Use |
| --- | --- | --- | --- |
| **All-reduce** | `[N]` partial | `[N]` summed | Gradient sync (DP), final reduce in TP |
| **All-gather** | `[N/P]` shard | `[N]` full | "Unshard" a tensor before a global op |
| **Reduce-scatter** | `[N]` partial | `[N/P]` shard | Distribute a reduction's output |

A useful identity: **all-reduce = reduce-scatter + all-gather**. Modern systems split a logical all-reduce into RS+AG so that intermediate compute can overlap with communication.

### Cost model (rough)

The lab models collective time in `src/sharding/all_reduce.py` as `bytes / ici_bandwidth + per_collective_overhead`. This is a "ring" approximation; real implementations use tree or hierarchical algorithms. _[INFER]_ The ratio remains useful for predicting which workloads are communication-bound.

```python
# Simplified mental model
collective_time_s = tensor_bytes / ici_bandwidth_Bps
```

For v5p ICI = 600 GB/s and an 8 GB gradient buffer: `8e9 / 6e11 ≈ 13 ms`. _[Computation; [DOC] catalog ICI value.]_

---

## 8. Scaling efficiency intuition

A useful framing: as you add chips,

- **Compute scales linearly** (in the best case).
- **Communication scales sub-linearly or worse** — adding chips usually adds more bytes to move.
- **Step time** is `max(compute, communication, input_pipeline)`.

Strong-scaling efficiency for a fixed global problem usually plateaus when communication catches up with compute. You can estimate the plateau by computing:

```
efficiency(N) = 1 / (1 + comm_fraction_per_step)
```

Below 10 % comm fraction you're scaling cleanly. Above 30 % the bottleneck report in `src/profiling/bottleneck_report.py` will flag "communication-bound" — see the `"Collective communication"` block.

### Weak vs strong scaling

- **Strong scaling**: keep problem size fixed, add chips. Communication fraction grows.
- **Weak scaling**: scale problem with chips. Easier to keep efficiency high.

Pre-training language models is typically **weak scaling** — global batch grows with chip count. That's why DP scales so well in practice.

---

## 9. A worked sharding plan: 7B transformer on `v5p-8`

A back-of-envelope plan, using `[PUBLIC]` v5p HBM = 96 GB / chip.

Total chips: 8. Suppose we lay out as a `(2, 4)` mesh with axes `("data", "model")`.

Model: 7 B params × 2 bytes (bf16) = 14 GB params replicated. Plus optimizer state (Adam: 2 fp32 moments + fp32 master copy ≈ 12 bytes / param = 84 GB) — does **not** fit replicated on one chip's 96 GB.

Plan:

- Shard optimizer state across the `"model"` axis (size 4). 84 GB / 4 = 21 GB / chip.
- Replicate params across `"model"` (since 14 GB < 96 GB).
- Shard batch across `"data"` (size 2).
- Activations: shard along `"data"` (batch) and `"model"` (hidden).

Per-step communication:

- DP all-reduce of the 14 GB grad buffer across `"data"` (2 devices). Tiny.
- TP all-reduces within each layer across `"model"` (4 devices). One per MLP, one per attention output projection.

This is the "ZeRO-1 + tensor parallel" pattern.

The lab does not have a built-in tool that emits a plan like this, but the building blocks (Mesh, PartitionSpec, all-reduce cost) are all there in `src/sharding/`. You could prototype a planner in 100 lines.

---

## 10. Common sharding mistakes

1. **Sharding a small dim across many chips.** If hidden = 4096 and `model_axis_size = 16`, each chip gets 256 — too small to keep the MXU busy. _[INFER]_
2. **Mismatched PartitionSpecs across an op.** XLA inserts implicit collectives ("resharding") to fix it — this can be expensive and silent. Always trace your specs end-to-end.
3. **Sharding optimizer state but not parameters.** Re-syncing on every step is expensive. Either shard both, or neither.
4. **Forgetting to shard gradients.** If params are sharded but grads accidentally end up replicated, you'll OOM on big models.
5. **Using `pmap` for multi-axis parallelism.** `pmap` is single-axis. Use `Mesh` + `jit`. _[PUBLIC]_

---

## 11. Cross-references

- [`docs/02_cloud_tpu_architecture.md`](02_cloud_tpu_architecture.md) — why ICI bandwidth bounds tensor parallel.
- [`docs/04_jax_on_tpu.md`](04_jax_on_tpu.md) — the JAX API surface.
- [`docs/05_pytorch_xla_on_tpu.md`](05_pytorch_xla_on_tpu.md) — the PyTorch surface.
- [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md) — diagnosing comm-bound vs compute-bound.

Code:

- `src/sharding/mesh.py` — Mesh / PartitionSpec.
- `src/sharding/partitioner.py` — naive partitioner.
- `src/sharding/all_reduce.py` — collective cost model.
- `src/profiling/bottleneck_report.py` — the "collective communication" finding.

Official:

- JAX distributed arrays: <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>
- PyTorch/XLA SPMD: <https://github.com/pytorch/xla/blob/master/docs/spmd.md>
- Cloud TPU multi-host docs: <https://cloud.google.com/tpu/docs/jax-pods>

---

## 12. Exercises

1. **Predict scaling.** For an all-reduce of 4 GB across 8 chips on v5e (`[DOC]` ICI = 200 GB/s), estimate the time. Do the same on v5p (`[DOC]` ICI = 600 GB/s). At what step-time-per-step does this become unacceptable?

2. **Read PartitionSpecs.** For each spec, describe in plain English what it does to a `(B, S, D)` tensor on a `("data", "model")` mesh:
   - `P("data", None, None)`
   - `P("data", None, "model")`
   - `P(None, "data", None)`
   - `P(("data", "model"), None, None)`

3. **Design a sharding plan.** For a 1B-parameter model on `v5e-4` (4 chips), give a PartitionSpec for params, activations, and gradients. Will it fit in 16 GB/chip HBM with bf16 weights and an Adam optimizer? Show your math.

4. **Find a mismatch.** In the lab's `src/sharding/partitioner.py`, identify one op type whose PartitionSpec propagation is **not** handled. Write down what would happen if you ran it (paper exercise).
