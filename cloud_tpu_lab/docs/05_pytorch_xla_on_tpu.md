# 05 - PyTorch / XLA on Cloud TPU

> **Learning goal:** understand how PyTorch reaches a Cloud TPU through `torch_xla`, the lazy-tensor execution model, the `xm.xla_device()` / `xm.mark_step()` pattern, how `MpDeviceLoader` keeps the pipeline fed, and how SPMD sharding is expressed via `torch_xla.distributed.spmd`.

This doc complements [`docs/04_jax_on_tpu.md`](04_jax_on_tpu.md). If you know JAX already, the headline is: PyTorch reaches the **same XLA + PJRT** stack, but with **lazy tensors** rather than tracing-on-jit.

Reference: <https://cloud.google.com/tpu/docs/pytorch-xla-users-guide> and <https://github.com/pytorch/xla>.

---

## 1. What `torch_xla` is

`torch_xla` is the PyTorch bridge to the OpenXLA compiler and PJRT runtime. _[PUBLIC]_ Conceptually:

```
PyTorch ops --> torch_xla lazy tensors --> HLO --> XLA --> PJRT --> TPU chip
```

You write idiomatic PyTorch. The library intercepts ops, records them into a graph keyed by an XLA device, and only **materialises** (compiles + executes) the graph when you ask for the result.

Three high-level surfaces matter:

1. **Device selection** — `xm.xla_device()`.
2. **The lazy-tensor model** — `xm.mark_step()`, materialisation boundaries.
3. **SPMD** — `torch_xla.distributed.spmd` for partitioning across multiple chips.

---

## 2. Sanity check: is the device a TPU?

```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print("device:", device)            # 'xla:0' on TPU VM
print("world size:", xm.xrt_world_size())
print("ordinal:", xm.get_ordinal())
```

On a clean machine without `torch_xla` plumbing, this raises. On Colab with the TPU runtime active or on a TPU VM, you get a real XLA device.

`xm.xla_device()` is **a torch device** — passing it to `torch.tensor(..., device=device)` allocates on that device. _[PUBLIC]_

---

## 3. The lazy-tensor model in one minute

PyTorch eager-mode evaluates every op immediately. **PyTorch/XLA does not.** Each op is appended to an internal graph. The graph is materialised — i.e. lowered to HLO, compiled, and executed — only when one of:

- `xm.mark_step()` is called.
- A tensor's value is needed on host (e.g. `tensor.item()`, `.cpu()`, a Python `print(tensor)`).
- A control-flow barrier is hit.

This is the single biggest mental shift for PyTorch users moving to TPU.

```python
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()

x = torch.ones(1024, 1024, device=device)
y = x @ x
z = (y * 2).sum()
# Nothing has executed on the TPU yet.

xm.mark_step()
# *Now* the graph (build a 1024x1024, matmul, mul, sum) runs.

print(z.item())
# This forces materialisation if mark_step hadn't been called.
```

### Materialisation traps

- **`.item()` / `.cpu()` / Python `if tensor > 0:` in the hot loop** force materialisation early. Each one is a graph boundary; you'll fragment one big efficient graph into many small slow ones.
- **`print(tensor)` for debugging** also forces materialisation.

The remedy is to keep loss values, metrics, etc. on-device, only printing every K steps after an `xm.mark_step()`.

> Reference: <https://github.com/pytorch/xla#advanced-topics>

---

## 4. The training loop pattern

A canonical single-host training step on TPU:

```python
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm

device = xm.xla_device()

model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)).to(device)
opt = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_step(batch):
    x, y = batch
    x, y = x.to(device), y.to(device)
    opt.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()
    xm.mark_step()      # <-- materialise once per step
    return loss
```

Two patterns to note:

1. **One `mark_step()` per logical training step.** That's the unit you want compiled into a single graph.
2. **`opt.step()` is included in the materialised graph.** Otherwise the optimiser ops would land in a separate (smaller, slower) graph.

---

## 5. `MpDeviceLoader` — keeping the pipeline fed

Vanilla `torch.utils.data.DataLoader` returns CPU tensors. Moving them to the XLA device costs PCIe time. `torch_xla.distributed.parallel_loader.MpDeviceLoader` (or `ParallelLoader`) does the move asynchronously, overlapping host preprocessing with device compute. _[PUBLIC]_

```python
from torch_xla.distributed.parallel_loader import MpDeviceLoader

cpu_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)
device_loader = MpDeviceLoader(cpu_loader, device)

for batch in device_loader:
    loss = train_step(batch)
```

The `MpDeviceLoader` does three things for you:

- Pulls batches from the underlying `DataLoader`.
- Prefetches and transfers them to `device` asynchronously.
- Yields ready-on-device tensors.

This is the moral equivalent of `tf.data.AUTOTUNE.prefetch()` for PyTorch/XLA. _[PUBLIC]_

Without it, the bottleneck-report rule in `src/profiling/bottleneck_report.py` (the `"Input pipeline"` block) will fire constantly: the TPU finishes its step and idles waiting for the next batch to land in HBM.

---

## 6. Multi-host / multi-process: `xmp.spawn`

On a multi-host TPU slice, each host runs a Python process, and each process can drive its local chips. The recommended entry point is `torch_xla.distributed.xla_multiprocessing.spawn` (or in newer versions, the `torchrun`-style launchers). _[PUBLIC]_

```python
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
    device = xm.xla_device()
    # build model, dataloader, train loop using `device`
    ...

if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())
```

Within each worker, `xm.xrt_world_size()` and `xm.get_ordinal()` give you the global ranks. Gradients are typically all-reduced via `xm.all_reduce()` or the optimizer wrapper.

---

## 7. `torch_xla.distributed.spmd` — modern SPMD

Older PyTorch/XLA code used `xm.all_reduce` and explicit data parallelism. The modern path mirrors JAX: declare a **device mesh**, attach a **sharding spec** to each tensor, and let XLA lower to SPMD HLO. _[PUBLIC]_

```python
import torch_xla.distributed.spmd as xs

# Build a 1D mesh across all available chips along the "data" axis.
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

num_devices = xr.global_runtime_device_count()
mesh_shape = (num_devices,)
device_ids = np.arange(num_devices)
mesh = xs.Mesh(device_ids, mesh_shape, axis_names=("data",))

# Shard input across the "data" axis on its first dim.
x = torch.ones(1024, 512, device=xm.xla_device())
xs.mark_sharding(x, mesh, ("data", None))
```

The conceptual mapping to JAX's API:

| JAX | PyTorch / XLA |
| --- | --- |
| `jax.sharding.Mesh` | `torch_xla.distributed.spmd.Mesh` |
| `jax.sharding.PartitionSpec` | the tuple in `xs.mark_sharding(t, mesh, partition_spec)` |
| `jax.sharding.NamedSharding` | the combination of mesh + partition spec |
| `jax.jit` | implicit via the lazy-tensor graph + `mark_step` |

The lab's `Mesh` / `PartitionSpec` in `src/sharding/mesh.py` is framework-neutral and is a reasonable mental model for either.

> Reference: <https://github.com/pytorch/xla/blob/master/docs/spmd.md>

---

## 8. Profiling notes

PyTorch/XLA exposes its profiler via the same backend XLA uses (Cloud TPU Profiler / TensorBoard plugin).

```python
import torch_xla.debug.profiler as xp

server = xp.start_server(9012)
# ... train ...
# Capture a trace either via TensorBoard's Profile tab pointed at localhost:9012,
# or:
xp.trace_detached("http://localhost:9012", logdir="./tb_logs", duration_ms=5000)
```

What to look for:

- **Step time** in the device timeline — should be one big block per `mark_step`.
- **Many small graphs** instead of one big one — usually means you have a stray `.item()` or `print(tensor)` inside the loop.
- **Recompiles** — set `PT_XLA_DEBUG=1` to see when graphs are compiled. _[PUBLIC]_

The lab's [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md) covers profile-reading in more depth.

---

## 9. Common PyTorch/XLA pitfalls

A short list, each with a "fix" hint:

1. **No `mark_step()`** in the loop → the graph grows unbounded, compile time explodes. Add one per step.
2. **`tensor.item()` in the metrics print** → forces materialisation. Move printing to every N steps after `mark_step`.
3. **Dynamic shapes (variable seq lengths, last-batch-smaller)** → recompiles every shape. Pad. _[PUBLIC]_
4. **CPU `DataLoader` directly used** → host-device copy stalls. Wrap in `MpDeviceLoader`.
5. **Optimizer step outside the materialised graph** → split graphs. Keep `opt.step()` _before_ `mark_step()`.
6. **Custom CUDA ops** → not portable; you need a PyTorch op with an XLA lowering. Replace or rewrite.
7. **In-place ops on shared tensors** → can defeat fusion; benchmark before optimising.

---

## 10. Running shape: CPU vs TPU

| Snippet | CPU laptop | Colab TPU | TPU VM |
| --- | --- | --- | --- |
| §2 device print | only with `torch_xla` installed and an `xla:0` available; uncommon on plain CPU | yes | yes |
| §3 lazy-tensor demo | partial (`torch_xla` on CPU works but is academic) | yes | yes |
| §4 training step | yes (CPU-only training will be slow) | yes | yes |
| §5 `MpDeviceLoader` | works but pointless | yes | yes |
| §6 `xmp.spawn` | single-process only | usually single process | full multi-host |
| §7 SPMD mesh | requires multi-device; use TPU | yes (small slice) | yes |

The simplest "I want to learn PyTorch/XLA without paying" path is **Colab connected to a TPU runtime**. That gives you a small free TPU for short experiments.

---

## 11. Mapping to the lab's simulator

Even though `torch_xla` doesn't itself appear in this lab's `src/`, the same pieces show up:

- The lazy-tensor → HLO step ≈ `src/xla_sim/lowering.py` (model → HLO).
- The compiled executable cached on first run ≈ `src/xla_sim/compile_cache.py`.
- The `mark_step` boundary ≈ "one trace step" in `src/pjrt_sim/runtime.py`.
- The sharding API ≈ `src/sharding/mesh.py`.
- The `MpDeviceLoader` overlap ≈ `src/input_pipeline/prefetch_sim.py`.

So your mental model carries over: graph → compile → execute → measure, with the input pipeline kept full by an async loader.

---

## 12. Cross-references

- [`docs/03_xla_pjrt_runtime.md`](03_xla_pjrt_runtime.md) — XLA + PJRT, the shared backbone.
- [`docs/04_jax_on_tpu.md`](04_jax_on_tpu.md) — JAX side-by-side.
- [`docs/07_sharding_and_spmd.md`](07_sharding_and_spmd.md) — Mesh + PartitionSpec.
- [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md) — TensorBoard / Cloud TPU profiler.

Code in this repo:

- `src/sharding/mesh.py`
- `src/input_pipeline/prefetch_sim.py`
- `src/pjrt_sim/runtime.py`

Official:

- PyTorch/XLA user guide: <https://cloud.google.com/tpu/docs/pytorch-xla-users-guide>
- PyTorch/XLA on GitHub: <https://github.com/pytorch/xla>
- SPMD docs: <https://github.com/pytorch/xla/blob/master/docs/spmd.md>

---

## 13. Exercises

1. **Spot the missing `mark_step`.** Take any small PyTorch training script you have. Port it to `xla_device()`. Run it once **without** `xm.mark_step()` in the loop. Run it again **with** `mark_step()` after `opt.step()`. Compare step-time stability.

2. **Find the stray materialisation.** In a `for batch in loader:` loop, log `loss.item()` every step. Then change to `if step % 50 == 0: print(loss.item())`. Measure throughput before/after. Why does the first version stall?

3. **MpDeviceLoader vs DataLoader.** With a synthetic dataset (just `torch.randn(...)` per index), compare wall-clock for 100 steps using a plain `DataLoader` versus `MpDeviceLoader(loader, device)`. Where does the gap come from?

4. **SPMD sketch.** Without running it, write the call sequence you would use to take an existing single-device PyTorch script and shard the model across 4 chips along a `"data"` axis using `torch_xla.distributed.spmd`. List every line that has to change.
