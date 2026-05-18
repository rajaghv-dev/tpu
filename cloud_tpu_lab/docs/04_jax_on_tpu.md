> **Note:** this doc predates the real-TPU pivot. References to `src/xla_sim/`, `src/pjrt_sim/`, `src/sharding/`, `src/memory/`, `src/input_pipeline/`, and `examples/run_cpu_simulation_demo.py` are historical — those modules were removed. The TPU architecture / XLA / observability concepts below are still accurate. Current run flow lives in [README.md](../README.md) and [16_runbook_real_tpu.md](16_runbook_real_tpu.md).

# 04 - JAX on Cloud TPU

> **Learning goal:** build a working knowledge of the JAX APIs that matter on Cloud TPU — `jax.jit`, `jax.device_put`, `jax.sharding.Mesh`, `NamedSharding`, `PartitionSpec`, the transition from `pjit` to `jit`, and the (mostly historical) `pmap`. Identify which snippets run on CPU only, on a real TPU VM, or both.

This doc focuses on JAX. PyTorch/XLA and TF have their own docs in this series.

---

## 1. Why JAX is the natural fit for TPU

JAX was built around XLA from day one. Every primitive in JAX has an HLO lowering, and `jax.jit` traces to HLO directly. This means:

- Less impedance mismatch than PyTorch's eager-by-default model.
- Sharding is first-class via `jax.sharding`.
- Multi-host orchestration is handled by `jax.distributed`.

What this does **not** mean:

- JAX is not "the only way" to use TPU. PyTorch/XLA and TF are first-class on Cloud TPU per Google's official docs. _[PUBLIC]_

Reference: <https://cloud.google.com/tpu/docs/jax-pods>, <https://jax.readthedocs.io/>

---

## 2. The minimal sanity check

This snippet runs on **any** environment — CPU only, Colab TPU, or a TPU VM. `jax.devices()` reflects whatever backend JAX detected.

```python
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("local_device_count:", jax.local_device_count())
```

- On a clean CPU laptop: `backend: cpu`, `devices: [CpuDevice(id=0)]`.
- On Colab connected to a TPU runtime: `backend: tpu`, `devices: [TpuDevice(...)...]`. _[PUBLIC]_
- On a TPU VM: similar to Colab, but with the full slice's chips visible.

**Always run this first** when something looks off. Half of "JAX is slow on TPU" stories turn out to be JAX silently running on CPU.

---

## 3. `jax.jit` — what it actually does

`jax.jit(f)` returns a JIT-compiled version of `f`. The first call traces, compiles, caches, and executes. Subsequent calls with the same input shapes / dtypes hit the cache.

```python
import jax
import jax.numpy as jnp

@jax.jit
def step(params, x):
    return params @ x

# CPU OK; TPU OK.
params = jnp.ones((128, 128))
x = jnp.ones((128, 1))
y = step(params, x)
print(y.shape)
```

Things that surprise new users:

1. **The first call is slow.** Compile time can dominate small workloads. Don't benchmark the first call.
2. **Tracers, not arrays.** Inside the function, `params` and `x` are tracer objects. Printing them shows abstract shapes, not values. This is by design.
3. **Side effects don't survive.** A `print(params[0, 0])` inside the jit'd function will print during tracing and never again.

### `static_argnums` and `static_argnames`

If a parameter affects the graph shape (a Python `int` for `n_heads`, say), declare it static so it becomes part of the cache key:

```python
@jax.jit(static_argnames=("n_heads",))
def attn(x, n_heads):
    ...
```

Each new value of `n_heads` recompiles. That is the price of correctness — better than a wrong-shape kernel.

---

## 4. `jax.device_put` — where does a tensor live?

`jax.device_put(x, device)` moves a host array onto a device (or onto a specific sharding).

```python
import jax
import jax.numpy as jnp

dev = jax.devices()[0]
x = jnp.ones((1024, 1024))
x_on_device = jax.device_put(x, dev)
print(x_on_device.device())  # CPU or TPU device
```

On a real TPU VM, this triggers an HBM allocation + a PCIe DMA from host RAM. _[PUBLIC]_

On CPU, it's a no-op-ish copy that lives in the local CPU "device".

Patterns:

- **Don't `device_put` inside the training loop** for the same tensor every step. Put once, reuse.
- **Use it to control initial sharding** when you have multiple devices (see below).

---

## 5. `jax.sharding.Mesh` and `PartitionSpec`

This is the modern API for SPMD parallelism in JAX. _[PUBLIC]_ See <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>.

A `Mesh` is an N-D arrangement of devices with **named axes**. A `PartitionSpec` says, per tensor dim, which mesh axis to shard along (or `None` for replicated).

This lab's analogue is in `src/sharding/mesh.py`:

```python
@dataclass(frozen=True)
class Mesh:
    shape: Tuple[int, ...]
    axis_names: Tuple[str, ...]
```

That dataclass is exactly the mental model. Real JAX adds the device list, but the API surface (`shape`, `axis_names`, `axis_size`) is the same.

### A 2D mesh: data × model

```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

# Real run (TPU VM, 8 chips): pretend devices are arranged as (2 data, 4 model).
devices = np.array(jax.devices()).reshape(2, 4)
mesh = Mesh(devices, axis_names=("data", "model"))

# Replicate a tensor along data, shard along model:
sharding = NamedSharding(mesh, P(None, "model"))
```

If you run this on CPU with `jax.devices() == [CpuDevice]`, the reshape will fail because you have only one device. That's the boundary: **mesh examples in this doc generally need ≥2 devices**, so they require a multi-device backend (TPU VM, GPU host, or a CPU faked into multi-device via `XLA_FLAGS="--xla_force_host_platform_device_count=8"`). _[PUBLIC]_

### `PartitionSpec` walkthrough

For a tensor `x` of shape `(B, S, D)` on a mesh `("data", "model")`:

| PartitionSpec | Meaning |
| --- | --- |
| `P(None, None, None)` | Fully replicated everywhere. |
| `P("data", None, None)` | Sharded across `data` along axis 0; replicated along `model`. |
| `P("data", None, "model")` | Sharded across `data` (axis 0) and across `model` (axis 2). |
| `P(("data", "model"), None, None)` | Sharded across **both** mesh axes along axis 0. |

The lab's `PartitionSpec` (in `src/sharding/mesh.py`) carries the same semantics:

```python
@dataclass(frozen=True)
class PartitionSpec:
    spec: Tuple[Optional[str], ...]
    def is_replicated(self) -> bool: ...
```

---

## 6. `pjit` → `jit` — the historical note

For a long time, sharded JIT was `pjit`. In modern JAX (2024+), the recommended call is plain `jax.jit` with `in_shardings` / `out_shardings` arguments. _[PUBLIC]_ See <https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html> and the migration guide.

```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Sketch only — needs >=2 devices to actually run.
mesh = Mesh(jax.devices(), ("data",))
in_sh = NamedSharding(mesh, P("data", None))
out_sh = NamedSharding(mesh, P("data", None))

@jax.jit
def step(x):
    return x * 2

# Run with named sharding:
# x = ... # produced via jax.device_put(x, in_sh)
# y = step(x)
```

If you read older JAX code, `pjit(f, in_axis_resources=..., out_axis_resources=...)` is the legacy form. It still works, but new code should use `jit` with `NamedSharding`.

---

## 7. `pmap` — old, mostly avoid

`jax.pmap` predates the modern sharding API. It is "data parallel mapped over devices" — every device runs the same function with a slice of the input along axis 0.

```python
import jax
import jax.numpy as jnp

@jax.pmap
def step(x):
    return x * 2

# CPU: requires --xla_force_host_platform_device_count=N to fake N devices.
# TPU: maps across local devices.
```

When to use `pmap` today (_[INFER]_, your mileage may vary):

- Quick scripts where you really just want data-parallel over `n_devices` and don't want to set up a mesh.
- Legacy code you're maintaining.

When to **not** use `pmap`:

- Anything model-parallel.
- Anything that needs a multi-axis mesh.
- Anything multi-host (the modern path is `jax.jit` + `Mesh` + `jax.distributed.initialize`).

The JAX team recommends migrating to `jax.jit` + sharding for new code. _[PUBLIC]_

---

## 8. `jax.distributed` for multi-host slices

On a multi-host slice, each host runs its own Python process, and they coordinate through `jax.distributed.initialize()`. _[PUBLIC]_

```python
import jax
import jax.distributed as jd

jd.initialize()  # auto-detects on TPU VM
print("process_index:", jax.process_index(),
      "/", jax.process_count())
print("global devices:", len(jax.devices()))
print("local devices:", len(jax.local_devices()))
```

Concepts:

- `jax.process_index()` — your rank.
- `jax.process_count()` — total processes.
- `jax.devices()` — all chips, all hosts.
- `jax.local_devices()` — chips on your host.

On a single-host slice (e.g. `v5e-4`), `process_count() == 1` and you can skip `initialize()`. _[PUBLIC]_

---

## 9. Running shape: CPU vs TPU snippets

A practical table of what runs where, for the snippets in this doc:

| Snippet | CPU laptop | Colab TPU runtime | TPU VM |
| --- | --- | --- | --- |
| §2 sanity check | yes (backend=cpu) | yes | yes |
| §3 `jax.jit` matmul | yes | yes | yes |
| §4 `device_put` | yes (trivial) | yes | yes |
| §5 mesh on 2 devices | only with `--xla_force_host_platform_device_count=N` | yes (small slice) | yes |
| §6 `pjit`/`jit` with sharding | same as §5 | yes | yes |
| §7 `pmap` | same as §5 | yes | yes |
| §8 `jax.distributed.initialize()` | no | (single-host: trivially yes; multi-host: no) | yes |

The "multi-device on CPU" trick:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=8" python my_script.py
```

This gives you 8 fake CPU devices, enough to exercise the mesh + sharding APIs on a laptop. The simulated step time is meaningless, but the partitioning logic is real.

---

## 10. A canonical training-step skeleton

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(jax.devices(), ("data",))
sharded = NamedSharding(mesh, P("data", None))

@jax.jit
def loss_fn(params, batch):
    x, y = batch
    pred = x @ params
    return jnp.mean((pred - y) ** 2)

grad_fn = jax.value_and_grad(loss_fn)

@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = grad_fn(params, batch)
    new_params = jax.tree.map(lambda p, g: p - 1e-3 * g, params, grads)
    return new_params, opt_state, loss
```

That's the shape every modern JAX training script has. Variants add `optax` for optimisers, `flax` for layer abstractions, and `orbax` for checkpointing — but the spine is `(params, opt_state, batch) -> (params, opt_state, loss)` inside a jit.

---

## 11. Common JAX-on-TPU mistakes

A short list, each cross-referenced to the right fix:

1. **Forgetting to jit.** Without `jit`, every op dispatches separately; XLA gets to fuse nothing. Always wrap the training step.
2. **Variable input shapes.** Every new shape recompiles. Pad. See [`docs/03_xla_pjrt_runtime.md`](03_xla_pjrt_runtime.md), §8.
3. **Side effects (printing tensors, mutating Python lists) inside jit.** Use `jax.debug.print` instead. _[PUBLIC]_
4. **Calling `.block_until_ready()` in the hot loop.** It serialises dispatch — fine for benchmarks, bad for steady-state throughput.
5. **Holding references to old tracer arrays.** Memory leaks across compiles.
6. **Confusing `pmap` with sharding.** Pick one; for new code, sharding.
7. **Treating PRNGKey as global state.** Pass it in and split it. _[PUBLIC]_

---

## 12. Cross-references

- [`docs/03_xla_pjrt_runtime.md`](03_xla_pjrt_runtime.md) — what jit traces to.
- [`docs/05_pytorch_xla_on_tpu.md`](05_pytorch_xla_on_tpu.md) — the PyTorch view of the same stack.
- [`docs/07_sharding_and_spmd.md`](07_sharding_and_spmd.md) — deeper on Mesh and PartitionSpec.
- [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md) — `jax.profiler` and recompile detection.

Code in this repo:

- `src/sharding/mesh.py` — the `Mesh` / `PartitionSpec` dataclasses mirror JAX's API.
- `src/sharding/partitioner.py` — naive partitioner.
- `src/sharding/all_reduce.py` — collective cost.
- `src/xla_sim/lowering.py` — what jit conceptually does.

Official:

- JAX: <https://jax.readthedocs.io/>
- Distributed arrays and automatic parallelization: <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>
- JAX on TPU pods: <https://cloud.google.com/tpu/docs/jax-pods>
- Persistent compile cache: <https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html>

---

## 13. Exercises

1. **Sanity-check on your machine.** Run the §2 snippet. What backend does JAX detect? If you have a TPU VM available, repeat. If not, set `XLA_FLAGS="--xla_force_host_platform_device_count=4"` and confirm `jax.local_device_count() == 4`.

2. **Force a recompile.** Write a `@jax.jit` function `f(x)` that does a matmul. Call it with shapes `(8, 8)`, `(16, 8)`, `(8, 8)` in that order. Enable `jax.config.update("jax_log_compiles", True)`. How many compiles do you see? Why?

3. **Mesh + PartitionSpec on fake CPU devices.** With `--xla_force_host_platform_device_count=8`, build a `(2, 4)` mesh with axes `("data", "model")`. Place a `(1024, 1024)` tensor with `PartitionSpec("data", "model")`. Print `tensor.sharding` and confirm both axes are sharded.

4. **Mirror the lab's `Mesh` API.** Without modifying any code, write a one-liner that constructs both a JAX `Mesh` and this lab's `Mesh` (from `src/sharding/mesh.py`) with matching shape and names. Confirm that `n_devices` matches `np.prod(jax.devices().shape)`.
