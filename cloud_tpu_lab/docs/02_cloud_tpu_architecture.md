# 02 - Cloud TPU Architecture: Chip, Host, Slice, Pod

> **Learning goal:** build an accurate, honest mental model of how a Cloud TPU is physically organised — from the MXU inside the chip all the way up to a pod — and how the three interconnect tiers (ICI, PCIe, DCN) drive both performance and price. Unknowns are marked unknown; we do not invent.

Every concrete number in this doc is sourced from the catalog at `src/tpu_versions/cloud_tpu_catalog.py`, or carries a `[PUBLIC]` / `[DOC]` / `[INFER]` / `[UNKNOWN]` marker.

---

## 1. The hierarchy

```
+---------------------------------------------------------------+
|  POD       (full ICI-connected fabric in a zone)              |
|  +---------------------------------------------------------+  |
|  |  SLICE   (a contiguous sub-rectangle of a pod)          |  |
|  |  +---------------------------------------------------+  |  |
|  |  |  HOST   (a VM that owns N chips over PCIe)        |  |  |
|  |  |  +-----------------------------------------------+ |  |  |
|  |  |  |  CHIP   (one ASIC, HBM + TensorCore + MXU)    | |  |  |
|  |  |  +-----------------------------------------------+ |  |  |
|  |  +---------------------------------------------------+ |  |
|  +---------------------------------------------------------+ |
+---------------------------------------------------------------+
```

The unit you write code for is **the slice**. The unit you pay for is **the slice**. The unit Google schedules is **the slice**. Remembering that saves you from a lot of confused conversations.

---

## 2. Chip — the smallest publicly named unit

A Cloud TPU chip contains, at minimum (per Google's public docs):

- **One or more TensorCores.** _[PUBLIC]_ For v4 _[PUBLIC]_ and v5p _[PUBLIC]_ each chip exposes 2 TensorCores; on v5e and v6e the chip exposes 1. See <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm>. The exact count by version may evolve — verify on the version page.
- **Matrix Multiplier Unit(s) — MXU.** _[PUBLIC]_ A systolic array specialised for matmul.
- **High-Bandwidth Memory — HBM.** _[PUBLIC]_ The on-package DRAM the chip reads tensors from.
- **Vector and scalar units.** _[PUBLIC]_ For elementwise and reduction ops that don't fit the MXU.
- **(v5p, v6e) SparseCore** for large embedding-table lookups. _[PUBLIC]_

What we **do not** assert in this doc, because the values are either proprietary or not consistently public:

- MXU width / height in lanes — _[UNKNOWN per Google's public docs in general; see system architecture page for the values that are published.]_
- Register file sizes — _[UNKNOWN]_
- Warp / wavefront scheduling details — `TPUs don't use the warp model; this is a GPU concept. Don't import it.`
- L1/L2 cache sizes — _[UNKNOWN]_ for most generations.

If you find yourself reading a confident blog post quoting these, check the citation. If it's not from Google's own docs or a peer-reviewed paper, treat it as folklore.

### MXU concept (without inventing dimensions)

A systolic MXU is conceptually: a fixed-size 2D mesh of multiply-accumulate cells through which data flows in a wavefront pattern. You feed it two operands (A, B) and it produces a partial output that streams through a column of accumulators. _[DOC]_ See the original TPU paper (Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit", 2017) for the canonical description.

What you need to know as a user:

1. **The MXU is the throughput engine.** Peak bf16 TFLOPS (catalog values) is dominated by the MXU. Hitting roofline means feeding the MXU big enough matmuls.
2. **Operand shapes must align with the MXU's native dimensions for peak throughput.** XLA handles this for you, but tiny matmuls (`m, n, k < 128` say) leave throughput on the table _[INFER]_.
3. **Non-matmul ops** (softmax, layernorm, elementwise) run on the vector unit. They consume HBM bandwidth but very few of them saturate the MXU.

### HBM — capacity and bandwidth

From the catalog, the per-chip HBM values:

| Version | HBM/chip (GB) | HBM bandwidth (GB/s) | Source markers |
| --- | --- | --- | --- |
| v4 | 32.0 | 1200 | `[PUBLIC]` |
| v5e | 16.0 | 820 | `[PUBLIC]` |
| v5p | 96.0 | 2765 | `[PUBLIC]` |
| v6e | 32.0 | 1640 | `[PUBLIC]` |

A useful derived quantity is the **arithmetic intensity** at which the chip is exactly HBM-bound:

```
arith_intensity_break_even = peak_FLOPS / HBM_bandwidth
```

For v5p: `459 TFLOPS / 2765 GB/s ≈ 166 FLOP / byte`. Any op with arithmetic intensity below that is HBM-bound; above it is compute-bound. This is the simulator's roofline model in `src/pjrt_sim/device.py`.

---

## 3. Host — the VM you SSH into

A **host** in TPU VM mode is a single Linux VM that owns `chips_per_host` chips, attached via **PCIe**. _[PUBLIC]_ See <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm>.

From the catalog:

| Version | chips_per_host |
| --- | --- |
| v4 | 4 |
| v5e | 4 |
| v5p | 4 |
| v6e | 8 |

This is where Python, JAX/Torch-XLA/TF, and your data pipeline live. The host has its own CPU cores and RAM (much smaller than HBM, but vastly more than the typical Python process needs).

### PCIe — the host↔chip bus

PCIe is **the slow path**. It connects the host CPU and chips, and is used for:

- Loading initial weights from CPU memory or disk into HBM.
- Streaming input tensors per step (the dataloader → device path).
- Pulling out metrics / loss values.

PCIe bandwidth on a host is **orders of magnitude lower than HBM bandwidth and lower than ICI bandwidth**. _[INFER per the system architecture page; exact PCIe Gen and lane count varies by host generation, partially [UNKNOWN].]_

This is why input pipeline starvation is a real problem on TPU. If you need to push N GB/sec of fresh tensor data per step, your dataloader pipeline must keep up. See the simulator's prefetch model at `src/input_pipeline/prefetch_sim.py` and the bottleneck rule at `src/profiling/bottleneck_report.py` (the `"Input pipeline"` block fires above 10 % of step time).

### Multi-host slices

Once a slice has more than `chips_per_host` chips, you have **multiple hosts**. Each host's Python process runs as one peer in a SPMD program. The framework (JAX especially) handles coordination — you write code as if it were one big device, and under the hood it is N processes talking over ICI.

---

## 4. Slice — the unit you actually rent

A **slice** is a contiguous rectangle (2D) or cuboid (3D) of chips inside a pod. From the catalog:

| Version | Typical slice shapes | Topology |
| --- | --- | --- |
| v4 | `(2,2,1)`, `(2,2,2)`, `(4,4,4)` | 3D torus |
| v5e | `(1,1)`, `(2,2)`, `(4,4)`, `(8,8)` | 2D torus |
| v5p | `(2,2,1)`, `(4,4,4)`, `(8,8,8)` | 3D torus |
| v6e | `(1,1)`, `(2,4)`, `(4,4)`, `(8,8)`, `(16,16)` | 2D-ish |

The convention `vX-N` means "N chips of vX". A `v5e-16` is 16 chips arranged as `(4,4)`. _[PUBLIC]_

### ICI — the fast interconnect

ICI is the high-bandwidth interconnect linking chips _inside_ a slice. It is a **torus** — each chip has bidirectional links to its neighbours on each axis. From the catalog:

| Version | ICI bandwidth (GB/s, per-link aggregate) | Source |
| --- | --- | --- |
| v4 | 270 | `[DOC]` |
| v5e | 200 | `[DOC]` |
| v5p | 600 | `[DOC]` |
| v6e | 800 | `[DOC]` |

> The catalog tags these as `[DOC]` rather than `[PUBLIC]` because the per-link vs aggregate framing varies across Google's communications. **Use these for relative comparison, not for capacity planning** — for that, run a real bandwidth probe on the actual slice.

ICI is what makes collectives like all-reduce, all-gather, reduce-scatter cheap inside a slice. It is also what fundamentally limits how big a model-parallel dimension can be before communication eats your throughput. The simulator at `src/sharding/all_reduce.py` models collective time as `tensor_bytes / ici_bandwidth` plus a fixed per-collective overhead — coarse, but enough to teach the shape of the tradeoff.

---

## 5. Pod — the fabric you never directly rent

A **pod** is the entire ICI-connected fabric of one TPU generation in one zone. You normally do not rent "a pod"; you rent a slice of it. Google publishes the maximum slice sizes per generation _[PUBLIC]_ on the version pages.

What matters for your mental model:

- Within a slice, communication is **ICI**.
- Across slices (multi-slice training), communication is **DCN** — data-center network, much slower per byte than ICI.

### DCN — the slowest interconnect tier

DCN is standard Ethernet-class data-center networking. _[INFER]_ exact per-VM throughput varies by host SKU and zone. It is used for:

- Loading data from GCS into the host.
- Multi-slice training (large jobs that span multiple slices in the same region).
- Out-of-band coordination (saving checkpoints, telemetry).

Multi-slice training works (see `MaxText` examples published by Google, _[DOC]_), but you must architect for it — large all-reduces _between_ slices are expensive. The usual pattern is **data parallel across slices, model parallel within a slice**, so DCN handles only gradient sync.

---

## 6. Three tiers of bandwidth — the picture to memorise

A practitioner's mental model:

```
HBM   >>   ICI   >>   PCIe   >   DCN
```

Order-of-magnitude rules of thumb (drawn from the catalog and from public docs):

- **HBM bandwidth** is hundreds-to-thousands of GB/s per chip. `[PUBLIC]`
- **ICI bandwidth** is hundreds of GB/s per chip, inside a slice. `[DOC]`
- **PCIe bandwidth** is tens of GB/s per host. `[INFER; verify by hardware generation]`
- **DCN bandwidth** is single-digit-to-tens of GB/s per VM. `[INFER]`

Every performance problem you will diagnose maps to one of those tiers being the bottleneck. The bottleneck report in `src/profiling/bottleneck_report.py` is organised exactly along these lines: `input_pipeline` (PCIe / DCN), `collective` (ICI), `hbm` (HBM capacity), `host` (CPU / Python), `xla` (compile).

---

## 7. Tensor layout, briefly

XLA assigns a physical layout to every tensor. _[PUBLIC]_ A "layout" specifies which logical dimension is the minor-most (innermost in memory). For maximal MXU throughput, the contracting dimensions of a matmul need to be laid out such that the MXU can stream them efficiently.

You usually do not pick layouts manually. XLA does layout assignment for you (see [`docs/03_xla_pjrt_runtime.md`](03_xla_pjrt_runtime.md)). What you do control:

- **Tensor shapes.** Tiny tensors are wasteful. Multiples of 128 along the contracting dim are a good rule of thumb _[INFER per common XLA-on-TPU guidance; verify with profiling]_.
- **Dtype.** bf16 is the default fast path. fp32 works but uses 2× HBM and is slower _[DOC]_. fp8 is available on newer generations _[PUBLIC, version-specific]_.
- **Sharding.** What you express with `PartitionSpec` (see [`docs/07_sharding_and_spmd.md`](07_sharding_and_spmd.md)) constrains layout.

---

## 8. What we deliberately leave unspecified

The lab refuses to invent. Specifically, you will **not** see in this repo:

- Quoted MXU lane counts unless they are public on Google's TPU page.
- Per-generation TDP, sustained-vs-peak clocks, register sizes.
- "Warps" or any GPU-borrowed schedule abstraction.
- Made-up cache line sizes.
- Pricing as a defaulted constant.

If a downstream consumer wants those, they belong in a separate, version-pinned "hardware reference" doc maintained against current Google sources. For learning, the publicly available shape (MXU + HBM + ICI + slice shape) is enough.

---

## 9. Worked example: where does the time go?

Pick a v5e-16 slice running a single dense matmul `[1024, 4096] @ [4096, 4096]` in bf16:

- Output shape: `[1024, 4096]`.
- FLOPS: `2 * 1024 * 4096 * 4096 ≈ 3.43e10`.
- Bytes moved (read both operands + write out, bf16): `(1024*4096 + 4096*4096 + 1024*4096) * 2 ≈ 4.2e7 B`.
- Arithmetic intensity: `3.43e10 / 4.2e7 ≈ 817 FLOP/B`. Above v5e's break-even (`197 TF / 820 GB/s ≈ 240 FLOP/B`), so this matmul is compute-bound. _[Computation; uses [PUBLIC] catalog values.]_
- Roofline time: `3.43e10 / 197e12 ≈ 0.17 ms` of pure MXU work. Real measured time would be slower due to dispatch overhead, but this is the order of magnitude.

This calculation is exactly what `src/pjrt_sim/device.py:roofline_op_time_s` does for every op in the fake HLO module.

---

## 10. Cross-references

- [`docs/00_big_picture.md`](00_big_picture.md) — when to use a TPU at all.
- [`docs/01_cloud_tpu_versions.md`](01_cloud_tpu_versions.md) — per-version specs.
- [`docs/03_xla_pjrt_runtime.md`](03_xla_pjrt_runtime.md) — how XLA assigns layout.
- [`docs/07_sharding_and_spmd.md`](07_sharding_and_spmd.md) — how slice shape constrains sharding.
- [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md) — diagnosing which tier is bottlenecking.

Code in this repo:

- `src/tpu_versions/cloud_tpu_catalog.py` — the catalog values cited above.
- `src/pjrt_sim/device.py` — the roofline model.
- `src/sharding/all_reduce.py` — ICI-based collective timing.
- `src/profiling/bottleneck_report.py` — the bottleneck classifier.

Official docs:

- System architecture: <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm>
- v4 / v5e / v5p / v6e: see [`docs/01_cloud_tpu_versions.md`](01_cloud_tpu_versions.md) for direct URLs.
- The original TPU paper: <https://arxiv.org/abs/1704.04760> (v1, but the systolic-MXU concept is unchanged).

---

## 11. Exercises

1. **Break-even arithmetic intensity.** For each version in the catalog, compute `peak_TFLOPS * 1e12 / (hbm_bandwidth_gbps * 1e9)`. Order versions by that number. Which is "most compute-bound at roofline"?

2. **Slice shape inventory.** From the catalog's `typical_slice_shapes`, compute `n_chips` for each tuple by product, then group by version. Which generation has the widest range of slice sizes available in the catalog? Cross-check against the official version page URLs from doc 01.

3. **Bandwidth budget.** A training step ships 4 GB of activations + 8 GB of gradients in collectives. Estimate, on v5p (`[DOC]` ICI = 600 GB/s), the minimum communication-only time. Repeat for v5e (`[DOC]` ICI = 200 GB/s). What does that tell you about which chip to choose for a communication-bound workload?

4. **Spot the made-up number.** Find one blog post on the open web that quotes an MXU dimension or register-file size for v5p. Trace its citation. If you cannot reach a Google source, what should you label that number in your own notes?
