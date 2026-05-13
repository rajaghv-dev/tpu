# 01 - Cloud TPU Versions: v4, v5e, v5p, v6e

> **Learning goal:** for each of the four Cloud TPU generations covered in this lab (v4, v5e, v5p, v6e/Trillium), be able to state — without inventing numbers — the intended workload, HBM and ICI characteristics, software family, cost positioning, and the situations in which you should and should not pick it.

Throughout this doc, every concrete number is tagged with the same provenance markers as `src/tpu_versions/cloud_tpu_catalog.py`:

- `[PUBLIC]` — published on cloud.google.com/tpu
- `[DOC]` — documented in Google whitepapers or Cloud blog
- `[SIM]` — value used by the simulator (best-effort, not authoritative)
- `[INFER]` — reasonable inference, clearly labeled
- `[UNKNOWN]` — proprietary or not publicly disclosed

> Every spec page below is a learning surface, not a procurement quote. Always check the official version page before deciding on a deployment. URLs are in each section.

---

## 1. How to read this doc

For each generation:

1. **One-line positioning** — Google's own framing.
2. **Catalog values** — pulled from `src/tpu_versions/cloud_tpu_catalog.py` with source markers.
3. **Software family** — which frameworks are first-class.
4. **Use cases** — what it's best at.
5. **Cost positioning** — relative, not absolute. Never trust this for billing.
6. **Known limitations.**
7. **When to choose / not.**

The catalog is the single source of truth in this repo. If a number disagrees with this doc, the catalog wins, because that's the file the simulator reads.

---

## 2. TPU v4

**Positioning:** Google's first widely available "training generation" with HBM2, 3D-torus topology, optimised for very large language model training. _[PUBLIC]_

### Catalog values

From `src/tpu_versions/cloud_tpu_catalog.py`, with the same markers the file declares:

| Field | Value | Source |
| --- | --- | --- |
| `hbm_per_chip_gb` | 32.0 | `[PUBLIC]` |
| `hbm_bandwidth_gbps` | 1200.0 | `[PUBLIC]` |
| `peak_bf16_tflops` | 275.0 | `[PUBLIC]` |
| `ici_bandwidth_gbps` | 270.0 | `[DOC]` |
| `chips_per_host` | 4 | `[PUBLIC]` |
| `typical_slice_shapes` | `(2,2,1)`, `(2,2,2)`, `(4,4,4)` | `[PUBLIC]` |

> Catalog note: _"3D torus topology; best for very large training; HBM = HBM2."_

### Software family

- JAX — first-class _[PUBLIC]_
- PyTorch / XLA — supported _[PUBLIC]_
- TensorFlow with `TPUStrategy` — supported _[PUBLIC]_

### Use cases where v4 fits

- Very large dense transformers (PaLM-class training was famously on v4 _[DOC]_).
- Workloads where you want a 3D-torus slice shape and don't care about Trillium-only features (SparseCore on v6e, fp8 path on v5p/v6e).

### Limitations

- HBM2, not HBM3 — capacity and bandwidth lower than v5p and v6e.
- Older generation; capacity is increasingly being shifted toward v5p / v6e _[INFER]_ (verify on <https://cloud.google.com/tpu/docs/regions-zones>).
- No SparseCore (not publicly available on v4) _[INFER]_.

### When to choose / not

- **Choose** when v4 is the only one your region/quota offers and you have an established v4 codebase.
- **Don't choose** for new projects in 2025+ if v5p or v6e capacity is available — newer generations have higher per-chip and per-dollar throughput _[INFER, verify against pricing page]_.

Reference: <https://cloud.google.com/tpu/docs/v4>

---

## 3. TPU v5e

**Positioning:** _"Cost-optimised for inference and medium-scale training."_ Google's own framing. _[PUBLIC]_ <https://cloud.google.com/tpu/docs/v5e>

### Catalog values

| Field | Value | Source |
| --- | --- | --- |
| `hbm_per_chip_gb` | 16.0 | `[PUBLIC]` |
| `hbm_bandwidth_gbps` | 820.0 | `[PUBLIC]` |
| `peak_bf16_tflops` | 197.0 | `[PUBLIC]` |
| `ici_bandwidth_gbps` | 200.0 | `[DOC]` |
| `chips_per_host` | 4 | `[PUBLIC]` |
| `typical_slice_shapes` | `(1,1)`, `(2,2)`, `(4,4)`, `(8,8)` | `[PUBLIC]` |

> Catalog note: _"2D torus; cost-optimised for inference + medium training."_

### Software family

Same JAX / PyTorch-XLA / TF support. Saxml and JetStream are commonly used for serving _[PUBLIC]_ (search "JetStream TPU v5e" on cloud.google.com).

### Use cases where v5e shines

- LLM **inference** — its cost/perf for serving is the headline. _[PUBLIC]_
- Fine-tuning models in the 1B–8B range.
- Cost-sensitive training of small/medium dense models.

### Limitations

- 2D torus, not 3D — large-scale training topologies are less flexible than v5p.
- HBM = 16 GB / chip. Half of v4, one-sixth of v5p. _[PUBLIC]_ This is the single biggest constraint: large models will require aggressive sharding or simply won't fit.
- Lower ICI bandwidth than v5p, so communication-bound workloads (model parallel, MoE) suffer relatively.

### When to choose / not

- **Choose** for inference servers, for small/medium fine-tunes, and for tight-budget training of models that fit in 16 GB × chip_count of HBM.
- **Don't choose** for from-scratch pre-training of models above the low-billion-param range, or for workloads with heavy all-reduce / all-to-all.

The simulator's bottleneck rule reflects this — if collectives exceed 30 % of step time on a multi-chip run, the report suggests moving from v5e to v5p (see `src/profiling/bottleneck_report.py`, in the "Collective communication" block).

---

## 4. TPU v5p

**Positioning:** _"Performance-optimised"_ — Google's flagship training chip in the v5 generation. _[PUBLIC]_ <https://cloud.google.com/tpu/docs/v5p>

### Catalog values

| Field | Value | Source |
| --- | --- | --- |
| `hbm_per_chip_gb` | 96.0 | `[PUBLIC]` |
| `hbm_bandwidth_gbps` | 2765.0 | `[PUBLIC]` |
| `peak_bf16_tflops` | 459.0 | `[PUBLIC]` |
| `ici_bandwidth_gbps` | 600.0 | `[DOC]` |
| `chips_per_host` | 4 | `[PUBLIC]` |
| `typical_slice_shapes` | `(2,2,1)`, `(4,4,4)`, `(8,8,8)` | `[PUBLIC]` |

> Catalog note: _"3D torus; performance-optimised; HBM = HBM3."_

### Software family

Same trio (JAX / PyTorch-XLA / TF). v5p is the chip where you most often see large MaxText / T5x / Praxis training runs in published Cloud blog posts _[DOC]_.

### Use cases

- Large-scale pre-training of dense transformers (LLMs, multimodal).
- MoE training where ICI bandwidth matters.
- Workloads needing 96 GB / chip of HBM — the largest in the lab's catalog.

### Limitations

- Most expensive per-chip-hour of the four generations _[INFER, check pricing page]_.
- Capacity is concentrated in specific regions _[PUBLIC]_, so you may need to plan around `regions-zones`.

### When to choose / not

- **Choose** when you are budgeted to do real pre-training, or fine-tunes that need >16 GB/chip of HBM.
- **Don't choose** for inference-serving — v5e or v6e will give you better $/throughput.

Reference: <https://cloud.google.com/tpu/docs/v5p>

---

## 5. TPU v6e (Trillium)

**Positioning:** Trillium — Google's "next-gen" v6 entry-class. _"~4× peak vs v5e"_ per Google's own framing in the catalog. _[DOC]_ <https://cloud.google.com/tpu/docs/v6e>

### Catalog values

| Field | Value | Source |
| --- | --- | --- |
| `hbm_per_chip_gb` | 32.0 | `[PUBLIC]` |
| `hbm_bandwidth_gbps` | 1640.0 | `[PUBLIC]` |
| `peak_bf16_tflops` | 918.0 | `[PUBLIC]` |
| `ici_bandwidth_gbps` | 800.0 | `[DOC]` |
| `chips_per_host` | 8 | `[PUBLIC]` |
| `typical_slice_shapes` | `(1,1)`, `(2,4)`, `(4,4)`, `(8,8)`, `(16,16)` | `[PUBLIC]` |

> Catalog note: _"Trillium generation; ~4× peak vs v5e; SparseCore available."_

Note `chips_per_host = 8` — twice as many as v5e/v5p/v4. This matters: a single host is now a more interesting parallelism unit. _[PUBLIC]_

### Software family

Same trio. JAX support is the most mature for new generations _[INFER]_; PyTorch/XLA support is improving — check current release notes.

### Use cases

- The default "next project" choice in 2025+ for both training and inference _[INFER]_.
- Recommendation systems with large embedding tables — **SparseCore** is available on v6e. _[PUBLIC]_
- Workloads that previously fit v5e but want more throughput per chip.

### Limitations

- HBM per chip is 32 GB — same as v4, only 1/3 of v5p. _[PUBLIC]_ For massive single-host models, v5p still wins on memory.
- Newer means software ecosystem is still maturing — some libraries may lag behind v5p support. _[INFER, verify in release notes.]_

### When to choose / not

- **Choose** for new projects where v6e capacity is available in your region.
- **Choose** for recommender / embedding-heavy workloads thanks to SparseCore.
- **Don't choose** if your code is hard-pinned to a specific older JAX or PyTorch-XLA wheel; verify version matrix first.

Reference: <https://cloud.google.com/tpu/docs/v6e>

---

## 6. Side-by-side comparison

A consolidated view of the catalog. Every number's marker is identical to what the code reports — see `src/tpu_versions/cloud_tpu_catalog.py`.

| Spec | v4 | v5e | v5p | v6e |
| --- | --- | --- | --- | --- |
| HBM per chip (GB) `[PUBLIC]` | 32.0 | 16.0 | 96.0 | 32.0 |
| HBM bandwidth (GB/s) `[PUBLIC]` | 1200 | 820 | 2765 | 1640 |
| Peak bf16 TFLOPS `[PUBLIC]` | 275 | 197 | 459 | 918 |
| ICI bandwidth (GB/s) `[DOC]` | 270 | 200 | 600 | 800 |
| chips_per_host `[PUBLIC]` | 4 | 4 | 4 | 8 |
| Topology | 3D torus | 2D torus | 3D torus | 2D-ish (see slice shapes) |
| HBM generation `[DOC, catalog note]` | HBM2 | (see public docs) | HBM3 | (see public docs) |
| SparseCore `[PUBLIC]` | no | no | yes | yes |

What we do **not** list, on purpose:

- Hourly USD price — **never hardcoded.** Look up at <https://cloud.google.com/tpu/pricing>.
- Per-chip TDP, sustained vs peak frequency, register file sizes, scratchpad sizes, MXU dimensions — these are either `[UNKNOWN]` (not publicly disclosed) or covered by official whitepapers but outside this lab's scope. We refuse to invent them.

---

## 7. Reading the catalog programmatically

```python
from cloud_tpu_lab.src.tpu_versions.cloud_tpu_catalog import (
    get_spec, list_versions, list_specs,
)

for v in list_versions():
    spec = get_spec(v)
    print(
        f"{v:5s}  hbm={spec.hbm_per_chip_gb:5.1f} GB  "
        f"bf16={spec.peak_bf16_tflops:6.1f} TFLOPS  "
        f"ici={spec.ici_bandwidth_gbps:5.1f} GB/s"
    )
```

This is exactly the call pattern the demo uses with `--show-versions`. Every value's `sources` dict tells you the provenance marker.

---

## 8. Choosing in three questions

A pragmatic decision tree for "which version?". Answer top-down.

1. **Do I need >16 GB HBM per chip after sharding?**
   - Yes → v4, v5p, or v6e (v5e is out).
   - No → v5e is in play.
2. **Is my workload communication-bound (lots of all-reduce / all-to-all)?**
   - Yes → favour v5p (highest ICI in the catalog) or v6e.
   - No → cost-driven choice.
3. **Do I need SparseCore for large embedding tables?**
   - Yes → v6e (or v5p, per public docs).
   - No → any.

Then resolve with availability (regions-zones) and price (pricing page).

---

## 9. Cross-references

- [`docs/00_big_picture.md`](00_big_picture.md) — workload fit matrix.
- [`docs/02_cloud_tpu_architecture.md`](02_cloud_tpu_architecture.md) — what HBM / ICI / MXU actually mean.
- [`docs/07_sharding_and_spmd.md`](07_sharding_and_spmd.md) — the relationship between ICI and collective cost.
- [`docs/09_cost_performance_methodology.md`](09_cost_performance_methodology.md) — how to convert chip-hours into USD honestly.

Official Cloud TPU version pages:

- v4: <https://cloud.google.com/tpu/docs/v4>
- v5e: <https://cloud.google.com/tpu/docs/v5e>
- v5p: <https://cloud.google.com/tpu/docs/v5p>
- v6e: <https://cloud.google.com/tpu/docs/v6e>
- System architecture: <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm>
- Regions and zones: <https://cloud.google.com/tpu/docs/regions-zones>
- Pricing: <https://cloud.google.com/tpu/pricing>

---

## 10. Exercises

1. **Catalog round-trip.**
   ```python
   from cloud_tpu_lab.src.tpu_versions.cloud_tpu_catalog import get_spec
   for v in ("v4", "v5e", "v5p", "v6e"):
       s = get_spec(v)
       arith_intensity = s.peak_bf16_tflops * 1e12 / (s.hbm_bandwidth_gbps * 1e9)
       print(v, "arithmetic intensity (FLOP/byte at roofline):", round(arith_intensity, 2))
   ```
   Which generation needs the highest arithmetic intensity (FLOPs per byte) to saturate the chip from HBM alone? Connect this back to why kernel fusion matters.

2. **Pick a chip for three workloads.** For each, give a one-sentence answer and one Google-doc URL backing it up:
   - Serving a 7B chat model at low latency, 100 req/s.
   - Pre-training a 30B dense transformer on 1T tokens.
   - Training a recommender with a 500 M-row embedding table.

3. **Find the catalog's `[DOC]` fields and decide which would you most like upgraded to `[PUBLIC]`.** Why? (Hint: look at the simulator and see which one most strongly drives a downstream finding in `bottleneck_report.py`.)

4. **Compile-time prediction.** Without running anything, predict whether a workload with effective batch = 1024, seq = 4096, d_model = 4096 fits in v5e HBM per chip. Show your arithmetic; flag every assumption.
