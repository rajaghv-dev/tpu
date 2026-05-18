> **Note:** this doc predates the real-TPU pivot. References to `src/xla_sim/`, `src/pjrt_sim/`, `src/sharding/`, `src/memory/`, `src/input_pipeline/`, and `examples/run_cpu_simulation_demo.py` are historical — those modules were removed. The TPU architecture / XLA / observability concepts below are still accurate. Current run flow lives in [README.md](../README.md) and [16_runbook_real_tpu.md](16_runbook_real_tpu.md).

# 00 - The Big Picture: Cloud TPU

> **Learning goal:** build an honest mental model of what a Google Cloud TPU is, where it sits in the wider ML accelerator landscape, when it is the right tool, and — equally important — when it is the wrong tool. By the end of this doc you should be able to look at a workload description and predict, without running anything, whether Cloud TPU is a sensible target.

---

## 1. What "Cloud TPU" actually means

"TPU" is overloaded. In this lab, "Cloud TPU" always refers to **Google Cloud's data-center TPU** — the ASIC family designed for dense linear algebra on bf16/fp8, exposed through Google Cloud as a managed accelerator.

This doc, and every other doc in the `docs/` series, is **strictly about Cloud TPU**. It is not about:

- Edge TPU / Coral USB / Coral dev board.
- Pixel Tensor / mobile NPUs.
- Generic "AI chips" or "NPUs" found in laptops.

Those are all separate product lines with different ISAs, compilers, and runtimes. None of them share a binary or a programming model with Cloud TPU. If you want learning material on those, look elsewhere; this lab will only confuse you about them.

### The four artefacts you actually touch

When you "use a Cloud TPU" you typically interact with four things at once:

1. **The hardware** — a TPU chip, packaged in 4-chip or 8-chip hosts, wired into a slice of a pod via the Inter-Chip Interconnect (ICI). _[PUBLIC]_
2. **A VM** — a Linux VM that owns the host's chips. The framework runs here.
3. **A framework** — JAX, PyTorch/XLA, or TensorFlow. You write code against one of these.
4. **A compiler+runtime** — XLA produces a compiled module; PJRT loads and executes it on the chip(s). You usually never see this layer directly, but every performance question eventually lands here.

The simulator in this repo models each of those four layers separately so you can poke them without paying for a real TPU. See [`src/tpu_versions/`](../src/tpu_versions/), [`src/xla_sim/`](../src/xla_sim/), and [`src/pjrt_sim/`](../src/pjrt_sim/).

---

## 2. TPU VM vs TPU Node — the architectural split that matters

Google has shipped two architectures for accessing TPUs:

| Architecture | What runs your Python | What the TPU sees | Status |
| --- | --- | --- | --- |
| **TPU Node** (legacy) | A separate CPU VM, talking over gRPC to a remote TPU "node" | Network-attached | Deprecated for most use cases _[PUBLIC]_ |
| **TPU VM** (current) | The same VM that owns the TPU chips | PCIe-attached, local | The default since 2021 _[PUBLIC]_ |

**TPU VM is the model you should learn and use.** In TPU VM you `ssh` into the VM that has the chips, and your Python process talks to the chips directly through PJRT. There is no gRPC hop. Debugging is dramatically easier: you can `strace`, `py-spy`, `nvidia-smi`-equivalent inspection (`gcloud compute tpus tpu-vm ssh` then framework-specific tooling), and you can `pip install` whatever you want.

> _[PUBLIC]_ See: <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm>

In this repo every script in `gcp/` assumes TPU VM — see `gcp/provision_tpu.sh` and `gcp/delete_tpu_vm.sh`.

---

## 3. Chip, host, slice, pod — the units you actually pay for

A Cloud TPU has a strict hierarchy. You should hold this in your head whenever someone says "TPU":

```
chip       — one ASIC, contains TensorCores + HBM stacks
host       — a single VM that owns N chips (N is version-specific)
slice      — a contiguous sub-rectangle of a pod, scheduled as one unit
pod        — the full ICI-connected fabric in a zone
```

Per-version chip-per-host counts come from the catalog at `src/tpu_versions/cloud_tpu_catalog.py`:

| Version | chips_per_host | Source marker |
| --- | --- | --- |
| v4 | 4 | _[PUBLIC]_ |
| v5e | 4 | _[PUBLIC]_ |
| v5p | 4 | _[PUBLIC]_ |
| v6e (Trillium) | 8 | _[PUBLIC]_ |

A **slice** is the unit of scheduling and pricing. You ask for, say, a `v5e-16` (16 chips of v5e) — that is a slice. You do not, in normal use, ask for a "pod" — pods are infrastructure, slices are products.

The ICI is the high-bandwidth toroidal interconnect that makes a slice behave like one big accelerator. Going off-slice means going over **DCN (Data Center Network)**, which is orders of magnitude slower. So "slice" is also the natural boundary for **fast collectives** like all-reduce.

> _[INFER]_ "Orders of magnitude slower" — ICI bandwidth on v5p is _[DOC]_ `~600 GB/s`-class per-link from the catalog; standard DCN per-VM bandwidth is single-digit-to-tens of GB/s. The exact ratio depends on link counts and slice shape. See <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm>.

---

## 4. Workload fit matrix

The most useful "should I use Cloud TPU?" tool is a small fit matrix. Each row below maps a workload class to the TPU generation that fits best, with caveats.

| Workload | Good fit | Bad fit | Notes |
| --- | --- | --- | --- |
| **Pre-training large transformer** (≥7B params) | v4 / v5p / v6e | v5e | You want HBM capacity + ICI bandwidth. v5p was Google's flagship training chip; v6e (Trillium) is its successor. _[PUBLIC]_ |
| **Fine-tuning mid-size models** (1B–7B) | v5e / v6e | n/a | v5e is cost-optimised; v6e is faster per chip. |
| **Inference / serving** | v5e / v6e | v5p | v5e is the "inference & medium training" chip per Google's positioning. _[PUBLIC]_ See <https://cloud.google.com/tpu/docs/v5e>. |
| **Recommendation models with large embedding tables** | v6e with **SparseCore** | v4 (no SparseCore) | SparseCore is a specialised unit on v5p/v6e for embedding lookups. _[PUBLIC]_ The catalog's v6e entry notes "SparseCore available." |
| **Small models (<100M)** | none — use CPU/GPU | all TPU | Compile time + ICI setup dominate; you will not amortise it. |
| **Heavy dynamic control flow / RL** | n/a | all TPU | XLA prefers static shapes. Dynamic-shape recompiles are the #1 silent killer. See [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md). |
| **Pure data-parallel CNN training** | v5e | v5p | CNNs are activation-heavy but parameter-light; v5e is cheaper per FLOP. |
| **MoE / sparse routing** | v5p / v6e | v5e | All-to-all collective bandwidth matters; the higher-ICI chips win. |

Use this matrix as a starting hypothesis, not a final answer. **Always run a small profiling job before committing budget.**

---

## 5. When NOT to use Cloud TPU

A short, honest list:

1. **Your model is small.** If a single CPU or a single GPU can finish your epoch in minutes, a TPU is administrative overhead. The compile + warm-up costs alone will dominate.
2. **You depend on operators XLA does not support, or supports only via slow fallbacks.** Custom CUDA kernels, exotic dynamic-shape ops, and Python-side control flow inside the hot loop all fight the compiler. _[DOC]_ See <https://cloud.google.com/tpu/docs/troubleshooting/known-issues>.
3. **You need NCCL-specific behaviour.** TPU collectives go through XLA; if your code is hard-wired to NCCL semantics, porting is real work.
4. **You need GPU-only libraries.** CUDA-only kernels (e.g. some Triton kernels, some custom attention variants) won't run as-is. Mainstream attention variants now have JAX/XLA equivalents, but verify before betting on it.
5. **Your debugging style is heavy in eager-mode tensor inspection.** XLA compiles whole graphs. You can drop out of jit for inspection, but if your daily workflow assumes "print every intermediate", expect friction.
6. **You can't tolerate region or quota constraints.** TPU capacity is regional and quota-gated. _[PUBLIC]_ See <https://cloud.google.com/tpu/docs/regions-zones>.

If two or more of the above apply, GPU (A100/H100/L4 on GCE, AWS, Lambda, etc.) is probably a better answer. There is no shame in that — it's a fit question, not a loyalty test.

---

## 6. A grounded mental model

A useful one-paragraph mental model:

> A Cloud TPU is a **dense-linear-algebra ASIC** with very large on-chip HBM, accessed through XLA — a whole-program compiler that needs static shapes. The natural unit is a **slice** of chips connected by a fast toroidal ICI. The framework (JAX / PyTorch-XLA / TF) lowers your model to HLO; XLA fuses and schedules it; PJRT executes it. You pay per chip-hour, regardless of how busy the chip is.

The last sentence is the one most people forget. **Idle TPU time is still billed time.** Every minute the chip waits on a slow input pipeline is paid for. The [cost model in `src/common/cost.py`](../src/common/cost.py) makes this explicit with a `utilization_adjusted_usd` column.

---

## 7. What this lab simulates

The lab is built so you can learn the stack without having a real TPU. The simulator covers:

| Real component | Lab module | What's simulated |
| --- | --- | --- |
| Cloud TPU catalog (per-version specs) | `src/tpu_versions/cloud_tpu_catalog.py` | HBM, peak FLOPS, ICI bandwidth, chips/host — values marked PUBLIC / DOC / SIM / INFER / UNKNOWN. |
| XLA lowering (model → HLO) | `src/xla_sim/lowering.py` | `Linear → DotGeneral + Add`, `Conv → Convolution`, `Attention → DotGeneral + Softmax + DotGeneral`, `LayerNorm → ReduceMean + Variance + Normalize`. |
| Compile cache | `src/xla_sim/compile_cache.py` | Cache-key shape, recompile detection. |
| PJRT runtime | `src/pjrt_sim/runtime.py` | Single-threaded synchronous executor; roofline timing per op. |
| Sharding | `src/sharding/{mesh,partitioner,all_reduce}.py` | Mesh / PartitionSpec / collective cost. |
| Profiling | `src/profiling/{profiler_trace,trace_analyzer,bottleneck_report}.py` | Chrome-trace JSON + bottleneck diagnosis. |
| Cost | `src/common/cost.py` | Step / sample / token / epoch cost, utilization-adjusted. |

The simulator is intentionally conservative and clearly marks anything not publicly documented as `UNKNOWN`. **You will not find made-up microarchitecture details in this repo.** When real numbers are not public, the code says so and uses a placeholder for teaching purposes.

---

## 8. Cross-references

- [`docs/01_cloud_tpu_versions.md`](01_cloud_tpu_versions.md) — per-version deep dive on v4, v5e, v5p, v6e.
- [`docs/02_cloud_tpu_architecture.md`](02_cloud_tpu_architecture.md) — chip / host / slice / pod and ICI vs PCIe vs DCN.
- [`docs/03_xla_pjrt_runtime.md`](03_xla_pjrt_runtime.md) — what happens from Python op to compiled executable.
- [`docs/09_cost_performance_methodology.md`](09_cost_performance_methodology.md) — the actual math behind `chip_count × step_time × hourly_rate`.

Official docs you will refer to constantly:

- System architecture: <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm>
- Pricing: <https://cloud.google.com/tpu/pricing>
- Regions and zones: <https://cloud.google.com/tpu/docs/regions-zones>
- v5e overview: <https://cloud.google.com/tpu/docs/v5e>
- v5p overview: <https://cloud.google.com/tpu/docs/v5p>
- v6e (Trillium): <https://cloud.google.com/tpu/docs/v6e>

---

## 9. Exercises

Do these before moving to doc 01. They take ~30 minutes total and do not need a real TPU.

1. **Run the end-to-end CPU simulation and read the report.**
   ```bash
   cd cloud_tpu_lab
   python3 examples/run_cpu_simulation_demo.py
   ```
   Open `artifacts/reports/run_<trace_id>.md`. Identify the dominant cost contributor.

2. **Diff two TPU versions on the same workload.**
   ```bash
   python3 examples/run_cpu_simulation_demo.py --tpu-version v5e --chip-count 4
   python3 examples/run_cpu_simulation_demo.py --tpu-version v5p --chip-count 4
   ```
   Which gives the lower simulated step time? Which gives lower simulated USD/step?
   Why? (Hint: peak FLOPS vs hourly rate.)

3. **Predict, then check.** Without running anything, predict which TPU version the workload below most plausibly fits, then verify by reading [`docs/01_cloud_tpu_versions.md`](01_cloud_tpu_versions.md):
   - 13B-parameter transformer, batch size 1M tokens per step, train for 1 trillion tokens, embedding table 200M rows.

4. **Stress-test your mental model.** Write one paragraph (in your own notes) explaining why a Cloud TPU is a poor choice for a Reinforcement Learning loop with environment stepping in Python. Cross-reference at least one official Google doc URL from section 8.
