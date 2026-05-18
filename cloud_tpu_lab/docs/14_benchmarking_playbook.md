> **Note:** this doc predates the real-TPU pivot. References to `src/xla_sim/`, `src/pjrt_sim/`, `src/sharding/`, `src/memory/`, `src/input_pipeline/`, and `examples/run_cpu_simulation_demo.py` are historical — those modules were removed. The TPU architecture / XLA / observability concepts below are still accurate. Current run flow lives in [README.md](../README.md) and [16_runbook_real_tpu.md](16_runbook_real_tpu.md).

# 14 — Benchmarking Playbook

> **Learning goal:** produce Cloud TPU benchmarks that another person can
> reproduce and trust. Understand why a first-step timing is essentially
> always wrong, how to separate cold compile from steady-state, how many
> runs you need before a number means anything, what to control, what to
> report, and how to capture enough lineage that the run can be re-done six
> months later.

The methodology here is the basis for the thresholds in
`src/profiling/bottleneck_report.py` and the metric definitions in
`13_oct_metrics_dictionary.md`. Use this doc before pressing "run" on a
real (PAID) Cloud TPU VM — it's free to do the planning offline.

---

## 1. Why benchmarking Cloud TPU is hard

A naive `time.perf_counter()` around `model.step()` is wrong for at least
six independent reasons. Each one is fixable; together they're a
methodology.

1. **First step is misleading.** It includes XLA compile, kernel
   autotuning, JAX/Torch-XLA cache misses, and possibly even
   PJRT-runtime startup. Reporting it as "the step time" is the most
   common rookie mistake.
2. **Compile cache asymmetry.** A second invocation in the same process
   may be ~free; a second invocation in a *new* process may be slow again
   unless `JAX_COMPILATION_CACHE_DIR` is set.
3. **Async dispatch.** TPU runtimes dispatch work asynchronously. If you
   don't `block_until_ready()` the result, you're timing how fast Python
   can enqueue, not how fast the chip computes.
4. **Input pipeline starvation.** A slow data loader can stretch step
   time arbitrarily without any change in the chip.
5. **Single-sample noise.** TPU step times have a non-trivial variance
   even at steady state (warm cache misses, neighbour noise on shared
   pods, OS scheduling on the host). A 1-run number is uncalibrated.
6. **Different SKU, different units.** Comparing `v5e-1` step time to
   `v5p-8` step time without normalising by chip count or batch is
   comparing two different things.

Sections 2–9 fix one of these each.

---

## 2. Warmup steps — what to discard

Pattern: throw away enough steps that compile, cache, and autotuning are
finished. Then start timing.

Rules of thumb (mirrored in `src/profiling/`):

- **Discard at least 5 steps** as a floor. Even for tiny models, runtime
  caches need a few step shapes to settle.
- **Discard until step time is stable to within 5%** over a sliding
  window of 5 consecutive steps. The simulator uses this rule when it
  decides what "steady state" means.
- **For larger models / higher chip counts**, discard 10–20 steps. The
  bigger the graph, the more compile / autotune surface area to traverse.

Minimal pattern (JAX flavour):

```python
warmup = 10
N      = 50

# Warmup pass — outputs are discarded.
for _ in range(warmup):
    out = step(x, y)
    out.block_until_ready()

# Steady-state timing.
ts = []
for _ in range(N):
    t0 = time.perf_counter()
    out = step(x, y)
    out.block_until_ready()
    ts.append(time.perf_counter() - t0)
```

Observability counterpart: `cloud_tpu_step_time_seconds` should be
reported **after** the warmup window. Tag the first `warmup` events with
a `warmup=true` extra field in the JSONL log so they're filterable from
dashboards.

---

## 3. Compile cache — cold vs warm

Treat "cold" and "warm" as two different experiments, not noise on one.

**Cold start** (representative of CI, first-time deploy, fresh VM):

```bash
# No persistent compile cache.
unset JAX_COMPILATION_CACHE_DIR
```

Expect first-step time to include the full compile. Report it
separately as `compile_seconds_first_run`.

**Warm start** (representative of long-running training):

```bash
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-compile"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
```

Run twice. The first invocation populates the cache; the second
invocation should skip compile. The delta is the cache benefit.

Both cases produce `cloud_tpu_compile_time_seconds` events. The
`cloud_tpu_recompile_count_total` counter helps you tell "one compile" from
"compile on every step" — a recompile loop is a methodology bug
documented in `src/profiling/bottleneck_report.py` and triggers the `xla`
finding at >20% compile fraction.

---

## 4. Steady-state vs first-step — what to report

Always report **both**, labelled:

| Field                          | Defined as                                          |
|--------------------------------|-----------------------------------------------------|
| `step_time_first_seconds`      | The single first-step wall time.                    |
| `compile_time_seconds`         | The compile time included in that first step.       |
| `step_time_median_seconds`     | Median of the post-warmup timing window (N≥30).     |
| `step_time_p95_seconds`        | 95th percentile of the same window.                 |
| `step_time_p99_seconds`        | 99th percentile.                                    |
| `tokens_per_second`            | `tokens_per_step / step_time_median_seconds`.       |

Why median and percentiles, not mean? Step time distributions have a long
tail (cache misses, GC pauses, host scheduling). The mean is dragged
around by tail events. The median tells you what a typical step costs;
p95/p99 tell you how bad the bad steps are.

---

## 5. Confidence intervals — three runs minimum

Within a single process, you have N timing samples. Across processes you
have **one** sample of the warm-cache behaviour. To trust a comparison,
repeat the **whole process** at least 3 times.

Minimum protocol:

1. Reset to a clean state (clear compile cache *or* keep it stable —
   document which).
2. Run, capture `step_time_median_seconds`.
3. Repeat 3+ times.
4. Report median across runs, plus the min and max as a range. A 5–10%
   spread is normal on healthy TPU; >20% means something is moving you
   don't control.

For pairwise comparisons (e.g. v5e-1 vs v6e-1), use the same N and the
same procedure on each side. If the per-run spread exceeds the across-SKU
gap, the comparison is not yet conclusive — collect more runs before
publishing the number.

The lab's bottleneck report intentionally **does not** rely on a single-
sample number to call a finding "high" — see the >0.25 / >0.30 / >0.85
thresholds in `bottleneck_report.py`, all chosen well above typical noise.

---

## 6. What to control — the experimental setup

A benchmark is only meaningful if everything except the variable under
test is held fixed. The five things to pin every time:

### 6.1 Random seed

```python
import random, numpy as np
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

# JAX:
key = jax.random.PRNGKey(SEED)
# Torch:
torch.manual_seed(SEED)
```

Without this, weight init / data shuffle change the workload silently.

### 6.2 Batch size

Pin both **local** batch (per-chip) and **global** batch (sum across
chips). Doubling chip count without changing local batch is the *weak*
scaling experiment; doubling global batch is the *strong* scaling
experiment. They give different answers; don't mix them.

### 6.3 Sequence length / image resolution

Sequence-padding tricks change wall time materially. Pin the exact
sequence length (or padding-policy) for the duration of the benchmark.

### 6.4 Sharding strategy

`src/sharding/mesh.py` records the mesh shape + partition spec used for
each run. Two different shardings can produce the same step time but
wildly different `cloud_tpu_collective_time_seconds` — document which one
you're measuring.

### 6.5 Precision

bf16 / fp32 / fp8 affect both throughput and HBM. Pin and report.

A complete control block:

```python
config = {
    "seed": 12345,
    "global_batch": 64,
    "local_batch": 8,
    "seq_len": 1024,
    "precision": "bf16",
    "mesh": {"axes": ("data", "model"), "shape": (2, 4)},
    "framework": "jax",
    "tpu_version": "v5p",
    "chip_count": 8,
}
```

This dict goes into the JSONL stream as the first event of the run
(`event="config"`) — see `src/common/config.py`.

---

## 7. What to report — the publishable number

Aim for a single tabular row per (workload, SKU, chip-count) configuration.

| Column                            | Source                                              |
|-----------------------------------|-----------------------------------------------------|
| `workload_name`                   | Config block.                                       |
| `framework`                       | Config block.                                       |
| `tpu_version`, `chip_count`       | Config block.                                       |
| `global_batch`                    | Config block.                                       |
| `precision`                       | Config block.                                       |
| `step_time_median_seconds`        | Section 4.                                          |
| `step_time_p95_seconds`           | Section 4.                                          |
| `tokens_per_second`               | Derived from median.                                |
| `samples_per_second`              | Derived from median.                                |
| `cloud_tpu_hbm_utilization_ratio` | `src/memory/hbm_sim.py` / framework profiler.       |
| `cloud_tpu_matrix_unit_utilization_ratio` (MXU util) | Profiler / sim approximation.    |
| `mfu` (model-FLOPs utilisation)   | `flops_per_step / (peak_flops * step_time)`.        |
| `cost_per_step_usd`               | `src/common/cost.py` with `--hourly-usd-per-chip`.  |
| `cost_per_token_usd`              | Derived from cost-per-step / tokens-per-step.       |
| `run_count`                       | The N from section 5 (must be ≥3).                  |
| `step_time_range_seconds`         | min..max across runs from section 5.                |

About MFU: it requires knowing the theoretical peak FLOPs of the SKU and
the FLOPs your workload consumes per step. Both are documented in
`src/tpu_versions/` and in the workload definition. MFU is a single
number that lets you compare wildly different setups apples-to-apples,
which is why it's worth the bookkeeping.

Cost is computed from `--hourly-usd-per-chip` — never hardcoded. Look up
the rate at https://cloud.google.com/tpu/pricing and pass it in.

---

## 8. Reproducibility — lineage and environment hash

A six-month-later replay should produce a number within the noise range
in section 5. To make that possible, capture lineage as part of the
artefact tree:

```
artifacts/<trace_id>/
├── config.json           # the section-6 control block
├── env.json              # software versions
├── git.json              # repo state (sha + dirty flag)
├── hardware.json         # SKU, chip count, runtime version
├── pricing.json          # the --hourly-usd-per-chip value + date
├── logs/run_<trace_id>.jsonl
├── metrics/run_<trace_id>.csv
├── traces/run_<trace_id>.json
└── reports/run_<trace_id>.md
```

`env.json` minimum content:

```json
{
  "python": "3.11.7",
  "jax": "0.4.x",
  "libtpu": "...",
  "framework": "jax",
  "os": "Ubuntu 22.04",
  "platform": "linux-x86_64",
  "compile_cache_dir": "/home/user/.cache/jax-compile"
}
```

`git.json` minimum content:

```json
{
  "sha": "abcd1234",
  "dirty": false,
  "branch": "main"
}
```

`src/common/config.py` already serialises this block as the first JSONL
event when the run starts. The downstream report uses it.

---

## 9. Methodology summary — a checklist

Before pressing "run":

- [ ] Random seed pinned.
- [ ] Global / local batch documented.
- [ ] Sequence length / resolution documented.
- [ ] Sharding mesh documented.
- [ ] Precision documented.
- [ ] Warmup count chosen (>= 5; >= 10 if first run).
- [ ] Timing window size N chosen (>= 30 for percentiles).
- [ ] Repeat count chosen (>= 3 whole-process runs).
- [ ] Compile cache state documented (cold vs warm).
- [ ] `--hourly-usd-per-chip` value looked up from the pricing page.
- [ ] Artefact tree path planned (and there's room on disk for it).

After the run:

- [ ] Report median + p95 + p99, not just mean.
- [ ] Compile-time figure reported separately from step time.
- [ ] HBM utilisation reported.
- [ ] MXU / MFU reported.
- [ ] Cost-per-step and cost-per-token reported.
- [ ] Lineage (`env.json`, `git.json`, `pricing.json`) saved.
- [ ] Bottleneck report regenerated and read end-to-end.
- [ ] TPU VM deleted (see `11_cleanup_and_cost_safety.md`).

---

## 10. Anti-patterns — avoid these

- Reporting the first step as "the step time".
- Reporting mean step time without p95.
- Comparing two SKUs with different batch sizes "to be fair".
- Skipping `block_until_ready()` and timing async dispatch.
- Reporting a single-run number without a run count.
- Hardcoding pricing in the script that computes cost.
- Sharing a step-time number without the corresponding HBM utilisation
  (the workload could be silently OOMing and falling back).
- Quoting "MFU = X" without documenting `peak_flops` for the SKU.

---

## 11. Worked example (simulator-side)

The simulator exists precisely to let you rehearse this methodology
without paying for a TPU. The full vertical slice:

```bash
python3 examples/run_cpu_simulation_demo.py \
  --tpu-version v5p \
  --chip-count 8 \
  --batch-size 64 \
  --seq-len 1024 \
  --num-steps 60 \
  --warmup-steps 10 \
  --hourly-usd-per-chip "<value from https://cloud.google.com/tpu/pricing>"
```

Read the resulting `artifacts/reports/run_TRACE-NNNN.md`. The headline
fields there are the ones you'd report — median step time, p95, MFU, HBM
util, cost-per-step, cost-per-token — with the methodology block at the
top. Use it as the template for your real-TPU runs.

---

## 12. Cross-references

- `13_oct_metrics_dictionary.md` — units, ranges, and meanings of every
  metric mentioned here.
- `10_cloud_tpu_setup_playbook.md` — how to actually run the benchmark
  on a real TPU VM.
- `11_cleanup_and_cost_safety.md` — the always-cleanup rule.
- `12_observability_with_grafana_prometheus.md` — viewing the numbers
  on a dashboard.
- `src/profiling/trace_analyzer.py` — implementation of the timing
  breakdown.
- `src/profiling/bottleneck_report.py` — thresholds for the findings
  this playbook lets you trust.
- `src/common/cost.py` — cost arithmetic with no hardcoded prices.

---

## 13. Exercises / TODOs

1. Take an existing simulator config. Run it once. Run it again. Then
   wipe the in-process compile cache and run a third time. Tabulate
   `step_time_first` vs `step_time_median` for each — the third run
   should look like the first, not the second.
2. Build a 3×3 grid of (chip_count ∈ {1, 4, 8}) × (batch ∈ {16, 32, 64})
   on a single SKU. For each cell report median, p95, MFU, cost-per-token.
   Identify the "sweet spot".
3. Reproduce one of your earlier real-TPU benchmarks from cold lineage
   only — i.e. just `config.json` + `git.json` + `env.json` + the
   pricing value. Note any number that is more than 10% off and explain
   why.
4. Inject a deliberate variable-batch op into your training step.
   Confirm `cloud_tpu_recompile_count_total` rises and the bottleneck
   report fires the `xla` finding. Remove it; confirm it returns to zero.
5. Pick two SKUs from `src/tpu_versions/`. Predict cost-per-token from
   the simulator. Then run on real hardware and compare. Document the
   delta (it should be modest if your methodology is solid).
