# 09 - Cost and Performance Methodology

> **Learning goal:** know how to honestly compute and compare the cost of a Cloud TPU workload — per step, per sample, per token, per epoch. Read `step_time × chip_count × hourly_rate` correctly. Avoid the common pricing traps: idle billing, preemptible/Spot, region differences, committed-use discounts. All grounded in this lab's `src/common/cost.py`.

This is the doc to re-read **before every** "should we use TPU for this?" decision.

> Pricing is never hardcoded as authoritative in this repo. **Always look up the current rate** at <https://cloud.google.com/tpu/pricing>. Per-region availability is at <https://cloud.google.com/tpu/docs/regions-zones>.

---

## 1. The single equation

Every TPU cost number is a derivation from one base equation:

```
total_usd = (n_steps × step_time_s) / 3600  ×  chip_count  ×  hourly_usd_per_chip
```

That is exactly what `src/common/cost.py:estimate_cost` computes:

```python
total_wall_s = inputs.n_steps * inputs.step_time_s
chip_hours = (total_wall_s / 3600.0) * inputs.chip_count
total_usd = chip_hours * inputs.hourly_usd_per_chip
```

Everything else — cost per step, cost per sample, cost per token, cost per epoch — is a derived view of that single number.

**The single most important consequence:** total cost is **linear in wall-clock time** and **linear in chip count**. If you double your chip count and your step time halves, you spent the same money — but you finished in half the wall time. That's the "weak vs strong scaling" tradeoff in dollars.

---

## 2. The four derived costs

`CostReport` in `src/common/cost.py` carries four derived per-unit costs:

| Field | Formula | When to use |
| --- | --- | --- |
| `cost_per_step_usd` | `total_usd / n_steps` | Comparing two configurations at the same step count |
| `cost_per_sample_usd` | `cost_per_step / samples_per_step` | Throughput-style comparisons |
| `cost_per_token_usd` | `cost_per_step / tokens_per_step` | LLM training / inference cost normalisation |
| `cost_per_epoch_usd` | `cost_per_sample × samples_per_epoch` | Multi-epoch training budgets |

A worked example:

```python
from cloud_tpu_lab.src.common.cost import CostInputs, estimate_cost

inputs = CostInputs(
    chip_count=8,
    n_steps=10_000,
    step_time_s=0.25,
    hourly_usd_per_chip=2.0,    # PLACEHOLDER — check the real rate.
    samples_per_step=256,
    tokens_per_step=256 * 1024, # 1024-token sequences
)
report = estimate_cost(inputs, samples_per_epoch=1_000_000)
print(report.to_dict())
```

For this configuration:

- `total_run_wall_s = 10_000 × 0.25 = 2500 s ≈ 0.694 h`.
- `chip_hours = 0.694 × 8 = 5.56`.
- `total_run_usd = 5.56 × 2 = 11.11 USD`.
- `cost_per_step = 11.11 / 10_000 = 0.00111`.
- `cost_per_sample = 0.00111 / 256 ≈ 4.34e-6`.
- `cost_per_token ≈ 4.24e-9`.
- `cost_per_epoch ≈ 4.34 USD`.

All of which are bookkeeping on the same 11.11 USD.

---

## 3. Utilization — the part most people miss

`CostReport.utilization_adjusted_usd` is the most clarifying field. From `src/common/cost.py`:

```python
util = max(min(inputs.utilization, 1.0), 1e-6)
util_adjusted = total_usd / util
```

> _"Utilization-adjusted: what would the same workload cost if we used the accelerator 100% of wall time? Lower number = 'you're being charged for X but only getting Y of useful work.'"_

If your TPU duty cycle (per the profiler's Overview page) is 50 %, your `utilization_adjusted_usd` is 2× your `total_run_usd`. The chip was idle for half the time and you paid for that half.

This is the link from §1's pure equation to the real world. **Idle TPU is paid TPU.** Half of cost optimisation is just keeping the chip busy.

---

## 4. The four pricing gotchas

### 4.1 Idle billing

When your TPU VM is created, you pay from the moment it's `READY`. Not when you start training. Not when you start any process. The clock starts at provisioning.

- The `gcp/provision_tpu.sh` script in this repo provisions on demand.
- `gcp/delete_tpu_vm.sh` deletes when done. **Run it.**

A common pattern: provision, debug for two hours via SSH, run a 20-minute training job, forget to delete. You paid for 2 h 20 min of chip time but did 20 min of useful work.

### 4.2 Spot / preemptible

Cloud TPU offers **Spot** (formerly "preemptible") capacity at a substantial discount. _[PUBLIC]_ <https://cloud.google.com/tpu/docs/preemptible> and <https://cloud.google.com/tpu/pricing>.

Tradeoffs:

- Pro: lower hourly rate.
- Con: can be **preempted** with little warning. You **must** checkpoint frequently to GCS.
- Con: capacity is opportunistic; you might not be able to acquire what you want.

A typical Spot workflow:

1. Checkpoint every N minutes to GCS.
2. On startup, restore latest checkpoint.
3. Make N small enough that re-doing the last N minutes is acceptable.

For research workloads, Spot can cut cost by ~50–70 % _[PUBLIC, see pricing page for current discount]_. For latency-sensitive serving, Spot is rarely a fit.

### 4.3 Region differences

TPU pricing differs by region. _[PUBLIC]_ Per-region availability is at <https://cloud.google.com/tpu/docs/regions-zones>.

Considerations beyond the per-chip rate:

- **Data residency.** Your dataset may be locked to a region for compliance.
- **GCS egress.** If your training data is in `us-central1` and you provision TPU in `europe-west4`, GCS reads can cost network egress. Always put TPU in the same region as your data.
- **Capacity.** A region might list a TPU version but currently have no quota.

Per-chip price comparison between regions for the same TPU generation: **always check the official pricing page** the day of provisioning.

### 4.4 Committed-use discounts

For sustained workloads, Google offers committed-use discounts (CUDs) — a multi-month or multi-year commitment in exchange for a per-hour discount. _[PUBLIC]_ <https://cloud.google.com/docs/cuds>.

- Pro: substantial discount when you'll keep N chips busy for a year.
- Con: you pay even if you don't use the capacity.

CUDs are a procurement decision, not an engineering decision. But they affect your effective `hourly_usd_per_chip` and therefore your cost reports. The lab's cost model takes `hourly_usd_per_chip` as an input; pass the CUD-effective rate, not the list price.

---

## 5. How to estimate before provisioning

A reliable workflow:

1. **Run the lab simulator** on your target config.
   ```bash
   python3 examples/run_cpu_simulation_demo.py \
     --tpu-version v5e --chip-count 8 \
     --hidden-size 4096 --num-layers 24
   ```
2. Note the simulated `step_time_s` and `samples_per_step`.
3. Compute via `CostInputs` for `n_steps = total_tokens / tokens_per_step`.
4. Compare across TPU versions.

The simulator is **not** authoritative on absolute step time — its roofline model is conservative. But the **ratio** between two versions is roughly meaningful, and that's what cost decisions usually come down to.

A complement: **always do a 5-minute paid pilot** on the real TPU at the smallest viable slice. Run 100 steps, measure step time, then back-of-envelope the full run cost. That single 5-minute spend prevents the "we provisioned for a week, got the math wrong, spent 10× our estimate" disaster.

---

## 6. Cost per step in a multi-cost world

A real training run has costs beyond chip-hours:

| Cost line | Typically | How it scales |
| --- | --- | --- |
| TPU chip-hours | dominant | linear in time × chip_count |
| Storage (GCS) | small | per-GB-month, mostly fixed |
| Egress (GCS → TPU VM) | small if same region | per GB if cross-region |
| Checkpoints (GCS) | small | per checkpoint size × frequency |
| Compute Engine for ancillary VMs (orchestrator, logging) | small | usually negligible |

`src/common/cost.py` only models the chip-hours portion. For a production budget, sum the rest. But for typical training, the chip-hours line is 90 %+ of total. _[INFER]_

---

## 7. Walk-through: comparing v5e vs v5p

A back-of-envelope, all values placeholder for illustration:

| Config | Step time (sim) | Chip count | Hourly rate (placeholder) | $/step | $/sample (at 256/step) |
| --- | --- | --- | --- | --- | --- |
| v5e, 8 chips | 0.40 s | 8 | $1.20 | $0.00107 | $4.17e-6 |
| v5p, 4 chips | 0.25 s | 4 | $4.00 | $0.00111 | $4.34e-6 |

In this hypothetical, v5e is fractionally cheaper per step but v5p finishes in 62 % of the wall time. **Either could be correct depending on the deadline.** That's the kind of comparison the cost model makes routine.

> **Reminder:** these numbers are placeholders. Pull current per-chip prices from <https://cloud.google.com/tpu/pricing> and per-version step time from your own simulator runs (`examples/run_cpu_simulation_demo.py --tpu-version vX`).

---

## 8. The cost-driven debugging loop

When a workload is more expensive than expected, the diagnosis order is:

1. **Check `utilization_adjusted_usd`.** If much higher than `total_run_usd`, the chip is idle. Profile.
2. **Check `cost_per_token` (if applicable).** If much higher than peer workloads, something is structurally inefficient — bad sharding, recompile loop, undersized batch.
3. **Check `step_time_s`.** Compare to a roofline estimate (sum of FLOPS / peak TFLOPS). If you're 5× off roofline, kernels are HBM- or comm-bound.
4. **Check `chip_count`.** Are you over-provisioned? If `(scaling_efficiency × chip_count)` is well below `chip_count`, you're buying chips that are mostly idle.

This is the same diagnostic flow as [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md), but framed in dollars.

---

## 9. Cost safety culture

The repo's README is explicit:

> _"Every `gcp/*.sh` script that creates a paid resource has a matching cleanup script in the same directory. Idle TPU VMs accrue cost even when idle — always run `delete_tpu_vm.sh` when done."_

A short checklist before any provisioning:

- [ ] I have the **cleanup command** memorised: `bash gcp/delete_tpu_vm.sh`.
- [ ] I checked **current pricing** at <https://cloud.google.com/tpu/pricing>.
- [ ] I picked a region with **my data nearby** (<https://cloud.google.com/tpu/docs/regions-zones>).
- [ ] I considered **Spot** for non-critical workloads.
- [ ] My script writes checkpoints to **GCS**, not local disk.
- [ ] My script writes **on `KeyboardInterrupt`** so I can stop without losing state.

A common discipline: set a **billing alert** at 50 % of your expected budget, and an automatic shutdown trigger at 100 %. Cheap to set up; pays for itself the first time.

---

## 10. Cross-references

- [`docs/00_big_picture.md`](00_big_picture.md) — when TPU is the wrong tool (and you save 100 % of the cost by not using it).
- [`docs/01_cloud_tpu_versions.md`](01_cloud_tpu_versions.md) — per-version positioning that drives cost-per-step.
- [`docs/02_cloud_tpu_architecture.md`](02_cloud_tpu_architecture.md) — why utilization matters.
- [`docs/08_profiling_and_debugging.md`](08_profiling_and_debugging.md) — finding the idle time to convert to throughput.

Code:

- `src/common/cost.py` — `CostInputs`, `CostReport`, `estimate_cost`.
- `src/profiling/bottleneck_report.py` — the cost-sanity rule fires at $10 total.
- `gcp/provision_tpu.sh` and `gcp/delete_tpu_vm.sh` — provisioning lifecycle.

Official:

- Pricing: <https://cloud.google.com/tpu/pricing>
- Regions & zones: <https://cloud.google.com/tpu/docs/regions-zones>
- Preemptible (Spot) TPU: <https://cloud.google.com/tpu/docs/preemptible>
- Committed-use discounts: <https://cloud.google.com/docs/cuds>

---

## 11. Exercises

1. **Roofline cost.** For a model with `total_flops = 5e18` (e.g. a small training run), and target TPU `peak_bf16_tflops = 197` (v5e), what is the lower-bound wall-clock at perfect utilization? Multiply by 8 chips and a placeholder $1.20 / chip-hr. Cross-check against actual current pricing.

2. **Find the break-even.** You can train on v5e-16 in 24 h or on v5p-8 in 14 h. Using the placeholder rates v5e=$1.20, v5p=$4.00 (placeholders only — verify), which is cheaper? At what v5p price does v5e become equally cost-effective?

3. **Utilization-adjusted reality check.** Set `utilization=0.5` in `CostInputs`. Compute both `total_run_usd` and `utilization_adjusted_usd` for a run of `n_steps=1000, step_time_s=0.3, chip_count=4, hourly_usd_per_chip=1.5`. Explain what each number means in one sentence.

4. **Spot strategy.** Sketch a training loop that:
   - Saves a checkpoint to GCS every 10 minutes.
   - Restores latest on startup.
   - Logs only every N steps.
   - Catches `SIGTERM` (Spot preemption) and saves before exiting.
   
   Which doc in this series does the "save before exiting" pattern reuse from? (Hint: profiling and debugging covers checkpoint stalls.)
