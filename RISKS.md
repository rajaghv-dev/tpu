# Risk Register

> **Format.** ID · Risk · Likelihood (L/M/H) · Impact (L/M/H) · Root cause · Early warning sign · Mitigation · Contingency.
> **Sections.** Technical (R-T) · Measurement Quality (R-M) · Cost & Access (R-C) · Infrastructure (R-I) · Learning/Comprehension (R-L).
> **Heat-map convention.** L×L = note; M×M or higher = active mitigation in current stage; H×H = block stage exit until resolved.
> **Last reviewed.** 2026-04-26.

---

## Technical risks (R-T)

### R-T01 — XLA recompilation from shape changes
- **Likelihood.** High. **Impact.** Medium.
- **Root cause.** XLA caches compiled HLO keyed on input shapes/dtypes. Any non-static dimension (batch, seq_len, padding mask shape) triggers a fresh compile, which can cost 30 s – 5 min.
- **Early warning.** A run's `compile_time_s` field is non-zero on the second and third repeat of the same cell, OR the second cell of the suite has a 30+ second pause before its first measured token.
- **Mitigation.** Pad to fixed shapes. `drop_remainder=True` on every batched axis. Use `jax.jit` with explicit `static_argnames`. Pre-warm the JIT cache at suite start with one dummy call per (shape, dtype) tuple. `JAX_COMPILATION_CACHE_DIR` set to a persistent path (GCS for cross-VM reuse — see ADR-006).
- **Contingency.** If a model recompiles every step (dynamic shapes inside the model, e.g., MoE routing), tag it `xla_hostile` and either pad statically (capacity-factor approach, P32) or remove from the TPU path and benchmark only on GPU.

### R-T02 — Thermal throttling on B200
- **Likelihood.** Medium. **Impact.** Medium-High.
- **Root cause.** B200 sustained at 700 W in a Dell DGX desktop chassis can exceed cooling capacity during long sequential runs, triggering clock throttle (1965 → 1500 MHz observed in some workloads).
- **Early warning.** `nvidia-smi --query-gpu=temperature.gpu` rises above 80 °C, or `pynvml.nvmlDeviceGetClockInfo` shows clock drop, or per-cell throughput visibly degrades over a 1-hour run.
- **Mitigation.** Thermal pre-flight: abort any cell where start-temp > 80 °C. 30-second cool-down between cells. Power cap to 600 W via `nvidia-smi -pl 600` if sustained throttle observed. Log temp + clock at 1 Hz during run.
- **Contingency.** If throttle persists, split suite into shorter sessions with explicit cool-down breaks. Re-run any cell where peak temp > 85 °C (untrustworthy).

### R-T03 — Mamba has no XLA primitive for selective_scan
- **Likelihood.** High (it's a known limitation). **Impact.** Medium.
- **Root cause.** Mamba/Mamba-2's `selective_scan` is a custom CUDA kernel; XLA emits a sequential `lax.scan` fallback that's 10–20× slower than the native CUDA path on the same GPU. There is no equivalent on TPU.
- **Early warning.** Mamba-2.8B throughput on JAX+GPU is dramatically lower than PyTorch+GPU, OR Mamba on TPU is far below its parameter-count peers.
- **Mitigation.** Document expected failure mode upfront. Run Mamba via PyTorch+GPU as the canonical path. On TPU, expect — and report — the underperformance as the finding it is.
- **Contingency.** Skip Mamba on TPU paths past Stage 3. Keep PyTorch+GPU result as the published number.

### R-T04 — MoE static-shape failure
- **Likelihood.** High (for Phi-3.5-MoE, DeepSeek-Coder-V2-Lite). **Impact.** Medium.
- **Root cause.** Top-k routing produces variable token counts per expert; XLA needs static shapes; a naive port either recompiles per token or runs all experts (negating MoE).
- **Early warning.** Compile time per token on a routed forward pass; OR all experts active in profiler trace.
- **Mitigation.** Capacity-factor padding (each expert gets a fixed slot count, drop overflow). Document the throughput cost vs dense Phi-3.5-mini. P32 covers this investigation.
- **Contingency.** If padding overhead exceeds the MoE benefit, mark MoE TPU-incompatible and benchmark only on GPU. Phi-3.5-mini (dense) carries the family signal in the headline table.

### R-T05 — SSM/Mamba returning wrong outputs silently
- **Likelihood.** Low. **Impact.** High.
- **Root cause.** Custom kernel ports may have off-by-one indexing on the recurrent state, producing plausible-looking but numerically wrong logits. Inference benchmarks measure throughput, not accuracy, so the bug hides.
- **Early warning.** Logit-MAE between JAX-Mamba and PyTorch-Mamba on a fixed input exceeds 1e-3 (any larger than this is a code bug).
- **Mitigation.** Cross-framework parity test in `tests/parity/` for every Mamba/SSM model: load same weights in JAX and PyTorch, run on same synthetic input, assert MAE < 1e-4 on output logits. Run before ANY Mamba result is published.
- **Contingency.** If parity fails, freeze the Mamba result to "NOT VERIFIED" in the dashboard, open a debugging issue, and do not publish until parity passes.

### R-T06 — First-compile cost spiking >5 min
- **Likelihood.** Medium. **Impact.** Low (delay) – Medium (preemption).
- **Root cause.** Some models (large transformers, dense XL nets) trigger long XLA compile passes — especially with auto-fusion enabled. On v5e-1 we've seen 4-min compiles for 1.5B-parameter models.
- **Early warning.** First call to `jax.jit(model)` doesn't return inside 90 seconds.
- **Mitigation.** Persistent compilation cache on GCS so each model compiles once across all VMs. Suite runner emits a "compile-only" pre-pass at start for all models.
- **Contingency.** Increase per-cell timeout to 10 minutes for the first occurrence. Cache the result. Subsequent runs are <1 s.

### R-T07 — HF gated model access failure
- **Likelihood.** Medium. **Impact.** Medium.
- **Root cause.** Models like Llama-3, Gemma, Stable Diffusion XL are "gated" — require accepting a license on HF web. Tokens fail with 403 until accepted.
- **Early warning.** `huggingface_hub.snapshot_download` raises `GatedRepoError`.
- **Mitigation.** At suite-start pre-flight: for every gated model in the registry, try `HfApi().model_info(model_id, token=...)`. Print missing acceptances and exit before burning TPU minutes.
- **Contingency.** Each gated model has a non-gated alternative declared in the registry (e.g., Mistral-7B as Llama alternative). Pre-flight prints the substitution.

### R-T08 — Preemptible VM preemption mid-run
- **Likelihood.** Medium. **Impact.** Low (with recovery) – High (without).
- **Root cause.** Preemptible VMs receive ~30 s warning then are terminated. Probability rises sharply during US business hours.
- **Early warning.** `gcloud` metadata server `v1/instance/preempted` returns true; SIGTERM delivered to processes.
- **Mitigation.** SIGTERM handler in `runner.py` flushes `runs.jsonl` and writes `last_completed_cell.txt`. Runner reads that file at startup and resumes from the next cell. JSONL append-only ensures no partial-row corruption (ADR-007).
- **Contingency.** New VM, resume. If preemption rate exceeds 30% (multiple preemptions per suite), switch to non-preemptible at ~3× cost for the remainder of the stage.

---

## Measurement quality risks (R-M)

### R-M01 — CV > 10% systemic (hardware vs run variability)
- **Likelihood.** Medium. **Impact.** Medium.
- **Root cause.** A specific model on a specific device persistently yields CV > 10%. Could be hardware (noisy neighbor on shared TPU pod), thermal drift, or genuine input-dependent control flow.
- **Early warning.** Three consecutive smoke-suite invocations of the same cell all show CV > 10%, with different mean throughputs (not just noise around a stable mean).
- **Mitigation.** Run the cell on a **fresh VM** in a different physical zone — if CV drops, original VM was noisy. Pin to a specific zone with lower contention. Add a 30 s cool-down before the cell.
- **Contingency.** Mark the cell `unstable` in `runs.jsonl`, exclude from headline charts, document in a `flaky_cells.md` runbook with hypothesised cause.

### R-M02 — Compile cache contaminating "cold" measurements
- **Likelihood.** Medium. **Impact.** High.
- **Root cause.** A "cold" run that's actually warm (because XLA cache wasn't cleared) reports a falsely-low compile time.
- **Early warning.** `compile_time_s` reported as <1 s on a 1B+ parameter model — physically implausible.
- **Mitigation.** `compile_controller.clear_xla_cache()` (P27) before every measured cold run, including deletion of `JAX_COMPILATION_CACHE_DIR`. Validate via the test that asserts `first_call > 1s` on a reference model.
- **Contingency.** Re-run from a fresh VM (no cache present at all) and compare. If still suspicious, log the cache directory size before/after the call.

### R-M03 — Thermal drift between n=3 runs
- **Likelihood.** Medium. **Impact.** Low-Medium.
- **Root cause.** First run on a cold device is often slower than runs 2–3 on a now-warm device (clocks higher, caches populated, host pages faulted in). This biases CV slightly low (since runs 2–3 cluster).
- **Early warning.** `monotonic increase` in throughput across run_idx 1→2→3 on multiple cells.
- **Mitigation.** Warmup runs (≥2) before measured runs (≥3). The cold/warm split (P27) explicitly separates compile-dominated first run from steady-state.
- **Contingency.** If drift is large (>5%), increase warmup count to 3 and document; if it's small, accept as part of CV.

### R-M04 — Batch-size sweep OOM not cleanly caught
- **Likelihood.** Medium. **Impact.** Low (cell skipped) – Medium (suite crash).
- **Root cause.** A batch=1024 cell on a 4B model hits HBM ceiling and the harness crashes the whole runner instead of marking the cell `oom` and continuing.
- **Early warning.** Suite halts mid-run; last cell in JSONL shows no completion record.
- **Mitigation.** Per-cell `try/except` around the harness call. HBM-fit precheck before invocation. On `OutOfMemoryError`, write a row with `status: oom` and continue.
- **Contingency.** Skip the failing cell, log it, restart suite from next cell.

### R-M05 — India → us-central1 latency inflating HF API path unfairly
- **Likelihood.** High (geographically guaranteed). **Impact.** Medium.
- **Root cause.** The HF Inference API path measures end-to-end latency including India ↔ us-central1 round-trip (~250 ms). A purely-local v5e-1 invocation has ~1 ms host-to-device. Comparing them on raw latency is unfair.
- **Early warning.** HF API path latencies are systematically ~250 ms larger than local with no compute correlation.
- **Mitigation.** For HF API runs, separate `network_rtt_ms` from `compute_ms` in the result row. Subtract baseline ping at run start. Report both.
- **Contingency.** Always include a "geographic-baseline" caveat on charts comparing HF API to local paths.

---

## Cost & access risks (R-C)

### R-C01 — GCS egress charges exceeding budget
- **Likelihood.** Low (we read in-region). **Impact.** Medium.
- **Root cause.** Accidentally triggering an `india ← us-central1` read (e.g., debugging a JSONL from a laptop) costs 0.12 USD/GB.
- **Early warning.** `gcloud billing` egress line item rises above 1 USD/month.
- **Mitigation.** All bulk reads happen from us-central1 VMs (free). Local laptop access uses `gsutil` for small files only. Set GCS lifecycle to alert on >1 GB egress in 24 h.
- **Contingency.** Tighten IAM to deny egress reads from non-us-central1 IPs if the pattern recurs.

### R-C02 — Preemptible TPU unavailability in us-central1
- **Likelihood.** Medium (during US business hours). **Impact.** Medium.
- **Root cause.** Preemptible v5e-1 capacity is shared; demand spikes mean `gcloud compute tpus tpu-vm create` returns `RESOURCE_EXHAUSTED`.
- **Early warning.** Provisioning fails with that exact error code, multiple zones.
- **Mitigation.** Try multiple zones (`us-central1-a`, `us-central1-b`, `us-central1-c`) in sequence. Schedule big runs for IST 18:00–23:00 (US night, low contention).
- **Contingency.** Fall back to non-preemptible at ~3× cost for time-sensitive runs; otherwise wait and retry in 1 h.

### R-C03 — B200 electricity cost at sustained load
- **Likelihood.** Medium. **Impact.** Low.
- **Root cause.** B200 at ~700 W × 4 hours × INR 8/kWh ≈ INR 22 (~0.27 USD) per multi-hour run. Cumulatively non-trivial over a year of weekly runs.
- **Early warning.** Monthly electricity bill rises noticeably.
- **Mitigation.** Cap sustained runs to ≤2 hours; schedule runs during off-peak electricity rates if available.
- **Contingency.** Reduce B200 frequency in favor of cloud H100 spot if local cost exceeds cloud cost.

### R-C04 — Colab Pro session timeout mid-suite
- **Likelihood.** High. **Impact.** Low (with recovery).
- **Root cause.** Colab Pro disconnects after ~12 hours idle / hard cap on TPU minutes. A 70-min smoke suite with a flaky network can exceed this cap.
- **Early warning.** Cell loses connection; runtime resets.
- **Mitigation.** Persist `runs.jsonl` to GCS at every cell (not just at suite end). Resume logic from last completed cell. Keep Colab suites strictly ≤50 min.
- **Contingency.** Use Colab for smoke and quick suites only. Full suites run on dedicated GCP TPU VMs.

---

## Infrastructure risks (R-I)

### R-I01 — GCS model cache corruption
- **Likelihood.** Low. **Impact.** Medium-High.
- **Root cause.** A partial `gsutil cp` (interrupted) leaves a truncated weight file. HF then loads it and gets a torch RuntimeError or — worse — a silent value mismatch.
- **Early warning.** Loading a model raises `RuntimeError: PytorchStreamReader failed reading zip archive`, OR logits diverge from PyTorch reference.
- **Mitigation.** Always upload via `gsutil cp` (atomic in GCS), never via writes through `gcsfuse` mount. Verify SHA256 of every weight file against HF's published hash on first download. Re-run hash check weekly.
- **Contingency.** Delete the corrupt file, re-pull from HF, re-upload via `gsutil cp`. Re-hash.

### R-I02 — runs.jsonl corruption on preemptible VM termination
- **Likelihood.** Low. **Impact.** Medium.
- **Root cause.** SIGTERM during a partial JSON line write leaves a truncated row that breaks all downstream readers.
- **Early warning.** `pandas.read_json(lines=True)` raises `JSONDecodeError`.
- **Mitigation.** Buffer one full row in memory; write atomically (`open(..., 'a')` + single `write(json + '\n')` is line-atomic on POSIX up to ~4 KB). Flush after every write.
- **Contingency.** A `tools/repair_jsonl.py` script that strips the last malformed line and reports the lost cell, which is then re-run.

### R-I03 — GitHub Pages dashboard not rendering
- **Likelihood.** Low. **Impact.** Low-Medium.
- **Root cause.** Vega-Lite CDN URL changes, or `runs.jsonl` exceeds GitHub Pages' file-size soft limit, or a CSP header blocks CDN script.
- **Early warning.** Browser console shows fetch error, OR Vega-Lite "spec failed to render".
- **Mitigation.** Pin Vega-Lite version in `<script>` tag. Test dashboard in a local browser (`python -m http.server`) before every push. Shard `runs.jsonl` yearly.
- **Contingency.** Inline Vega-Lite (~300 KB) into the HTML if CDN unstable. Serve via Netlify/Cloudflare Pages as a backup if GitHub Pages flakes.

### R-I04 — runs.jsonl growing too large to diff
- **Likelihood.** Medium (long-term). **Impact.** Low.
- **Root cause.** Weekly full suite × 800 rows × 2 KB × 52 weeks = ~80 MB/year. PR diffs become unreviewable; git operations slow.
- **Early warning.** File >50 MB; PR review tool truncates diff.
- **Mitigation.** Yearly shard rotation: `runs-2026.jsonl`, `runs-2027.jsonl`, …. Old shards remain in repo for traceability but are not appended.
- **Contingency.** Move historical shards to a release artifact or GCS, keep only current year in repo.

---

## Learning / comprehension risks (R-L)

### R-L01 — Misinterpreting "MXU% high = good performance"
- **Likelihood.** High. **Impact.** Medium.
- **Root cause.** TPU MXU utilization is a busy-percentage metric; a model can pin MXU at 95% while doing useless work (e.g., matrix ops on padded tokens). High MXU% with low throughput indicates inefficient compute, not good compute.
- **Early warning.** A cell with `mxu_util_pct > 90` but `throughput_samples_per_sec` lower than a peer at `mxu_util_pct = 70`.
- **Mitigation.** Always pair MXU% with throughput. Define "efficient" as `throughput_per_TFLOP` (achieved samples/s ÷ peak TFLOP) and report on dashboards.
- **Contingency.** A glossary entry in `LESSON_PLAN.md` and on the dashboard explaining the trap with a worked example.

### R-L02 — Confusing first-compile latency with model inference latency
- **Likelihood.** High. **Impact.** Medium.
- **Root cause.** A reader sees "ResNet-50 took 45 s" and assumes 45 s/sample. The 45 s was almost entirely XLA compile.
- **Early warning.** `latency_ms` near `compile_time_s × 1000` (i.e., compile dominates the reported number).
- **Mitigation.** Schema strictly separates `compile_time_s` from `latency_p50_ms`. Dashboards never show a single "latency" field — always cold (first call), warm (steady state), and compile broken out.
- **Contingency.** Per-row visual flag: a warning icon on any row where compile > 50% of measured time.

### R-L03 — Conflating "BF16 2× on GPU" with "BF16 always 2×"
- **Likelihood.** Medium. **Impact.** Medium.
- **Root cause.** BF16 vs FP32 speedup depends on whether the device's matrix engine has dedicated BF16 throughput (Ampere+ Tensor Cores: 2× BF16 over FP32; TPU MXU: BF16 native, FP32 falls back). Reader generalizes from one device.
- **Early warning.** A claim "BF16 is 2× on Path X" without a Path-Y comparison.
- **Mitigation.** Every dtype claim in evidence chain is qualified by device. P31 (BF16-free-on-TPU validation) sets the template.
- **Contingency.** A "BF16 myths" entry in the lesson plan with the actual device-by-device speedup table.

### R-L04 — Assuming "all TPU models are faster than GPU"
- **Likelihood.** Medium. **Impact.** Medium.
- **Root cause.** TPU marketing emphasises TFLOPs; reader concludes TPU > GPU universally. In fact TPU loses to GPU on (a) memory-bound decode at small batch, (b) Mamba/SSM, (c) MoE, (d) anything with lots of small ops or dynamic shapes.
- **Early warning.** Reader requests "just give me the TPU number" without specifying workload.
- **Mitigation.** Headline dashboard always shows TPU-vs-GPU side by side per model. Document the four GPU-wins-here cases prominently.
- **Contingency.** A short "When TPU loses" section in `context.md` with the four cases and the measurement that proves each.

---