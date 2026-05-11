# Architectural Decision Records (ADRs)

> **Format.** Each decision is one Architecture Decision Record. Sections: Decision · Status · Context · Decision rationale · Alternatives considered · Consequences · Risks · Revisit trigger.
> **Scope.** These are the locked decisions for the inference benchmark repo. Any reversal must be a new ADR (ADR-NN-superseded-by) — do not edit history.
> **Last reviewed.** 2026-04-26.

---

## ADR-001 — Inference-only scope

**Decision.** This repo benchmarks **inference only**. No training loops, no fine-tuning, no LoRA adapters, no RLHF.

**Status.** Accepted (locked at P9, 2026-04-25).

**Context.** Initial framing was open-ended ("examples on TPU"). As we costed the work and clarified the learning goal — understanding hardware, compilers, and dataflow on a single card with preemptible VMs and consumer GPUs — training simply does not fit. A 4B-parameter training run on v5e-1 would either OOM at typical batch sizes or take days; on RTX 4090 it requires gradient checkpointing and ZeRO-style sharding the box doesn't natively offer. Inference, by contrast, fits the 1–3 minute per-experiment budget cleanly.

**Decision rationale.**
- Inference exercises the same compiler pipeline (XLA, Inductor, TensorRT) and the same memory hierarchy (HBM, on-chip SRAM, MXU/Tensor-Core feeds) as training, so the learning yield is preserved.
- Inference is the deployment workload that dominates real production cost — the conclusions are directly transferable.
- A 1–3 minute run gives clean cold/warm cache separation with n=3 inside a 10-minute budget per cell, which a training step cannot do.
- Training would force a per-experiment cost an order of magnitude above the 0.36 USD/hr v5e-1 preemptible budget.

**Alternatives considered.**
1. *Inference + fine-tuning hybrid.* Rejected — adds a second axis of variability (optimizer, LR, gradient accumulation) without adding compiler-pipeline insight.
2. *Inference + small training (e.g., 100-step LR sweeps).* Rejected — small training is dominated by warm-up and noisy on n=3.
3. *Training-only.* Rejected — does not fit the hardware budget on v5e-1 or RTX 3080.

**Consequences.**
- Enables: synthetic inputs (no dataset pipeline), determinism via greedy decoding, KV-cache focus, prefill/decode split studies, quantization ablation.
- Constrains: no claims about convergence behaviour; no claims about training throughput; no comparison with reported MLPerf-Training numbers.

**Risks.**
- Reader may assume our "BF16 is fast on TPU" claim transfers to training — it does not (training adds gradient/optimizer-state pressure on HBM). Risk mitigated by stating the inference-only scope on every results page.

**Revisit trigger.** A future stage (10+) explicitly funded for fine-tuning, with v6e-8 or larger access. Not on the roadmap.

---

## ADR-002 — JAX AND PyTorch (both, not one)

**Decision.** The repo runs experiments on **both** JAX (with XLA) and PyTorch (with `torch.compile`/Inductor and `torch_xla` for TPU), as separate execution paths.

**Status.** Accepted (P8/P10).

**Context.** A single-framework project would be simpler and faster to build. The argument for JAX-only is its first-class TPU support and cleaner compiler story; the argument for PyTorch-only is the larger model ecosystem and the fact that real-world deployment is mostly PyTorch. The learning goal here is comparative — understanding when each compiler emits worse or better code, where the framework abstraction leaks, and how `torch_xla` differs from native PyTorch on the same accelerator.

**Decision rationale.**
- Compiler comparison is a primary learning objective — XLA-via-JAX vs Inductor-via-PyTorch on the same GPU is the cleanest controlled experiment we can run.
- HuggingFace dominance means most pretrained weights are PyTorch-native; removing PyTorch would force conversion gymnastics for every new model.
- `torch_xla` lets us isolate "XLA the compiler" from "JAX the frontend" — without PyTorch we cannot tell which framework's tracing semantics caused a slowdown.
- Five execution paths (JAX+TPU, JAX+GPU, PyTorch+GPU, torch_xla+TPU, HF API) all ride this dual-framework foundation.

**Alternatives considered.**
1. *JAX-only.* Rejected — loses the Inductor comparison and forces ad-hoc weight conversion for every HF model.
2. *PyTorch-only with torch_xla on TPU.* Rejected — torch_xla on TPU is known to leave performance on the table compared to JAX; we'd be measuring framework overhead, not hardware.
3. *PyTorch + JAX-only-on-TPU.* Considered. Rejected because we want JAX-on-GPU as a control to attribute differences to the compiler, not the device.

**Consequences.**
- Enables: cleanest available compiler attribution; access to both pretrained-weight ecosystems; comparison with both JAX-numpy idioms and PyTorch eager idioms.
- Constrains: 2× the harness surface area; weight loading paths are not symmetric (Flax/Penzai for JAX, native for PyTorch); pinned versions across both frameworks must coexist in `requirements.txt`.

**Risks.**
- Version-skew: a JAX upgrade forces a jaxlib upgrade which may conflict with a torch_xla pin. Mitigated by separate `pyproject` extras (`[jax]`, `[torch]`, `[torch-xla]`) and CI matrix testing.
- Weight conversion bugs: a model loaded in JAX may give different logits than the same weights in PyTorch due to layer-norm epsilon or rotary-embedding sign conventions. Mitigated by a `tests/parity/` suite asserting logit-MAE < 1e-4 between paths.

**Revisit trigger.** A path becomes obviously dominant for our hardware (e.g., torch_xla 2.6 reaches JAX-on-TPU parity for our model set) AND maintaining the other path is consuming >20% of time. We do not anticipate this in 2026.

---

## ADR-003 — Primary TPU target: v5e-1 preemptible

**Decision.** **v5e-1 preemptible** in `us-central1` is the canonical TPU target for Stages 1–6. v6e-1 is added in Stage 7 for selected experiments only. v4 is not used.

**Status.** Accepted (P6/P8).

**Context.** Personal-account TPU access from India is limited to v5e and v6e Cloud TPU VMs in us-central1 (and a few other US/EU regions). v5e-1 (single-chip) is the cheapest preemptible TPU at ~0.36 USD/hr. v6e-1 is roughly 2× the cost (~0.75 USD/hr) with 2× the HBM (32 vs 16 GB) and ~2× the bandwidth (1.64 TB/s vs 0.82 TB/s).

**Decision rationale.**
- v5e-1 fits our 4B-parameter ceiling: a 4B-BF16 model is ~8 GB of weights, leaves ~6–7 GB for KV cache and activations on 16 GB HBM.
- 0.36 USD/hr × ~70 min smoke suite = 0.42 USD per smoke suite — fits the personal budget.
- Preemption is acceptable for inference (no checkpoint state to lose); experiments are 1–3 min and can resume from the last completed cell.
- v4 is not generally available on personal accounts in 2026.
- Buying v6e-1 first would obscure where the v5e-1 ceiling is — an explicit Stage 7 escalation is the right learning path.

**Alternatives considered.**
1. *v6e-1 from Stage 1.* Rejected — masks the OOM/bandwidth ceiling that is itself pedagogically important.
2. *v4-8 (multi-chip).* Rejected — not single-card and not generally available to us.
3. *v5p (production TPU).* Rejected — no personal access from India in 2026.

**Consequences.**
- Enables: cheap iteration; clean comparison with consumer GPUs (RTX 3080 16 GB has the same HBM size).
- Constrains: model ceiling at 4B (stretch to 7B-quantized in Stage 4); preemption-recovery code is mandatory from Stage 1.

**Risks.**
- Preemptible v5e-1 unavailability during peak hours (US business). Contingency: fall back to non-preemptible at ~3× cost or reschedule.
- us-central1 → India read latency for HF model pulls. Mitigated by GCS cache (ADR-006).

**Revisit trigger.** A consumer-priced v6e or v7 lands at ≤0.50 USD/hr preemptible AND v5e-1 ceiling is binding on more than 30% of our suite.

---

## ADR-004 — Synthetic inputs with fixed seed=42

**Decision.** All inference benchmarks use **synthetic inputs** generated from `numpy.random.default_rng(42)`. No real-dataset samples (no ImageNet JPEGs, no Wikipedia text) are used in benchmark runs.

**Status.** Accepted (P8).

**Context.** Real datasets introduce I/O variability (decode time, network) and license/distribution complexity. The benchmark goal is _compute and memory characterization_, not accuracy.

**Decision rationale.**
- Determinism: seed=42 produces byte-identical inputs across runs and across machines, so any throughput delta is attributable to compute, not inputs.
- Repeatability for any future agent — no need to host a 100 GB dataset mirror.
- Avoids licensing footguns (ImageNet redistribution, Common Crawl ToS).
- Inference latency for a transformer at fixed seq-length is input-content-independent at >99% precision (no early-exit, no KV-cache compression).

**Alternatives considered.**
1. *Subsampled real data.* Rejected — adds an I/O variable to every measurement.
2. *Random per-run inputs.* Rejected — destroys cross-run comparability.
3. *Per-modality canonical inputs (e.g., the same JPEG always).* Considered for a future "realism" suite, but not for the headline benchmark — would re-introduce distribution.

**Consequences.**
- Enables: deterministic comparisons; no dataset hosting; clean cross-framework parity tests.
- Constrains: cannot make accuracy claims; cannot study input-distribution effects (e.g., unusual sequence content triggering attention sparsity); no perplexity numbers.

**Risks.**
- Reader confuses synthetic-input throughput with real-deployment throughput. Mitigated by the "synthetic inputs" badge on every chart.
- Some models (e.g., Whisper) behave subtly differently on white noise vs speech for KV-cache reasons. Documented as a per-model caveat.

**Revisit trigger.** Adding an explicit accuracy-validation suite (Stage 10+, not on roadmap).

---

## ADR-005 — HuggingFace pretrained weights

**Decision.** All models load **HuggingFace pretrained weights** (pinned by `revision=` SHA), not randomly initialized weights.

**Status.** Accepted (P10).

**Context.** Inference benchmarks could legitimately use random weights (compute is the same), but this divorces the work from any real model. Pretrained weights make every result a statement about a real artifact people deploy.

**Decision rationale.**
- Real deployments use pretrained weights — our numbers transfer.
- Pretrained weights catch numerical bugs (e.g., a wrong layer-norm epsilon shows up as garbage logits, while random weights produce garbage either way).
- HuggingFace Hub provides versioned (revision-pinned) weights with model cards, licensing, and reproducibility guarantees.
- Quantization studies (Stage 5) require real weights — random weights have no calibration distribution.

**Alternatives considered.**
1. *Random initialization.* Rejected — loses parity tests, loses quantization story, loses real-world transfer.
2. *Self-hosted weight mirror.* Rejected — duplicates HF infrastructure for no gain.

**Consequences.**
- Enables: real-world transfer of conclusions; numerical-correctness sanity checks; quantization studies.
- Constrains: gated-model access (Llama, Gemma) requires HF token; first download per model is large (1–10 GB) — addressed by ADR-006.

**Risks.**
- HF takes a model down or rotates a revision SHA. Mitigated by GCS cache (ADR-006), pinning every revision in `registry.yaml`.
- HF gated-access denial. Mitigated by maintaining a non-gated alternative for every gated model in the registry.

**Revisit trigger.** HF Hub becomes unavailable from India (geopolitical), or our 4B-ceiling expands beyond what HF reasonably hosts.

---

## ADR-006 — GCS bucket for model weight cache

**Decision.** Model weights are cached in a single-region GCS bucket (`gs://rajaghv-tpu-cache`, `us-central1`, standard storage), mounted into every TPU/GPU VM via `gcsfuse`. HF cache directory points into the mount.

**Status.** Accepted (P12).

**Context.** Re-downloading a 4B-parameter HF model (~8 GB BF16) per preemptible VM lifetime is 8 GB × 0.12 USD/GB egress (us-central1 → India) ≈ 1 USD per pull, plus 5–15 minutes wall-clock — both unacceptable at our cadence.

**Decision rationale.**
- One-time pull from HF to GCS (us-central1 → us-central1, free egress to TPU) amortizes across all subsequent runs.
- Single-region GCS at us-central1 matches v5e-1/v6e-1 placement for free intra-region read.
- `gcsfuse` is officially supported on Cloud TPU VMs and presents a POSIX filesystem to HF's cache code, no library changes needed.
- Standard storage class is the cheapest access tier; we never use Nearline because we read weekly+.

**Alternatives considered.**
1. *Persistent disk attached to TPU VM.* Rejected — preemptible VMs lose disks, and reattaching across regions is messy.
2. *HF Hub direct, no cache.* Rejected — egress cost and cold-pull latency.
3. *Nearline/Coldline storage.* Rejected — retrieval fees on a weekly read pattern exceed standard-tier storage cost.
4. *Multi-region GCS.* Rejected — 2× storage cost without benefit; we run in a single region.

**Consequences.**
- Enables: <30s model load on warm cache; portable across VMs; cheap.
- Constrains: a manual `gsutil cp` step to seed each new model; a single point of failure for weight availability (mitigated by HF as fallback).

**Risks.**
- gcsfuse silent corruption on a partially completed write — addressed by always seeding via `gsutil cp` (atomic) not by writing through the fuse mount.
- Bucket policy misconfiguration leaking model files. Mitigated by `uniform_bucket_level_access=true` and IAM scoped to the user's account only.

**Revisit trigger.** Monthly storage cost exceeds 5 USD (it would not — 100 GB × 0.020 USD/GB = 2 USD/mo) or HF Hub releases a documented mirror in `asia-south1`.

---

## ADR-007 — Append-only JSONL results format

**Decision.** Benchmark results are written as **append-only JSONL** to `results/runs.jsonl`. One row per (model, batch, seq_len, dtype, path, run_idx) tuple. Schema version is in row 0 of every file.

**Status.** Accepted (P14).

**Context.** Results need to be queryable, diffable in git, recoverable on preemption, and convertible to dashboards. The candidates are JSONL, SQLite, Parquet, and a managed warehouse.

**Decision rationale.**
- Append-only is preemption-safe: a killed VM mid-write loses at most one row, not the file.
- JSONL is git-diffable — a PR adding 50 new rows is reviewable.
- One row per experiment is the right grain; aggregation happens downstream in `observe/stats.py`.
- A static dashboard (ADR-008) can `fetch()` JSONL directly from GitHub Pages without a backend.
- Schema version tag in row 0 lets us evolve the schema without breaking old readers.

**Alternatives considered.**
1. *SQLite.* Rejected — binary file is not git-diff-friendly; concurrent writes need WAL setup; preemption can corrupt.
2. *Parquet.* Rejected — column-store value is real for >1M rows but our scale is ~10K rows total over years; columnar tooling adds dependencies.
3. *BigQuery.* Rejected — adds a paid service, network dependency, and authentication footgun for a dataset that fits in a single 50 MB file.
4. *CSV.* Rejected — no nested structure for telemetry summaries; quote-handling pain.

**Consequences.**
- Enables: trivial recovery; git-tracked results; dashboard with no backend; offline analysis with `pandas.read_json(lines=True)`.
- Constrains: file size — at ~800 rows × 2 KB = 1.6 MB per full suite, ~80 MB after 50 weeks. Acceptable; sharding by year can be added later if needed.

**Risks.**
- Git LFS not used; if file grows >100 MB GitHub starts complaining. Mitigated by yearly shard rotation (`runs-2026.jsonl`, `runs-2027.jsonl`).
- Schema drift across rows. Mitigated by `schema_version` field per row + a `tools/migrate.py` script.

**Revisit trigger.** File size > 50 MB or query patterns require true columnar access (e.g., aggregating over 1M+ rows).

---

## ADR-008 — Static HTML + Vega-Lite dashboard

**Decision.** The results dashboard is a **single static HTML file** at `results/dashboard/index.html`, using Vega-Lite v5 (CDN), served via GitHub Pages.

**Status.** Accepted (P14).

**Context.** Visualization needs to be shareable (a URL), zero-cost (no server), versioned (in git), and rich enough for filtered tables and roofline charts.

**Decision rationale.**
- GitHub Pages is free, integrates with the repo, and serves a static file in <50 ms globally.
- Vega-Lite v5 declarative spec is enough for tables, line charts, scatter (roofline), and faceting — no React or D3 needed.
- A single HTML file is reviewable in a PR and runs offline.
- No server means no hosting cost, no auth surface, no uptime concern.
- A future agent can rebuild the dashboard from `runs.jsonl` alone.

**Alternatives considered.**
1. *Grafana.* Rejected — requires a Prometheus or similar backend; designed for time-series, not benchmark-row data.
2. *Streamlit.* Rejected — requires a running Python process; not zero-cost.
3. *Jupyter notebook outputs.* Rejected — not interactive in a stateless URL.
4. *Plotly Dash.* Rejected — needs a server.

**Consequences.**
- Enables: shareable URL, zero-cost, in-repo, offline-capable.
- Constrains: client-side rendering only — large datasets (>10K rows) start to lag in browser; we shard or aggregate before render. No real auth — all dashboards are public (acceptable; no PII or secrets in benchmark results).

**Risks.**
- Browser rendering lag at 800+ rows. Mitigated by Vega-Lite pagination/streaming and by aggregating to per-(model,path) rows for headline charts.
- CDN drift breaking old dashboards. Mitigated by pinning Vega-Lite version in the script tag.

**Revisit trigger.** Result count exceeds 10K rows AND interactive filtering becomes the primary use, OR a multi-user write path appears (it won't — it's a personal benchmark).

---

## ADR-009 — n=3 independent runs with CV<10% threshold

**Decision.** Every benchmark cell runs **n=3 independent times** (cold-cache cleared between runs). A cell is "clean" iff `coefficient_of_variation < 10%` AND no Grubbs outlier.

**Status.** Accepted (P14).

**Context.** Cloud accelerators and consumer GPUs have variability from thermal state, neighbor noise (preemptible TPU), and JIT cache effects. Reporting a single number is misleading; reporting n=10 is wasteful. We need the smallest defensible n.

**Decision rationale.**
- n=3 is the minimum where Grubbs outlier detection is meaningful (n=2 has no concept of outlier).
- 10% CV is the empirical noise floor on shared cloud accelerators per literature (MLPerf-Inference reports similar tolerances).
- Cost: at 1–3 min per run, n=3 is 3–9 min per cell, fitting our budget.
- Lower n (n=1) loses the variance signal; higher n (n=5+) triples cost without proportional confidence gain at this noise level.

**Alternatives considered.**
1. *n=1 with a tightened cold-warm protocol.* Rejected — no variance estimate; can't distinguish outlier from systematic.
2. *n=5.* Rejected — 67% more cost for ~20% tighter CI on n=3-already-clean data.
3. *Adaptive n (start n=3, escalate if CV>10%).* Considered. Adopted as a Tier-2 refinement: re-run once if CV>10%, then mark `unstable`.

**Consequences.**
- Enables: every published number has a CV; outliers are flagged not silently averaged; CI_95 reported.
- Constrains: 3× the run-time of single-shot benchmarking; full suite time = 3 × #cells × 90 s.

**Risks.**
- Three runs is too few to trust Grubbs on borderline cases. Mitigated by also reporting raw values and treating Grubbs as a flag, not a delete.
- Persistent CV>10% indicates a hardware/config issue, not a stats issue. Runbook in `docs/runbooks/high_cv.md` (P30) covers diagnosis.

**Revisit trigger.** A class of models (e.g., MoE) consistently produces CV>15% on n=3 — escalate to n=5 for that class only.

---

## ADR-010 — Sequential single-experiment execution

**Decision.** Experiments run **sequentially**, one at a time, on a single accelerator. No parallel runs across cells, no multi-process, no distributed.

**Status.** Accepted (P8).

**Context.** Parallel execution is faster but introduces resource contention (HBM, PCIe, host CPU) that contaminates measurements. Our budget allows sequential.

**Decision rationale.**
- A clean number requires the device be exclusive to the run — sequential gives that for free.
- Single-card focus (ADR-003) and a 1–3 min run time mean total wall-clock for full suite (~800 cells × 3 runs × 90 s ≈ 60 hours) is tolerable on a weekly schedule.
- No multiprocess host overhead; no race in JSONL append (only one writer ever).
- Simpler harness, simpler debugging.

**Alternatives considered.**
1. *Parallel cells on the same device.* Rejected — contamination.
2. *Parallel cells across devices (TPU + GPU simultaneously).* Considered. Acceptable in principle but operationally complex (two harnesses, two cost streams, two failure modes); deferred until Stage 8+.

**Consequences.**
- Enables: clean measurements; simple harness; no contention bugs.
- Constrains: total wall-clock; cannot finish a full suite in a single Colab session.

**Risks.**
- Wall-clock drift from accumulated thermal load on a long sequential run. Mitigated by 30-second cool-down between cells, plus thermal pre-flight checks.

**Revisit trigger.** A future "scaling study" stage (10+) explicitly investigating parallel inference behaviour.

---

## ADR-011 — 9-stage incremental build (not big-bang)

**Decision.** The repo is built in **9 staged increments**. Each stage produces a runnable, end-to-end benchmark for a strict subset of models and paths, and informs the design of the next stage.

**Status.** Accepted (P14). **Stage 1 delivered 2026-04-29.**

**Context.** A 75-model × 5-path × multi-batch matrix built upfront is high-risk: by the time anything runs, dozens of design choices are locked in based on guesses, not measurements.

**Decision rationale.**
- Stage 1 produces a working harness with 5 models and 1 path, in days, not weeks.
- Each subsequent stage is informed by real failure modes from the previous (e.g., MoE failure in Stage 3 informs Stage 4's quantization scope).
- Risk of total project death is bounded — even if we stop after Stage 2 we have a real artifact.
- The repo is always in a runnable state; it always produces results; results inform the next ADR.

**Alternatives considered.**
1. *Build everything, run once.* Rejected — too much code without feedback; high probability of unused features.
2. *Two stages (MVP + full).* Considered. Rejected because the design space is too large for one mid-build pivot.

**Consequences.**
- Enables: feedback-driven refinement; bounded risk; always-runnable repo; clear stop-points.
- Constrains: requires discipline to not prematurely add Stage-N+1 features in Stage N.

**Risks.**
- Stage drift — Stage 1 takes 4 weeks instead of 1. Mitigated by exit-criteria gates per stage in `LESSON_PLAN.md`.
- Over-fitting Stage-1 design choices to 5 models. Mitigated by Stage 2 explicitly validating Path 1 generalises before extending.

**Revisit trigger.** A stage exit-criterion is missed twice — pause, re-plan.

---

## ADR-012 — 75-model registry with 4B parameter ceiling

**Decision.** The model registry contains **75 curated models**, with a hard parameter ceiling of **~4B**. Models above 4B are admitted only with explicit quantization (INT8, INT4) bringing weight memory under 8 GB.

**Status.** Accepted (P10/P14).

**Context.** v5e-1 is the primary target with 16 GB HBM. A 4B-BF16 model weighs ~8 GB, leaving ~6 GB for KV cache, activations, framework overhead. A 7B-BF16 model is 14 GB, leaving almost nothing.

**Decision rationale.**
- 4B is the largest size where every model fits BF16 on v5e-1 with realistic batch+seq workloads.
- 75 is the curated count from our model-justification pass — chosen for architecture diversity (attention variants, SSM, MoE, conv, hybrid), modality coverage (vision, NLP, audio, multimodal), and family coverage (Qwen, DeepSeek, Gemma, Phi, Llama, Mistral).
- Above 4B we admit selectively (e.g., Llama-3-8B-INT4) when the quantization story itself is the experiment.
- Larger ceiling would require v6e-1 from Stage 1, undoing ADR-003.

**Alternatives considered.**
1. *2B ceiling (matches RTX 3080 16 GB cleanly).* Rejected — excludes Phi-3-mini (3.8B), Gemma-2-2B-it ceiling, half the popular models.
2. *7B ceiling.* Rejected — forces v6e-1 from Stage 1, doubles cost, masks the bandwidth ceiling that's pedagogically important.
3. *Open ceiling, ad-hoc admission.* Rejected — encourages fishing for the biggest model that fits, not the most informative set.

**Consequences.**
- Enables: every BF16 model fits v5e-1; cost stays bounded; comparison surface stays meaningful.
- Constrains: no Llama-3-70B, no Mixtral-8x22B, no SDXL — these are not in scope for this hardware.

**Risks.**
- A reader expects "all popular models" and is disappointed by the ceiling. Mitigated by explicit ceiling statement on the model-list page.
- A high-value model lands at 5B and we want it. Mitigated by the quantization-admission clause.

**Revisit trigger.** v6e-1 becomes the primary target (ADR-003 revisit) OR a 5–7B model becomes pedagogically critical AND admits to INT4 cleanly.

---

## ADR-013 — GQA/MQA/SSM/MoE inclusion despite XLA challenges

**Decision.** The registry **deliberately includes** models with novel architectures that are XLA-hostile or otherwise hard to benchmark: GQA (Grouped Query Attention), MQA, SSM (Mamba/Mamba-2, RWKV), and MoE (Phi-3.5-MoE, DeepSeek-Coder-V2-Lite).

**Status.** Accepted (P11).

**Context.** "Novel architectures break our compiler" is itself a finding worth measuring. Excluding them would make the benchmark a tour of well-trodden CNN/transformer territory.

**Decision rationale.**
- A primary learning goal is _compiler/architecture interaction_. Models that break XLA are precisely where the lesson lives.
- GQA/MQA test KV-cache reduction in practice — claim transfer from the Llama-3 paper to our hardware.
- SSM (Mamba) tests the "no XLA primitive for selective_scan" hypothesis — directly supports the claim that some workloads should run on GPU not TPU.
- MoE tests dynamic-routing-vs-static-shape, the most-cited reason XLA struggles with large models.
- Excluding them would leave the conclusion "XLA is fine for everything" untested.

**Alternatives considered.**
1. *Transformer-only.* Rejected — leaves the most interesting compiler lessons unstudied.
2. *Include but disable.* Rejected — disabled benchmarks decay.
3. *Include in a separate "research suite", off the main matrix.* Considered. Accepted as a tagging refinement: these models carry a `risk: high` tag in `registry.yaml` so the smoke suite skips them by default.

**Consequences.**
- Enables: documented, evidence-backed claims about which architectures suit which device; a story worth telling.
- Constrains: more failure modes per stage; per-model investigation budget required (e.g., P32 for MoE).

**Risks.**
- A novel-arch failure blocks the whole suite. Mitigated by per-cell timeout + try/except + graceful skip + flagged "incompatible" entry in JSONL.
- Wasted hours debugging an architecture that's not central. Mitigated by the 4-hour-per-investigation cap (see P32 constraint).

**Revisit trigger.** A novel-arch class consumes >20% of total stage time without producing transferable insight — demote it.

---

## ADR-014 — Local OTel + Grafana for TPU-run observability

**Decision.** Instrument the benchmark harness with the OpenTelemetry SDK. On the TPU VM, a local `otelcol-contrib` (v0.105.0) receives OTLP/gRPC on `localhost:4317` and writes OTLP-JSON files to `results/otel/`. After the run, the user `scp`s those files back to the laptop (`./scripts/otel_collect.sh`) and replays them into a single-container Grafana stack (`grafana/otel-lgtm`) via a sidecar otelcol with the `otlpjsonfile` receiver (`./scripts/otel_view.sh`). No cloud, no live streaming.

**Status.** Accepted (Session 4, 2026-05-11).

**Context.** ADR-008 rejected Grafana for *results visualization* — the cross-experiment table of benchmark rows (model × precision × device × throughput) belongs in a static HTML + Vega-Lite dashboard served from GitHub Pages. That decision stands. ADR-014 addresses a different concern: **per-run telemetry** — phase timings, latency distributions, compile-cache hit/miss, and the internal Gantt of a single experiment. The static dashboard cannot answer "why did this one cell take 47 seconds in the compile phase?"; OTel + a query language (PromQL/TraceQL) can. The user requirement is strict: no cloud dependency, no always-on local service, and the workflow must survive losing the preemptible TPU mid-run.

**Decision rationale.**
- OTLP is the standard observability wire protocol — every backend (Tempo, Prometheus, Jaeger, even a no-op JSON file) accepts it. Future swaps are free.
- Writing OTLP-JSON to disk on the TPU and replaying locally decouples capture from visualization — a preempted TPU still leaves usable telemetry on disk up to the last flush.
- `grafana/otel-lgtm` packages Loki, Grafana, Tempo, and Prometheus in one container — operationally cheaper than running four services for a personal benchmark.
- Pinned `otelcol-contrib` v0.105.0 has both the `otlpjsonfile` receiver (replay side) and the `file` exporter (capture side) stable.
- Replay-from-files is debug-friendly: the same run can be inspected next week, on a different laptop, without re-running on a TPU.

**Alternatives considered.**
1. *Grafana Cloud.* Rejected by user — zero cloud telemetry dependency is a hard constraint.
2. *`jax.profiler` + TensorBoard.* Kept for Stage 3 (HLO/op-level analysis). Does not cover host-side phase timings or aggregate distributions across many runs, and TensorBoard's UI is not queryable.
3. *Custom timeline written to `runs.jsonl`.* Rejected — JSONL can record summary stats but cannot drive PromQL/TraceQL queries; we'd be reinventing a query engine.
4. *Live OTLP push from TPU to laptop over the gcloud SSH tunnel.* Rejected — flaky on preemption, requires tunnel to stay open for the full run, and adds a network failure mode to every benchmark.

**Consequences.**
- Enables: per-phase Gantt for a single experiment, latency p50/p95/p99 heatmaps over time, throughput-vs-precision compare in Grafana, compile-time breakdown (cold vs warm) — none of which fit the static dashboard.
- Constrains: requires Docker locally (already a soft assumption for dev work); adds `opentelemetry-*` to `requirements.txt`; adds ~30s to `provision_tpu.sh` (one-time otelcol-contrib download per VM); adds `results/otel/` to repo `.gitignore`.

**Risks.**
- `otelcol-contrib` version drift breaks the `otlpjsonfile` receiver or `file` exporter. Mitigation: pin v0.105.0 in both `provision_tpu.sh` and `infra/docker-compose.yml`; revisit on each minor release.
- `grafana/otel-lgtm` internal layout (mount paths, port assignments) shifts across releases. Mitigation: pin image tag in `infra/docker-compose.yml` with an explanatory comment.
- User forgets to start the collector on the TPU before the run. Mitigation: harness logs `OTel: writing to localhost:4317` at startup; the post-run `otel_collect.sh` warns if `results/otel/` is empty.

**Revisit trigger.** Per-run metric volume exceeds 1M records per `runs.jsonl` (summarization layer required), OR the user wants live-during-run monitoring (would push us back to Grafana Cloud, persistent Prometheus, or an always-on tunnel — re-cost at that point).

---