# Session State — TPU × GPU Benchmark Repo

Read this file at the start of every new session before doing anything else.
It contains the complete state of the project, decisions made, and what comes next.

---

## Identity

| Item | Value |
|------|-------|
| Repo | https://github.com/rajaghv-dev/tpu |
| Local path | `/home/raja/tpu` |
| GitHub user | `rajaghv-dev` (email: rajaghv.dev@gmail.com) |
| GitHub auth | `gh` CLI authenticated, HTTPS protocol |
| Branch | `main` |
| Last commit | `[update at commit time]` — Stage 1.5 observability layer + first successful TPU smoke run |
| Shell | WSL2 (Linux on Windows), bash, /home/raja/tpu |

---

## What This Repo Is

Single repo to understand and benchmark inference across:
- **Hardware:** Google TPU (v2/v3/v4/v5e/v5p/v6e) vs NVIDIA GPU (RTX 3080, RTX 4090, DGX B200)
- **Frameworks:** JAX/Flax, PyTorch, torch_xla, HuggingFace Inference API (5 execution paths)
- **Models:** ~75 models across vision, NLP (encoder/decoder/SSM/MoE), audio, multimodal
- **Dimensions:** precision (FP32/BF16/INT8/INT4/FP8), compilation, sparsity, LLM decode modes
- **Goal:** evidence-backed claims with full traceability (profiler traces, lineage, statistics)

---

## User Profile

| Item | Detail |
|------|--------|
| Name | Raja |
| Location | India |
| Hardware | RTX 3080 (16GB), RTX 4090 (24GB), DGX Dell B200 Blackwell (192GB HBM3e, 256GB system RAM) |
| Cloud | Google Colab Pro, GCP personal account (rajaghv@gmail.com), $300 free credits |
| HuggingFace | Paid PRO account — gated model access (Gemma, PaliGemma, LLaMA) + Inference API |
| Expertise | Learning from essentials to expert; wants depth across hardware/compiler/model |
| Style | Prefers discussion before coding; wants MD documentation files; no fluff |

---

## Current Repo File Structure

```
/home/raja/tpu/
├── 01_hello_tpu/                    hello_tpu.py + README.md
├── 02_mnist_classification/         train.py + README.md
├── 03_resnet_imagenet/              model.py + train.py + README.md
├── 04_bert_finetuning/              train.py + README.md
├── 05_gpt_pretraining/              model.py + train.py + README.md
├── 06_data_pipeline/                pipeline.py + README.md
├── 07_custom_training_loop/         train.py + README.md
├── 08_multi_host/                   train.py + README.md
├── benchmarks/                      ✅ Stage 1 — harness.py + runner.py (BenchmarkError + phase ctx)
├── models/                          ✅ Stage 1 — registry.yaml (5 models)
├── observe/                         ✅ Stage 1+1.5 — stats.py + lineage.py + compile_controller.py
│                                       + probe.py (ABC + registry + 5 fanout helpers)
│                                       + otel_probe.py + cloud_monitoring_probe.py
│                                       + timing_probe.py + memory_probe.py
│                                       + input_fingerprint_probe.py + hlo_dump_probe.py
│                                       + jax_profiler_probe.py
├── results/                         runs.jsonl + dashboard/index.html
│   ├── dashboard/grafana/           ✅ Stage 1.5 — 5 importable JSON dashboards
│   │   (roofline, mxu_heatmap, latency_violins, failures, cost)
│   └── run_logs/<run_id>/           per-run probe outputs + error.json on failure
├── tests/                           ✅ ~180 unit tests (was 124 before probe layer)
├── scripts/                         ✅ 22 staged scripts + run_all.sh master + lib/
│   ├── 00_validate_local.sh         01_validate_gcp.sh   02_validate_bucket.sh
│   ├── 10_setup_bucket.sh           11_setup_budget.sh
│   ├── 20_provision_tpu.sh          21_provision_fallback.sh   22_wait_ready.sh
│   ├── 30_deploy_code.sh            31_install_deps.sh   32_verify_env.sh
│   ├── 40_smoke_jax.sh              41_smoke_model.sh   42_dry_run.sh
│   ├── 50_run_smoke.sh              51_run_quick.sh
│   ├── 60_pull_results.sh           61_render_results.sh
│   ├── 70_teardown.sh               71_verify_teardown.sh
│   ├── 90_cost_report.sh            91_dump_logs.sh   92_idle_check.sh
│   ├── run_all.sh                   master pipeline
│   ├── render_results.py            generates RESULTS.md + per-run REPORT.md
│   └── lib/{common,config}.sh
├── README.md                        Landing page — Quick Start, harness table, all domains
├── LESSON_PLAN.md                   15-module beginner→expert curriculum
├── DECISIONS.md                     13 ADRs
├── RISKS.md                         25+ risks
├── QUESTIONS.md                     23 open questions
├── RECOMMENDATIONS.md               3-tier prioritised actions
├── context.md                       Full project context (700+ lines)
├── prompts.md                       All prompts P1–P49 + standing instructions
├── SESSION.md                       This file — session continuity
├── MEMORY.md                        Dense 3-min fast startup reference
└── requirements.txt
```

**Session 4 additions (2026-05-06) — Stage 1.5 + first TPU smoke run COMPLETE:**

- GCP setup: gcloud CLI installed, authenticated as rajaghv@gmail.com, project `nellaiappar-001`.
  v5e quota confirmed in us-east5 / us-west1 / us-west4 (per-region, 1536-chip preemptible limit).
- 22 staged scripts: `scripts/00_validate_local.sh` … `92_idle_check.sh` + `run_all.sh` master +
  `lib/{common,config}.sh`. Pipeline = validate → bucket → provision (multi-zone fallback) →
  deploy → install → verify → smoke/quick → pull results → teardown → verify $0/hr.
- BenchmarkError class in `benchmarks/runner.py` (phase + error_category attrs); per-phase context
  manager; `error.json` written to `results/run_logs/<run_id>/`; failure stub appended to
  `runs.jsonl` with `status="failed"` and `run_id` so REPORT.md can link.
- Probe-based observability layer (the biggest addition):
  - `observe/probe.py` — `Probe` ABC, registry, 5 fanout helpers wired into
    `runner.run_experiment` and `phase()`. Contract: `before_run`, `after_run`,
    `before_phase`, `after_phase`, `on_error`, `write_log`. Fanout swallows probe
    exceptions so a buggy probe never fails a benchmark.
  - 7 built-in probes: `OTelProbe`, `CloudMonitoringProbe`, `TimingProbe`, `MemoryProbe`,
    `InputFingerprintProbe`, `HloDumpProbe`, `JaxProfilerProbe`. Each writes
    `<probe_name>.json` to the run_log dir.
  - 5 importable Grafana dashboards at `results/dashboard/grafana/`
    (roofline, mxu_heatmap, latency_violins, failures, cost).
  - Test count: 124 → ~180 (+60 new probe tests).
- `scripts/render_results.py` — generates RESULTS.md + per-run REPORT.md from runs.jsonl + run_logs.
- Bug fixes during the session:
  - `01_validate_gcp.sh` IAM heuristic: older gcloud lacks `test-iam-permissions`; replaced
    with read-probe heuristic.
  - `42_dry_run.sh` / `50_run_smoke.sh` / `51_run_quick.sh`: switched from
    `python3 benchmarks/harness.py` to `python3 -m benchmarks.harness` for proper package resolution.
  - `run_all.sh` stage order: removed `02_validate_bucket.sh` from the pipeline because it ran
    before `10_setup_bucket.sh`; soft-fail handling for stage 11 budget setup.
  - `harness.py` failure stub now includes `run_id` so REPORT.md can link.

**First successful TPU smoke run (2026-05-06, v5e-1 in us-west4-a):**

| Metric | Value |
|---|---|
| Model / precision | BERT-base, BF16 |
| Latency p50 / p95 / p99 | 0.64 / 0.66 / 0.66 ms (CV 1.31%, well under the 10% gate) |
| Throughput | 5,261 samples/sec ± 7.8 |
| Compile (cold / warm) | 5.20 s / 0.0008 s (cache hit working) |
| Cost per experiment / per 1k samples | $0.00056 / $0.000019 |
| Test session total / post-teardown $/hr | ~$0.12 / $0/hr verified clean |

Versions pinned: jax 0.6.2, transformers 4.44.2 (4.45+ removed Flax).

**Session 3 additions (2026-04-29) — Stage 1 COMPLETE:**
- benchmarks/harness.py — CLI: --suite smoke/quick, --model, --dry-run, --device, --precision
- benchmarks/runner.py — ExperimentConfig dataclass, make_synthetic_inputs, run_experiment (9-phase)
- models/registry.yaml — 5 Stage 1 models with full input specs
- observe/stats.py — MAD-based outlier removal, p50/p95/p99, CV<10% (gap C2)
- observe/lineage.py — git SHA + package versions + HF model revision + env hash
- observe/compile_controller.py — XLA cache clear, cold + warm compile timing (gap C3)
- results/dashboard/index.html — static sortable/filterable HTML table dashboard
- tests/ — 97 unit tests, all pass (no JAX/GPU required)
- All MD files updated: README.md, SESSION.md, MEMORY.md, context.md, DECISIONS.md,
  RECOMMENDATIONS.md, prompts.md — aligned with actual code state

**Session 2 additions (2026-04-26):**
- prompts.md: P1–P21 Opus-rewritten; P22–P47 added; Standing Instructions section; Session 2 raw prompts recorded
- context.md: §16 Artifact Catalog, §17 Colab Pro+HF Workflow, §18 Open Questions, comparability column, exit criteria per stage
- All 8 example READMEs: hardware context, real metrics, "What to observe", benchmark connections
- DECISIONS.md (new): 13 ADRs with full rationale, alternatives, risks, revisit triggers
- RISKS.md (new): 25+ risks across 5 categories with mitigation + contingency
- QUESTIONS.md (new): 23 open questions with expected answers and test plans
- RECOMMENDATIONS.md (new): 22+ recommendations in 3 tiers with effort estimates
- LESSON_PLAN.md: Module 0 added (repo organisation, ADR table, framework hierarchy, terminology index)

---

## Committed but Not Yet Pushed in This Session

At time of writing: LESSON_PLAN.md, SESSION.md, MEMORY.md, prompts.md updates are
staged but not committed. The push command is at the end of this session's work.

---

## The 5 Execution Paths

| Path | Framework | Compiler | Hardware |
|------|-----------|---------|---------|
| 1 | JAX/Flax | XLA | TPU |
| 2 | JAX/Flax | XLA (CUDA) | GPU |
| 3 | PyTorch | CUDA/cuDNN | GPU |
| 4 | PyTorch + torch_xla | XLA | TPU |
| 5 | HTTP | HF-managed | Unknown |

---

## Staged Build Plan — Current Status

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Foundation: harness.py, runner.py, 5 models, Path 1, table dashboard | **COMPLETE (2026-04-29)** |
| 1.5 | Probe layer (Probe ABC + 7 probes + Grafana dashboards) + GCP scripts + first TPU smoke run | **COMPLETE 2026-05-06** |
| 2 | Multi-path: Paths 2+3, system_monitor, 15 models, heatmap dashboard | Not started |
| 3 | Profiler + Roofline: flops_counter, tracer, hlo_analyser | Not started |
| 4 | torch_xla: Path 4 | Not started |
| 5 | Novel architectures: Mamba, RWKV, DiT | Not started |
| 6 | Precision + Quantization: INT8, FP8, numerics | Not started |
| 7 | HF Inference API: Path 5 | Not started |
| 8 | Sparsity + Pruning | Not started |
| 9 | Full registry + GitHub Actions automation | Not started |

**Next coding session starts at Stage 2** (Paths 2+3, observe/system_monitor.py, 15 models, heatmap dashboard).

---

## Stage 1.5 — Observability Layer (this session)

The probe layer is the extension point for everything observability-shaped. Runner core stays small;
new telemetry is added by writing a Probe subclass and registering it — no edits to `runner.py`.

**Probe contract (from `observe/probe.py`):**

```
class Probe(ABC):
    name: str
    def before_run(ctx): ...     # called once at run_experiment start
    def after_run(ctx, result): ...
    def before_phase(ctx, phase_name): ...
    def after_phase(ctx, phase_name, elapsed): ...
    def on_error(ctx, exc): ...
    def write_log(run_log_dir): ...   # called at end; emits <name>.json
```

**Built-in probes (in `observe/`):**

| Probe | Captures |
|---|---|
| OTelProbe | Spans + histograms + counters via OTLP |
| CloudMonitoringProbe | TPU MXU% / GPU SM% / power / thermal from cloud APIs |
| TimingProbe | Per-phase wall-clock, cold-vs-warm split |
| MemoryProbe | Peak HBM, allocator timeline |
| InputFingerprintProbe | Input tensor SHA + shape + dtype (lineage) |
| HloDumpProbe | XLA HLO text + after-optimization dump |
| JaxProfilerProbe | jax.profiler trace.pb (open in TensorBoard) |

**Register a probe in 5 lines:**

```python
from observe.probe import register
from observe.timing_probe import TimingProbe
register(TimingProbe())
# now run benchmarks; <run_log_dir>/timing_probe.json appears automatically
```

---

## Key Decisions Made (do not re-discuss)

| Decision | Choice | Reason |
|----------|--------|--------|
| Framework | JAX + PyTorch (both) | JAX for clean hardware comparison; PyTorch for real-world GPU parity |
| TPU target | v5e-1 preemptible ($0.36/hr) primary; v6e-1 secondary | Single-chip VMs; cost-effective |
| Model scope | ~75 models; up to 4B params on v5e-1 BF16 | v5e-1 has 16GB; 4B BF16 = ~8GB weights |
| Input data | Synthetic random (numpy seed = 42) | No data download; hardware-only comparison |
| Model weights | HuggingFace pretrained | Real weights give realistic compute patterns |
| Model cache | GCS bucket (rajaghv@gmail.com) | Download once; reuse across VM restarts |
| Results format | Append-only JSONL | Simple, grep-able, version-controllable |
| Dashboard | Static HTML + Vega-Lite (GitHub Pages) | No server; zero cost; shareable |
| Run mode | Sequential, 1–3 min per experiment | Colab-compatible; no parallel complexity |
| Statistics | n=3 independent runs; CV < 10% threshold | Evidence-grade measurements |
| Inference only | Yes — no training benchmarks in harness | Cleaner; faster experiments |

---

## Hardware Reference (Quick)

| Card | Memory | BW | BF16 TFLOPs | Notes |
|------|--------|-----|-------------|-------|
| RTX 3080 | 16GB GDDR6X | 760 GB/s | 119 | Local |
| RTX 4090 | 24GB GDDR6X | 1008 GB/s | 330 | Local; FP8 |
| DGX B200 | 192GB HBM3e | 4000 GB/s | 2250 | Local; Blackwell; 1000W TDP |
| TPU v5e-1 | 16GB HBM | 820 GB/s | 394 | $0.36/hr preemptible |
| TPU v6e-1 | 32GB HBM | 1640 GB/s | 918 | ~$0.75/hr preemptible |

---

## All 15 Gaps — Status

All 15 gaps identified and assigned. See `context.md` Section 13 for full table.
Short summary: C1–C5 (critical) addressed in Stages 1–3; I1–I7 (important) in Stages 1–7; N1–N4 in later stages.

---

## How to Push to GitHub

```bash
git add -A
git commit -m "message"
git push origin main
```

`gh` CLI is authenticated as `rajaghv-dev`. No password prompt needed.

---

## Stage 1 Build — Completed 2026-04-29

**Files created:**
- `benchmarks/harness.py` — CLI: `--suite smoke/quick`, `--model`, `--dry-run`, appends to JSONL
- `benchmarks/runner.py` — ExperimentConfig dataclass, make_synthetic_inputs, run_experiment (9 phases)
- `models/registry.yaml` — 5 Stage 1 models: bert_base, vit_b16, gpt2, whisper_base, clip_vit_b32
- `observe/stats.py` — MAD-based iterative outlier removal, p50/p95/p99, CV check (gap C2)
- `observe/lineage.py` — git SHA, package versions, HF model revision, environment hash (gap C3 support)
- `observe/compile_controller.py` — XLA cache clear, cold + warm compile timing (gap C3)
- `results/dashboard/index.html` — static sortable HTML table dashboard (no server needed)
- `results/runs.jsonl` — empty; populated when harness runs on TPU
- `tests/` — 97 unit tests covering all Stage 1 modules (no JAX/GPU required)

**Test status:** 97/97 pass (`pytest tests/`)

**Stage 1 status:** Code complete. Harness ready to run on v5e-1.
**Stage 2 next:** Paths 2+3, system_monitor.py, 15 models, heatmap dashboard.

---

## What to Do at Session Start

1. Read MEMORY.md (2 min) for the dense summary
2. Read this SESSION.md file (5 min) for complete state
3. Check `git log --oneline` to see what was last committed
4. Check `git status` to see if anything is uncommitted
5. Ask the user what they want to do today, then proceed

Do NOT re-read context.md from scratch every session — it is 700+ lines.
Use MEMORY.md as the quick reference; context.md as the deep reference.
