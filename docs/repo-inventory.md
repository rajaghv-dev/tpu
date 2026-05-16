# Repo Inventory

Generated: 2026-05-16 — Phase 0 baseline.

---

## Repo purpose

A rigorous, reproducible inference benchmark comparing Google Cloud TPU and NVIDIA GPU performance across ~75 models, 5 execution paths, and 9 experiment dimensions. Every claim is backed by profiler traces, statistical analysis, and full lineage. A companion educational curriculum (15 modules) teaches the full TPU/GPU stack from hardware through XLA to production cost.

Two sub-projects live in the same repo:

| Sub-project | Root | Purpose |
|---|---|---|
| **Inference benchmark** | `/` (root) | Benchmark harness, observe/ probes, 75-model registry, results |
| **cloud_tpu_lab** | `cloud_tpu_lab/` | Hands-on learning lab with CPU simulation, no TPU required |

---

## Languages and frameworks detected

| Language | Files | Use |
|---|---|---|
| Python 3.12 | `benchmarks/`, `observe/`, `tests/`, `train/`, `scripts/*.py` | All source code |
| YAML | `models/registry.yaml`, `train/registry.yaml`, `cloud_tpu_lab/configs/` | Config + CI |
| Bash | `scripts/*.sh` | GCP provisioning, TPU lifecycle, cache management |
| Jupyter | `colab/tpu_benchmark.ipynb`, `cloud_tpu_lab/notebooks/*.ipynb` | Interactive runs |
| JSON | `results/`, `infra/grafana/dashboards/` | Results data + Grafana dashboards |
| Markdown | `docs/`, `DECISIONS.md`, `MEMORY.md`, `context.md`, etc. | Documentation |
| TOML | `cloud_tpu_lab/pyproject.toml` | Package build config |
| Docker Compose | `infra/docker-compose.yml` | Grafana + OTel Collector stack |

Key frameworks:
- **JAX / Flax / Optax / Orbax** — primary ML runtime
- **HuggingFace Transformers** (pinned `>=4.40,<4.45` — Flax model constraint)
- **OpenTelemetry** (`opentelemetry-api/sdk/exporter-otlp-proto-grpc`)
- **Grafana + Prometheus** — local observability stack via Docker Compose
- **pytest** — test runner

---

## Main entry points

| Entry point | Command | Purpose |
|---|---|---|
| Inference harness | `python3 -m benchmarks.harness --suite smoke --device cpu` | Run benchmarks |
| Training harness | `python3 -m train.harness --suite smoke --device cpu` | Training observability |
| CPU demo (cloud_tpu_lab) | `python3 cloud_tpu_lab/examples/run_cpu_simulation_demo.py` | No-TPU demo |
| Results dashboard | `make dashboard` → `http://localhost:8080` | View results |
| Grafana stack | `make otel-view` → `http://localhost:3000` | OTel + Grafana |
| Dry-run harness | `make dry-run` | Validate config without running |

---

## Build system

| Tool | File | Purpose |
|---|---|---|
| `make` | `Makefile` (root) | Test, benchmark, lint, GCP, observability targets |
| `make` | `cloud_tpu_lab/Makefile` | Lab-specific targets |
| `setuptools` | `cloud_tpu_lab/pyproject.toml` | cloud_tpu_lab installable package |
| `pip` | `requirements.txt` | Root project dependencies |
| `pip` | `requirements.stage1.lock.txt` | Locked Stage 1 deps (used by CI) |

No root-level `pyproject.toml` — root project is **not packaged**, only cloud_tpu_lab is.

---

## Runtime dependencies

From `requirements.txt` (root):

| Package | Version constraint | Use |
|---|---|---|
| `jax[tpu]` | `>=0.4.25` | Core ML runtime |
| `flax` | `>=0.8.3` | Neural net library |
| `optax` | `>=0.2.2` | Optimizers |
| `orbax-checkpoint` | `>=0.6.0` | Checkpointing |
| `opentelemetry-api/sdk` | `>=1.27` | Observability |
| `opentelemetry-exporter-otlp-proto-grpc` | `>=1.27` | OTLP export |
| `tensorflow` | `>=2.15.0` | Data pipeline |
| `transformers` | `>=4.40,<4.45` | **Pinned** — Flax model compatibility |
| `datasets` | `>=2.19.0` | HF datasets |
| `pyyaml` | `>=6.0` | Registry loading |
| `rich` | `>=13.7` | Progress display |
| `pytest` / `pytest-cov` | `>=7.4 / >=4.1` | Testing |

`requirements.stage1.lock.txt` is the pip-frozen snapshot used by GitHub CI.

---

## Test framework

| Location | Test count (baseline) | Runner |
|---|---|---|
| `tests/` (root) | 272 total — **255 pass, 17 fail** | `python3 -m pytest tests/ -q` |
| `cloud_tpu_lab/tests/` | 8 test files | `cd cloud_tpu_lab && python3 -m pytest tests/ -q` |

**17 failing tests** at baseline — all caused by `opentelemetry` not installed in `.tpu` venv:
- `tests/test_otel.py` (10 failures)
- `tests/test_runner.py` (7 failures — runner eagerly imports OTel on startup)

These pass in CI via `requirements.stage1.lock.txt` (which pins OTel). Local `.tpu` venv is missing OTel.

---

## CI/CD workflows

| File | Trigger | Jobs | Coverage |
|---|---|---|---|
| `.github/workflows/smoke_on_push.yml` | push/PR to `main`, manual | `unit-tests`, `dry-run-harness` | pytest + lineage sanity + harness dry-run |

**Gaps:**
- No lint CI (`flake8` is only in `Makefile`)
- No cloud_tpu_lab test CI
- No release workflow
- No dependabot config
- No PR/issue templates

---

## Documentation files

| File | Purpose | Quality |
|---|---|---|
| `README.md` | Project overview, hardware specs, quick start | Excellent — detailed and current |
| `MEMORY.md` | Dense session startup reference | Excellent — comprehensive |
| `DECISIONS.md` | 13 ADRs with rationale | Excellent |
| `RISKS.md` | Risk register | Good |
| `QUESTIONS.md` | 23 open questions | Good |
| `RECOMMENDATIONS.md` | Tier 1/2/3 action items | Good |
| `SESSION.md` | Session state / identity | Good, but has stale `[update at commit time]` placeholder |
| `context.md` | Deep reference doc (hardware, models, protocol, costs) | Excellent, 2026-04-29 |
| `LESSON_PLAN.md` | 15-module curriculum | Good |
| `todo.md` | Provisioning bug report + fix plan | Situational |
| `prompts.md` | P22-P40 prompt library | Good |
| `observe/README.md` | Probe layer overview | Good |
| `scripts/README.md` | Script pipeline overview | Present |
| `docs/runbooks/tier3_tpu_session.md` | Tier 3 TPU session runbook | Present |
| `cloud_tpu_lab/docs/` | 16 learning modules (00-15) | Good — 16 markdown files |
| `cloud_tpu_lab/README.md` | Lab quick start | Good |
| `cloud_tpu_lab/CHANGELOG.md` | Lab version history | Good |
| `cloud_tpu_lab/TODO.md` | Lab TODO list | Present |
| `results/RESULTS.md` | Auto-generated benchmark results | 2 runs (1 pass, 1 fail) |

**Missing:**
- `CONTRIBUTING.md` — no contributor guide
- `CHANGELOG.md` at root — no root-level changelog
- `LICENSE` — no license file at root (cloud_tpu_lab pyproject.toml says MIT but no LICENSE file)
- `docs/architecture.md` — not present
- `docs/setup.md` — not present (setup info is in README)
- `.env.example` — not present
- `AGENTS.md` — not present

---

## Public APIs / CLIs / services

| Interface | Type | Module | Status |
|---|---|---|---|
| `benchmarks/harness.py` | CLI (`--suite`, `--device`, `--dry-run`) | root | ✅ |
| `benchmarks/runner.py` | Python API (`run_experiment`) | root | ✅ |
| `train/harness.py` | CLI (`--suite`, `--device`) | root | ✅ |
| `train/runner.py` | Python API | root | ✅ |
| `observe/probe.py` | Probe ABC + `register_probe` | root | ✅ |
| `observe/lineage.py` | `get_git_sha()` | root | ✅ |
| `observe/stats.py` | `compute_stats()` | root | ✅ |
| `observe/otel.py` | `get_meter()`, `get_instruments()` | root | ✅ |
| `cloud_tpu_lab/examples/run_cpu_simulation_demo.py` | CLI | cloud_tpu_lab | ✅ |
| Results dashboard | HTTP server at port 8080 | `results/dashboard/` | ✅ |
| Grafana stack | HTTP server at port 3000 | `infra/docker-compose.yml` | ✅ |

---

## Configuration files

| File | Purpose |
|---|---|
| `models/registry.yaml` | Inference model registry (~75 models) |
| `train/registry.yaml` | Training model registry (bert_finetune) |
| `cloud_tpu_lab/configs/` | Lab simulation configs |
| `infra/otelcol-tpu-config.yaml` | OTel Collector config for TPU VM |
| `infra/otelcol-replay-config.yaml` | OTel Collector config for local replay |
| `.github/workflows/smoke_on_push.yml` | CI workflow |
| `.claude/settings.local.json` | Claude Code local settings |
| `scripts/lib/config.sh` | Shared bash config (GCP project, zone, bucket) |
| `scripts/lib/common.sh` | Shared bash helpers |

**Security note:** `.claude/settings.local.json` was previously SCP'd to TPU VMs (see `todo.md`). It is now in `.gitignore`.

---

## Examples and demos

| Example | Location | Status |
|---|---|---|
| CPU simulation demo | `cloud_tpu_lab/examples/run_cpu_simulation_demo.py` | ✅ Runnable (no TPU) |
| 12 Jupyter notebooks | `cloud_tpu_lab/notebooks/*.ipynb` | Present — execution status unknown |
| Colab notebook | `colab/tpu_benchmark.ipynb` | Present — TPU/GPU runtime needed |
| Training examples | `01_hello_tpu/` through `08_multi_host/` | Present — require TPU/JAX install |
| Harness dry-run | `make dry-run` | ✅ Runnable (CPU, no download) |

---

## Deployment artifacts

| Artifact | Location | Purpose |
|---|---|---|
| GCP provisioning scripts | `scripts/20_provision_tpu.sh`, etc. | TPU VM lifecycle |
| GCS cache scripts | `scripts/cache_*.sh` | Wheel / model / XLA cache |
| Docker Compose stack | `infra/docker-compose.yml` | Local Grafana + OTelCol |
| Grafana dashboards | `infra/grafana/dashboards/` + `results/dashboard/grafana/` | Visualization |
| Lock file | `requirements.stage1.lock.txt` | Reproducible CI env |

---

## Observability / logging / metrics / tracing

| Component | Technology | Status |
|---|---|---|
| Per-run JSONL log | `results/run_logs/<run_id>/` | ✅ |
| OTel spans + metrics | `observe/otel.py` + OTLP exporter | ✅ (requires OTel deps) |
| Grafana dashboards | `infra/grafana/` + `results/dashboard/` | ✅ (5 dashboards) |
| Probe layer | `observe/probe.py` + 10 probes | ✅ |
| Chrome-trace profiler | `cloud_tpu_lab/src/profiling/` | ✅ (simulation only) |
| Cloud Monitoring | `observe/cloud_monitoring_probe.py` | ✅ (requires GCP auth) |
| JAX profiler | `observe/jax_profiler_probe.py` | ✅ (TPU VM only) |
| HLO dump | `observe/hlo_dump_probe.py` | ✅ (TPU VM only) |

---

## Security-sensitive files

| File/Pattern | Sensitivity | Gitignore? |
|---|---|---|
| `.env` | API keys | ✅ |
| `.hf-token`, `*.hf-token` | HuggingFace token | ✅ |
| `.tpu/` | Local venv | ✅ |
| `.claude/` | Claude Code session state | ✅ |
| `results/otel/` | Run telemetry | ✅ |
| `scripts/lib/config.sh` | GCP project ID, bucket, zone | Not secret but env-specific |
| GCP credentials | gcloud auth (system-level) | Not in repo |

**Gaps:**
- No `.env.example` to document required env vars
- `scripts/lib/config.sh` hardcodes GCP project / zone / bucket values — not secret but should be templated

---

## Known gaps

| Gap | Severity | Notes |
|---|---|---|
| 17 failing tests (OTel missing from `.tpu` venv) | High | `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc` fixes locally |
| No `LICENSE` file at root | Medium | cloud_tpu_lab declares MIT in pyproject.toml |
| No `CONTRIBUTING.md` | Medium | No contributor guide |
| No root `CHANGELOG.md` | Low | cloud_tpu_lab has one; root does not |
| No `.env.example` | Medium | Required env vars undocumented |
| No `AGENTS.md` | Medium | No agent-friendly repo guide |
| No `docs/architecture.md` | Medium | Architecture is in README and context.md only |
| No lint CI | Low | `make lint` exists but not in CI |
| No cloud_tpu_lab tests in CI | Low | Only root tests are CI-validated |
| No PR/issue templates | Low | `.github/` only has workflows |
| `SESSION.md` has `[update at commit time]` placeholder | Low | Stale |
| `transformers<4.45` pin | Known risk | Flax model removal — documented in requirements.txt |
| Stage 2–9 not implemented | By design | Staged build plan, Stage 1.6 is current |

---

## Initial git state

```
Branch: main
Remote: https://github.com/rajaghv-dev/tpu.git
Status: clean (nothing to commit)
Last commits:
  f725bf8  train + observe: stage 1.6+ — multi-task training + deeper probes
  f2b1b19  cloud_tpu_lab: phase 2 notebooks — 12 run-along Jupyter notebooks
  0a7aefc  cloud_tpu_lab: phase 2 — docs, GCP scripts, observability stack, more examples
  031f2d9  cloud_tpu_lab: initial vertical slice (CPU simulation, no TPU needed)
  0854538  Fix: strip inner quote-wrapping from all --command= values
```

Test baseline (`.tpu` venv, `python3 -m pytest tests/ -q`):
- 255 passed, 17 failed, 3 skipped
- All 17 failures: `ModuleNotFoundError: No module named 'opentelemetry'`
- CI uses `requirements.stage1.lock.txt` which pins OTel — CI is unaffected
