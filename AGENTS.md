# AGENTS.md

TPU × GPU Inference Benchmark — agent quick-reference.
Read docs/agent-handoff.md for the full handoff state.

## Repo purpose

Rigorous TPU vs GPU inference benchmark. 75 models. 5 execution paths. 9 experiment dimensions. Full traceability. Companion 15-module curriculum.

Two sub-projects:
- Root `/` — benchmark harness, observe/ probes, results
- `cloud_tpu_lab/` — CPU-simulation learning lab (no TPU needed)

## Safe files to edit

- `docs/` — any documentation file
- `observe/*.py` — probe implementations (no public API changes without updating tests)
- `tests/` — add or fix tests
- `benchmarks/harness.py` — suites dict, DEVICE_COSTS (not run_experiment)
- `train/harness.py` — suites dict (not run_training)
- `Makefile` — targets and commands
- `.github/workflows/` — CI jobs
- `scripts/` — shell scripts (test with --dry-run first)
- `models/registry.yaml` — add models (don't remove existing)
- `train/registry.yaml` — add tasks (don't remove existing)
- `MEMORY.md` — update session notes
- `SESSION.md` — update state

## Files requiring human approval

- `benchmarks/runner.py` — core experiment logic; behavioral changes need approval
- `train/runner.py` — core training logic; behavioral changes need approval
- `observe/probe.py` — Probe ABC; changes affect all probes
- `observe/otel.py` — OTel init; changes affect all probes that use OTel
- `results/runs.jsonl` — append-only; never edit, never delete
- `requirements.txt` — dependency changes need approval (Flax/transformers pin)
- `requirements.stage1.lock.txt` — regenerate only after full install validation
- `scripts/70_teardown_tpu.sh` / `scripts/kill_all_tpus.sh` — destructive GCP ops

## Build command

```bash
# No build step for root project (not packaged)
# cloud_tpu_lab:
cd cloud_tpu_lab && pip install -e .
```

## Test command

```bash
# Root tests (no GPU/TPU needed):
python3 -m pytest tests/ -q   # 269 passed, 0 failed, 6 skipped

# cloud_tpu_lab tests:
cd cloud_tpu_lab && python3 -m pytest tests/ -q   # requires: pip install -e .

# With coverage:
python3 -m pytest tests/ --cov=benchmarks --cov=observe --cov=train -q
```

## Validation command

```bash
# Harness dry-run (validates config, registry, CLI parsing — no downloads):
JAX_PLATFORMS=cpu python3 -m benchmarks.harness --suite quick --device cpu --dry-run

# Training harness dry-run:
JAX_PLATFORMS=cpu python3 -m train.harness --suite smoke --device cpu --dry-run

# Lint:
python3 -m flake8 benchmarks/ observe/ tests/ train/ --max-line-length 100

# Lineage check:
python3 -c "from observe.lineage import get_git_sha; print(get_git_sha())"
```

## Coding style

- Python 3.12, type annotations on all public functions
- No comments unless the WHY is non-obvious
- No extra abstractions beyond what the task requires
- Lazy imports for optional dependencies (opentelemetry, psutil, jax)
- Tests must run without GPU/TPU (use mocks in conftest.py)
- `python3` not `python` in all commands

## Refactor rules

- Do not rename public CLIs or Python APIs without updating all references
- Do not change ExperimentConfig or TrainingExperimentConfig field names (results schema impact)
- Do not change runs.jsonl schema (append-only, historical data)
- Do not remove models from registry.yaml
- Do not change probe ABC hooks without updating all 14 probe implementations
- Do not change probe output filenames (<name>.json) without updating RESULTS.md generator

## Security rules

- Never print or log secret values (HF tokens, GCP keys)
- Never commit .env, .hf-token, .claude/ contents
- Never bypass CI (--no-verify)
- Exclude .claude/ and .tpu/ from any remote copy commands
- Validate TPU_NAME and ZONE inputs before use in shell commands

## Known risks

- `transformers>=4.40,<4.45` pin — Flax model classes removed in 4.45+; do not lift without migrating harness
- `observe/otel.py` lazy-import fix applied (2026-05-16) — 0 test failures; opentelemetry now in requirements.stage1.lock.txt
- `requirements.stage1.lock.txt` regenerated 2026-05-16 with 24 packages including opentelemetry
- `scripts/30_deploy_repo.sh` does not exclude .claude/ from deployment tarball

## Current TODOs

1. ✅ Fixed observe/otel.py eager import — 269 passed, 0 failed, 6 skipped (2026-05-16)
2. ✅ Regenerated requirements.stage1.lock.txt — 24 packages including opentelemetry (2026-05-16)
3. Create LICENSE, CONTRIBUTING.md, .env.example
4. Add 4 undocumented probes to docs (DeterminismProbe, DeviceInfoProbe, PowerThermalProbe, XlaCompileProbe)
5. Fix Makefile: python → python3
6. Fix 30_deploy_repo.sh: add .claude/ exclusion
7. Add Stage 2: Paths 2+3 (JAX+GPU, PyTorch+GPU), system_monitor.py

## Next recommended tasks

1. Create missing root files (LICENSE, CONTRIBUTING.md, .env.example)
2. Update observe/README.md with 4 new probes
3. Start Stage 2: add observe/system_monitor.py for GPU SM% / MXU% eager counters
