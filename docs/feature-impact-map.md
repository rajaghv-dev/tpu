# Feature Impact Map

Generated: 2026-05-16. Use this before any feature modification to minimize token cost.

## How to use

Before modifying any feature:
1. Read `AGENTS.md` (rules)
2. Read `docs/repo-inventory.md` (structure)
3. Read `docs/architecture.md` (data flow)
4. Read `docs/interfaces.md` (public APIs)
5. Read this file — find the feature in the map below
6. Read ONLY the 3–7 files listed as "files likely affected"
7. Create/update this file with a specific section for the feature
8. Make the smallest safe change
9. Run narrow tests first, then broader tests

## File ownership map

| Component | Primary files | Test files | Doc files |
|---|---|---|---|
| Inference runner | benchmarks/runner.py | tests/test_runner.py | docs/architecture.md |
| Inference harness | benchmarks/harness.py | tests/test_harness.py | docs/interfaces.md |
| Training runner | train/runner.py | tests/test_train_runner.py | docs/architecture.md |
| Training harness | train/harness.py | tests/test_harness.py | docs/interfaces.md |
| Probe ABC | observe/probe.py | tests/test_app_probes.py | observe/README.md |
| OTel init | observe/otel.py | tests/test_otel.py | docs/observability.md |
| Statistics | observe/stats.py | tests/test_stats.py | docs/interfaces.md |
| Lineage | observe/lineage.py | tests/test_lineage.py | docs/interfaces.md |
| Compile control | observe/compile_controller.py | tests/test_compile_controller.py | — |
| Model registry | models/registry.yaml | tests/test_registry.py | docs/interfaces.md |
| Training registry | train/registry.yaml | tests/test_train_runner.py | docs/interfaces.md |
| Results render | scripts/render_results.py | tests/test_render_results.py | — |
| GCP lifecycle | scripts/20_*.sh–71_*.sh | — | docs/runbooks/ |
| Observability stack | infra/docker-compose.yml | — | docs/observability.md |

## Feature: Add a new probe

Files likely affected (6):
- `observe/<probe_name>.py` — new file to create
- `observe/probe.py` — verify hooks match what you need (read only)
- `observe/README.md` — add probe to table
- `README.md` — add probe to Probes section
- `MEMORY.md` — add probe to quick reference
- `tests/test_app_probes.py` — add test

Files unlikely to touch:
- benchmarks/runner.py, train/runner.py, benchmarks/harness.py, train/harness.py

Validation plan:
```bash
python3 -m pytest tests/test_app_probes.py -q
python3 -m pytest tests/ -q
JAX_PLATFORMS=cpu python3 -m benchmarks.harness --suite smoke --device cpu --dry-run
```

## Feature: Add a new model to the inference registry

Files likely affected (3):
- `models/registry.yaml` — add entry
- `tests/test_registry.py` — add test if new schema fields
- `README.md` — add to model table if significant

Validation plan:
```bash
python3 -m pytest tests/test_registry.py -q
JAX_PLATFORMS=cpu python3 -m benchmarks.harness --model <new_model_id> --device cpu --dry-run
```

## Feature: Fix OTel behavior (enabled path)

Files likely affected (4):
- `observe/otel.py` — main OTel init
- `observe/otel_probe.py` — probe wrapper
- `tests/test_otel.py` — verify (requires opentelemetry installed)
- `tests/test_otel_probe.py` — probe tests

Note: get_tracer() and get_meter() now guard against eager imports when disabled.
The `if not _state["enabled"]` path returns _NoOpTracer/_NoOpMeter — don't remove this guard.

## Feature: Change results schema (runs.jsonl fields)

Files likely affected (5):
- `benchmarks/runner.py` — where result dict is built (~line 600+)
- `tests/test_runner.py` — update expected field assertions
- `scripts/render_results.py` — may need column updates
- `docs/interfaces.md` — update schema section
- `results/RESULTS.md` — regenerate after change

Warning: runs.jsonl is append-only. Schema changes are forward-only. Old rows will be missing new fields — ensure render_results.py handles missing fields gracefully.

## Feature: Change the probe ABC (add/remove/rename hooks)

Files likely affected (many — HIGH RISK):
- `observe/probe.py` — hook definitions and fanout functions
- All 14 probe implementations — must update any broken hooks
- All test files referencing probes
- `docs/interfaces.md`, `observe/README.md`

Risk: HIGH — breaking change to all 14 probes. Get human approval first.

## Feature: Add a new suite (e.g., "medium" between quick and domain)

Files likely affected (3):
- `benchmarks/harness.py` — SUITES dict (~line 60)
- `tests/test_harness.py` — add suite test
- `README.md` — add suite to Suite Definitions table

## Feature: Add Stage 2 (Paths 2+3, system_monitor)

Files likely affected (10+):
- `observe/system_monitor.py` — new file (GPU SM%/MXU% eager counters)
- `benchmarks/runner.py` — add Path 2 (JAX+GPU) and Path 3 (PyTorch+GPU) branches
- `benchmarks/harness.py` — update framework choices
- `models/registry.yaml` — add 15 more models
- `tests/test_runner.py` — extend for new paths
- `requirements.txt` — add torch, torch_xla (carefully — don't break Flax pin)
- `docs/architecture.md` — update data flow for new paths
- `results/dashboard/` — add heatmap view

Validation plan:
```bash
python3 -m pytest tests/test_runner.py -q
python3 -m pytest tests/ -q
JAX_PLATFORMS=cpu python3 -m benchmarks.harness --suite smoke --device cpu
```

## Token-saving checklist

Before any feature work:
- [ ] Read AGENTS.md — understand safe vs approval-required files
- [ ] Find feature in this map above
- [ ] Read ONLY the 3–7 files listed as affected
- [ ] Grep for the specific function/class, don't read whole files
- [ ] Make the smallest safe change
- [ ] Run narrow test first (e.g., test_registry.py not tests/)
- [ ] Run full test suite only after narrow test passes
- [ ] Update only the docs listed as affected
