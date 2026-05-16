# Testing

Generated: 2026-05-16.

## Test types

| Type | Location | Count | GPU/TPU needed |
|---|---|---|---|
| Unit tests (root) | tests/ | 272 total (255 pass + 17 fail + 3 skip at baseline) | No |
| Unit tests (cloud_tpu_lab) | cloud_tpu_lab/tests/ | 8 test files | No — Requires `pip install -e .` from cloud_tpu_lab/ first |
| Integration (dry-run) | CI: `make dry-run` | 1 harness invocation | No |
| Integration (smoke-cpu) | `make smoke-cpu` | 1 model end-to-end | No (CPU-JAX) |
| Smoke (TPU) | `make smoke-tpu` | 1 model end-to-end | Yes (v5e-1) |

## How to run all tests

```bash
# Root tests (CPU, no GPU/TPU)
python3 -m pytest tests/ -v

# cloud_tpu_lab tests (requires package install first)
cd cloud_tpu_lab && pip install -e . && python3 -m pytest tests/ -q
```

## How to run unit tests

```bash
python3 -m pytest tests/ -q --tb=short
```

## How to run specific test files

```bash
python3 -m pytest tests/test_stats.py -v
python3 -m pytest tests/test_otel.py -v        # requires opentelemetry
python3 -m pytest tests/test_runner.py -v       # requires opentelemetry
python3 -m pytest tests/test_lineage.py -v
```

## How to run integration tests

```bash
# No-download dry-run (fastest):
make dry-run

# CPU inference (downloads BERT-base):
make smoke-cpu

# Training smoke (CPU):
python3 -m train.harness --suite smoke --device cpu
```

## Test data

All tests use synthetic data — no real datasets, no model downloads. The conftest.py mocks JAX/numpy when not installed.

## Known failing tests (baseline 2026-05-16)

| Test file | Failures | Root cause | Fix |
|---|---|---|---|
| tests/test_otel.py | 10 | `No module named 'opentelemetry'` | Install opentelemetry OR fix otel.py eager import |
| tests/test_runner.py | 7 | `No module named 'opentelemetry'` (via otel.py) | Same as above |

After fixing `observe/otel.py` to lazily import opentelemetry, all 17 failures resolve.

## Known missing tests

| Gap | Notes |
|---|---|
| DeterminismProbe | No tests in tests/ |
| DeviceInfoProbe | No tests in tests/ |
| PowerThermalProbe | No tests in tests/ |
| XlaCompileProbe | No tests in tests/ |
| Training harness CLI | --dry-run path not CI-tested |
| cloud_tpu_lab tests | Not in CI workflow — also not installable without `pip install -e .` from cloud_tpu_lab/, so not CI-validated |
| cloud_tpu_lab/ not installable without pip install -e . | Not CI-validated — run `cd cloud_tpu_lab && pip install -e . && python3 -m pytest tests/ -q` manually |
| examples 01–08 | Not tested |
| Notebooks | Not tested |

## CI validation

See `.github/workflows/smoke_on_push.yml`. Jobs:
1. `unit-tests` — `pytest tests/ -q --tb=short` on ubuntu-latest, Python 3.12
2. `dry-run-harness` — smoke + quick harness dry-run
3. `lineage-sanity` — confirms get_git_sha() matches git log

**CI does not currently cover:**
- cloud_tpu_lab tests
- lint (flake8)
- Training harness

## Test strategy

- Prefer narrow tests over broad ones
- Mock JAX when not available (conftest.py)
- All tests must pass without GPU/TPU
- Add failing test first when fixing a bug
- Cover every new probe with at least one test
