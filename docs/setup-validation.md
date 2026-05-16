# Setup Validation

Generated: 2026-05-16 — validated in WSL2, Python 3.12, .tpu venv.

## Validation Table

| Step | Command | Expected | Actual | Status | Fix |
|---|---|---|---|---|---|
| Python available | `python3 --version` | Python 3.10+ | Python 3.12.x | ✅ Pass | — |
| python alias | `python --version` | Python 3.x | Not found | ❌ Fail | Use `python3` everywhere; fix Makefile |
| Repo clone | `git clone https://github.com/rajaghv-dev/tpu && cd tpu` | Clean clone | N/A — already cloned | ✅ (assumed) | — |
| Install dependencies | `pip install -r requirements.txt` | All packages installed | Requires libtpu index for jax[tpu] | ⚠️ Partial | Add note: `--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html` (already in requirements.txt) |
| Run tests (minimal) | `python3 -m pytest tests/ -q` | 272 total (0 failures with full deps) | 255 pass, 17 fail (opentelemetry missing), 3 skip | ❌ 17 failures | Fix observe/otel.py eager import; or install opentelemetry separately |
| Run tests (CI lock) | `pip install -r requirements.stage1.lock.txt && python3 -m pytest tests/ -q` | All tests pass | Lock file missing opentelemetry → same 17 failures | ❌ 17 failures | Regenerate lock file after full requirements.txt install |
| Dry-run harness | `python3 -m benchmarks.harness --suite quick --device cpu --dry-run` | Plan printed | Works | ✅ Pass | — |
| Smoke CPU | `JAX_PLATFORMS=cpu python3 -m benchmarks.harness --suite smoke --device cpu` | 1 model run completes | Requires jax, transformers, flax installed | ⚠️ Env-dependent | Run in .tpu venv with full install |
| cloud_tpu_lab install | `cd cloud_tpu_lab && pip install -e .` | Package installed in venv | ModuleNotFoundError without this step | ❌ Fail without install (ModuleNotFoundError: cloud_tpu_lab) | `cd cloud_tpu_lab && pip install -e .` then run tests |
| cloud_tpu_lab tests | `cd cloud_tpu_lab && pip install -e . && python3 -m pytest tests/ -q` | 8 test files pass | Fails without prior `pip install -e .` | ❌ Fail without install (ModuleNotFoundError: cloud_tpu_lab) | `cd cloud_tpu_lab && pip install -e .` then run tests |
| CPU simulation demo | `cd cloud_tpu_lab && python3 examples/run_cpu_simulation_demo.py` | Demo artifacts written | Should work (stdlib-only sim) | ✅ Expected | — |
| Lint | `make lint` | 0 flake8 errors | Not tested in this session | ⚠️ Unknown | Run `python3 -m flake8 benchmarks/ observe/ tests/ --max-line-length 100` |
| make test | `make test` | pytest runs | `python` not found → Makefile fails | ❌ Fail | Fix Makefile python → python3 |
| Dashboard | `make dashboard` → http://localhost:8080 | HTML page served | Not tested | ⚠️ Unknown | Requires `cd results/dashboard && python3 -m http.server 8080` |
| Grafana stack | `make otel-view` → http://localhost:3000 | Grafana starts | Requires Docker | ⚠️ Docker-dependent | Ensure Docker Desktop running |

## Environment Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.12 tested |
| pip 22+ | For `--find-links` support |
| jax[tpu] | Only installs on TPU VMs; on CPU: `pip install jax` (CPU-only) |
| opentelemetry-api, opentelemetry-sdk | Required for test_otel.py and test_runner.py |
| Docker | Required for Grafana stack only |
| gcloud CLI | Required for GCP provisioning scripts |
| git | Required for lineage tracking |

## Quick Valid Local Setup

```bash
# 1. Clone
git clone https://github.com/rajaghv-dev/tpu && cd tpu

# 2. Create venv
python3 -m venv .venv && source .venv/bin/activate

# 3. Install CPU-compatible subset
pip install jax flax optax orbax-checkpoint transformers datasets pyyaml rich tqdm \
            opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc \
            numpy pytest pytest-cov

# 4. Run tests
JAX_PLATFORMS=cpu python3 -m pytest tests/ -q
# Expected: 272 passed (after otel.py fix), 3 skipped

# 4b. Run cloud_tpu_lab tests (requires separate install step)
# Note: cloud_tpu_lab tests require the package to be installed first.
# Run `pip install -e .` from within the cloud_tpu_lab/ directory before running tests.
cd cloud_tpu_lab && pip install -e . && python3 -m pytest tests/ -q && cd ..

# 5. Dry-run harness
JAX_PLATFORMS=cpu python3 -m benchmarks.harness --suite quick --device cpu --dry-run
```

## TPU VM Setup

```bash
# Automated (recommended):
./scripts/run_all.sh --suite smoke

# Manual:
./scripts/20_provision_tpu.sh
./scripts/21_wait_tpu_ready.sh
./scripts/30_deploy_repo.sh
./scripts/31_install_deps.sh
./scripts/40_verify_jax.sh      # confirm jax.devices() shows TPU
./scripts/41_run_pytests.sh
./scripts/50_run_smoke.sh
./scripts/60_pull_results.sh
./scripts/70_teardown_tpu.sh
./scripts/71_verify_teardown.sh  # confirm $0/hr
```
