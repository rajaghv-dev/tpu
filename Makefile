.PHONY: test test-fast test-verbose dry-run smoke-cpu lint dashboard help

# ── Testing ────────────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -q

test-verbose:
	python -m pytest tests/ -v --tb=long

# ── Benchmark ──────────────────────────────────────────────────────────────────

dry-run:
	python benchmarks/harness.py --suite quick --device cpu --dry-run

smoke-cpu:
	## Run smoke suite on CPU-JAX (no TPU/GPU download needed — for local dev)
	JAX_PLATFORMS=cpu python benchmarks/harness.py --suite smoke --device cpu

smoke-tpu:
	python benchmarks/harness.py --suite smoke --device tpu

quick-tpu:
	python benchmarks/harness.py --suite quick --device tpu

# ── Development ────────────────────────────────────────────────────────────────

lint:
	python -m flake8 benchmarks/ observe/ tests/ --max-line-length 100 --extend-ignore=E501

dashboard:
	## Serve dashboard at http://localhost:8080
	cd results/dashboard && python -m http.server 8080

lock:
	## Snapshot current environment to requirements.stage1.lock.txt
	pip freeze > requirements.stage1.lock.txt
	@echo "Lock file written. Tag the commit: git tag stage1-complete"

# ── Help ───────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make test          Run all 121 unit tests (verbose)"
	@echo "  make test-fast     Run tests in quiet mode"
	@echo "  make dry-run       Show what quick suite would run (no downloads)"
	@echo "  make smoke-cpu     Run smoke suite on CPU-JAX (no TPU needed)"
	@echo "  make smoke-tpu     Run smoke suite on TPU v5e-1"
	@echo "  make quick-tpu     Run quick suite on TPU v5e-1 (~50 min)"
	@echo "  make lint          Run flake8 on source files"
	@echo "  make dashboard     Serve results dashboard at localhost:8080"
	@echo "  make lock          Freeze current env to requirements.stage1.lock.txt"
	@echo ""
