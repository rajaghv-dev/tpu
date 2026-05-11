.PHONY: test test-fast test-verbose dry-run smoke-cpu lint dashboard help otel-collect otel-view otel-down gcp-bootstrap gcp-check gcp-budget tpus-kill tpus-list

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

# ── Observability (OTel + Grafana, see ADR-016) ────────────────────────────────

otel-collect:
	## Pull results/otel/ and runs.jsonl from the TPU VM
	./scripts/otel_collect.sh

otel-view:
	## Start local Grafana stack and open the browser (http://localhost:3000)
	./scripts/otel_view.sh

otel-down:
	## Stop the local Grafana stack
	./scripts/otel_view.sh --down

# ── GCP setup (one-time + safety nets) ─────────────────────────────────────────

gcp-bootstrap:
	## Install gcloud, auth, link billing, enable APIs, create bucket (idempotent)
	./scripts/gcp_bootstrap.sh

gcp-check:
	## Verify the GCP setup without making changes
	./scripts/gcp_bootstrap.sh --check

gcp-budget:
	## Create a $$20/mo budget with alerts at 50/90/100% (override with AMOUNT=...)
	./scripts/gcp_set_budget.sh

tpus-list:
	## List all TPU VMs across all zones (no deletion)
	./scripts/kill_all_tpus.sh --dry-run

tpus-kill:
	## PANIC BUTTON: delete all TPU VMs across all zones (with confirmation)
	./scripts/kill_all_tpus.sh

# ── Help ───────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  Testing & benchmarks:"
	@echo "    make test          Run all unit tests (verbose)"
	@echo "    make test-fast     Run tests in quiet mode"
	@echo "    make dry-run       Show what quick suite would run (no downloads)"
	@echo "    make smoke-cpu     Run smoke suite on CPU-JAX (no TPU needed)"
	@echo "    make smoke-tpu     Run smoke suite on TPU v5e-1"
	@echo "    make quick-tpu     Run quick suite on TPU v5e-1 (~50 min)"
	@echo ""
	@echo "  Dev:"
	@echo "    make lint          Run flake8 on source files"
	@echo "    make dashboard     Serve results dashboard at localhost:8080"
	@echo "    make lock          Freeze current env to requirements.stage1.lock.txt"
	@echo ""
	@echo "  Observability (OTel + Grafana, ADR-016):"
	@echo "    make otel-collect  Pull OTel traces + runs.jsonl from TPU VM"
	@echo "    make otel-view     Start local Grafana stack at localhost:3000"
	@echo "    make otel-down     Stop the local Grafana stack"
	@echo ""
	@echo "  GCP one-time setup + safety nets:"
	@echo "    make gcp-bootstrap Install gcloud, auth, billing, APIs, bucket"
	@echo "    make gcp-check     Verify GCP setup without changes"
	@echo "    make gcp-budget    Create a \$$20/mo billing budget with alerts"
	@echo "    make tpus-list     List all TPU VMs across all zones"
	@echo "    make tpus-kill     PANIC BUTTON: delete all TPU VMs"
	@echo ""
