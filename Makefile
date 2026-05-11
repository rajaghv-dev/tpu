.PHONY: test test-fast test-verbose dry-run smoke-cpu lint dashboard help otel-collect otel-view otel-down gcp-bootstrap gcp-check gcp-budget tpus-kill tpus-list cache-env cache-status cache-wheels-build cache-models-upload cache-xla-upload colab-notebook colab-open setup-hf check-hf

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

# ── HuggingFace auth (gated models: Gemma/LLaMA/PaliGemma) ─────────────────────

setup-hf:
	## Capture HF PRO token; stores locally + GCP Secret Manager (idempotent)
	./scripts/setup_hf.sh

check-hf:
	## Verify HF token (where stored, validity, whoami)
	./scripts/setup_hf.sh --check

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

# ── GCS caches (Tier 2 cost reduction) ─────────────────────────────────────────

cache-env:
	## Print the cache URLs for the current project
	./scripts/setup_cache_env.sh

cache-status:
	## Inventory of all GCS caches (size, items, last sync)
	./scripts/cache_status.sh

cache-wheels-build:
	## Build pip wheel cache (run after a successful TPU pip install)
	./scripts/cache_wheels.sh --build

cache-models-upload:
	## Sync local HF model cache to GCS (run after a successful run)
	./scripts/cache_models.sh --upload

cache-xla-upload:
	## Sync XLA compile cache to GCS
	./scripts/cache_xla.sh --upload

# ── Colab Pro path (free TPU iteration, todo.md Tier 3 #9) ─────────────────────

colab-notebook:
	## Open the Colab notebook URL in default browser
	@echo "Opening: https://colab.research.google.com/github/rajaghv-dev/tpu/blob/main/colab/tpu_benchmark.ipynb"
	@xdg-open "https://colab.research.google.com/github/rajaghv-dev/tpu/blob/main/colab/tpu_benchmark.ipynb" 2>/dev/null \
	  || open "https://colab.research.google.com/github/rajaghv-dev/tpu/blob/main/colab/tpu_benchmark.ipynb" 2>/dev/null \
	  || wslview "https://colab.research.google.com/github/rajaghv-dev/tpu/blob/main/colab/tpu_benchmark.ipynb" 2>/dev/null \
	  || echo "(no browser opener found — paste the URL manually)"

colab-open: colab-notebook
	## Alias for colab-notebook

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
	@echo "  HuggingFace (gated models — Gemma/LLaMA/PaliGemma):"
	@echo "    make setup-hf      Capture + validate + store HF PRO token"
	@echo "    make check-hf      Verify HF token (whoami + storage status)"
	@echo ""
	@echo "  Colab Pro path (free TPU iteration, todo.md Tier 3 #9):"
	@echo "    make colab-notebook  Open the Colab notebook URL in default browser"
	@echo ""
	@echo "  GCS caches (Tier 2 cost reduction — see todo.md):"
	@echo "    make cache-env             Print cache URLs for current project"
	@echo "    make cache-status          Size + items + last sync of all 3 caches"
	@echo "    make cache-wheels-build    Build pip wheel cache (run on TPU VM)"
	@echo "    make cache-models-upload   Sync HF model cache to GCS"
	@echo "    make cache-xla-upload      Sync XLA compile cache to GCS"
	@echo ""
