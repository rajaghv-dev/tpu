.PHONY: demo test smoke versions clean install-dev help

PYTHON ?= python3

help:
	@echo "Targets:"
	@echo "  make smoke     Run only the smoke tests (fast)."
	@echo "  make test      Run the full pytest suite."
	@echo "  make versions  Print the TPU version comparison table."
	@echo "  make clean     Remove artifacts/ and __pycache__/."
	@echo ""
	@echo "Real-TPU runs are orchestrated by gcp/run_real_demo.sh (PAID)."

smoke:
	$(PYTHON) -m pytest cloud_tpu_lab/tests/test_imports.py -q

test:
	$(PYTHON) -m pytest cloud_tpu_lab/tests/ -q

versions:
	$(PYTHON) -m cloud_tpu_lab.src.tpu_versions.version_compare

clean:
	rm -rf cloud_tpu_lab/artifacts/logs/* \
	       cloud_tpu_lab/artifacts/metrics/* \
	       cloud_tpu_lab/artifacts/traces/* \
	       cloud_tpu_lab/artifacts/reports/* \
	       cloud_tpu_lab/artifacts/plots/*
	find cloud_tpu_lab -name __pycache__ -type d -exec rm -rf {} +
