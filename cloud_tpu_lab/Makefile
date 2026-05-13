.PHONY: demo test smoke versions clean install-dev help

PYTHON ?= python3

help:
	@echo "Targets:"
	@echo "  make demo      Run the CPU simulation demo (no TPU needed)."
	@echo "  make smoke     Run only the smoke tests (fast)."
	@echo "  make test      Run the full pytest suite."
	@echo "  make versions  Print the TPU version comparison table."
	@echo "  make clean     Remove artifacts/ and __pycache__/."

demo:
	$(PYTHON) -m cloud_tpu_lab.examples.run_cpu_simulation_demo

smoke:
	$(PYTHON) -m pytest cloud_tpu_lab/tests/test_imports.py \
	                    cloud_tpu_lab/tests/test_cpu_simulation_smoke.py -q

test:
	$(PYTHON) -m pytest cloud_tpu_lab/tests/ -q

versions:
	$(PYTHON) -m cloud_tpu_lab.examples.run_cpu_simulation_demo --show-versions

clean:
	rm -rf cloud_tpu_lab/artifacts/logs/* \
	       cloud_tpu_lab/artifacts/metrics/* \
	       cloud_tpu_lab/artifacts/traces/* \
	       cloud_tpu_lab/artifacts/reports/* \
	       cloud_tpu_lab/artifacts/plots/*
	find cloud_tpu_lab -name __pycache__ -type d -exec rm -rf {} +
