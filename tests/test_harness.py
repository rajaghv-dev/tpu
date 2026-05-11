"""Tests for benchmarks/harness.py — registry loading, config building, CLI."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

yaml = pytest.importorskip("yaml")

from benchmarks.harness import (
    DEVICE_COSTS,
    SUITES,
    append_result,
    build_config,
    filter_registry,
    load_registry,
    run_suite,
)

_REGISTRY_PATH = Path(__file__).parent.parent / "models" / "registry.yaml"


class TestLoadRegistry:
    def test_loads_five_models(self):
        models = load_registry(str(_REGISTRY_PATH))
        assert len(models) == 5

    def test_models_have_id(self):
        models = load_registry(str(_REGISTRY_PATH))
        for m in models:
            assert "id" in m

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_registry("/tmp/__nonexistent_registry__.yaml")


class TestFilterRegistry:
    @pytest.fixture
    def registry(self):
        return load_registry(str(_REGISTRY_PATH))

    def test_filter_single(self, registry):
        filtered = filter_registry(registry, model_ids=["bert_base"])
        assert len(filtered) == 1
        assert filtered[0]["id"] == "bert_base"

    def test_filter_multiple(self, registry):
        filtered = filter_registry(registry, model_ids=["bert_base", "gpt2"])
        assert len(filtered) == 2
        ids = {m["id"] for m in filtered}
        assert ids == {"bert_base", "gpt2"}

    def test_filter_none_returns_all(self, registry):
        filtered = filter_registry(registry, model_ids=None)
        assert len(filtered) == len(registry)

    def test_filter_unknown_id_returns_empty(self, registry):
        filtered = filter_registry(registry, model_ids=["nonexistent_model"])
        assert filtered == []


class TestBuildConfig:
    @pytest.fixture
    def bert_entry(self):
        return load_registry(str(_REGISTRY_PATH))[0]

    def test_model_id_matches(self, bert_entry):
        cfg = build_config(bert_entry, precision="bf16", device="tpu")
        assert cfg.model_id == bert_entry["id"]

    def test_precision_stored(self, bert_entry):
        cfg = build_config(bert_entry, precision="fp32", device="tpu")
        assert cfg.precision == "fp32"

    def test_device_stored(self, bert_entry):
        cfg = build_config(bert_entry, precision="bf16", device="gpu")
        assert cfg.device == "gpu"

    def test_tpu_has_cost(self, bert_entry):
        cfg = build_config(bert_entry, precision="bf16", device="tpu")
        assert cfg.device_cost_usd_per_hr == DEVICE_COSTS["tpu"]

    def test_local_gpu_is_free(self, bert_entry):
        cfg = build_config(bert_entry, precision="bf16", device="b200")
        assert cfg.device_cost_usd_per_hr == 0.0

    def test_framework_default_jax(self, bert_entry):
        cfg = build_config(bert_entry, precision="bf16", device="tpu")
        assert cfg.framework == "jax"


class TestSuiteDefinitions:
    def test_smoke_has_one_model(self):
        assert len(SUITES["smoke"]["model_ids"]) == 1

    def test_quick_has_five_models(self):
        assert len(SUITES["quick"]["model_ids"]) == 5

    def test_smoke_bert_base(self):
        assert "bert_base" in SUITES["smoke"]["model_ids"]

    def test_all_quick_models_in_registry(self):
        registry = load_registry(str(_REGISTRY_PATH))
        registry_ids = {m["id"] for m in registry}
        for mid in SUITES["quick"]["model_ids"]:
            assert mid in registry_ids, f"Suite model {mid} not in registry"

    def test_suites_have_description(self):
        for name, suite in SUITES.items():
            assert "description" in suite, f"Suite {name} missing description"


class TestAppendResult:
    def test_appends_valid_json_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            result = {"run_id": "test-123", "model": "bert_base", "latency_p50_ms": 9.1}
            append_result(result, path)

            lines = path.read_text().strip().splitlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["run_id"] == "test-123"
            assert parsed["model"] == "bert_base"
        finally:
            path.unlink(missing_ok=True)

    def test_multiple_appends(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            for i in range(3):
                append_result({"run_id": str(i)}, path)

            lines = path.read_text().strip().splitlines()
            assert len(lines) == 3
        finally:
            path.unlink(missing_ok=True)


class TestRunSuiteErrorPaths:
    """Verify that run_suite returns -1 on configuration errors."""

    def test_unknown_model_returns_minus_one(self, tmp_path):
        n = run_suite(
            suite_name=None,
            model_id="totally_unknown_model_xyz",
            device="cpu",
            framework="jax",
            precision="bf16",
            output_path=tmp_path / "out.jsonl",
            registry_path=str(_REGISTRY_PATH),
            dry_run=False,
        )
        assert n == -1

    def test_unknown_suite_returns_minus_one(self, tmp_path):
        n = run_suite(
            suite_name="not_a_real_suite",
            model_id=None,
            device="cpu",
            framework="jax",
            precision="bf16",
            output_path=tmp_path / "out.jsonl",
            registry_path=str(_REGISTRY_PATH),
            dry_run=False,
        )
        assert n == -1

    def test_no_suite_no_model_returns_minus_one(self, tmp_path):
        n = run_suite(
            suite_name=None,
            model_id=None,
            device="cpu",
            framework="jax",
            precision="bf16",
            output_path=tmp_path / "out.jsonl",
            registry_path=str(_REGISTRY_PATH),
            dry_run=False,
        )
        assert n == -1

    def test_failed_experiment_appends_failure_stub(self, tmp_path):
        """When run_experiment raises BenchmarkError, run_suite should:
          - record 0 successes;
          - append a failure-stub row to the JSONL output.
        """
        from benchmarks.runner import BenchmarkError

        # Build a synthetic BenchmarkError as the runner would produce.
        try:
            raise OSError("simulated network failure")
        except OSError as raw:
            be = BenchmarkError("model_load", raw, "network")

        out = tmp_path / "out.jsonl"
        with mock.patch("benchmarks.harness.run_experiment", side_effect=be):
            n = run_suite(
                suite_name=None,
                model_id="bert_base",
                device="cpu",
                framework="jax",
                precision="bf16",
                output_path=out,
                registry_path=str(_REGISTRY_PATH),
                dry_run=False,
            )
        # 0 successes (one configuration, one failure → returns 0)
        assert n == 0
        # Failure stub should be the only row in the JSONL.
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["phase"] == "model_load"
        assert rows[0]["error_category"] == "network"
        assert rows[0]["model"] == "bert_base"

    def test_dry_run_returns_zero(self, tmp_path):
        n = run_suite(
            suite_name="smoke",
            model_id=None,
            device="cpu",
            framework="jax",
            precision="bf16",
            output_path=tmp_path / "out.jsonl",
            registry_path=str(_REGISTRY_PATH),
            dry_run=True,
        )
        assert n == 0


class TestMainExitCode:
    """Verify main() exits 1 on error, 0 on success/dry-run."""

    def test_dry_run_exits_zero(self):
        from benchmarks.harness import main
        code = main(["--suite", "smoke", "--device", "cpu", "--dry-run"])
        assert code == 0

    def test_unknown_model_exits_one(self):
        from benchmarks.harness import main
        code = main(["--model", "nonexistent_xyz", "--device", "cpu"])
        assert code == 1

    def test_unknown_suite_exits_one(self):
        from benchmarks.harness import main
        # argparse will catch unknown suite before run_suite — exits with SystemExit(2)
        with pytest.raises(SystemExit) as exc:
            main(["--suite", "nonexistent_suite"])
        assert exc.value.code != 0
