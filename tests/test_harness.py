"""Tests for benchmarks/harness.py — registry loading, config building, CLI."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

yaml = pytest.importorskip("yaml")

from benchmarks.harness import (
    SUITES,
    build_config,
    filter_registry,
    load_registry,
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
        return load_registry(str(_REGISTRY_PATH))[0]  # bert_base is first

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
        assert cfg.device_cost_usd_per_hr == 0.36

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
        from benchmarks.harness import append_result

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
        from benchmarks.harness import append_result

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            for i in range(3):
                append_result({"run_id": str(i)}, path)

            lines = path.read_text().strip().splitlines()
            assert len(lines) == 3
        finally:
            path.unlink(missing_ok=True)
