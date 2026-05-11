"""Tests for observe/otel.py — OTel init, file exporters, instruments."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import is safe even without OTel envvar set.
from observe import otel as otel_mod
from observe.otel import (
    _METRIC_SCHEMA,
    get_instruments,
    get_meter,
    get_tracer,
    init_otel,
    is_enabled,
    shutdown_otel,
)


# ── Isolation fixture ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_otel():
    """Wipe global OTel + module state around every test."""
    otel_mod._reset_for_tests()
    yield
    otel_mod._reset_for_tests()


def _force_flush_all() -> None:
    """Flush both providers so the file exporters write to disk."""
    tp = otel_mod._state.get("tracer_provider")
    mp = otel_mod._state.get("meter_provider")
    if tp is not None:
        tp.force_flush()
    if mp is not None:
        mp.force_flush()


# ── Off mode ─────────────────────────────────────────────────────────────────

class TestOffMode:
    def test_default_env_disabled(self, monkeypatch):
        monkeypatch.delenv("TPU_BENCH_OTEL", raising=False)
        init_otel()
        assert is_enabled() is False

    def test_explicit_off_disabled(self, monkeypatch):
        monkeypatch.setenv("TPU_BENCH_OTEL", "off")
        init_otel()
        assert is_enabled() is False

    def test_off_tracer_span_is_noop(self, monkeypatch):
        monkeypatch.setenv("TPU_BENCH_OTEL", "off")
        init_otel()
        tracer = get_tracer()
        # Should not raise and should accept set_attribute calls.
        with tracer.start_as_current_span("foo") as s:
            s.set_attribute("k", "v")

    def test_unknown_value_treated_as_off(self, monkeypatch):
        monkeypatch.setenv("TPU_BENCH_OTEL", "bogus")
        init_otel()
        assert is_enabled() is False


# ── File mode ────────────────────────────────────────────────────────────────

class TestFileMode:
    def test_files_created_on_shutdown(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TPU_BENCH_OTEL", "file")
        monkeypatch.setenv("TPU_BENCH_OTEL_DIR", str(tmp_path))
        init_otel(resource_attrs={"run_id": "abc123"})
        assert is_enabled() is True

        tracer = get_tracer()
        with tracer.start_as_current_span("phase.preflight") as s:
            s.set_attribute("device", "tpu")

        inst = get_instruments()
        inst["latency_ms"].record(12.5, {"model_id": "bert"})

        _force_flush_all()
        shutdown_otel()

        spans_path = tmp_path / "abc123_spans.jsonl"
        metrics_path = tmp_path / "abc123_metrics.jsonl"
        assert spans_path.exists(), "spans jsonl should be created"
        assert metrics_path.exists(), "metrics jsonl should be created"

    def test_span_jsonl_contains_name_and_attrs(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TPU_BENCH_OTEL", "file")
        monkeypatch.setenv("TPU_BENCH_OTEL_DIR", str(tmp_path))
        init_otel(resource_attrs={"run_id": "spntest"})

        tracer = get_tracer()
        with tracer.start_as_current_span("phase.latency") as s:
            s.set_attribute("block_index", 1)

        _force_flush_all()
        shutdown_otel()

        lines = (tmp_path / "spntest_spans.jsonl").read_text().splitlines()
        assert lines, "at least one span line expected"
        rows = [json.loads(line) for line in lines]
        names = [r.get("name") for r in rows]
        assert "phase.latency" in names

        latency_row = next(r for r in rows if r.get("name") == "phase.latency")
        attrs = latency_row.get("attributes", {})
        # attributes may be a dict on ReadableSpan.to_json output.
        assert attrs.get("block_index") == 1

    def test_metric_jsonl_contains_instrument_name(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TPU_BENCH_OTEL", "file")
        monkeypatch.setenv("TPU_BENCH_OTEL_DIR", str(tmp_path))
        init_otel(resource_attrs={"run_id": "mtest"})

        inst = get_instruments()
        inst["throughput_samples_sec"].record(123.4, {"model_id": "bert"})
        inst["compile_cold_s"].record(2.5)

        _force_flush_all()
        shutdown_otel()

        text = (tmp_path / "mtest_metrics.jsonl").read_text()
        assert "benchmark.throughput.samples_sec" in text
        assert "benchmark.compile.cold.seconds" in text

    def test_all_five_instruments_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TPU_BENCH_OTEL", "file")
        monkeypatch.setenv("TPU_BENCH_OTEL_DIR", str(tmp_path))
        init_otel(resource_attrs={"run_id": "instest"})

        inst = get_instruments()
        expected_keys = {key for key, _, _ in _METRIC_SCHEMA}
        assert set(inst.keys()) == expected_keys
        assert len(inst) == 5

    def test_instrument_names_and_units(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TPU_BENCH_OTEL", "file")
        monkeypatch.setenv("TPU_BENCH_OTEL_DIR", str(tmp_path))
        init_otel(resource_attrs={"run_id": "u_test"})

        # Record one sample from each instrument so it lands in the export.
        inst = get_instruments()
        inst["latency_ms"].record(1.0)
        inst["throughput_samples_sec"].record(1.0)
        inst["compile_cold_s"].record(1.0)
        inst["compile_warm_s"].record(1.0)
        inst["latency_cv_pct"].record(1.0)

        _force_flush_all()
        shutdown_otel()

        text = (tmp_path / "u_test_metrics.jsonl").read_text()
        for _, full_name, _ in _METRIC_SCHEMA:
            assert full_name in text, f"missing metric name in export: {full_name}"
        # Spot-check units appear at least once.
        assert '"ms"' in text
        assert '"s"' in text
        assert '"%"' in text

    def test_get_instruments_is_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TPU_BENCH_OTEL", "file")
        monkeypatch.setenv("TPU_BENCH_OTEL_DIR", str(tmp_path))
        init_otel(resource_attrs={"run_id": "cache_test"})
        a = get_instruments()
        b = get_instruments()
        assert a is b


# ── Lifecycle ────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_shutdown_without_init_is_safe(self):
        # _reset_for_tests already ran, so nothing is initialised.
        shutdown_otel()  # must not raise
        shutdown_otel()  # idempotent

    def test_tracer_before_init_works_as_noop(self, monkeypatch):
        monkeypatch.delenv("TPU_BENCH_OTEL", raising=False)
        # No init_otel call yet — get_tracer should still work as a no-op.
        tracer = get_tracer()
        with tracer.start_as_current_span("x"):
            pass

    def test_meter_before_init_works_as_noop(self, monkeypatch):
        monkeypatch.delenv("TPU_BENCH_OTEL", raising=False)
        meter = get_meter()
        # The meter should at least permit instrument creation (no-op meter).
        h = meter.create_histogram(name="dummy.metric", unit="ms")
        h.record(1.0)

    def test_init_is_idempotent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TPU_BENCH_OTEL", "file")
        monkeypatch.setenv("TPU_BENCH_OTEL_DIR", str(tmp_path))
        init_otel(resource_attrs={"run_id": "idem"})
        first_tp = otel_mod._state["tracer_provider"]
        init_otel(resource_attrs={"run_id": "idem"})
        second_tp = otel_mod._state["tracer_provider"]
        assert first_tp is second_tp
