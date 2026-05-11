"""
Tests for observe/otel_probe.py.

The CI/dev host doesn't have ``opentelemetry`` installed (it's an opt-in
extra — see the module docstring), so the primary path under test is the
no-op fallback. We additionally exercise a fake-SDK injection to cover the
"available" path without requiring the real package.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from observe import otel_probe as op
from observe.otel_probe import OTelProbe, _parse_resource_attrs


# ── Tiny fake config so we don't drag in benchmarks.runner ────────────────────

class _FakeConfig:
    model_id = "bert_base"
    device = "tpu"
    precision = "bf16"
    framework = "jax"
    total_params_M = 110


# ── No-op (OTel-unavailable) path ─────────────────────────────────────────────

class TestOTelProbeUnavailable:
    """
    These tests assume ``opentelemetry`` is NOT installed. If a future host
    DOES have it installed we skip rather than fail — the available-path
    suite below covers the other branch.
    """

    def setup_method(self) -> None:
        if op._OTEL_AVAILABLE:
            pytest.skip("opentelemetry is installed; no-op tests N/A")

    def test_construct_when_unavailable(self):
        probe = OTelProbe()
        assert probe._available is False

    def test_name_is_otel(self):
        assert OTelProbe.name == "otel"

    def test_all_hooks_are_callable_noop(self, tmp_path: Path):
        probe = OTelProbe()
        # None of these should raise.
        probe.before_run("run-1", _FakeConfig(), tmp_path)
        probe.before_phase("compile")
        probe.after_phase("compile", 0.123)
        probe.on_error("latency", RuntimeError("boom"))
        probe.after_run("run-1", {"latency_p50_ms": 1.0})
        probe.after_run("run-1", None)

    def test_write_log_returns_dict_with_available_false(self):
        probe = OTelProbe()
        payload = probe.write_log()
        assert isinstance(payload, dict)
        assert payload["available"] is False
        # Stable schema — these keys must exist even in no-op mode so the
        # dashboard / run_log consumer doesn't have to special-case it.
        for key in ("otel_endpoint", "service_name",
                    "spans_emitted", "metrics_emitted"):
            assert key in payload
        assert payload["spans_emitted"] == 0
        assert payload["metrics_emitted"] == 0

    def test_endpoint_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://example:4318")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "custom-service")
        probe = OTelProbe()
        log = probe.write_log()
        assert log["otel_endpoint"] == "http://example:4318"
        assert log["service_name"] == "custom-service"

    def test_endpoint_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        probe = OTelProbe()
        log = probe.write_log()
        assert log["otel_endpoint"] == "http://localhost:4318"
        assert log["service_name"] == "tpu-bench"


# ── Resource-attribute parsing (pure, runs in any environment) ────────────────

class TestParseResourceAttrs:
    def test_empty_string(self):
        assert _parse_resource_attrs("") == {}

    def test_single_pair(self):
        assert _parse_resource_attrs("env=prod") == {"env": "prod"}

    def test_multiple_pairs(self):
        out = _parse_resource_attrs("env=prod,team=ml,region=us-central1")
        assert out == {"env": "prod", "team": "ml", "region": "us-central1"}

    def test_skips_malformed_entries(self):
        out = _parse_resource_attrs("good=1,broken,also_broken=,=novalue")
        assert out == {"good": "1", "also_broken": ""}

    def test_strips_whitespace(self):
        out = _parse_resource_attrs("  k1 = v1 , k2= v2 ")
        assert out == {"k1": "v1", "k2": "v2"}


# ── Fake-SDK available path ───────────────────────────────────────────────────

class _FakeSpan:
    def __init__(self) -> None:
        self.exceptions = []
        self.status = None

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status


class _FakeSpanCM:
    def __init__(self, registry: list, name: str):
        self.span = _FakeSpan()
        self.name = name
        self.registry = registry
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        self.registry.append(("enter", self.name))
        return self.span

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        self.registry.append(("exit", self.name))
        return False


class _FakeTracer:
    def __init__(self) -> None:
        self.events: list = []
        self.spans: list = []

    def start_as_current_span(self, name, attributes=None):
        cm = _FakeSpanCM(self.events, name)
        self.spans.append((name, attributes, cm))
        return cm


class _FakeHistogram:
    def __init__(self) -> None:
        self.observations: list = []

    def record(self, value, attributes=None):
        self.observations.append((value, attributes))


class _FakeCounter:
    def __init__(self) -> None:
        self.observations: list = []

    def add(self, value, attributes=None):
        self.observations.append((value, attributes))


class _FakeMeter:
    def __init__(self) -> None:
        self.histograms: dict = {}
        self.counters: dict = {}

    def create_histogram(self, name, unit=None, description=None):
        h = _FakeHistogram()
        self.histograms[name] = h
        return h

    def create_counter(self, name, description=None):
        c = _FakeCounter()
        self.counters[name] = c
        return c


class TestOTelProbeAvailablePath:
    """
    Force ``_available=True`` and inject fake tracer/meter to cover the
    record-and-export branches without needing the real SDK installed.

    We bypass ``_configure_providers`` because that path needs the real OTel
    classes; instead we patch the tracer/meter directly on the constructed
    probe. This isolates the Probe's own logic (instrument creation,
    span open/close, metric record) from OTel SDK behaviour.
    """

    def _make_probe(self) -> OTelProbe:
        # Patch the module-level _OTEL_AVAILABLE so __init__ takes the
        # available branch, then immediately stub _configure_providers so it
        # doesn't try to import real OTel classes.
        with (
            mock.patch.object(op, "_OTEL_AVAILABLE", True),
            mock.patch.object(OTelProbe, "_configure_providers", lambda self: None),
        ):
            probe = OTelProbe()
        probe._tracer = _FakeTracer()
        probe._meter = _FakeMeter()
        # Provide minimal Status/StatusCode shims so on_error works without
        # the real OTel symbols being imported.
        op.Status = lambda code: ("status", code)  # type: ignore[attr-defined]
        op.StatusCode = types.SimpleNamespace(ERROR="ERROR")  # type: ignore[attr-defined]
        return probe

    def test_before_run_opens_span_with_attrs(self, tmp_path: Path):
        probe = self._make_probe()
        probe.before_run("run-xyz", _FakeConfig(), tmp_path)
        assert probe._run_span is not None
        names = [name for name, _, _ in probe._tracer.spans]
        assert "benchmark.run" in names
        # Attribute carried through.
        attrs = probe._tracer.spans[0][1]
        assert attrs["model_id"] == "bert_base"
        assert attrs["run_id"] == "run-xyz"
        assert probe._spans_emitted == 1

    def test_phase_lifecycle_records_histogram(self, tmp_path: Path):
        probe = self._make_probe()
        probe.before_run("run-1", _FakeConfig(), tmp_path)
        probe.before_phase("compile")
        probe.after_phase("compile", 0.250)
        hist = probe._meter.histograms["benchmark.phase.duration_ms"]
        assert hist.observations == [(250.0, {"phase": "compile"})]
        # Phase span closed.
        assert probe._phase_span is None
        assert probe._metrics_emitted >= 1

    def test_on_error_records_exception_and_counter(self, tmp_path: Path):
        probe = self._make_probe()
        probe.before_run("run-1", _FakeConfig(), tmp_path)
        probe.before_phase("latency")
        # Capture the live span before on_error closes it.
        live_span = probe._phase_span
        exc = RuntimeError("kapow")
        probe.on_error("latency", exc)
        assert exc in live_span.exceptions
        counter = probe._meter.counters["benchmark.errors_total"]
        assert counter.observations == [(1, {"phase": "latency"})]
        assert probe._phase_span is None

    def test_after_run_records_result_metrics(self, tmp_path: Path):
        probe = self._make_probe()
        probe.before_run("run-1", _FakeConfig(), tmp_path)
        result = {
            "latency_p50_ms": 1.5,
            "latency_p95_ms": 2.0,
            "latency_p99_ms": 3.0,
            "throughput_mean_samples_sec": 1234.5,
            "experiment_cost_usd": 0.001,
        }
        probe.after_run("run-1", result)

        lat = probe._meter.histograms["benchmark.latency_ms"]
        recorded = {attrs["quantile"]: v for v, attrs in lat.observations}
        assert recorded == {"p50": 1.5, "p95": 2.0, "p99": 3.0}

        tp = probe._meter.histograms["benchmark.throughput_samples_per_sec"]
        assert tp.observations == [(1234.5, None)]

        cost = probe._meter.histograms["benchmark.experiment_cost_usd"]
        assert cost.observations == [(0.001, None)]

        # Run span closed.
        assert probe._run_span is None

    def test_after_run_with_none_result_does_not_record_metrics(self, tmp_path: Path):
        probe = self._make_probe()
        probe.before_run("run-1", _FakeConfig(), tmp_path)
        probe.after_run("run-1", None)
        # No latency histogram created (would only be created on first record).
        assert "benchmark.latency_ms" not in probe._meter.histograms
        assert probe._run_span is None

    def test_write_log_reflects_emissions(self, tmp_path: Path):
        probe = self._make_probe()
        probe.before_run("run-1", _FakeConfig(), tmp_path)
        probe.before_phase("preflight")
        probe.after_phase("preflight", 0.01)
        log = probe.write_log()
        assert log["available"] is True
        assert log["spans_emitted"] >= 2  # run + phase
        assert log["metrics_emitted"] >= 1
