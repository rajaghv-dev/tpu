"""
OpenTelemetry instrumentation for the benchmark harness.

Provides lazy-initialised tracer + meter providers, with three modes:
'off' (no-ops), 'otlp' (gRPC export), 'file' (OTLP-JSON to disk).
Env: TPU_BENCH_OTEL, TPU_BENCH_OTEL_ENDPOINT, TPU_BENCH_OTEL_DIR.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


# ── Pure-Python no-op stubs (no opentelemetry import required) ───────────────

class _NoOpSpan:
    """Minimal span stand-in that satisfies set_attribute / set_status calls."""
    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NoOpTracer:
    """Tracer stand-in returned when OTel is disabled or not installed."""

    @contextlib.contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any):
        yield _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpHistogram:
    """Histogram stand-in whose record() is a no-op."""

    def record(self, amount: float, attributes: Any = None, **kwargs: Any) -> None:
        pass


class _NoOpMeter:
    """Meter stand-in that returns _NoOpHistogram for any instrument creation."""

    def create_histogram(self, name: str, **kwargs: Any) -> _NoOpHistogram:
        return _NoOpHistogram()

    def create_counter(self, name: str, **kwargs: Any) -> _NoOpHistogram:
        return _NoOpHistogram()

    def create_up_down_counter(self, name: str, **kwargs: Any) -> _NoOpHistogram:
        return _NoOpHistogram()

    def create_observable_gauge(self, name: str, **kwargs: Any) -> _NoOpHistogram:
        return _NoOpHistogram()


# ── Module state ─────────────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "initialized": False,
    "enabled": False,
    "mode": "off",
    "tracer_provider": None,
    "meter_provider": None,
    "file_span_exporter": None,
    "file_metric_exporter": None,
    "instruments": None,
    "run_id": None,
    "lock": threading.Lock(),
}


# ── Public configuration ─────────────────────────────────────────────────────

# Metric instrument schema (name, unit, key in get_instruments dict).
_METRIC_SCHEMA = (
    ("latency_ms",             "benchmark.latency.ms",                "ms"),
    ("throughput_samples_sec", "benchmark.throughput.samples_sec",    ""),
    ("compile_cold_s",         "benchmark.compile.cold.seconds",      "s"),
    ("compile_warm_s",         "benchmark.compile.warm.seconds",      "s"),
    ("latency_cv_pct",         "benchmark.latency.cv.percent",        "%"),
)


def is_enabled() -> bool:
    """Return True when OTel has been initialised in a non-off mode."""
    return bool(_state["enabled"])


def _read_mode() -> str:
    """Resolve the configured mode from env (off|otlp|file)."""
    val = (os.environ.get("TPU_BENCH_OTEL") or "off").lower().strip()
    if val not in ("off", "otlp", "file"):
        val = "off"
    return val


# ── File exporters (used when TPU_BENCH_OTEL=file) ───────────────────────────

def _make_file_span_exporter(path: Path):
    """Build a SpanExporter that writes one JSON object per line."""
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _JsonlSpanExporter(SpanExporter):
        def __init__(self, p: Path) -> None:
            self._path = p
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()
            self._fh = self._path.open("a", encoding="utf-8")

        def export(self, spans):  # type: ignore[override]
            with self._lock:
                for span in spans:
                    try:
                        line = span.to_json(indent=None)
                    except Exception:
                        line = json.dumps({"name": getattr(span, "name", "unknown")})
                    self._fh.write(line.replace("\n", " ") + "\n")
                self._fh.flush()
            return SpanExportResult.SUCCESS

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            with self._lock:
                self._fh.flush()
            return True

        def shutdown(self) -> None:
            with self._lock:
                if not self._fh.closed:
                    self._fh.flush()
                    self._fh.close()

    return _JsonlSpanExporter(path)


def _make_file_metric_exporter(path: Path):
    """Build a MetricExporter that writes one JSON object per export call."""
    from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult

    class _JsonlMetricExporter(MetricExporter):
        def __init__(self, p: Path) -> None:
            super().__init__()
            self._path = p
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()
            self._fh = self._path.open("a", encoding="utf-8")

        def export(self, metrics_data, timeout_millis: float = 10000, **kwargs):  # type: ignore[override]
            with self._lock:
                try:
                    line = metrics_data.to_json(indent=None)
                except Exception:
                    line = json.dumps({"resource_metrics": []})
                self._fh.write(line.replace("\n", " ") + "\n")
                self._fh.flush()
            return MetricExportResult.SUCCESS

        def force_flush(self, timeout_millis: float = 10000) -> bool:
            with self._lock:
                self._fh.flush()
            return True

        def shutdown(self, timeout_millis: float = 30000, **kwargs) -> None:
            with self._lock:
                if not self._fh.closed:
                    self._fh.flush()
                    self._fh.close()

    return _JsonlMetricExporter(path)


# ── Initialisation ───────────────────────────────────────────────────────────

def init_otel(resource_attrs: Optional[Dict[str, Any]] = None) -> None:
    """Initialise tracer + meter providers based on env config (idempotent)."""
    with _state["lock"]:
        if _state["initialized"]:
            return
        mode = _read_mode()
        _state["mode"] = mode
        _state["run_id"] = (resource_attrs or {}).get("run_id") or uuid.uuid4().hex[:12]

        if mode == "off":
            _state["enabled"] = False
            _state["initialized"] = True
            return

        # Lazy imports — only required when actually enabled.
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        attrs: Dict[str, Any] = {
            "service.name": "tpu-benchmark",
            "service.version": "stage1",
        }
        if resource_attrs:
            for k, v in resource_attrs.items():
                attrs[str(k)] = v
        resource = Resource.create(attrs)

        if mode == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            endpoint = os.environ.get("TPU_BENCH_OTEL_ENDPOINT", "localhost:4317")
            span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        else:  # mode == "file"
            out_dir = Path(os.environ.get("TPU_BENCH_OTEL_DIR", "results/otel"))
            run_id = _state["run_id"]
            span_path = out_dir / f"{run_id}_spans.jsonl"
            metric_path = out_dir / f"{run_id}_metrics.jsonl"
            span_exporter = _make_file_span_exporter(span_path)
            metric_exporter = _make_file_metric_exporter(metric_path)
            _state["file_span_exporter"] = span_exporter
            _state["file_metric_exporter"] = metric_exporter

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

        reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=60_000
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

        _state["tracer_provider"] = tracer_provider
        _state["meter_provider"] = meter_provider
        _state["enabled"] = True
        _state["initialized"] = True


def get_tracer():
    """Return a Tracer (real or no-op when disabled)."""
    if not _state["enabled"]:
        return _NoOpTracer()
    from opentelemetry import trace
    return trace.get_tracer("tpu-benchmark", "stage1")


def get_meter():
    """Return a Meter (real or no-op when disabled)."""
    if not _state["enabled"]:
        return _NoOpMeter()
    from opentelemetry import metrics
    return metrics.get_meter("tpu-benchmark", "stage1")


def get_instruments() -> Dict[str, Any]:
    """Return cached dict of histogram instruments keyed by short name."""
    if _state["instruments"] is not None:
        return _state["instruments"]

    meter = get_meter()
    instruments: Dict[str, Any] = {}
    for key, name, unit in _METRIC_SCHEMA:
        instruments[key] = meter.create_histogram(
            name=name,
            unit=unit,
            description=f"Benchmark histogram for {key}",
        )
    _state["instruments"] = instruments
    return instruments


def shutdown_otel() -> None:
    """Flush + close all exporters. Safe to call multiple times."""
    with _state["lock"]:
        if not _state["initialized"]:
            return
        tp = _state.get("tracer_provider")
        mp = _state.get("meter_provider")
        try:
            if tp is not None:
                tp.shutdown()
        except Exception:
            pass
        try:
            if mp is not None:
                mp.shutdown()
        except Exception:
            pass
        # File exporters' shutdown is called via providers, but be defensive:
        for key in ("file_span_exporter", "file_metric_exporter"):
            exp = _state.get(key)
            if exp is not None:
                try:
                    exp.shutdown()
                except Exception:
                    pass
        _state["initialized"] = False
        _state["enabled"] = False
        _state["tracer_provider"] = None
        _state["meter_provider"] = None
        _state["file_span_exporter"] = None
        _state["file_metric_exporter"] = None
        _state["instruments"] = None


def _reset_for_tests() -> None:
    """Test-only: drop cached providers so the next init_otel re-runs."""
    try:
        shutdown_otel()
    except Exception:
        pass
    # Reset the global trace + metrics providers. OTel exposes the trace
    # globals on `opentelemetry.trace` directly, and the metrics globals on
    # `opentelemetry.metrics._internal` — handle both.
    try:
        from opentelemetry import trace as _t  # type: ignore
        if hasattr(_t, "_TRACER_PROVIDER"):
            _t._TRACER_PROVIDER = None
        if hasattr(_t, "_TRACER_PROVIDER_SET_ONCE"):
            _t._TRACER_PROVIDER_SET_ONCE._done = False
    except Exception:
        pass
    try:
        from opentelemetry.metrics import _internal as m_internal  # type: ignore
        if hasattr(m_internal, "_METER_PROVIDER"):
            m_internal._METER_PROVIDER = None
        if hasattr(m_internal, "_METER_PROVIDER_SET_ONCE"):
            m_internal._METER_PROVIDER_SET_ONCE._done = False
    except Exception:
        pass
    _state["initialized"] = False
    _state["enabled"] = False
    _state["mode"] = "off"
    _state["tracer_provider"] = None
    _state["meter_provider"] = None
    _state["file_span_exporter"] = None
    _state["file_metric_exporter"] = None
    _state["instruments"] = None
    _state["run_id"] = None
