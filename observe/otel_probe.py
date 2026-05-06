"""
observe/otel_probe.py — OpenTelemetry probe for the benchmark runner.

This probe emits one parent span per run, one child span per phase, and a
small set of metrics (phase duration, latency quantiles, throughput, cost,
error count). It is intentionally a pure consumer of the Probe contract in
``observe/probe.py``: nothing in the runner needs to change to use it.

## Why lazy, optional imports

Stage 1 deliberately keeps OTel out of ``requirements.txt``. The benchmark
host may be a CPU-only laptop, a TPU pod, or CI — most of those don't have
an OTLP collector reachable, and we don't want to force every dev to install
a tracing stack just to run ``pytest``.

The strategy:

  1.  At module import we attempt the OTel imports inside a single try/except
      and set ``_OTEL_AVAILABLE``. Any failure (ImportError, missing
      transitive deps, version skew) collapses to ``False``. The probe class
      is still defined so callers can construct it unconditionally.
  2.  Every public method short-circuits on ``not self._available`` BEFORE
      touching any OTel object. This keeps the no-op path branch-free of
      AttributeErrors.
  3.  The OTLP exporter is imported separately from the API/SDK because the
      ``opentelemetry-exporter-otlp`` package is the most likely to be
      missing in dev environments. If only the exporter is unavailable we
      fall back to ``ConsoleSpanExporter`` so a developer still sees
      something on stderr.

The runner's ``_safe_call`` already swallows probe exceptions, so we don't
need belt-and-braces try/except inside every hook — but we still avoid
raising on the happy path because failed instrumentation should never look
like a benchmark failure in logs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from observe.probe import Probe

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from benchmarks.runner import ExperimentConfig


_log = logging.getLogger(__name__)


# ── Lazy OTel import ─────────────────────────────────────────────────────────
# We attempt the full SDK + OTLP exporter import here. If any of these fails
# (missing package, version skew), `_OTEL_AVAILABLE` stays False and the probe
# becomes a no-op. The split between OTLP and Console exporter lets us degrade
# gracefully when the exporter alone is missing.
_OTEL_AVAILABLE = False
_OTLP_AVAILABLE = False

try:
    from opentelemetry import metrics as _otel_metrics  # type: ignore
    from opentelemetry import trace as _otel_trace  # type: ignore
    from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
    from opentelemetry.sdk.metrics.export import (  # type: ignore
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import (  # type: ignore
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.trace import Status, StatusCode  # type: ignore

    _OTEL_AVAILABLE = True

    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # type: ignore
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
            OTLPSpanExporter,
        )

        _OTLP_AVAILABLE = True
    except Exception:  # noqa: BLE001 — exporter is optional, console fallback covers us
        _OTLP_AVAILABLE = False
except Exception:  # noqa: BLE001 — any failure → no-op probe
    _OTEL_AVAILABLE = False


# Phases recognised by the runner. Listed for documentation; the probe does
# not gate on this set — it accepts any phase name the runner emits.
_KNOWN_PHASES = (
    "preflight", "model_load", "compile", "warmup",
    "latency", "throughput", "postflight",
)


class OTelProbe(Probe):
    """
    Emit OpenTelemetry traces + metrics for each benchmark run.

    Construct unconditionally; the probe self-disables if OTel is missing.
    """

    name = "otel"

    def __init__(self) -> None:
        # Bookkeeping always exists, even in no-op mode, so write_log always
        # produces a stable schema.
        self._endpoint: str = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
        )
        self._service_name: str = os.environ.get("OTEL_SERVICE_NAME", "tpu-bench")
        self._spans_emitted: int = 0
        self._metrics_emitted: int = 0
        self._available: bool = _OTEL_AVAILABLE

        # Per-run state (set in before_run / before_phase, cleared in
        # after_run / after_phase). Lives outside the available branch so
        # __getattr__-style access is consistent in tests.
        self._run_span_cm: Any = None
        self._run_span: Any = None
        self._phase_span_cm: Any = None
        self._phase_span: Any = None
        self._current_phase: Optional[str] = None

        # Lazily-created instruments. We don't create them eagerly because
        # construction order matters (MeterProvider must be registered first)
        # and because in no-op mode we want zero OTel calls.
        self._tracer: Any = None
        self._meter: Any = None
        self._hist_phase_ms: Any = None
        self._hist_latency_ms: Any = None
        self._hist_throughput: Any = None
        self._hist_cost: Any = None
        self._counter_errors: Any = None

        if not self._available:
            return

        # Provider setup. We register globally because OTel's API expects
        # singleton providers — multiple OTelProbe instances would race
        # otherwise. In practice only one probe is registered per process.
        self._configure_providers()

    # ── Provider configuration ──────────────────────────────────────────────

    def _configure_providers(self) -> None:
        """
        Wire up TracerProvider + MeterProvider with OTLP exporters (or the
        Console fallback). Idempotent at the OTel level: if the user has
        already configured providers we still get back working tracers and
        meters, just attached to whatever provider is current.
        """
        try:
            resource = Resource.create({
                "service.name": self._service_name,
                **_parse_resource_attrs(
                    os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
                ),
            })

            # ── Tracer ────────────────────────────────────────────────────
            tracer_provider = TracerProvider(resource=resource)
            if _OTLP_AVAILABLE:
                span_exporter: Any = OTLPSpanExporter(
                    endpoint=f"{self._endpoint.rstrip('/')}/v1/traces"
                )
            else:
                # Dev-friendly fallback so output isn't silently dropped when
                # the OTLP package isn't installed.
                span_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            _otel_trace.set_tracer_provider(tracer_provider)
            self._tracer = _otel_trace.get_tracer("tpu-bench.runner")

            # ── Meter ─────────────────────────────────────────────────────
            if _OTLP_AVAILABLE:
                metric_exporter = OTLPMetricExporter(
                    endpoint=f"{self._endpoint.rstrip('/')}/v1/metrics"
                )
                reader = PeriodicExportingMetricReader(metric_exporter)
                meter_provider = MeterProvider(
                    resource=resource, metric_readers=[reader]
                )
            else:
                # No-op meter provider keeps instrument creation working
                # without sending anything; user sees Console spans only.
                meter_provider = MeterProvider(resource=resource)
            _otel_metrics.set_meter_provider(meter_provider)
            self._meter = _otel_metrics.get_meter("tpu-bench.runner")
        except Exception as exc:  # noqa: BLE001 — degrade to no-op on any setup failure
            _log.warning("OTelProbe provider setup failed: %s; disabling probe", exc)
            self._available = False

    # ── Instrument helpers ──────────────────────────────────────────────────

    def _phase_histogram(self) -> Any:
        if self._hist_phase_ms is None and self._meter is not None:
            self._hist_phase_ms = self._meter.create_histogram(
                name="benchmark.phase.duration_ms",
                unit="ms",
                description="Wall-clock duration of each benchmark phase.",
            )
        return self._hist_phase_ms

    def _latency_histogram(self) -> Any:
        if self._hist_latency_ms is None and self._meter is not None:
            self._hist_latency_ms = self._meter.create_histogram(
                name="benchmark.latency_ms",
                unit="ms",
                description="Per-quantile latency of one inference pass.",
            )
        return self._hist_latency_ms

    def _throughput_histogram(self) -> Any:
        if self._hist_throughput is None and self._meter is not None:
            self._hist_throughput = self._meter.create_histogram(
                name="benchmark.throughput_samples_per_sec",
                unit="samples/s",
                description="Mean samples/sec under load.",
            )
        return self._hist_throughput

    def _cost_histogram(self) -> Any:
        if self._hist_cost is None and self._meter is not None:
            self._hist_cost = self._meter.create_histogram(
                name="benchmark.experiment_cost_usd",
                unit="USD",
                description="Estimated dollar cost of one experiment.",
            )
        return self._hist_cost

    def _error_counter(self) -> Any:
        if self._counter_errors is None and self._meter is not None:
            self._counter_errors = self._meter.create_counter(
                name="benchmark.errors_total",
                description="Count of phase failures per phase.",
            )
        return self._counter_errors

    # ── Probe lifecycle hooks ───────────────────────────────────────────────

    def before_run(
        self,
        run_id: str,
        config: "ExperimentConfig",
        log_dir: Path,
    ) -> None:
        if not self._available:
            return

        # Start a long-lived span using the context-manager form so that
        # nested phase spans become children automatically (OTel uses the
        # active context, which start_as_current_span sets).
        attrs = {
            "run_id": run_id,
            "model_id": getattr(config, "model_id", ""),
            "device": getattr(config, "device", ""),
            "precision": getattr(config, "precision", ""),
            "framework": getattr(config, "framework", ""),
            "total_params_M": getattr(config, "total_params_M", 0),
        }
        self._run_span_cm = self._tracer.start_as_current_span(
            "benchmark.run", attributes=attrs
        )
        self._run_span = self._run_span_cm.__enter__()
        self._spans_emitted += 1

    def before_phase(self, phase_name: str) -> None:
        if not self._available:
            return
        self._current_phase = phase_name
        self._phase_span_cm = self._tracer.start_as_current_span(
            f"phase.{phase_name}", attributes={"phase": phase_name}
        )
        self._phase_span = self._phase_span_cm.__enter__()
        self._spans_emitted += 1

    def after_phase(self, phase_name: str, duration_s: float) -> None:
        if not self._available:
            return
        try:
            hist = self._phase_histogram()
            if hist is not None:
                hist.record(duration_s * 1000.0, attributes={"phase": phase_name})
                self._metrics_emitted += 1
        finally:
            self._close_phase_span(error=False)

    def on_error(self, phase_name: str, exc: BaseException) -> None:
        if not self._available:
            return
        # Tag the phase span with the exception before closing it. We use
        # record_exception so the trace backend gets the full type+stack.
        try:
            if self._phase_span is not None:
                if isinstance(exc, BaseException):
                    self._phase_span.record_exception(exc)
                self._phase_span.set_status(Status(StatusCode.ERROR))
            counter = self._error_counter()
            if counter is not None:
                counter.add(1, attributes={"phase": phase_name})
                self._metrics_emitted += 1
        finally:
            self._close_phase_span(error=True)

    def after_run(
        self,
        run_id: str,
        result: Optional[Dict[str, Any]],
    ) -> None:
        if not self._available:
            return
        try:
            if result is not None:
                self._record_result_metrics(result)
                # Final structured log so OTel's logs SDK (if configured by
                # the user) can pick it up. We use the standard logger; OTel
                # auto-instrumentation hooks the logging module.
                logging.getLogger("benchmark").info(
                    "benchmark.run.complete",
                    extra={k: v for k, v in result.items()
                           if isinstance(v, (str, int, float, bool, type(None)))},
                )
        finally:
            self._close_run_span()

    def write_log(self) -> Optional[Dict[str, Any]]:
        # Always return a dict — even in no-op mode — so the run_log retains
        # evidence of *what the probe did*. This is how a user discovers
        # "OTel wasn't installed" after the fact, without re-running.
        return {
            "otel_endpoint": self._endpoint,
            "service_name": self._service_name,
            "spans_emitted": self._spans_emitted,
            "metrics_emitted": self._metrics_emitted,
            "available": self._available,
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _close_phase_span(self, error: bool) -> None:
        """Exit the phase span CM. Tolerates double-close from on_error+after_phase races."""
        if self._phase_span_cm is None:
            return
        try:
            self._phase_span_cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 — never propagate exporter errors
            pass
        finally:
            self._phase_span_cm = None
            self._phase_span = None
            self._current_phase = None

    def _close_run_span(self) -> None:
        if self._run_span_cm is None:
            return
        try:
            self._run_span_cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._run_span_cm = None
            self._run_span = None

    def _record_result_metrics(self, result: Dict[str, Any]) -> None:
        """Translate the runner's result dict into histogram observations."""
        latency = self._latency_histogram()
        if latency is not None:
            for q_field, q_label in (
                ("latency_p50_ms", "p50"),
                ("latency_p95_ms", "p95"),
                ("latency_p99_ms", "p99"),
            ):
                v = result.get(q_field)
                if isinstance(v, (int, float)):
                    latency.record(float(v), attributes={"quantile": q_label})
                    self._metrics_emitted += 1

        throughput = self._throughput_histogram()
        tp = result.get("throughput_mean_samples_sec")
        if throughput is not None and isinstance(tp, (int, float)):
            throughput.record(float(tp))
            self._metrics_emitted += 1

        cost = self._cost_histogram()
        cv = result.get("experiment_cost_usd")
        if cost is not None and isinstance(cv, (int, float)):
            cost.record(float(cv))
            self._metrics_emitted += 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_resource_attrs(spec: str) -> Dict[str, str]:
    """
    Parse an ``OTEL_RESOURCE_ATTRIBUTES``-style string (``k1=v1,k2=v2``).

    Lenient on purpose — malformed entries are skipped rather than raising,
    because a typo in env config should not disable telemetry entirely.
    """
    out: Dict[str, str] = {}
    if not spec:
        return out
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece or "=" not in piece:
            continue
        k, _, v = piece.partition("=")
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out
