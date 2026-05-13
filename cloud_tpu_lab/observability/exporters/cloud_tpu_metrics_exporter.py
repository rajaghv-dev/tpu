"""
cloud_tpu_metrics_exporter — Prometheus exporter for cloud_tpu_lab.

What this script does
---------------------
- Stands up an HTTP `/metrics` endpoint that Prometheus can scrape.
- Registers exactly the metric names declared in
  `src/observability/metrics.py:METRIC_NAMES`. Nothing else.
- Tails one or more JSONL log files (the artefacts emitted by
  `src/observability/logger.py`) and updates the gauges / counters
  according to the `event` field on each line.

Cardinality discipline
----------------------
Every label has a finite, small set of values:

    workload_name  - dozen-ish strings per project
    framework      - {jax, torch_xla, tf, cpu_sim}
    tpu_version    - {v4, v5e, v5p, v6e, cpu_sim}
    run_mode       - {local_cpu, colab, cloud_tpu_vm}
    operation_type - small enum (used only on HBM gauges)
    error_type     - small enum (oom, unsupported_op, slow_step, recompile)

High-cardinality fields from the log stream — `trace_id`, `step_id`,
`hlo_op_id`, `executable_id`, `tensor_id`, `shard_id` — are intentionally
NOT propagated to Prometheus labels. They stay in the JSONL log lines
where Loki indexes them per-stream rather than per-distinct-value. This
keeps the Prometheus active-series count bounded regardless of how long
the lab runs. If you fork this exporter and add `trace_id` as a label,
the active-series cardinality grows linearly with run count and the
index eventually exceeds available memory. Don't.

Optional dependencies
---------------------
- `prometheus_client` is preferred. If missing, we fall back to a tiny
  stdlib HTTP server that emits the Prometheus text exposition format
  directly. Either way the `/metrics` endpoint works.

Usage
-----
    python cloud_tpu_metrics_exporter.py \\
        --port 9100 \\
        --log-path "cloud_tpu_lab/artifacts/logs/*.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------------------------------------------------
# Canonical metric + label catalogue.
# These must match `src/observability/metrics.py` exactly.
# ----------------------------------------------------------------------

METRIC_NAMES: Tuple[str, ...] = (
    "cloud_tpu_step_time_seconds",
    "cloud_tpu_compile_time_seconds",
    "cloud_tpu_recompile_count_total",
    "cloud_tpu_input_wait_time_seconds",
    "cloud_tpu_device_execution_time_seconds",
    "cloud_tpu_hbm_used_bytes",
    "cloud_tpu_hbm_capacity_bytes",
    "cloud_tpu_hbm_utilization_ratio",
    "cloud_tpu_collective_time_seconds",
    "cloud_tpu_all_reduce_time_seconds",
    "cloud_tpu_all_gather_time_seconds",
    "cloud_tpu_reduce_scatter_time_seconds",
    "cloud_tpu_matrix_unit_utilization_ratio",
    "cloud_tpu_memory_stall_time_seconds",
    "cloud_tpu_host_overhead_seconds",
    "cloud_tpu_checkpoint_time_seconds",
    "cloud_tpu_tokens_per_second",
    "cloud_tpu_samples_per_second",
    "cloud_tpu_cost_per_step",
    "cloud_tpu_cost_per_token",
    "cloud_tpu_error_count_total",
)

SAFE_LABELS: Tuple[str, ...] = (
    "workload_name",
    "framework",
    "tpu_version",
    "run_mode",
)

# Metrics that need an extra low-cardinality label.
HBM_LABELS = SAFE_LABELS + ("operation_type",)
ERROR_LABELS = SAFE_LABELS + ("error_type",)

# Metrics whose semantics are "always increasing" -> Counter,
# everything else is a Gauge.
COUNTER_METRICS = frozenset(
    {"cloud_tpu_recompile_count_total", "cloud_tpu_error_count_total"}
)


# ----------------------------------------------------------------------
# Prometheus client setup with graceful fallback.
# ----------------------------------------------------------------------

try:
    from prometheus_client import (  # type: ignore[import-not-found]
        CollectorRegistry,
        Counter,
        Gauge,
        start_http_server,
    )

    HAVE_PROM_CLIENT = True
except Exception:  # pragma: no cover - optional dep absent
    HAVE_PROM_CLIENT = False


log = logging.getLogger("cloud_tpu_metrics_exporter")


# ----------------------------------------------------------------------
# Fallback exporter — only used when prometheus_client is unavailable.
# ----------------------------------------------------------------------


@dataclass
class _FallbackMetric:
    name: str
    kind: str  # "counter" or "gauge"
    help: str
    label_names: Tuple[str, ...]
    samples: Dict[Tuple[str, ...], float] = field(default_factory=dict)

    def _key(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        return tuple(labels.get(n, "") for n in self.label_names)

    def set(self, value: float, labels: Dict[str, str]) -> None:
        self.samples[self._key(labels)] = float(value)

    def inc(self, amount: float, labels: Dict[str, str]) -> None:
        k = self._key(labels)
        self.samples[k] = self.samples.get(k, 0.0) + float(amount)

    def render(self) -> str:
        lines: List[str] = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} {self.kind}",
        ]
        for key, value in self.samples.items():
            if self.label_names:
                pairs = ",".join(
                    f'{n}="{_escape(v)}"' for n, v in zip(self.label_names, key)
                )
                lines.append(f"{self.name}{{{pairs}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines) + "\n"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _FallbackRegistry:
    def __init__(self) -> None:
        self.metrics: Dict[str, _FallbackMetric] = {}

    def register(
        self, name: str, kind: str, help_text: str, label_names: Tuple[str, ...]
    ) -> _FallbackMetric:
        m = _FallbackMetric(name=name, kind=kind, help=help_text, label_names=label_names)
        self.metrics[name] = m
        return m

    def render(self) -> str:
        return "".join(m.render() for m in self.metrics.values())


def _start_fallback_http(registry: _FallbackRegistry, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            body = registry.render().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # silence access log

    httpd = HTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()


# ----------------------------------------------------------------------
# Metric registry — single source of truth for type/labels per metric.
# ----------------------------------------------------------------------


@dataclass
class MetricSpec:
    name: str
    help: str
    label_names: Tuple[str, ...]
    kind: str  # counter | gauge


def _metric_specs() -> List[MetricSpec]:
    specs: List[MetricSpec] = []
    for name in METRIC_NAMES:
        if name in COUNTER_METRICS:
            kind = "counter"
        else:
            kind = "gauge"
        if name.startswith("cloud_tpu_hbm_") and name != "cloud_tpu_hbm_utilization_ratio":
            labels = HBM_LABELS
        elif name == "cloud_tpu_error_count_total":
            labels = ERROR_LABELS
        else:
            labels = SAFE_LABELS
        specs.append(MetricSpec(name=name, help=_help_for(name), label_names=labels, kind=kind))
    return specs


def _help_for(name: str) -> str:
    return {
        "cloud_tpu_step_time_seconds": "Wall-clock step time on the TPU.",
        "cloud_tpu_compile_time_seconds": "XLA compile time (first step / on cache miss).",
        "cloud_tpu_recompile_count_total": "Number of XLA recompilations since process start.",
        "cloud_tpu_input_wait_time_seconds": "Time the device spent waiting for input.",
        "cloud_tpu_device_execution_time_seconds": "Time the device spent actually executing.",
        "cloud_tpu_hbm_used_bytes": "HBM bytes in use.",
        "cloud_tpu_hbm_capacity_bytes": "HBM capacity in bytes.",
        "cloud_tpu_hbm_utilization_ratio": "HBM used / capacity, in [0, 1].",
        "cloud_tpu_collective_time_seconds": "Total time spent in collectives.",
        "cloud_tpu_all_reduce_time_seconds": "Time spent in all-reduce collectives.",
        "cloud_tpu_all_gather_time_seconds": "Time spent in all-gather collectives.",
        "cloud_tpu_reduce_scatter_time_seconds": "Time spent in reduce-scatter collectives.",
        "cloud_tpu_matrix_unit_utilization_ratio": "MXU utilization ratio, in [0, 1].",
        "cloud_tpu_memory_stall_time_seconds": "Time the device stalled on HBM bandwidth.",
        "cloud_tpu_host_overhead_seconds": "Host-side overhead per step (driver, dataloader, etc.).",
        "cloud_tpu_checkpoint_time_seconds": "Time spent saving / loading checkpoints.",
        "cloud_tpu_tokens_per_second": "Throughput in tokens per second.",
        "cloud_tpu_samples_per_second": "Throughput in samples per second.",
        "cloud_tpu_cost_per_step": "Estimated USD cost per training step.",
        "cloud_tpu_cost_per_token": "Estimated USD cost per token.",
        "cloud_tpu_error_count_total": "Errors observed in the log stream, by error_type.",
    }.get(name, "cloud_tpu_lab metric.")


class MetricRegistry:
    """Wraps either prometheus_client or the fallback registry uniformly."""

    def __init__(self) -> None:
        self.specs = _metric_specs()
        if HAVE_PROM_CLIENT:
            self.registry = CollectorRegistry()
            self.objs: Dict[str, Any] = {}
            for s in self.specs:
                klass = Counter if s.kind == "counter" else Gauge
                self.objs[s.name] = klass(
                    s.name, s.help, labelnames=s.label_names, registry=self.registry
                )
        else:
            self.fallback = _FallbackRegistry()
            for s in self.specs:
                self.fallback.register(s.name, s.kind, s.help, s.label_names)

    def _safe_labels(self, name: str, labels: Dict[str, str]) -> Dict[str, str]:
        wanted = next(s.label_names for s in self.specs if s.name == name)
        return {k: str(labels.get(k, "unknown")) for k in wanted}

    def set(self, name: str, value: float, labels: Dict[str, str]) -> None:
        lbl = self._safe_labels(name, labels)
        if HAVE_PROM_CLIENT:
            m = self.objs[name]
            spec = next(s for s in self.specs if s.name == name)
            if spec.kind == "counter":
                # Prometheus counters are inc-only; emulate "set" as a delta to current.
                m.labels(**lbl).inc(max(0.0, float(value)))
            else:
                m.labels(**lbl).set(float(value))
        else:
            f = self.fallback.metrics[name]
            if f.kind == "counter":
                f.inc(max(0.0, float(value)), lbl)
            else:
                f.set(float(value), lbl)

    def inc(self, name: str, amount: float, labels: Dict[str, str]) -> None:
        lbl = self._safe_labels(name, labels)
        if HAVE_PROM_CLIENT:
            self.objs[name].labels(**lbl).inc(float(amount))
        else:
            self.fallback.metrics[name].inc(float(amount), lbl)


# ----------------------------------------------------------------------
# JSONL → metric updates.
# ----------------------------------------------------------------------


def _base_labels(line: Dict[str, Any]) -> Dict[str, str]:
    return {
        "workload_name": str(line.get("workload_name", line.get("model_name", "unknown"))),
        "framework": str(line.get("framework", "cpu_sim")),
        "tpu_version": str(line.get("tpu_version", "cpu_sim")),
        "run_mode": str(line.get("run_mode", "local_cpu")),
    }


def update_from_log_line(reg: MetricRegistry, line: Dict[str, Any]) -> None:
    """Update Prometheus metrics from a single JSONL log line.

    Looks at `event`, then promotes fields off `metrics` or the
    top-level dict. Unknown events are simply ignored — extending the
    mapping is additive and safe.
    """
    event = str(line.get("event", ""))
    metrics = line.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    base = _base_labels(line)

    def num(key: str, default: Optional[float] = None) -> Optional[float]:
        for src in (metrics, line):
            if key in src:
                try:
                    return float(src[key])
                except (TypeError, ValueError):
                    return default
        return default

    # XLA
    if event == "xla.compile":
        v = num("compile_time_s")
        if v is not None:
            reg.set("cloud_tpu_compile_time_seconds", v, base)
        cache_hit = line.get("cache_hit")
        if cache_hit is False and v is not None and v > 0:
            reg.inc("cloud_tpu_recompile_count_total", 1.0, base)

    elif event == "xla.recompile":
        reg.inc("cloud_tpu_recompile_count_total", 1.0, base)

    # PJRT / runtime
    elif event in ("pjrt.step", "runtime.step", "tpu.execute"):
        v = num("step_time_s")
        if v is not None:
            reg.set("cloud_tpu_step_time_seconds", v, base)
        v = num("device_execution_time_s")
        if v is not None:
            reg.set("cloud_tpu_device_execution_time_seconds", v, base)
        v = num("host_overhead_s")
        if v is not None:
            reg.set("cloud_tpu_host_overhead_seconds", v, base)
        v = num("tokens_per_second")
        if v is not None:
            reg.set("cloud_tpu_tokens_per_second", v, base)
        v = num("samples_per_second")
        if v is not None:
            reg.set("cloud_tpu_samples_per_second", v, base)

    # Input pipeline
    elif event in ("input_pipeline.load_batch", "input_pipeline.wait"):
        v = num("input_wait_s")
        if v is not None:
            reg.set("cloud_tpu_input_wait_time_seconds", v, base)

    # HBM
    elif event in ("hbm.snapshot", "hbm.read_write"):
        op = str(line.get("operation_type", "all"))
        hbm_labels = dict(base, operation_type=op)
        v = num("hbm_used_bytes")
        if v is not None:
            reg.set("cloud_tpu_hbm_used_bytes", v, hbm_labels)
        v = num("hbm_capacity_bytes")
        if v is not None:
            reg.set("cloud_tpu_hbm_capacity_bytes", v, hbm_labels)
        v = num("hbm_utilization_ratio")
        if v is not None:
            reg.set("cloud_tpu_hbm_utilization_ratio", v, base)
        v = num("memory_stall_s")
        if v is not None:
            reg.set("cloud_tpu_memory_stall_time_seconds", v, base)

    elif event == "hbm.oom":
        reg.inc(
            "cloud_tpu_error_count_total",
            1.0,
            dict(base, error_type="oom"),
        )

    # Collectives
    elif event.startswith("collective."):
        v = num("collective_time_s")
        if v is not None:
            reg.set("cloud_tpu_collective_time_seconds", v, base)
        if event == "collective.all_reduce" and (v := num("time_s")) is not None:
            reg.set("cloud_tpu_all_reduce_time_seconds", v, base)
        elif event == "collective.all_gather" and (v := num("time_s")) is not None:
            reg.set("cloud_tpu_all_gather_time_seconds", v, base)
        elif event == "collective.reduce_scatter" and (v := num("time_s")) is not None:
            reg.set("cloud_tpu_reduce_scatter_time_seconds", v, base)

    # Profiler / utilization
    elif event in ("profiler.collect", "profiler.summary"):
        v = num("matrix_unit_utilization_ratio")
        if v is not None:
            reg.set("cloud_tpu_matrix_unit_utilization_ratio", v, base)

    # Checkpoint
    elif event == "checkpoint.save":
        v = num("checkpoint_time_s")
        if v is not None:
            reg.set("cloud_tpu_checkpoint_time_seconds", v, base)

    # Cost
    elif event in ("cost.estimate", "report.cost"):
        v = num("cost_per_step")
        if v is not None:
            reg.set("cloud_tpu_cost_per_step", v, base)
        v = num("cost_per_token")
        if v is not None:
            reg.set("cloud_tpu_cost_per_token", v, base)

    # Errors / warnings
    elif event in ("error.unsupported_op",):
        reg.inc(
            "cloud_tpu_error_count_total",
            1.0,
            dict(base, error_type="unsupported_op"),
        )
    elif event in ("error.slow_step",):
        reg.inc(
            "cloud_tpu_error_count_total",
            1.0,
            dict(base, error_type="slow_step"),
        )
    elif str(line.get("level", "")).upper() == "ERROR":
        reg.inc(
            "cloud_tpu_error_count_total",
            1.0,
            dict(base, error_type="generic"),
        )


# ----------------------------------------------------------------------
# JSONL tailer.
# ----------------------------------------------------------------------


def _expand(paths: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        out.extend(sorted(glob.glob(p)))
    # de-dupe, keep order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def tail_jsonl(
    paths: Iterable[str],
    reg: MetricRegistry,
    poll_interval: float = 1.0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Tail one or more JSONL files and feed lines into the metric registry.

    Globs are re-expanded each iteration so newly-rotated files are picked
    up automatically. Each file is read from its last offset; on initial
    open we replay the existing contents so dashboards aren't empty on
    first scrape.
    """
    offsets: Dict[str, int] = {}
    while stop_event is None or not stop_event.is_set():
        for path in _expand(paths):
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            start = offsets.get(path, 0)
            if size < start:
                start = 0  # file was truncated / rotated
            if size == start:
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    fh.seek(start)
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            line = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        try:
                            update_from_log_line(reg, line)
                        except Exception as exc:  # noqa: BLE001
                            log.debug("update_from_log_line failed: %s", exc)
                    offsets[path] = fh.tell()
            except OSError as exc:
                log.debug("read %s failed: %s", path, exc)
        time.sleep(poll_interval)


# ----------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prometheus exporter for cloud_tpu_lab")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument(
        "--log-path",
        action="append",
        default=None,
        help=(
            "Glob of JSONL log files to tail. May be passed multiple times. "
            "Default: cloud_tpu_lab/artifacts/logs/*.jsonl"
        ),
    )
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log_paths = args.log_path or ["cloud_tpu_lab/artifacts/logs/*.jsonl"]

    reg = MetricRegistry()

    if HAVE_PROM_CLIENT:
        start_http_server(args.port, registry=reg.registry)  # type: ignore[arg-type]
        log.info("prometheus_client HTTP server listening on :%d", args.port)
    else:
        _start_fallback_http(reg.fallback, args.port)
        log.info(
            "prometheus_client not installed — fallback HTTP server on :%d", args.port
        )

    stop = threading.Event()
    try:
        tail_jsonl(log_paths, reg, poll_interval=args.poll_interval, stop_event=stop)
    except KeyboardInterrupt:
        stop.set()
        log.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
