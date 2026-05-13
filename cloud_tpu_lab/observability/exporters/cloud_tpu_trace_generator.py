"""
cloud_tpu_trace_generator — emit OpenTelemetry-shaped traces for the spans
the cloud_tpu_lab simulation produces.

Span catalogue (matches the layers in the OCT model):

    model.forward
    xla.lowering
    xla.compile
    pjrt.executable_create
    input_pipeline.load_batch
    runtime.device_put
    tpu.execute
    hbm.read_write
    collective.all_reduce
    checkpoint.save
    profiler.collect
    report.generate

If `opentelemetry-api` and `opentelemetry-exporter-otlp` are installed,
spans are emitted as real OTLP and sent to Tempo at the configured
endpoint. Otherwise we fall back to writing a JSON file in roughly the
OpenTelemetry "ResourceSpans" shape — good enough to import into Jaeger
/ Tempo via their JSON ingestion paths and easy to diff in tests.

Usage
-----
    # JSON fallback (no optional deps required):
    python cloud_tpu_trace_generator.py \\
        --out cloud_tpu_lab/artifacts/traces/run_TRACE-SYNTH.otlp.json

    # OTLP push (requires opentelemetry-sdk + opentelemetry-exporter-otlp):
    python cloud_tpu_trace_generator.py \\
        --otlp-endpoint http://localhost:4317
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SPAN_NAMES: Tuple[str, ...] = (
    "model.forward",
    "xla.lowering",
    "xla.compile",
    "pjrt.executable_create",
    "input_pipeline.load_batch",
    "runtime.device_put",
    "tpu.execute",
    "hbm.read_write",
    "collective.all_reduce",
    "checkpoint.save",
    "profiler.collect",
    "report.generate",
)


# ---- Optional OpenTelemetry SDK path ---------------------------------

def _try_import_otel():
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-not-found]

        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
        except Exception:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )

        return {
            "trace": trace,
            "Resource": Resource,
            "TracerProvider": TracerProvider,
            "BatchSpanProcessor": BatchSpanProcessor,
            "OTLPSpanExporter": OTLPSpanExporter,
        }
    except Exception:
        return None


def emit_otlp(*, endpoint: str, workload_name: str, tpu_version: str,
              framework: str, run_mode: str, seed: int = 0) -> bool:
    """Returns True if OTLP emit succeeded; False if dependencies missing."""
    mods = _try_import_otel()
    if mods is None:
        return False

    rng = random.Random(seed)

    resource = mods["Resource"].create({
        "service.name": "cloud_tpu_lab",
        "service.namespace": "observability",
        "workload_name": workload_name,
        "tpu_version": tpu_version,
        "framework": framework,
        "run_mode": run_mode,
    })
    provider = mods["TracerProvider"](resource=resource)
    exporter = mods["OTLPSpanExporter"](endpoint=endpoint, insecure=True)
    provider.add_span_processor(mods["BatchSpanProcessor"](exporter))
    mods["trace"].set_tracer_provider(provider)
    tracer = mods["trace"].get_tracer("cloud_tpu_lab")

    with tracer.start_as_current_span("model.forward") as root:
        root.set_attribute("layer", "model")
        for name in SPAN_NAMES[1:]:
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("layer", name.split(".")[0])
                time.sleep(max(0.001, rng.gauss(0.01, 0.003)))

    provider.shutdown()
    return True


# ---- JSON fallback ---------------------------------------------------

def _hex(byte_len: int) -> str:
    return uuid.uuid4().hex[: byte_len * 2]


@dataclass
class _Span:
    name: str
    span_id: str
    parent_id: Optional[str]
    start_ns: int
    end_ns: int
    attributes: Dict[str, Any] = field(default_factory=dict)


def _ns(t: float) -> int:
    return int(t * 1_000_000_000)


def build_json_trace(*, workload_name: str, tpu_version: str,
                     framework: str, run_mode: str, seed: int = 0) -> Dict[str, Any]:
    """Build a single trace in OpenTelemetry JSON shape (one ResourceSpan)."""
    rng = random.Random(seed)
    trace_id = _hex(16)
    t = time.time()

    spans: List[_Span] = []
    root_id = _hex(8)
    root_start = _ns(t)
    cursor = t
    children: List[_Span] = []
    for name in SPAN_NAMES[1:]:
        dur = max(0.001, rng.gauss(0.01, 0.003))
        s = _Span(
            name=name,
            span_id=_hex(8),
            parent_id=root_id,
            start_ns=_ns(cursor),
            end_ns=_ns(cursor + dur),
            attributes={"layer": name.split(".")[0]},
        )
        children.append(s)
        cursor += dur
    root_end = _ns(cursor)
    spans.append(
        _Span(
            name="model.forward",
            span_id=root_id,
            parent_id=None,
            start_ns=root_start,
            end_ns=root_end,
            attributes={"layer": "model"},
        )
    )
    spans.extend(children)

    def _to_otlp_span(sp: _Span) -> Dict[str, Any]:
        return {
            "traceId": trace_id,
            "spanId": sp.span_id,
            "parentSpanId": sp.parent_id or "",
            "name": sp.name,
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": str(sp.start_ns),
            "endTimeUnixNano": str(sp.end_ns),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in sp.attributes.items()
            ],
            "status": {"code": "STATUS_CODE_OK"},
        }

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "cloud_tpu_lab"}},
                        {"key": "service.namespace", "value": {"stringValue": "observability"}},
                        {"key": "workload_name", "value": {"stringValue": workload_name}},
                        {"key": "tpu_version", "value": {"stringValue": tpu_version}},
                        {"key": "framework", "value": {"stringValue": framework}},
                        {"key": "run_mode", "value": {"stringValue": run_mode}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "cloud_tpu_lab.trace_generator", "version": "0.1"},
                        "spans": [_to_otlp_span(s) for s in spans],
                    }
                ],
            }
        ]
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Emit OTel-shaped traces for cloud_tpu_lab")
    p.add_argument("--otlp-endpoint", default=None,
                   help="If set, push to this OTLP gRPC endpoint instead of writing JSON")
    p.add_argument("--out", type=Path,
                   default=Path("cloud_tpu_lab/artifacts/traces/run_TRACE-SYNTH.otlp.json"))
    p.add_argument("--workload-name", default="synthetic_workload")
    p.add_argument("--framework", default="cpu_sim")
    p.add_argument("--tpu-version", default="v5p")
    p.add_argument("--run-mode", default="local_cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.otlp_endpoint:
        ok = emit_otlp(
            endpoint=args.otlp_endpoint,
            workload_name=args.workload_name,
            tpu_version=args.tpu_version,
            framework=args.framework,
            run_mode=args.run_mode,
            seed=args.seed,
        )
        if ok:
            print(f"pushed OTLP traces to {args.otlp_endpoint}", file=sys.stderr)
            return 0
        print(
            "opentelemetry SDK not installed — falling back to JSON file.",
            file=sys.stderr,
        )

    payload = build_json_trace(
        workload_name=args.workload_name,
        tpu_version=args.tpu_version,
        framework=args.framework,
        run_mode=args.run_mode,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote OTLP-JSON trace to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
