"""
Trace and correlation IDs — the spine of the OCT (Observability /
Controllability / Traceability) model.

Every event in the simulation pipeline (model → HLO → executable → device
execution → HBM → sharding → collective) carries a small bundle of IDs so
you can join them across logs / metrics / traces. The same trace_id flows
from the top-level demo down through every layer.

ID hierarchy (parent → child):

    trace_id
    ├── step_id            (one per training step within a trace)
    ├── model_layer_id     (per model layer in the graph)
    │   └── hlo_op_id      (per HLO op produced by lowering that layer)
    │       └── executable_id (per compiled XLA module)
    │           └── device_event_id (per execution on a device)
    │               ├── tensor_id   (per logical tensor)
    │               │   └── shard_id (per tensor shard if sharded)
    │               └── collective_id (per all-reduce / all-gather / ...)

All IDs are short, human-readable strings. Avoid UUIDs in user-facing places
because they're hard to grep — instead use prefix + monotonic counter.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

_lock = threading.Lock()
_counters: dict[str, itertools.count] = {}
# Process-wide TRACE counter — NOT cleared by reset_counters() so two
# back-to-back runs in the same Python process produce distinct trace_ids.
_trace_counter = itertools.count(1)


def _next(prefix: str) -> str:
    """Return prefix-NNNN where NNNN is a monotonic counter per prefix."""
    with _lock:
        if prefix not in _counters:
            _counters[prefix] = itertools.count(1)
        n = next(_counters[prefix])
    return f"{prefix}-{n:04d}"


def reset_counters() -> None:
    """Reset all per-run counters — does NOT reset the trace_id counter."""
    with _lock:
        _counters.clear()


# ── ID minters ────────────────────────────────────────────────────────────────

def new_trace_id() -> str:
    """Top-level correlation ID for one end-to-end run (process-unique)."""
    with _lock:
        n = next(_trace_counter)
    return f"TRACE-{n:04d}"


def new_step_id() -> str:          return _next("STEP")
def new_model_layer_id() -> str:   return _next("LAYER")
def new_hlo_op_id() -> str:        return _next("HLO")
def new_executable_id() -> str:    return _next("EXE")
def new_device_event_id() -> str:  return _next("DEV")
def new_tensor_id() -> str:        return _next("TENSOR")
def new_shard_id() -> str:         return _next("SHARD")
def new_collective_id() -> str:    return _next("COLL")


# ── Trace context object ─────────────────────────────────────────────────────

@dataclass
class TraceContext:
    """
    Bundle of correlation IDs that flows through the run.

    Layers attach the subset of fields they care about to each event they
    emit. The observability layer writes whatever fields are populated.
    """
    trace_id: str = field(default_factory=new_trace_id)
    step_id: Optional[str] = None
    model_layer_id: Optional[str] = None
    hlo_op_id: Optional[str] = None
    executable_id: Optional[str] = None
    device_event_id: Optional[str] = None
    tensor_id: Optional[str] = None
    shard_id: Optional[str] = None
    collective_id: Optional[str] = None

    def with_step(self, step_id: str) -> "TraceContext":
        """Return a copy with `step_id` set — non-mutating; safe to fan out."""
        return TraceContext(
            trace_id=self.trace_id, step_id=step_id,
            model_layer_id=self.model_layer_id,
            hlo_op_id=self.hlo_op_id,
            executable_id=self.executable_id,
            device_event_id=self.device_event_id,
            tensor_id=self.tensor_id,
            shard_id=self.shard_id,
            collective_id=self.collective_id,
        )

    def as_log_fields(self) -> dict:
        """Render to the dict shape expected by the JSONL logger / Loki."""
        return {
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "model_layer_id": self.model_layer_id,
            "hlo_op_id": self.hlo_op_id,
            "executable_id": self.executable_id,
            "device_event_id": self.device_event_id,
            "tensor_id": self.tensor_id,
            "shard_id": self.shard_id,
            "collective_id": self.collective_id,
        }


def utc_now_iso() -> str:
    """ISO-8601 timestamp with millisecond precision, used in every log line."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z"
