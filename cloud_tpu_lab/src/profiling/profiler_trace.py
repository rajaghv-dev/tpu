"""
Profiler-style trace generator.

Emits a "perfetto/Chrome-trace"-shaped list of events that you can either
view directly in `chrome://tracing` or aggregate into a Matplotlib timeline.

Each event:
    {
      "name": "...", "cat": "...",
      "ts": <microseconds from trace start>,
      "dur": <microseconds>,
      "pid": 1, "tid": <thread or device id>,
      "ph": "X" (complete-event format),
      "args": { trace_id, step_id, hlo_op_id, ... }
    }

The simulator emits one event per:
  * phase (compile, warmup, train_step, eval, checkpoint)
  * op execution
  * collective
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    name: str
    cat: str
    ts_us: int
    dur_us: int
    tid: int
    args: Dict[str, Any]


@dataclass
class ProfilerTrace:
    trace_id: str
    events: List[TraceEvent] = field(default_factory=list)
    _t0_s: float = 0.0

    def start(self) -> None:
        self._t0_s = time.perf_counter()

    def _now_us(self) -> int:
        return int((time.perf_counter() - self._t0_s) * 1e6)

    def add_event(
        self,
        name: str, cat: str, dur_s: float,
        tid: int = 0, args: Optional[Dict[str, Any]] = None,
        when_us: Optional[int] = None,
    ) -> None:
        self.events.append(TraceEvent(
            name=name, cat=cat,
            ts_us=when_us if when_us is not None else self._now_us(),
            dur_us=int(dur_s * 1e6),
            tid=tid,
            args=args or {},
        ))

    # ── Export ──────────────────────────────────────────────────────────────

    def to_chrome_trace(self) -> Dict[str, Any]:
        return {
            "traceEvents": [
                {
                    "name": e.name, "cat": e.cat,
                    "ph": "X", "ts": e.ts_us, "dur": e.dur_us,
                    "pid": 1, "tid": e.tid,
                    "args": e.args,
                }
                for e in self.events
            ],
            "displayTimeUnit": "ms",
            "metadata": {"trace_id": self.trace_id},
        }

    def write_chrome_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_chrome_trace(), indent=2))

    # ── Aggregate helpers (used by the bottleneck report) ───────────────────

    def total_duration_s(self, cat: Optional[str] = None) -> float:
        return sum(e.dur_us for e in self.events
                   if cat is None or e.cat == cat) / 1e6

    def breakdown_by_cat(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for e in self.events:
            out[e.cat] = out.get(e.cat, 0.0) + e.dur_us / 1e6
        return out
