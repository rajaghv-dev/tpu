"""
JSONL logger — every event becomes one line in
`artifacts/logs/run_<trace_id>.jsonl`. Loki picks up that file via the
provided promtail / alloy config.

Schema is intentionally flat (one level of nesting at most) so Loki / jq /
pandas all consume it cleanly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.trace import TraceContext, utc_now_iso


class JsonlLogger:
    """Buffered JSONL logger — flush() writes to disk."""

    def __init__(self, app: str = "cloud_tpu_lab") -> None:
        self.app = app
        self.buffer: List[Dict[str, Any]] = []

    def log(
        self,
        layer: str,
        event: str,
        message: str = "",
        level: str = "INFO",
        trace: Optional[TraceContext] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        line: Dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "app": self.app,
            "level": level,
            "layer": layer,
            "event": event,
            "message": message,
            "metrics": metrics or {},
        }
        if trace is not None:
            for k, v in trace.as_log_fields().items():
                if v is not None:
                    line[k] = v
        for k, v in extra.items():
            line[k] = _jsonable(v)
        self.buffer.append(line)

    def flush(self, path: Path) -> int:
        """Write the buffered events to `path` (one JSON object per line)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for line in self.buffer:
                fh.write(json.dumps(line) + "\n")
        return len(self.buffer)


def _jsonable(v: Any) -> Any:
    """Best-effort JSON coercion for log fields."""
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)
