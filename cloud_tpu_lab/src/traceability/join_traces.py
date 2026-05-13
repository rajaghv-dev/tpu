"""
Traceability join — links the same trace_id across logs, metrics, and trace.

Given the trio of artefacts a run produces:

    artifacts/logs/run_<trace_id>.jsonl
    artifacts/metrics/run_<trace_id>.csv
    artifacts/traces/run_<trace_id>.json

`join_run` returns a Python dict that joins them by `trace_id` so the
report renderer can reference all three.

Useful when you have a TRACE ID and just want one object representing
that run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def join_run(
    trace_id: str,
    log_path: Path,
    metrics_path: Path,
    trace_path: Path,
) -> Dict[str, Any]:
    logs: List[dict] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    metrics_rows: List[dict] = []
    if metrics_path.exists():
        with metrics_path.open() as fh:
            metrics_rows = list(csv.DictReader(fh))

    trace_obj: dict = {}
    if trace_path.exists():
        try:
            trace_obj = json.loads(trace_path.read_text())
        except json.JSONDecodeError:
            trace_obj = {}

    return {
        "trace_id": trace_id,
        "logs": logs,
        "metrics_rows": metrics_rows,
        "trace": trace_obj,
        "summary": {
            "n_log_events": len(logs),
            "n_metric_rows": len(metrics_rows),
            "n_trace_events": len(trace_obj.get("traceEvents", [])),
        },
    }
