"""
Metrics — CSV per step + Prometheus-compatible name registry.

The same metric names are used by the on-disk CSV exporter (no install
needed) and the `observability/exporters/cloud_tpu_metrics_exporter.py`
that uses `prometheus_client` (optional install). Keep names in sync.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Canonical metric name list — used by Prometheus exporter and the
# Markdown metrics dictionary. Documented in docs/13_oct_metrics_dictionary.md.
METRIC_NAMES = (
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

# Safe labels — low cardinality.
SAFE_LABELS = ("workload_name", "framework", "tpu_version", "run_mode",
               "operation_type", "error_type")
# Dangerous labels — high cardinality, will blow up Prometheus index. Use
# these in Loki logs, not Prometheus metrics.
DANGEROUS_LABELS = ("trace_id", "step_id", "hlo_op_id",
                    "executable_id", "tensor_id", "shard_id")


@dataclass
class MetricRow:
    step: int
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricStream:
    rows: List[MetricRow] = field(default_factory=list)

    def record(self, step: int, **fields: Any) -> None:
        self.rows.append(MetricRow(step=step, fields=fields))

    def column_names(self) -> List[str]:
        cols = ["step"]
        seen = set(cols)
        for r in self.rows:
            for k in r.fields:
                if k not in seen:
                    cols.append(k)
                    seen.add(k)
        return cols

    def write_csv(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = self.column_names()
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                row = {"step": r.step}
                row.update(r.fields)
                w.writerow(row)
        return len(self.rows)
