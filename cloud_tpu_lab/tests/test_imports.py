"""Every public module imports cleanly with stdlib only."""
import importlib

MODULES = [
    "cloud_tpu_lab.src.common.trace",
    "cloud_tpu_lab.src.common.config",
    "cloud_tpu_lab.src.common.timing",
    "cloud_tpu_lab.src.common.cost",
    "cloud_tpu_lab.src.tpu_versions.cloud_tpu_catalog",
    "cloud_tpu_lab.src.tpu_versions.version_compare",
    "cloud_tpu_lab.src.profiling.profiler_trace",
    "cloud_tpu_lab.src.profiling.trace_analyzer",
    "cloud_tpu_lab.src.profiling.bottleneck_report",
    "cloud_tpu_lab.src.observability.logger",
    "cloud_tpu_lab.src.observability.metrics",
    "cloud_tpu_lab.src.observability.report",
    "cloud_tpu_lab.src.traceability.join_traces",
]


def test_all_modules_importable() -> None:
    for name in MODULES:
        importlib.import_module(name)
