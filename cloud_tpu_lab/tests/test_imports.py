"""Every public module imports cleanly with stdlib only."""
import importlib

MODULES = [
    "cloud_tpu_lab.src.common.trace",
    "cloud_tpu_lab.src.common.config",
    "cloud_tpu_lab.src.common.timing",
    "cloud_tpu_lab.src.common.cost",
    "cloud_tpu_lab.src.tpu_versions.cloud_tpu_catalog",
    "cloud_tpu_lab.src.tpu_versions.version_compare",
    "cloud_tpu_lab.src.xla_sim.fake_hlo",
    "cloud_tpu_lab.src.xla_sim.lowering",
    "cloud_tpu_lab.src.xla_sim.compile_cache",
    "cloud_tpu_lab.src.pjrt_sim.device",
    "cloud_tpu_lab.src.pjrt_sim.executable",
    "cloud_tpu_lab.src.pjrt_sim.runtime",
    "cloud_tpu_lab.src.memory.hbm_sim",
    "cloud_tpu_lab.src.memory.activation_memory",
    "cloud_tpu_lab.src.memory.checkpoint_memory",
    "cloud_tpu_lab.src.sharding.mesh",
    "cloud_tpu_lab.src.sharding.partitioner",
    "cloud_tpu_lab.src.sharding.all_reduce",
    "cloud_tpu_lab.src.input_pipeline.dataloader_sim",
    "cloud_tpu_lab.src.input_pipeline.prefetch_sim",
    "cloud_tpu_lab.src.profiling.profiler_trace",
    "cloud_tpu_lab.src.profiling.trace_analyzer",
    "cloud_tpu_lab.src.profiling.bottleneck_report",
    "cloud_tpu_lab.src.observability.logger",
    "cloud_tpu_lab.src.observability.metrics",
    "cloud_tpu_lab.src.observability.report",
    "cloud_tpu_lab.src.traceability.join_traces",
    "cloud_tpu_lab.src.model_examples.tiny_mlp_jax",
    "cloud_tpu_lab.src.model_examples.tiny_transformer_jax",
]


def test_all_modules_importable() -> None:
    for name in MODULES:
        importlib.import_module(name)
