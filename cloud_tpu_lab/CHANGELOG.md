# Changelog

## 0.1.0 — initial vertical slice

- Repo skeleton (docs / notebooks / src / gcp / examples / tests / artifacts / observability).
- `src/common`: trace IDs, workload config, simulation config, cost estimator.
- `src/tpu_versions`: catalog for v4 / v5e / v5p / v6e (Trillium) with source-marker fields.
- `src/xla_sim`: fake HLO IR + lowering rules (Linear / Conv / Attention / LayerNorm) + compile cache.
- `src/pjrt_sim`: fake device + executable + runtime with roofline op timing.
- `src/memory`: HBM simulator + activation-memory + checkpoint-memory estimators.
- `src/sharding`: mesh + partitioner + ring-all-reduce / all-gather / reduce-scatter cost model.
- `src/input_pipeline`: dataloader + prefetch-depth sim.
- `src/profiling`: Chrome-trace profiler + analyzer + bottleneck report.
- `src/observability`: JSONL logger + CSV metrics stream + Markdown run report.
- `src/traceability`: join-by-`trace_id` across logs / metrics / traces.
- `examples/run_cpu_simulation_demo.py`: end-to-end vertical slice (no TPU needed).
- `tests/test_imports.py`, `tests/test_cpu_simulation_smoke.py`,
  `tests/test_trace_ids.py`, `tests/test_xla_lowering_smoke.py`,
  `tests/test_sharding_simulation.py`, `tests/test_hbm_simulation.py`,
  `tests/test_cost_estimator.py`, `tests/test_report_generation.py`.
- `README.md`, `Makefile`, `pyproject.toml`, `requirements.txt`.
