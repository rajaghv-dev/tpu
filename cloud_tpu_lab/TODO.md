# TODO

## Done (Phase 1 — vertical slice)
- [x] Repo skeleton
- [x] Trace IDs, workload config, cost estimator
- [x] TPU version catalog
- [x] Fake HLO + lowering + compile cache
- [x] PJRT-sim runtime + device + executable
- [x] HBM + activation + checkpoint memory estimators
- [x] Mesh + partitioner + collectives
- [x] Input pipeline sim
- [x] Profiler trace + analyzer + bottleneck report
- [x] JSONL logger + CSV metrics + Markdown report
- [x] Traceability join
- [x] CPU simulation demo
- [x] Minimal tests

## Phase 2 — docs + Cloud TPU + observability stack
- [ ] docs/00_big_picture.md
- [ ] docs/01_cloud_tpu_versions.md
- [ ] docs/02_cloud_tpu_architecture.md
- [ ] docs/03_xla_pjrt_runtime.md
- [ ] docs/04–06_{jax,pytorch_xla,tensorflow}_on_tpu.md
- [ ] docs/07_sharding_and_spmd.md
- [ ] docs/08_profiling_and_debugging.md
- [ ] docs/09_cost_performance_methodology.md
- [ ] docs/10_cloud_tpu_setup_playbook.md
- [ ] docs/11_cleanup_and_cost_safety.md
- [ ] docs/12_observability_with_grafana_prometheus.md
- [ ] docs/13_oct_metrics_dictionary.md
- [ ] docs/14_benchmarking_playbook.md
- [ ] docs/15_paper_outline.md
- [ ] notebooks/*.ipynb (12 notebooks)
- [ ] gcp/*.sh (create / ssh / install / run / collect / delete)
- [ ] observability/docker-compose.yml + Prometheus + Grafana + Loki + Tempo
- [ ] observability/exporters/cloud_tpu_metrics_exporter.py

## Phase 3 — depth
- [ ] Matrix-unit / systolic-array simulator
- [ ] Pipeline-parallel sim
- [ ] SparseCore mental model (where officially documented)
- [ ] Pod-level scaling-efficiency plots
- [ ] OpenTelemetry trace export
