# train — training observability harness, parallel to benchmarks/.
#
# Mirrors the architecture of benchmarks/ but for training:
#   - train/runner.py — single-experiment runner with per-phase + per-step probes
#   - train/harness.py — CLI entry point
#   - train/registry.yaml — training task definitions
#
# Reuses observe/probe.py wholesale; training-specific probes live in
# observe/training_metrics_probe.py, observe/step_timing_probe.py,
# observe/checkpoint_probe.py.
