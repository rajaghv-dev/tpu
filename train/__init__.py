# train — training observability harness, parallel to benchmarks/.
#
# Mirrors the architecture of benchmarks/ but for training:
#   - train/runner.py    — single-experiment runner with per-phase + per-step probes
#   - train/harness.py   — CLI entry point with suites + controllability flags
#   - train/registry.yaml — training task definitions (14 tasks, 3 families)
#
# Supported task families (each with its own JIT'd train_step / eval_step):
#   * sequence-classification — BERT, RoBERTa, DistilBERT
#   * causal-lm               — GPT-2 family, Pythia (RoPE)
#   * image-classification    — ViT, ResNet, Swin
#
# Controllability surfaced via TrainingExperimentConfig + harness CLI:
#   optimizer (adamw | sgd | lion | adafactor) · lr_schedule (linear | cosine | constant)
#   · max_grad_norm · grad_accum_steps · eval_seed · deterministic
#
# Observability is the registered set of probes from observe/. Stage 1.6
# adds four new one-shot probes that capture the full SW/HW stack:
#   - observe/device_info_probe.py   — TPU/GPU/CPU + jax/jaxlib/libtpu identity
#   - observe/xla_compile_probe.py   — compile counters + cache + silent-recompile warns
#   - observe/power_thermal_probe.py — 1 Hz background power/temp/util sampler
#   - observe/determinism_probe.py   — XLA + cuBLAS + cuDNN reproducibility flags
# plus the existing TrainingMetrics, StepTiming, Checkpoint, Timing, Memory,
# and InputFingerprint probes.
