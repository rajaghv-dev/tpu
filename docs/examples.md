# Examples

Generated: 2026-05-16.

## Overview

Examples are split across three locations:
1. Root training examples (`01_hello_tpu/` through `08_multi_host/`) — require JAX/TPU install
2. `cloud_tpu_lab/examples/` — CPU simulation, no TPU needed
3. `colab/tpu_benchmark.ipynb` — Colab Pro notebook

---

## Example 1: CPU simulation demo (no TPU needed)

### Purpose
End-to-end vertical slice of the TPU simulation stack — fake HLO → fake XLA compile → fake PJRT runtime → HBM sim → sharding sim → profiler → report.

### Command
```bash
cd cloud_tpu_lab
python3 examples/run_cpu_simulation_demo.py
```

### Expected output
Artifacts in `cloud_tpu_lab/artifacts/`:
- `logs/run_<trace_id>.jsonl` — all events with correlation IDs
- `metrics/run_<trace_id>.csv` — Prometheus-style metric stream
- `traces/run_<trace_id>.json` — Chrome-trace JSON
- `reports/run_<trace_id>.md` — human-readable report

### Variants
```bash
python3 examples/run_cpu_simulation_demo.py --show-versions   # compare TPU versions
python3 examples/run_cpu_simulation_demo.py --tpu-version v5p --chip-count 4
python3 examples/run_cpu_simulation_demo.py --tpu-version v6e --batch-size 64
python3 examples/run_cpu_simulation_demo.py --hidden-size 1024 --num-layers 12
```

### Current status: Runnable (no TPU required)

---

## Example 2: Inference harness dry-run (no downloads)

### Purpose
Validates the harness config, suite resolution, and registry loading without downloading models.

### Command
```bash
make dry-run
# Equivalent: JAX_PLATFORMS=cpu python3 -m benchmarks.harness --suite quick --device cpu --dry-run
```

### Expected output
Printed plan: models to run, precisions, estimated wall time and cost.

### Current status: Runnable

---

## Example 3: Smoke suite on CPU

### Purpose
Runs 1 model (BERT-base) end-to-end on CPU-JAX with real inference (no GPU/TPU).

### Command
```bash
make smoke-cpu
# Equivalent: JAX_PLATFORMS=cpu python3 -m benchmarks.harness --suite smoke --device cpu
```

### Expected output
- Result appended to `results/runs.jsonl`
- Report in `results/run_logs/<run_id>/`
- Timing in `results/run_logs/<run_id>/timing.json` (if TimingProbe registered)

### Current status: Runnable (requires jax and transformers install)

---

## Example 4: Training smoke suite

### Purpose
Runs bert_finetune for 10 steps with training probes. Tests the training observability harness.

### Command
```bash
python3 -m train.harness --suite smoke --device cpu
```

### Expected output
- Result in `results/training_runs.jsonl`
- Per-step metrics in `results/run_logs/<run_id>/training_metrics.json`

### Current status: Runnable (requires jax, flax, optax install)

---

## Example 5: Colab Pro notebook

### Purpose
Run smoke/quick suites on Colab's free TPU (v2-8 or v3-8).

### Command
Open: https://colab.research.google.com/github/rajaghv-dev/tpu/blob/main/colab/tpu_benchmark.ipynb

Or bootstrap from any Colab cell:
```bash
!curl -sL https://raw.githubusercontent.com/rajaghv-dev/tpu/main/scripts/colab_bootstrap.sh | bash -s -- smoke default
```

### Current status: Present — requires Colab Pro TPU runtime

---

## Example 6: Training examples 01–08

### Purpose
Standalone JAX/Flax training loops demonstrating TPU use cases.

| Dir | Topic |
|---|---|
| 01_hello_tpu/ | Hello world — device detection |
| 02_mnist_classification/ | MNIST on JAX |
| 03_resnet_imagenet/ | ResNet on ImageNet |
| 04_bert_finetuning/ | BERT fine-tuning |
| 05_gpt_pretraining/ | GPT pre-training |
| 06_data_pipeline/ | tf.data pipeline |
| 07_custom_training_loop/ | Custom training loop |
| 08_multi_host/ | Multi-host training |

### Current status: Present — each requires JAX + TPU or CPU install. Not CI-validated.

---

## Example 7: Jupyter notebooks (cloud_tpu_lab)

12 run-along notebooks in `cloud_tpu_lab/notebooks/`:
1. Cloud TPU big picture
2. Matrix unit simulation
3. XLA compilation simulation
4. JAX: CPU to TPU-ready
5. PyTorch XLA: CPU to TPU-ready
6. TensorFlow: CPU to TPU-ready
7. Sharding simulation
8. HBM bandwidth simulation
9. Pod all-reduce simulation
10. Input pipeline bottleneck
11. Profiler trace analysis
12. Cost/performance analysis

### Current status: Present — execution status not CI-validated

---

## Gaps

| Gap | Notes |
|---|---|
| Examples 01–08 not CI-tested | Would require JAX install in CI |
| Notebooks not CI-tested | Jupyter execution not in CI |
| No example for probe registration | Should add to README |
| No example for OTel local Grafana full flow | End-to-end OTel example missing |
