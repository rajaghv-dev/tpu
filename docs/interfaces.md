# Interfaces

Generated: 2026-05-16 — confirmed from source inspection.

## CLI Interfaces

### benchmarks/harness.py

```
python3 -m benchmarks.harness [options]
  --suite    {smoke, quick}         Pre-defined suite
  --model    MODEL_ID               Single model (overrides suite)
  --device   {tpu, gpu, cpu, ...}  Target device
  --precision {fp32, bf16}          Default: bf16
  --framework {jax}                 Default: jax (Stage 1 only)
  --output   PATH                   Default: results/runs.jsonl
  --registry PATH                   Override registry.yaml path
  --dry-run                         Plan without model downloads
  --probes   {none, default, full}  Probe set
```

### train/harness.py

```
python3 -m train.harness [options]
  --suite        {smoke, quick}     Pre-defined suite
  --task         TASK_ID            Single task (overrides suite)
  --device       {tpu, gpu, cpu}
  --framework    {jax}
  --precision    {fp32, bf16}       Default: bf16
  --steps        N                  Override n_steps
  --eval-steps   N
  --save-checkpoint
  --optimizer    {adamw, sgd, lion, adafactor}
  --lr-schedule  {linear, cosine, constant}
  --max-grad-norm FLOAT
  --grad-accum   N
  --eval-seed    N
  --deterministic
  --probes       {none, default, full}
  --output       PATH               Default: results/training_runs.jsonl
  --registry     PATH
  --dry-run
```

## Python APIs

### benchmarks/runner.py — run_experiment

```python
def run_experiment(
    cfg: ExperimentConfig,
    results_dir: str = "results/run_logs",
    _loader: Callable | None = None,   # test injection point
) -> dict
```
Returns a dict with: run_id, timestamp, model, device, precision, latency_p50/p95/p99_ms, throughput_mean_samples_sec, first_compile_s, subsequent_compile_s, cost_per_1k_samples_usd, flags: list[str], error_category (if failed).

Raises `BenchmarkError` on phase failure.

### train/runner.py — run_training

```python
def run_training(
    cfg: TrainingExperimentConfig,
    results_dir: str = "results/run_logs",
) -> dict
```
Returns dict with: run_id, final_loss, mean_step_s, eval_loss, eval_accuracy, lineage fields.

### observe/probe.py — Probe ABC

```python
class Probe:
    name: str  # used as output filename stem

    # Inference hooks
    def before_run(self, run_id: str, config: Any, log_dir: Path) -> None: ...
    def before_phase(self, phase_name: str) -> None: ...
    def after_phase(self, phase_name: str, duration_s: float) -> None: ...
    def on_error(self, phase_name: str, exc: BaseException) -> None: ...
    def after_run(self, run_id: str, result: dict | None) -> None: ...
    def write_log(self) -> dict | None: ...  # return dict → written as <name>.json

    # Training hooks
    def before_step(self, step: int) -> None: ...
    def after_step(self, step: int, metrics: dict) -> None: ...
    def record_metric(self, name: str, value: Any, step: int | None) -> None: ...
```

### observe/probe.py — registry

```python
def register_probe(probe: Probe) -> None         # append to active list
def set_active_probes(probes: list[Probe]) -> None  # replace active list (use in tests)
def get_active_probes() -> list[Probe]
def clear_probes() -> None

# Fanout helpers (called by runner)
def fanout_before_run(run_id, config, log_dir) -> None
def fanout_after_run(run_id, result_or_none, log_dir) -> None
def fanout_before_phase(phase_name) -> None
def fanout_after_phase(phase_name, duration_s) -> None
def fanout_on_error(phase_name, exc) -> None
def fanout_before_step(step) -> None
def fanout_after_step(step, metrics) -> None
def fanout_record_metric(name, value, step=None) -> None
```

### observe/stats.py

```python
def compute_timing_stats(timings_ms: Sequence[float]) -> TimingStats
# Returns: p50, p95, p99, mean, std, cv_pct, n_outliers_removed, high_variance: bool

def throughput_stats(timings_ms: Sequence[float], batch_size: int) -> ThroughputStats
# Returns: mean_samples_sec, std_samples_sec
```

### observe/lineage.py

```python
def get_git_sha() -> str
def get_package_version(name: str) -> str
def build_environment_hash(...) -> str
def build_lineage(cfg, run_id) -> dict
```

## Configuration Files

### models/registry.yaml

Each entry:
```yaml
bert_base:
  hf_id: bert-base-uncased
  task: sequence-classification
  domain: nlp_encoder
  architecture_family: transformer_encoder
  attention_variant: mha
  positional_encoding: absolute
  is_moe: false
  total_params_M: 110
  active_params_M: 110
  input_type: text
  seq_len: 128
  vocab_size: 30522
  gated: false
  risk: low
```

### train/registry.yaml

Each entry:
```yaml
bert_finetune:
  hf_id: bert-base-uncased
  task: sequence-classification
  domain: nlp_encoder
  optimizer: adamw
  lr_schedule: linear
  num_steps: 100
  num_eval_steps: 10
  batch_size: 16
  grad_accum_steps: 1
  max_grad_norm: 1.0
```

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `TPU_BENCH_OTEL` | No | off | OTel mode: off \| otlp \| file |
| `TPU_BENCH_OTEL_ENDPOINT` | No | localhost:4317 | OTLP gRPC endpoint |
| `TPU_BENCH_OTEL_DIR` | No | results/otel/ | OTLP-JSON output directory |
| `JAX_PLATFORMS` | No | auto | Override: cpu \| tpu \| gpu |
| `HF_TOKEN` | For gated models | — | HuggingFace auth token |
| `GCP_PROJECT` | For GCP scripts | — | GCP project ID |
| `XLA_FLAGS` | Set by HloDumpProbe | — | XLA compilation flags |
| `WHEEL_CACHE_URL` | No | — | GCS URL for pip wheel cache |
| `HF_MODEL_CACHE_URL` | No | — | GCS URL for HF model cache |

## Results Schema (runs.jsonl)

Key fields per JSONL row:
```json
{
  "run_id": "uuid",
  "timestamp": "ISO-8601",
  "git_sha": "hex",
  "model": "bert_base",
  "device": "tpu",
  "framework": "jax",
  "precision": "bf16",
  "first_compile_s": 5.2,
  "subsequent_compile_s": 0.08,
  "latency_p50_ms": 0.64,
  "latency_p95_ms": 0.66,
  "latency_p99_ms": 0.67,
  "latency_cv_pct": 1.31,
  "throughput_mean_samples_sec": 5261,
  "cost_per_1k_samples_usd": 1.9e-5,
  "flags": [],
  "error_category": null
}
```
