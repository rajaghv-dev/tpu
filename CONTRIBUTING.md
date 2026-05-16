# Contributing

Thank you for your interest in contributing to the TPU x GPU Inference Benchmark.

## Before you start

Read AGENTS.md for the full agent guide. Read MEMORY.md for project state.

## Development setup

```bash
git clone https://github.com/rajaghv-dev/tpu && cd tpu
python3 -m venv .venv && source .venv/bin/activate
pip install jax flax optax orbax-checkpoint transformers datasets pyyaml rich tqdm \
            opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc \
            numpy pytest pytest-cov psutil
```

## Running tests

```bash
JAX_PLATFORMS=cpu python3 -m pytest tests/ -q
```

All tests must pass without GPU or TPU.

## Code style

- Python 3.10+ with type annotations on public functions
- Max line length: 100
- Run `python3 -m flake8 benchmarks/ observe/ tests/ train/ --max-line-length 100` before submitting
- No comments unless the WHY is non-obvious
- No unnecessary abstractions

## Adding a new probe

1. Create `observe/<probe_name>_probe.py` (exception: `observe/input_fingerprint.py`)
2. Implement the `Probe` ABC from `observe/probe.py`
3. Set `self.name = "<probe_name>"` — this becomes the output filename
4. Add at least one test in `tests/test_app_probes.py` or a new file
5. Add the probe to `observe/README.md` probe table
6. Add the probe to the main `README.md` probes section
7. Add the probe to `MEMORY.md` probe quick reference

## Adding a new model

1. Add an entry to `models/registry.yaml` following the existing schema
2. Ensure the model is publicly available on HuggingFace (or mark `gated: true`)
3. Test with `python3 -m benchmarks.harness --model <model_id> --device cpu --dry-run`

## Pull request checklist

- [ ] Tests pass: `python3 -m pytest tests/ -q`
- [ ] Lint passes: `python3 -m flake8 benchmarks/ observe/ tests/ train/ --max-line-length 100`
- [ ] Docs updated for any behavior changes
- [ ] No secrets committed
- [ ] New probes added to observe/README.md and MEMORY.md

## Commit message style

```
type(scope): short description

Types: feat, fix, docs, refactor, test, ci, chore
Scope: observe, benchmarks, train, scripts, docs, ci

Examples:
  fix(observe): lazy-import opentelemetry in get_tracer/get_meter
  feat(observe): add PowerThermalProbe
  docs(readme): update test count to 272
  ci(smoke): add explicit permissions block
```

## Questions

Open an issue or check QUESTIONS.md for known open research questions.
