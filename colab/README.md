# Colab Pro TPU path — free dev iteration (todo.md Tier 3 #9)

Run the same Stage 1 smoke / quick suites on **Colab Pro's TPU runtime** at $0
marginal cost. Reserve the paid GCP v5e-1 path for measurement-grade runs.

## When to use which

| Use case                                                          | Path           |
|-------------------------------------------------------------------|----------------|
| Iterating on harness / probes / registry changes                  | Colab Pro      |
| Smoke-testing a new model before adding to registry               | Colab Pro      |
| Debugging a flaky run                                             | Colab Pro      |
| Measurement-grade numbers for `runs.jsonl` shipped to the dashboard | **GCP v5e-1** |
| Anything quoted in a writeup or compared across runs              | **GCP v5e-1** |

Colab co-tenant variance + opaque hardware (`v2-8` vs `v5e-8` rotation) makes
it unsuitable for the final numbers; use it for the 80% of work that doesn't
need that.

## One-time setup

1. Active Colab Pro subscription (`colab.research.google.com/signup`).
2. (Optional) HuggingFace token for gated models — paste into a cell:
   `import os; os.environ['HF_TOKEN'] = '...'`.

## Open the notebook

Direct link (no local clone required):

```
https://colab.research.google.com/github/rajaghv-dev/tpu/blob/main/colab/tpu_benchmark.ipynb
```

Then **Runtime → Change runtime type → TPU** and run cells top-to-bottom.

Or paste a single bootstrap cell:

```python
!curl -sL https://raw.githubusercontent.com/rajaghv-dev/tpu/main/scripts/colab_bootstrap.sh | bash -s -- smoke default
```

Args are `[suite] [probes_set]`. Defaults: `smoke default`.

## Limitations vs the gcloud path

- **Session timeouts**: Pro is ~12–24 h; idle disconnect ~90 min. Long quick suites
  should be split or use `make quick-tpu` on a real v5e-1.
- **No SSH** → can't run the `CloudMonitoringProbe` (it needs `gcloud auth` against a
  GCP project + Cloud Monitoring API). The `--probes full` set will still register
  Timing+Memory+InputFingerprint+JaxProfiler+HloDump and skip CloudMonitoring.
- **No gcloud, no GCS mount** → models are pulled directly from HuggingFace each
  fresh session (no ADR-006 bucket cache). Set `HF_TOKEN` for gated models.
- **Hardware non-determinism**: TPU type rotates (`v2-8`, `v5e-8`, sometimes others).
  `device=tpu` is recorded but you cannot assert exact accelerator across sessions.
- **Ephemeral filesystem**: everything under `/content/` is wiped between sessions.
  Download the result zip before disconnecting.

## Bridge to local Grafana (ADR-016)

The notebook's "Download results" cell zips `results/` (OTel JSONL + `runs.jsonl`
+ `run_logs/`). On your laptop:

```bash
unzip -o ~/Downloads/tpu_bench_results.zip -d ./results/
./scripts/otel_view.sh        # local Grafana at http://localhost:3000
```

Dashboards work identically to the gcloud path — same OTLP-JSON schema.

## Sample commands

Inside any Colab cell after the bootstrap cells:

```python
# Single model, BF16
!cd /content/tpu && PYTHONPATH=. python benchmarks/harness.py \
    --model bert_base --device tpu --precision bf16 --probes default

# Smoke with full probe set (no CloudMonitoring on Colab)
!cd /content/tpu && PYTHONPATH=. python benchmarks/harness.py \
    --suite smoke --device tpu --probes full

# Dry-run (no model download, prints plan + per-config cost)
!cd /content/tpu && PYTHONPATH=. python benchmarks/harness.py \
    --suite quick --device tpu --dry-run
```
