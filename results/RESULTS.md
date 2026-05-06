# Results — TPU × GPU inference benchmark

Auto-generated from `results/runs.jsonl` + `results/run_logs/` by
[`scripts/render_results.py`](../scripts/render_results.py). Re-run after each benchmark session.

## Summary

- **Total rows.** 2
- **Succeeded.** 1
- **Failed.**    1

## Successful runs

| model | device | precision | first_compile_s | p50 ms | p95 ms | p99 ms | CV % | tput sps | cost/1k | flags | probes attached | report |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| bert_base | tpu | bf16 | 5.203 | 0.6407 | 0.6591 | 0.6634 | 1.31 | 5261 | 1.901e-05 | — | — | [6f049c5d…](run_logs/6f049c5d-d1fb-4f1b-aa9a-998c34d2e894/REPORT.md) |

## Failed runs

| model | device | precision | phase | category | exception | report |
|---|---|---|---|---|---|---|
| bert_base | tpu | bf16 | `model_load` | `other` | `ImportError: cannot import name 'FlaxAutoModelForSequenceClassification' from 'transformers' ` | — |

## Reproducing

Each row's `lineage` field captures the git SHA, JAX/transformers
versions, HF model revision, and input seed needed to reproduce
identically. See `scripts/run_all.sh --suite smoke` for the
orchestration that produces these rows.
