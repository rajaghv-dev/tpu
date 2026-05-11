# Run report — bert_base · bf16 · tpu (success)

**Status.** ✓ success.  **Timestamp.** 2026-05-06T17:48:41.411926Z.  **Run id.** `6f049c5d-d1fb-4f1b-aa9a-998c34d2e894`

## Identity

| Field | Value |
|---|---|
| run_id | 6f049c5d-d1fb-4f1b-aa9a-998c34d2e894 |
| timestamp | 2026-05-06T17:48:41.411926Z |
| device | tpu |
| framework | jax |
| path | 1 |

## Model

| Field | Value |
|---|---|
| model | bert_base |
| domain | nlp_encoder |
| architecture_family | transformer_encoder |
| attention_variant | mha |
| positional_encoding | absolute |
| is_moe | false |
| total_params_M | 110 |
| active_params_M | 110 |

## Variant

| Field | Value |
|---|---|
| precision | bf16 |
| pruning | dense |
| compiled | true |
| compile_mode | default |
| inference_mode | combined |
| batch_size | 1 |
| batch_size_throughput | 64 |
| seq_len | 128 |

## Compile

| Field | Value |
|---|---|
| first_compile_s | 5.203 |
| subsequent_compile_s | 0.0008 |
| compile_cache_hit | false |

## Latency

| Field | Value |
|---|---|
| latency_mean_ms | 0.6424 |
| latency_std_ms | 0.0084 |
| latency_cv_pct | 1.31 |
| latency_p50_ms | 0.6407 |
| latency_p95_ms | 0.6591 |
| latency_p99_ms | 0.6634 |

## Throughput

| Field | Value |
|---|---|
| throughput_mean_samples_sec | 5261 |
| throughput_std_samples_sec | 7.8 |

## Quality

| Field | Value |
|---|---|
| flags | — |

## Cost

| Field | Value |
|---|---|
| device_cost_usd_per_hr | 0.36 |
| experiment_cost_usd | 0.00056 |
| cost_per_1k_samples_usd | 1.901e-05 |

## Lineage

| Field | Value |
|---|---|
| git_sha | unknown |
| jax_version | 0.6.2 |
| torch_version | not_installed |
| transformers_version | 4.44.2 |
| hf_model_revision | 86b5e0934494bd15c9632b12f734a8a67f723594 |
| input_seed | 42 |
| n_independent_runs | 3 |
| environment_hash | d6419ffc16ace495 |

## Auxiliary log files

- `lineage.json` (290 bytes)