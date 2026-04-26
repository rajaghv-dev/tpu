# 05 – GPT Pre-training

Trains a GPT-2-small-scale (~124 M params) decoder-only transformer from scratch. The goal is to exercise the same architecture family that drives the benchmark's LLM workloads — and to make the prefill-vs-decode distinction concrete.

## Concepts

- Causal self-attention with lower-triangular mask
- `optax.warmup_cosine_decay_schedule` (init=0 → peak=3e-4 over 2000 warmup steps → cosine decay)
- Gradient clipping with `optax.clip_by_global_norm(1.0)` — essential for transformer stability
- `donate_argnums` to donate device buffers in-place (saves a full param copy each step)

## Prefill vs decode — the key LLM insight

This example trains (all-prefill), but the architecture is what the benchmark harness measures at **inference** time, where two very different regimes appear:

| Phase | What it does | Compute pattern | Bottleneck | Winning hardware |
|-------|-------------|----------------|-----------|-----------------|
| **Prefill** | Process the full prompt in one batched forward pass | Dense matmul over `[batch, prompt_len, d_model]` | Compute-bound (TFLOPs) | B200 (2250 TFLOPs), v6e (918), 4090 (330) |
| **Decode** | Generate one token at a time using KV-cache | Tiny matmul `[batch, 1, d_model]` against full KV-cache | Bandwidth-bound (GB/s) | B200 (4000 GB/s), v6e (1640), 4090 (1008) |

**The same model can have different winning hardware for prefill vs decode.** Training is essentially "all prefill" — every token is processed with full causal attention — so the hardware ranking here mirrors the prefill ranking. The decode story doesn't appear until the benchmark harness runs autoregressive generation.

## Hardware context

GPT-2-small at batch=8, seq=1024:

| Accelerator | HBM | BF16 TFLOPs | Bandwidth | Notes |
|-------------|-----|-------------|-----------|-------|
| RTX 3080 | 16 GB | 119 | 760 GB/s | Tight; activations at seq=1024 ≈ 8 GB. Reduce batch if OOM. |
| RTX 4090 | 24 GB | 330 | 1008 GB/s | Comfortable |
| B200 | 192 GB | 2250 | 4000 GB/s | Trivially fits; Tensor Cores fully exercised |
| v5e-1 | 16 GB | 394 | 820 GB/s | Same HBM pressure as 3080; BF16 free |
| v6e-1 | 32 GB | 918 | 1640 GB/s | Headroom for seq=2048 if wanted |

GPT-2-XL (1.5 B params) is the harder version in the registry: highest KV-cache pressure of any benchmark model because it predates GQA. On RTX 3080 at seq=1024, KV-cache alone is ~6 GB — the dramatic prefill-vs-decode winner flip shows up most visibly there.

## Run

```bash
# Quick smoke test (random tokens) — 500 iters, ~3 min on v5e-1
python train.py --iters=500

# On real text corpus — 600k iters, days of compute at GPT-2-small scale
python train.py --corpus=data/corpus.txt --iters=600000
```

## Files

- `model.py` — GPT Flax module: causal MHA, MLP (4× expand + GELU), learned position emb, vocab=50257
- `train.py` — training loop

## Expected output

| Stage | Loss | Interpretation |
|-------|------|----------------|
| Random tokens (smoke test) | ~10.8 | `ln(50257)` — uniform prediction baseline |
| Start of real corpus | ~7.0 | Unigram baseline forming |
| Converged (600k steps) | ~3.5 | GPT-2-small quality |

## What to observe

- **First-compile (25–40 s).** Causal mask + 12-layer transformer is the heaviest compile of any example. After it lands, step time is flat.
- **Gradient clip matters.** Without `clip_by_global_norm(1.0)` the loss can spike → NaN within the first 500 steps as the warmup ramps LR.
- **Donation working.** Remove `donate_argnums=(0,)` and peak HBM roughly doubles — old param buffer isn't reused. On v5e-1 / RTX 3080 this can OOM.
- **BF16 on TPU.** This example uses FP32 by default. Switch params to BF16 on v5e-1 and step time is unchanged. Same switch on a 4090 gives ~2× via Tensor Cores.
- **MXU utilization should be high.** Decoder-only at seq=1024 is essentially pure matmul. MXU/Tensor Core utilization >70% is expected. If you see <40%, suspect host data pipeline stall.

## Connection to the benchmark

The architecture here (decoder-only, MHA, learned positional emb) is the template for the registry's LLM models — GPT-2-XL most directly, and all Qwen / DeepSeek / Phi / Llama models by family.

The benchmark harness will:
1. Load a HF-pretrained checkpoint (e.g., `openai-community/gpt2-xl`)
2. Run **prefill** at prompt lengths 256 / 1024 / 4096 → tokens/sec
3. Run **decode** at generation lengths 256 / 1024 → tokens/sec, watch KV-cache HBM grow
4. Compare across all 5 paths

`observe/compile_controller.py` isolates first-compile from steady-state. `observe/stats.py` enforces n=3 with CV<10%. The prefill-vs-decode-winner-flip is one of the key Stage 2 deliverables.
