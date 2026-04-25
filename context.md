# Project Context — TPU vs GPU Inference Benchmark

Complete accumulated context for this project. Updated as decisions are made.
Last updated: 2026-04-25

---

## Project Goal

Build a rigorous, reproducible inference benchmark comparing Google TPU and NVIDIA GPU across:
- Multiple model families (vision, NLP, audio, multimodal, novel architectures)
- Multiple frameworks (JAX/Flax, PyTorch, torch_xla)
- Multiple precision/quantization variants
- Multiple compiler strategies
- Microarchitecture and dataflow understanding via roofline analysis

Results sync to GitHub automatically. Dashboard renders comparison charts statically (no server).

---

## User Hardware Inventory

### Local GPUs (zero cost to run)
| Card | VRAM | Memory BW | Peak BF16 | Notes |
|------|------|-----------|-----------|-------|
| RTX 3080 | 16 GB GDDR6X | 760 GB/s | 119 TFLOPs | Ampere, 2:4 sparsity support |
| RTX 4090 | 24 GB GDDR6X | 1008 GB/s | 330 TFLOPs | Ada Lovelace, FP8 support |
| DGX A100 (1 card) | 80 GB HBM2e | 2000 GB/s | 312 TFLOPs | SXM4, single card use |

### Cloud TPU (personal account: rajaghv@gmail.com)
Primary: **v5e-1 preemptible** ($0.36/hr) for 3080 comparison
Secondary: **v6e-1 preemptible** (~$0.75/hr) for 4090/A100 comparison

### Colab
- Colab Pro — v2-8 or v3-8 TPU runtime; A100/V100 GPU runtime
- Good for smoke/quick suites; free after $9.99/mo subscription

### External Services
- HuggingFace account: available (gated model access: Gemma, PaliGemma, LLaMA)
- HF Inference API: available (5th execution path — zero infra, compare managed serving)
- GCS bucket: model weight cache (download once, reuse across VM restarts; ~$1.60/month for 80GB)

---

## TPU Variants Accessible via Cloud VMs

| TPU | Min slice | HBM/chip | Mem BW/chip | Peak BF16/chip | On-demand | Preemptible | Single-chip VM |
|-----|-----------|----------|-------------|---------------|-----------|-------------|----------------|
| v2 | 8 chips | 8 GB | 600 GB/s | 45 TFLOPs | $4.50/hr | $1.35/hr | No |
| v3 | 8 chips | 16 GB | 900 GB/s | 123 TFLOPs | $8.00/hr | $2.40/hr | No |
| v4 | 8 chips | 32 GB | 1200 GB/s | 275 TFLOPs | $13.00/hr | $3.90/hr | No |
| **v5e** | **1 chip** | **16 GB** | **820 GB/s** | **394 TFLOPs** | **$1.20/hr** | **$0.36/hr** | **Yes — v5e-1** |
| v5p | 8 chips | 95 GB | 2765 GB/s | 459 TFLOPs | ~$32/hr | ~$9.60/hr | No |
| **v6e** | **1 chip** | **32 GB** | **1640 GB/s** | **918 TFLOPs** | **~$2.50/hr** | **~$0.75/hr** | **Yes — v6e-1** |

### Max Model Size Per Chip (Inference Only)
| Chip | HBM | BF16 comfortable | BF16 tight (bs=1) | INT8 |
|------|-----|-----------------|-------------------|------|
| v2 | 8 GB | 1.5B | 2.5B | 5B |
| v3 | 16 GB | 3B | 5B | 10B |
| v5e | 16 GB | 3B | 5B | 10B |
| v4 | 32 GB | 7B | 10B | 20B |
| v6e | 32 GB | 7B | 10B | 20B |
| v5p | 95 GB | 30B | 38B | 70B |

---

## The 4 (+1) Execution Paths

```
                   Same Model Weights (HuggingFace pretrained)
                              |
              ┌───────────────┼────────────────┐
           JAX/Flax        JAX/Flax         PyTorch
              │                │                │
         ┌────▼────┐     ┌─────▼─────┐   ┌─────┴──────┐
         │ Path 1  │     │  Path 2   │   │  Path 3    │ Path 4
         │JAX+TPU  │     │ JAX+GPU   │   │ PyTorch+GPU│ PT+torch_xla+TPU
         │ (XLA)   │     │  (CUDA)   │   │  (CUDA)    │ (XLA)
         └─────────┘     └───────────┘   └────────────┘
              │                                 │
         Path 5: HF Inference API (managed, serverless, no infra)
```

| Comparison pair | What it isolates |
|----------------|-----------------|
| Path 1 vs 2 | Pure hardware: TPU vs GPU, JAX constant |
| Path 2 vs 3 | Pure framework: JAX vs PyTorch, GPU constant |
| Path 1 vs 4 | Pure framework on TPU: JAX vs torch_xla |
| Path 3 vs 4 | Compiler on TPU: native CUDA vs XLA (PyTorch API constant) |
| Path 1 vs 3 | Real-world: production TPU vs production GPU |
| Path 5 vs all | Managed inference overhead vs self-hosted |

---

## HuggingFace Integration

### Model Loading
- All models loaded via `transformers` (FlaxAutoModel for JAX, AutoModel for PyTorch)
- Same pretrained weights across all 4 self-hosted paths — fair comparison
- `HF_TOKEN` env var for gated models (Gemma, PaliGemma, CodeGemma)
- `HF_HOME=/tmp/hf_cache` pointing to GCS-backed directory

### GCS Cache Strategy
```
One-time setup (local laptop):
  huggingface-cli download <model> --cache-dir ./hf_cache/
  gsutil -m cp -r ./hf_cache/ gs://rajaghv-bench-cache/models/

Every VM startup (~30s):
  gsutil -m cp -r gs://rajaghv-bench-cache/models/ /tmp/hf_cache/
  export HF_HOME=/tmp/hf_cache

GCS cost: ~80GB × $0.02/GB/month = $1.60/month
```

### HF Inference API (Path 5)
- Free tier: 1000 requests/day, rate-limited
- PRO tier ($9/month): 10× throughput, no rate limit on most models
- Serverless Inference: auto-scales, no VM management
- Dedicated Endpoints: deploy model on specific hardware (GPU/CPU), reserved capacity
- Use case: compare managed serving latency vs self-hosted TPU/GPU

---

## Full Model Registry

### Vision — Encoders (Image Classification / Feature Extraction)

| Model | Params | Architecture | HF ID | Key characteristic |
|-------|--------|-------------|-------|-------------------|
| ResNet-50 | 86M | Conv+BN+Residual | `microsoft/resnet-50` | Conv-heavy baseline |
| ConvNeXt-XL | 350M | Modern depthwise conv | `facebook/convnext-xlarge-224-22k` | Pure conv, no attn |
| EfficientNet-B7 | 66M | Depthwise+SE | `google/efficientnet-b7` | Kills MXU/Tensor Core |
| ViT-B/16 | 86M | Pure self-attention | `google/vit-base-patch16-224` | Pure attention |
| ViT-L/16 | 307M | Pure self-attention | `google/vit-large-patch16-224` | Larger ViT |
| DINOv2-L | 307M | ViT + self-sup | `facebook/dinov2-large` | SSL features |
| SigLIP-B/16 | 400M | ViT + sigmoid loss | `google/siglip-base-patch16-224` | Google, CLIP replacement |
| SAM-L | 312M | ViT + mask decoder | `facebook/sam-vit-large` | Segment anything |
| EVA-02-L | 307M | Improved ViT | `Yuxin-CV/EVA-02` | CLIP-pretrained |
| EfficientViT-L3 | 246M | Linear attn+conv | `mit-han-lab/efficientvit-l3` | Efficient hybrid |

### Vision — Object Detection

| Model | Params | Architecture | HF ID | Key characteristic |
|-------|--------|-------------|-------|-------------------|
| DETR-ResNet50 | 41M | Transformer decoder | `facebook/detr-resnet-50` | End-to-end transformer det. |
| RT-DETR-L | 32M | RT detection transformer | `PekingU/rtdetr_r50vd` | Real-time DETR |

### Vision — Generative (Diffusion)

| Model | Params | Architecture | HF ID | Key characteristic |
|-------|--------|-------------|-------|-------------------|
| SD-UNet | 860M | U-Net + cross-attn | `runwayml/stable-diffusion-v1-5` (UNet only) | Conv+attention hybrid |
| DiT-XL/2 | 675M | Pure transformer | `facebook/DiT-XL-2-256` | Pure matmul diffusion — great TPU fit |

### NLP — Encoders

| Model | Params | Architecture | HF ID | Key characteristic |
|-------|--------|-------------|-------|-------------------|
| BERT-base | 110M | Transformer enc | `bert-base-uncased` | Historical baseline |
| RoBERTa-large | 355M | BERT variant | `roberta-large` | Better pretraining |
| DeBERTa-v3-large | 400M | Disentangled attn | `microsoft/deberta-v3-large` | SOTA encoder |
| ModernBERT-base | 149M | Flash attn + RoPE | `answerdotai/ModernBERT-base` | Dec 2024, 8192 ctx |
| ModernBERT-large | 395M | Flash attn + RoPE | `answerdotai/ModernBERT-large` | Latest encoder SOTA |
| BGE-large-en | 335M | BERT-based | `BAAI/bge-large-en-v1.5` | Dense retrieval/RAG |
| E5-large-v2 | 335M | BERT-based | `intfloat/e5-large-v2` | Universal embedding |

### NLP — Decoders (Autoregressive LLMs)

| Model | Params | Architecture | HF ID | Gated | Key characteristic |
|-------|--------|-------------|-------|-------|--------------------|
| GPT-2 XL | 1.5B | Original transformer | `gpt2-xl` | No | Historical baseline |
| TinyLlama-1.1B | 1.1B | LLaMA arch, GQA | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | No | Efficient small LLM |
| SmolLM2-1.7B | 1.7B | LLaMA arch | `HuggingFaceTB/SmolLM2-1.7B` | No | HF flagship small model |
| OLMo-2-1B | 1.2B | Fully open | `allenai/OLMo-2-1124-7B` | No | Reproducible research |
| Gemma-2B | 2B | MQA + RoPE | `google/gemma-2b` | Yes | Google — TPU co-designed |
| Gemma-2-2B | 2.6B | GQA + sliding window | `google/gemma-2-2b` | Yes | Newer Gemma |
| RecurrentGemma-2B | 2B | RNN-hybrid (RGLRU) | `google/recurrentgemma-2b` | Yes | Google RNN vs transformer |
| Phi-2 | 2.7B | Transformer | `microsoft/phi-2` | No | Punches above weight |
| Phi-3-mini-4k | 3.8B | LLaMA arch | `microsoft/Phi-3-mini-4k-instruct` | No | Near 4B ceiling |
| Qwen2.5-3B | 3B | GQA + RoPE | `Qwen/Qwen2.5-3B` | No | Strong multilingual |
| StableLM-3B | 3B | LLaMA arch | `stabilityai/stablelm-3b-4e1t` | No | Open 3B |
| StableLM-2-1.6B | 1.6B | LLaMA arch | `stabilityai/stablelm-2-1_6b` | No | Small baseline |

### NLP — Code Models

| Model | Params | HF ID | Key characteristic |
|-------|--------|-------|-------------------|
| StarCoder2-3B | 3B | `bigcode/starcoder2-3b` | Code, 16k context |
| CodeGemma-2B | 2B | `google/codegemma-2b` | Google code model |

### NLP — Novel Architectures (Non-Transformer)

| Model | Params | Architecture | HF ID | Key characteristic |
|-------|--------|-------------|-------|-------------------|
| Mamba-2.8B | 2.8B | SSM (selective scan) | `state-spaces/mamba-2.8b` | Custom CUDA kernel; bad for XLA |
| Mamba2-2.7B | 2.7B | Structured SSM | `state-spaces/mamba2-2.7b` | Improved SSM |
| RWKV-4-3B | 3B | Linear RNN | `RWKV/rwkv-4-world-3b` | Linear transformer alternative |

### Audio

| Model | Params | Architecture | HF ID | Key characteristic |
|-------|--------|-------------|-------|-------------------|
| Whisper-base | 74M | Conv+transformer enc-dec | `openai/whisper-base` | Small, fast |
| Whisper-medium | 307M | Same | `openai/whisper-medium` | Balanced |
| Whisper-large-v3 | 1.5B | Same | `openai/whisper-large-v3` | Best quality |
| wav2vec2-large | 317M | Conv + transformer | `facebook/wav2vec2-large` | Conv feature ext |
| HuBERT-large | 316M | Conv + transformer | `facebook/hubert-large-ls960-ft` | SSL audio |
| SeamlessM4T-medium | 1.2B | Multi-task | `facebook/seamless-m4t-medium` | Speech translation |
| MMS-1B | 1B | wav2vec2-based | `facebook/mms-1b-all` | Multilingual |
| EnCodec-24kHz | 44M | Conv + LSTM | `facebook/encodec_24khz` | Neural audio codec |

### Multimodal — Vision-Language (≤4B)

| Model | Params | Architecture | HF ID | Gated | Key characteristic |
|-------|--------|-------------|-------|-------|--------------------|
| CLIP ViT-L/14 | 428M | Dual encoder | `openai/clip-vit-large-patch14` | No | Contrastive baseline |
| SigLIP-SO400M | 400M | Sigmoid CLIP | `google/siglip-so400m-patch14-384` | No | Google CLIP replacement |
| moondream2 | 1.86B | SigLIP + Phi | `vikhyatk/moondream2` | No | Tiny capable VLM |
| SmolVLM-2B | 2B | SigLIP + SmolLM | `HuggingFaceTB/SmolVLM-Instruct` | No | Very new (Dec 2024) |
| PaliGemma-3B | 3B | SigLIP + Gemma | `google/paligemma-3b-pt-224` | Yes | Google VLM — TPU co-designed |
| LLaVA-Phi3 | ~4B | CLIP + Phi-3-mini | `xtuner/llava-phi-3-mini-hf` | No | Phi-3 backbone |
| Idefics2-8B | 8B | Mistral + vision | — | No | Too big for v5e BF16 |

---

## Experiment Dimensions (Variants)

### Precision
| Format | TPU support | GPU support | Notes |
|--------|-------------|-------------|-------|
| FP32 | Yes | Yes | Baseline |
| BF16 | Native (same speed as FP32) | Via autocast (2× speedup) | Key differentiator |
| FP16 | Limited | Native | Different numerics from BF16 |
| INT8 | v5+ native | Tensor Cores (Ampere+) | Post-training quantization |
| INT4 | Limited | Tensor Cores (bitsandbytes) | GPU advantage |
| FP8 | v5e/v6e | 4090 (Ada), H100 | Very new |

### Compilation Strategy
| Strategy | JAX path | PyTorch path | What you learn |
|----------|----------|-------------|---------------|
| Eager | Disabled jit | Default PyTorch | Baseline, overhead visible |
| Compiled (default) | `jax.jit` | `torch.compile(mode='default')` | Standard production |
| Max-optimised | `jax.jit` (always max) | `torch.compile(mode='max-autotune')` | Peak performance ceiling |
| CUDA Graphs | N/A | `torch.cuda.make_graphed_callables` | GPU-specific kernel replay |
| XLA persistent cache | `JAX_COMPILATION_CACHE_DIR` | N/A | XLA reuse across runs |

### Sparsity / Pruning
| Variant | TPU behaviour | GPU behaviour | Key insight |
|---------|--------------|---------------|-------------|
| Dense (baseline) | Optimal | Optimal | Reference |
| Unstructured 50% | Slower (dense matmul still runs) | Slightly slower | Sparsity ≠ speed without hardware support |
| Unstructured 90% | Slower | Marginally faster | Same story, more dramatic |
| Structured (channel) | Linear speedup | Linear speedup | Both handle this |
| 2:4 structured | No hardware support | 2× Tensor Core speedup | NVIDIA-specific hardware win |

### Attention Variants (per-model where applicable)
- MHA (Multi-Head Attention) — standard
- MQA (Multi-Query) — fewer KV heads, less memory BW
- GQA (Grouped Query) — between MHA and MQA
- Sliding Window (Mistral/Gemma-2) — O(n) instead of O(n²)
- Flash Attention 2 (GPU) vs Splash Attention (TPU) — memory-efficient attention impl

### Serving Modes (LLMs only)
- Prefill only (process prompt, no generation)
- Single-token decode (KV-cache warm, measure token/sec)
- Batch decode (throughput mode)

---

## Per-Experiment Protocol (1–3 min)

```
Step 1 — Compile/trace      1 forward pass    Record compile_time_s separately
Step 2 — Warmup            20 forward passes  Discard; kernels/cache stabilise
Step 3 — Latency (bs=1)    50 forward passes  Record p50, p95, p99 ms
Step 4 — Throughput        50 forward passes  Max batch that fits; record samples/sec
Step 5 — Profile            5 forward passes  Full trace (jax.profiler / torch.profiler)
Step 6 — Memory sweep       1 pass per bs     bs=1,2,4,8,... until OOM; record ceiling
```

---

## Suite Definitions

| Suite | Models | Variants | Experiments | Estimated time | TPU v5e cost |
|-------|--------|----------|-------------|----------------|--------------|
| `smoke` | 1 (BERT-base) | FP32+BF16, bs=1 | 4 | ~8 min | $0.05 |
| `quick` | 6 (1 per domain) | BF16 only, bs=1+max | 24 | ~50 min | $0.30 |
| `domain` | All in one domain | BF16+FP32 | ~30 | ~60 min | $0.36 |
| `arch` | Novel architectures | BF16+FP32 | ~20 | ~40 min | $0.24 |
| `full` | All 45 models | All variants | ~200 | ~7 hrs | $2.52 |

Full suite ($2.52) can be run weekly for ~$10/month.

---

## Results Schema (JSONL — one line per experiment)

```json
{
  "timestamp": "ISO8601",
  "run_id": "uuid",
  "device": "tpu_v5e | tpu_v6e | rtx3080 | rtx4090 | a100_dgx",
  "framework": "jax | pytorch | torch_xla | hf_api",
  "path": 1,
  "model": "bert_base",
  "domain": "nlp_encoder",
  "architecture": "transformer_encoder",
  "params_M": 110,
  "precision": "bf16",
  "pruning": "dense",
  "compiled": true,
  "compile_mode": "default",
  "batch_size": 32,
  "seq_len": 128,
  "compile_time_s": 12.4,
  "warmup_time_s": 3.1,
  "latency_p50_ms": 8.2,
  "latency_p95_ms": 9.1,
  "latency_p99_ms": 9.8,
  "throughput_samples_sec": 3901,
  "tokens_per_sec": 499328,
  "peak_memory_gb": 4.2,
  "max_batch_before_oom": 256,
  "flops_per_sample_G": 22.4,
  "arithmetic_intensity": 312,
  "mxu_utilization_pct": 71,
  "sm_utilization_pct": null,
  "cost_per_1k_samples_usd": 0.00009
}
```

---

## Repository Structure

```
tpu-gpu-inference-bench/
├── benchmarks/
│   ├── harness.py              # CLI entry: --model --suite --device --framework
│   ├── runner.py               # 1 experiment → metrics dict
│   ├── models/
│   │   ├── registry.yaml       # all 45 models: HF ID, input spec, bs limits
│   │   ├── jax/                # Flax (Path 1 TPU + Path 2 GPU)
│   │   │   ├── vision.py
│   │   │   ├── nlp_encoder.py
│   │   │   ├── nlp_decoder.py
│   │   │   ├── audio.py
│   │   │   └── multimodal.py
│   │   └── torch/              # PyTorch (Path 3 GPU + Path 4 torch_xla TPU)
│   │       ├── vision.py
│   │       ├── nlp_encoder.py
│   │       ├── nlp_decoder.py
│   │       ├── audio.py
│   │       └── multimodal.py
│   ├── variants/
│   │   ├── precision.py
│   │   ├── pruning.py
│   │   └── compile.py
│   └── profiler/
│       ├── jax_profiler.py
│       ├── torch_profiler.py
│       ├── roofline.py
│       └── metrics.py
├── configs/
│   ├── hardware.yaml           # peak FLOPs, BW, memory per device
│   └── suites/
│       ├── smoke.yaml
│       ├── quick.yaml
│       ├── domain.yaml
│       ├── arch.yaml
│       └── full.yaml
├── results/
│   ├── runs.jsonl
│   └── dashboard/
│       ├── index.html
│       └── views/
│           ├── roofline.html
│           ├── throughput.html
│           ├── latency.html
│           ├── compiler.html
│           ├── precision.html
│           └── architecture.html
└── scripts/
    ├── cache_models_gcs.sh
    ├── provision_tpu.sh
    ├── run_suite.sh
    ├── teardown_tpu.sh
    └── sync_github.sh
```

---

## Learning Arcs (Strategic)

See strategy discussion in conversation for full detail. Seven arcs:

1. **Hardware DNA** — silicon-level roofline, memory BW vs compute ceilings
2. **The Compiler is the Hardware** — XLA fusion vs CUDA graphs vs torch.compile
3. **Architecture-Hardware Fit** — which model designs love which chips
4. **The Precision Story** — BF16 is not just half the bits on TPU
5. **Beyond Transformers** — SSMs, MoE, Diffusion, Detection
6. **The Ecosystem** — framework overhead, tooling, debugging cost
7. **Real-world TCO** — combining all metrics into cost/quality

---

## Key "Aha Moments" Engineered by the Benchmark

1. EfficientNet is slower than ViT on TPU despite fewer FLOPs — depthwise conv kills MXU
2. Mamba-2.8B is faster than GPT-2-XL on GPU, slower on TPU — custom CUDA kernel story
3. Gemma-2B is fastest 2B on TPU — hardware/software co-design pays off
4. BF16 is free on TPU (same latency), costs ~45% overhead on GPU — key differentiator
5. XLA compile takes 10–50× longer than torch.compile but runs faster after
6. Unstructured 90% sparse = slower on both; 2:4 = 2× faster on GPU only
7. DiT-XL/2 (transformer diffusion) is TPU's best diffusion story — pure matmuls
8. RecurrentGemma-2B shows RNN hybrid: worse throughput on TPU than transformer Gemma-2B
9. HF Inference API latency is 3–10× higher than self-hosted but zero infra
10. ModernBERT outperforms BERT on throughput too — Flash attention helps TPU and GPU
