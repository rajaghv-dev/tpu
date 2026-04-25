# Project Context — TPU × GPU Inference Benchmark

Full accumulated context, design decisions, and strategy.
Last updated: 2026-04-25

---

## 1. Project Goal

Build a rigorous, reproducible inference benchmark comparing Google TPU and NVIDIA GPU.
Every claim made by the benchmark must be backed by traceable evidence:
raw timings, hardware utilisation metrics, profiler traces, statistical confidence intervals,
and reproducibility metadata (git SHA, model revision, input seed, environment snapshot).

---

## 2. User Hardware

### Local GPUs (zero marginal cost)
| Card | VRAM | Mem BW | Peak BF16 TFLOPs | Notes |
|------|------|--------|-----------------|-------|
| RTX 3080 | 16 GB GDDR6X | 760 GB/s | 119 | Ampere — 2:4 structured sparsity |
| RTX 4090 | 24 GB GDDR6X | 1008 GB/s | 330 | Ada Lovelace — FP8, highest consumer BW |
| A100 SXM (DGX, 1 card) | 80 GB HBM2e | 2000 GB/s | 312 | Highest memory BW in local fleet |

### Cloud TPU — Single-Chip VMs (personal account: rajaghv@gmail.com)
| TPU | HBM | Mem BW | Peak BF16 | Preemptible | Single-chip VM |
|-----|-----|--------|-----------|-------------|----------------|
| v2 chip | 8 GB | 600 GB/s | 45 TFLOPs | $1.35/hr (8-chip min) | No |
| v3 chip | 16 GB | 900 GB/s | 123 TFLOPs | $2.40/hr (8-chip min) | No |
| v4 chip | 32 GB | 1200 GB/s | 275 TFLOPs | $3.90/hr (8-chip min) | No |
| **v5e-1** | **16 GB** | **820 GB/s** | **394 TFLOPs** | **$0.36/hr** | **Yes — primary** |
| v5p chip | 95 GB | 2765 GB/s | 459 TFLOPs | $9.60/hr (8-chip min) | No |
| **v6e-1 (Trillium)** | **32 GB** | **1640 GB/s** | **918 TFLOPs** | **~$0.75/hr** | **Yes — secondary** |

Primary pairing: **v5e-1 ↔ RTX 3080** (both 16 GB).
Secondary pairing: **v6e-1 ↔ RTX 4090** (32 GB vs 24 GB).

### Max Model Size Per Chip (Inference, BF16)
| Chip | Comfortable | Tight (bs=1) | INT8 |
|------|-------------|--------------|------|
| v2 / v3 / v5e (16 GB) | 3B | 5B | 10B |
| v4 / v6e (32 GB) | 7B | 10B | 20B |
| v5p (95 GB) | 30B | 38B | 70B |

### External Services
- HuggingFace account: available (gated model access; token = `HF_TOKEN` env var)
- HF Inference API: Serverless + Dedicated Endpoints (Path 5)
- GCS bucket: model weight cache; ~80 GB; ~$1.60/month
- Google Colab Pro: TPU v2-8 or v3-8; A100/V100 GPU

---

## 3. The 5 Execution Paths

```
            Same pretrained weights (HuggingFace Hub)
                           |
         ┌─────────────────┼──────────────────────┐
      JAX/Flax          JAX/Flax               PyTorch
         │                  │                      │
    ┌────▼────┐       ┌──────▼─────┐       ┌───────┴────┐    ┌────────────┐
    │ Path 1  │       │  Path 2    │       │  Path 3    │    │  Path 4    │
    │JAX+TPU  │       │ JAX+GPU    │       │ PyTorch+   │    │ PT+        │
    │  (XLA)  │       │  (CUDA)    │       │   GPU      │    │ torch_xla  │
    └─────────┘       └────────────┘       └────────────┘    └────────────┘
                                                              (TPU via XLA)

    Path 5:  HuggingFace Inference API  (serverless / dedicated endpoint)
```

| Pair | Isolates |
|------|---------|
| 1 vs 2 | Hardware only (TPU vs GPU), JAX constant |
| 2 vs 3 | Framework only (JAX vs PyTorch), GPU constant |
| 1 vs 4 | Framework on TPU (JAX/XLA vs torch_xla) |
| 3 vs 4 | Compiler only (CUDA vs XLA), PyTorch API constant |
| 1 vs 3 | Real-world (production TPU vs production GPU) |
| 5 vs all | Managed serving overhead vs self-hosted |

---

## 4. Full Model Registry (~75 Models)

### 4.1 Vision — Classification / Feature Extraction
| Model | Params | HF ID | Story |
|-------|--------|-------|-------|
| ResNet-50 | 86M | `microsoft/resnet-50` | Conv baseline |
| ConvNeXt-XL | 350M | `facebook/convnext-xlarge-224-22k` | Modern pure conv |
| EfficientNet-B7 | 66M | `google/efficientnet-b7` | Depthwise — kills MXU |
| EfficientViT-L3 | 246M | `mit-han-lab/efficientvit-l3` | Efficient hybrid |
| ViT-B/16 | 86M | `google/vit-base-patch16-224` | Pure attention baseline |
| ViT-L/16 | 307M | `google/vit-large-patch16-224` | Larger ViT |
| DINOv2-L | 307M | `facebook/dinov2-large` | SSL |
| SigLIP-B/16 | 400M | `google/siglip-base-patch16-224` | Google CLIP replacement |
| SAM-L | 312M | `facebook/sam-vit-large` | Segment-anything |
| EVA-02-L | 307M | `Yuxin-CV/EVA-02` | CLIP-pretrained ViT |

### 4.2 Vision — Detection
| Model | Params | HF ID | Story |
|-------|--------|-------|-------|
| DETR-ResNet50 | 41M | `facebook/detr-resnet-50` | Transformer detection |
| RT-DETR-L | 32M | `PekingU/rtdetr_r50vd` | Real-time DETR |

### 4.3 Vision — Diffusion / Generative
| Model | Params | HF ID | Story |
|-------|--------|-------|-------|
| DiT-XL/2 | 675M | `facebook/DiT-XL-2-256` | Pure transformer diffusion — TPU's best |
| SD-UNet | 860M | SD v1.5 UNet only | Conv+cross-attn hybrid |

### 4.4 NLP — Encoders
| Model | Params | HF ID | Story |
|-------|--------|-------|-------|
| BERT-base | 110M | `bert-base-uncased` | Historical baseline |
| RoBERTa-large | 355M | `roberta-large` | Better pretraining |
| DeBERTa-v3-large | 400M | `microsoft/deberta-v3-large` | SOTA encoder |
| ModernBERT-base | 149M | `answerdotai/ModernBERT-base` | Dec 2024, Flash attn |
| ModernBERT-large | 395M | `answerdotai/ModernBERT-large` | Latest encoder SOTA |
| BGE-large-en-v1.5 | 335M | `BAAI/bge-large-en-v1.5` | Dense retrieval — huge bs |
| E5-large-v2 | 335M | `intfloat/e5-large-v2` | Universal embedding |
| nomic-embed-v1.5 | 137M | `nomic-ai/nomic-embed-text-v1.5` | Matryoshka + RoPE |

### 4.5 NLP — Decoders (LLMs, autoregressive inference)
| Model | Params | HF ID | Gated | Story |
|-------|--------|-------|-------|-------|
| GPT-2 XL | 1.5B | `gpt2-xl` | No | Historical baseline |
| OPT-2.7B | 2.7B | `facebook/opt-2.7b` | No | ALiBi — no RoPE |
| BLOOM-3B | 3B | `bigscience/bloom-3b` | No | ALiBi positional encoding |
| Falcon-RW-1B | 1B | `tiiuae/falcon-rw-1b` | No | Multi-query attention |
| TinyLlama-1.1B | 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | No | Efficient small LLM |
| SmolLM2-1.7B | 1.7B | `HuggingFaceTB/SmolLM2-1.7B` | No | HF flagship small |
| OLMo-2-1B | 1.2B | `allenai/OLMo-2-1124-7B` | No | Fully open |
| Llama-3.2-1B | 1B | `meta-llama/Llama-3.2-1B` | Yes | Meta smallest, GQA |
| Llama-3.2-3B | 3B | `meta-llama/Llama-3.2-3B` | Yes | Meta 3B |
| Mistral-7B-v0.3 | 7B | `mistralai/Mistral-7B-v0.3` | No | Sliding window attn (v6e-1) |
| StableLM-2-1.6B | 1.6B | `stabilityai/stablelm-2-1_6b` | No | Open 1.6B |
| StableLM-3B | 3B | `stabilityai/stablelm-3b-4e1t` | No | Open 3B |
| **Gemma-2B** | **2B** | **`google/gemma-2b`** | Yes | **Google TPU co-designed** |
| **Gemma-2-2B** | **2.6B** | **`google/gemma-2-2b`** | Yes | GQA + sliding window |
| **Gemma-2-2B-IT** | **2.6B** | **`google/gemma-2-2b-it`** | Yes | Instruction-tuned variant |
| **Gemma-3-1B** | **1B** | **`google/gemma-3-1b`** | Yes | **Newest 2025, multimodal-capable** |
| **Gemma-3-4B** | **4B** | **`google/gemma-3-4b`** | Yes | **At the 4B ceiling** |
| RecurrentGemma-2B | 2B | `google/recurrentgemma-2b` | Yes | RNN-hybrid — recurrence vs attention |
| **Phi-1.5** | **1.3B** | **`microsoft/phi-1_5`** | No | Textbook data training story |
| **Phi-2** | **2.7B** | **`microsoft/phi-2`** | No | Quality/size ratio benchmark |
| **Phi-3-mini-4k** | **3.8B** | **`microsoft/Phi-3-mini-4k-instruct`** | No | Near 4B ceiling |
| **Phi-3.5-mini** | **3.8B** | **`microsoft/Phi-3.5-mini-instruct`** | No | 128k context |
| **Phi-3.5-MoE** | **6.6B / 2.7B active** | **`microsoft/Phi-3.5-MoE-instruct`** | No | **MoE sparse routing story** |
| **Qwen2.5-0.5B** | **500M** | **`Qwen/Qwen2.5-0.5B`** | No | Smallest modern LLM — latency floor |
| **Qwen2.5-1.5B** | **1.5B** | **`Qwen/Qwen2.5-1.5B`** | No | Best 1.5B on most benchmarks |
| **Qwen2.5-3B** | **3B** | **`Qwen/Qwen2.5-3B`** | No | Strong multilingual |
| **Qwen2.5-Coder-1.5B** | **1.5B** | **`Qwen/Qwen2.5-Coder-1.5B`** | No | Code — different token dist |
| **Qwen2.5-Coder-3B** | **3B** | **`Qwen/Qwen2.5-Coder-3B`** | No | Code 3B |
| **DeepSeek-R1-Distill-Qwen-1.5B** | **1.5B** | **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`** | No | **Reasoning — long chain-of-thought decode** |
| **DeepSeek-R1-Distill-Qwen-7B** | **7B** | **`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`** | No | v6e-1 only; #1 reasoning evals |
| **DeepSeek-Coder-V2-Lite** | **2.3B active / 16B total** | **`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`** | No | **MoE — sparse routing** |
| MPT-7B | 7B | `mosaicml/mpt-7b` | No | ALiBi + FlashAttn (v6e-1) |

### 4.6 NLP — Novel Architectures (Non-Transformer)
| Model | Params | HF ID | Story |
|-------|--------|-------|-------|
| Mamba-2.8B | 2.8B | `state-spaces/mamba-2.8b` | SSM — custom CUDA, no XLA prim |
| Mamba2-2.7B | 2.7B | `state-spaces/mamba2-2.7b` | Improved SSM |
| RWKV-4-3B | 3B | `RWKV/rwkv-4-world-3b` | Linear RNN — O(1) memory |

### 4.7 NLP — Code Models
| Model | Params | HF ID | Story |
|-------|--------|-------|-------|
| StarCoder2-3B | 3B | `bigcode/starcoder2-3b` | 16k context, infill mask |
| CodeGemma-2B | 2B | `google/codegemma-2b` | Google code — pairs with Gemma-2B |
| Qwen2.5-Coder-1.5B | 1.5B | `Qwen/Qwen2.5-Coder-1.5B` | see 4.5 |
| Qwen2.5-Coder-3B | 3B | `Qwen/Qwen2.5-Coder-3B` | see 4.5 |

### 4.8 Audio
| Model | Params | HF ID | Story |
|-------|--------|-------|-------|
| Whisper-base | 74M | `openai/whisper-base` | Small audio baseline |
| Whisper-medium | 307M | `openai/whisper-medium` | Balanced |
| Whisper-large-v3 | 1.5B | `openai/whisper-large-v3` | Best quality |
| wav2vec2-large | 317M | `facebook/wav2vec2-large` | Heavy conv front-end |
| HuBERT-large | 316M | `facebook/hubert-large-ls960-ft` | SSL audio |
| SeamlessM4T-medium | 1.2B | `facebook/seamless-m4t-medium` | Speech translation |
| MMS-1B | 1B | `facebook/mms-1b-all` | 1000+ language multilingual |
| EnCodec-24kHz | 44M | `facebook/encodec_24khz` | Conv + LSTM — sequential story |

### 4.9 Multimodal — Vision-Language (≤4B)
| Model | Params | HF ID | Gated | Story |
|-------|--------|-------|-------|-------|
| CLIP ViT-L/14 | 428M | `openai/clip-vit-large-patch14` | No | Contrastive baseline |
| SigLIP-SO400M | 400M | `google/siglip-so400m-patch14-384` | No | Google CLIP replacement |
| moondream2 | 1.86B | `vikhyatk/moondream2` | No | Tiny capable VLM |
| SmolVLM-2B | 2B | `HuggingFaceTB/SmolVLM-Instruct` | No | Dec 2024 |
| PaliGemma-3B | 3B | `google/paligemma-3b-pt-224` | Yes | All-Google: model+compiler+hardware |
| LLaVA-Phi3 | ~4B | `xtuner/llava-phi-3-mini-hf` | No | Phi-3 backbone |
| ImageBind | ~1.2B | `facebook/imagebind` | No | 6-modality encoder |

---

## 5. Experiment Dimensions

### Precision
| Format | TPU | GPU | Key insight |
|--------|-----|-----|-------------|
| FP32 | Yes | Yes | Baseline |
| BF16 | **Native = FP32 speed** | ~2× faster via Tensor Cores | TPU: free; GPU: a choice |
| INT8 | v5+ (~1.3×) | Tensor Cores (~2×) | GPU wins more |
| INT4 | No | bitsandbytes | GPU-exclusive |
| FP8 | v5e/v6e | 4090/H100 | Frontier |

### Compilation
| Mode | JAX | PyTorch | Measures |
|------|-----|---------|---------|
| Eager | disabled jit | default | Overhead baseline |
| Compiled | `jax.jit` | `torch.compile(default)` | Production mode |
| Max-optimised | `jax.jit` | `torch.compile(max-autotune)` | Peak ceiling |
| CUDA Graphs | — | `make_graphed_callables` | GPU kernel-replay at bs=1 |
| XLA persistent cache | `JAX_COMPILATION_CACHE_DIR` | — | Compile amortisation |

### Sparsity / Pruning
| Variant | TPU | GPU | Key insight |
|---------|-----|-----|-------------|
| Dense | Baseline | Baseline | — |
| Unstructured 50/90% | Same or slower | Marginally faster | No hardware benefit without structure |
| 2:4 structured | No support | **2× Sparse Tensor Core** | NVIDIA-exclusive |

### LLM Serving Modes
| Mode | Measures |
|------|---------|
| Prefill-only | Prompt processing throughput (tokens/sec) |
| Single-token decode | KV-cache warm, autoregressive token/sec |
| Batch decode | Throughput at multiple concurrency levels |

---

## 6. Per-Experiment Protocol

```
Phase           Passes   Measure
─────────────────────────────────────────────────────
1. Compile      1        compile_time_s (XLA trace or torch.compile)
2. Warmup       20       discard — kernels and caches stabilise
3. Latency      100      bs=1 → p50 / p95 / p99 / mean / std_dev (ms)
4. Throughput   100      max_batch → samples/sec ± std_dev
5. Profile      10       full trace → op breakdown, roofline data
6. Memory sweep 1/bs     bs=1,2,4,8,… → peak_memory_gb per level
7. Numerics     1        FP32 vs BF16 output L2 norm (correctness check)
```

---

## 7. Suite Definitions

| Suite | Models | Variants | Experiments | Wall time | v5e-1 cost |
|-------|--------|----------|-------------|-----------|------------|
| `smoke` | 1 (BERT-base) | FP32+BF16 | 4 | ~8 min | $0.05 |
| `quick` | 6 (1/domain) | BF16 | 24 | ~50 min | $0.30 |
| `domain` | All in 1 domain | FP32+BF16 | ~30 | ~60 min | $0.36 |
| `arch` | Novel arches | BF16 | ~20 | ~40 min | $0.24 |
| `full` | All ~75 models | All variants | ~800 | ~8 hrs | $2.88 |

---

## 8. Results Schema (JSONL — one line per experiment)

```jsonc
{
  // Identity
  "run_id": "uuid-v4",
  "experiment_id": "sha256-of-config",
  "timestamp": "2026-04-25T10:32:00Z",

  // Lineage (reproducibility anchors)
  "git_sha": "abc1234",
  "jax_version": "0.4.25",
  "torch_version": "2.3.0",
  "torch_xla_version": "2.3.0",
  "cuda_version": "12.4",
  "cudnn_version": "8.9",
  "tpu_runtime_version": "tpu-vm-base-v5e",
  "hf_model_revision": "sha256-of-weights",
  "input_seed": 42,
  "environment_hash": "sha256-of-key-versions",

  // Hardware
  "device": "tpu_v5e1 | tpu_v6e1 | rtx3080 | rtx4090 | a100_dgx",
  "framework": "jax | pytorch | torch_xla | hf_api",
  "path": 1,

  // Model
  "model": "bert_base",
  "model_hf_id": "bert-base-uncased",
  "domain": "nlp_encoder",
  "architecture_family": "transformer_encoder",
  "attention_variant": "mha",
  "positional_encoding": "absolute",
  "params_M": 110,

  // Variant
  "precision": "bf16",
  "pruning": "dense",
  "sparsity_ratio": 0.0,
  "compiled": true,
  "compile_mode": "default",
  "kv_cache": false,

  // Input spec
  "batch_size": 32,
  "seq_len": 128,
  "image_size": null,

  // Compile metrics
  "compile_time_s": 12.4,
  "compile_cache_hit": false,

  // Latency (100 passes, bs=1)
  "latency_mean_ms": 8.4,
  "latency_std_ms": 0.3,
  "latency_p50_ms": 8.2,
  "latency_p95_ms": 9.1,
  "latency_p99_ms": 9.8,
  "latency_cv_pct": 3.6,       // coefficient of variation — high = unstable

  // Throughput (100 passes, max batch)
  "throughput_mean_samples_sec": 3901,
  "throughput_std_samples_sec": 45,
  "tokens_per_sec": 499328,

  // Memory
  "peak_memory_gb": 4.2,
  "weight_memory_gb": 0.22,
  "activation_memory_gb": 3.98,
  "max_batch_before_oom": 256,

  // Compute analysis
  "flops_per_sample_G": 22.4,
  "arithmetic_intensity_flops_per_byte": 312,
  "achieved_tflops": 87.3,
  "peak_tflops_device": 394,
  "mfu_pct": 22.2,              // model FLOP utilisation

  // Hardware utilisation
  "mxu_utilization_pct": 71,    // TPU only — from Cloud Monitoring
  "sm_utilization_pct": null,   // GPU only — from pynvml
  "memory_bw_utilization_pct": 58,
  "device_power_w": null,       // GPU: pynvml; TPU: not exposed

  // Thermal / clock state
  "gpu_clock_mhz": null,
  "gpu_temp_c": null,
  "throttle_detected": false,

  // Numerical correctness
  "output_l2_vs_fp32": 0.0012, // null if this IS the fp32 run
  "output_cosine_sim_vs_fp32": 0.9998,

  // Quality flags
  "flags": [],                  // ["compile_slow","high_variance","near_oom"]

  // Cost
  "device_cost_usd_per_hr": 0.36,
  "experiment_cost_usd": 0.012,
  "cost_per_1k_samples_usd": 0.000092
}
```

---

## 9. Observability Infrastructure

### 9.1 System Monitor (`observe/system_monitor.py`)
Captures hardware state before and after each experiment:
- **GPU:** `pynvml` → utilisation %, memory used/free, power draw (W), temperature (°C), clock speeds, throttle reasons
- **TPU:** Google Cloud Monitoring API → MXU utilisation, HBM used, infeed queue depth, step time
- **Host CPU/RAM:** `psutil` → CPU %, RAM used, I/O wait

### 9.2 FLOPs Counter (`observe/flops_counter.py`)
- **JAX:** `jax.make_jaxpr()` → parse HLO operations, sum FLOPs per op
- **PyTorch:** `torch.profiler` with `profile_memory=True, with_flops=True`, or `fvcore.nn.FlopCountAnalysis`
- Output: `flops_per_sample_G`, `flops_breakdown_by_op_type` (matmul, conv, attn, norm, elementwise)

### 9.3 Memory Profiler (`observe/memory_profiler.py`)
- Peak HBM/VRAM (existing)
- Timeline: memory snapshot every N steps → detect spikes
- Per-layer breakdown: activation vs weight vs KV-cache (LLMs)
- JAX: `jax.profiler.device_memory_profile()` → pprof format
- PyTorch: `torch.cuda.memory_snapshot()` + `torch.cuda.memory_stats()`

### 9.4 Statistical Analyser (`observe/stats.py`)
- Collect all N pass timings as raw array
- Compute mean, std, p50, p95, p99, coefficient of variation
- Run Grubbs test for outliers, flag if CV > 10%
- Require minimum N=100 passes for latency claims (configurable)
- Output: `latency_distribution.json` per experiment

### 9.5 Numerical Validator (`observe/numerics.py`)
- Run each model in FP32 first (reference)
- For each precision variant, compare output tensor:
  - L2 norm of difference: `‖y_fp32 − y_variant‖₂`
  - Cosine similarity
  - Max absolute error
- Flags run if cosine sim < 0.99 (configurable threshold)
- Records `output_l2_vs_fp32` and `output_cosine_sim_vs_fp32` in schema

### 9.6 Lineage Tracker (`observe/lineage.py`)
Records at experiment start:
- `git_sha`: `git rev-parse HEAD`
- `hf_model_revision`: HuggingFace model card `sha` field
- `input_seed`: RNG seed used for synthetic inputs
- `environment_hash`: SHA256 of key version string (JAX + CUDA + driver)
- All framework versions via `importlib.metadata`

### 9.7 Profiler Traces (`observe/tracer.py`)
- **JAX:** `jax.profiler.trace(log_dir)` → 10 steps → TensorBoard + Perfetto
- **PyTorch GPU:** `torch.profiler.profile(activities=[CPU, CUDA], with_stack=True)` → Chrome JSON + TensorBoard
- **torch_xla:** `torch_xla.debug.profiler.trace(log_dir)` → same as JAX path
- Output: `results/run_logs/<run_id>/profiles/<model>_<precision>.pb`

---

## 10. Staged Repository Build Plan

Each stage is independently runnable and produces real benchmark data.

```
Stage 1 — Foundation                          Target: 1 day
  Files: harness.py, runner.py
         models/registry.yaml (5 models)
         observe/lineage.py, observe/stats.py
         results/dashboard/index.html (table only)
  Paths: Path 1 (JAX+TPU) only
  Output: first rows in runs.jsonl, working table dashboard

Stage 2 — Multi-path + GPU                    Target: 2 days
  Files: models/jax/*, models/torch/*
         observe/system_monitor.py
  Paths: Add Path 2 (JAX+GPU) and Path 3 (PyTorch+GPU)
  Models: expand to 15
  Dashboard: throughput heatmap, latency chart

Stage 3 — Profiler + Roofline                 Target: 2 days
  Files: observe/flops_counter.py
         observe/tracer.py
         observe/memory_profiler.py
         results/dashboard/views/roofline.html
  Adds: flops_per_sample, arithmetic_intensity, mfu_pct to schema

Stage 4 — torch_xla (Path 4)                  Target: 1 day
  Files: models/torch/xla_wrapper.py
  Dashboard: compiler comparison view (Path 1 vs 4, Path 3 vs 4)

Stage 5 — Novel Architectures                 Target: 2 days
  Files: variants/compile.py
  Models: Mamba, RWKV, RecurrentGemma, DiT — the "aha moment" models
  Dashboard: architecture-hardware fit view

Stage 6 — Precision + Quantization            Target: 2 days
  Files: variants/precision.py (INT8, FP8)
         observe/numerics.py
  Models: add Qwen, DeepSeek, Phi, Gemma-3
  Dashboard: precision speedup view

Stage 7 — HF Inference API (Path 5)           Target: 1 day
  Files: paths/hf_api.py
  Dashboard: TCO calculator

Stage 8 — Sparsity + Pruning                  Target: 1 day
  Files: variants/pruning.py
  Dashboard: sparsity impact view

Stage 9 — Full Registry + Automation          Target: ongoing
  Files: .github/workflows/bench.yml (scheduled runs)
         scripts/run_suite.sh with auto-push
  Models: all ~75
  Dashboard: full interactive explorer with claim → evidence links
```

---

## 11. Visualisation Plan

### Layer 1 — Static GitHub Pages Dashboard
Built with Vega-Lite, reads `results/runs.jsonl` via `fetch()`.

| View | Chart type | Primary claim it evidences |
|------|-----------|---------------------------|
| `throughput.html` | Heatmap: model × device | Who wins at what |
| `latency.html` | Box plot + CDF per device | Tail latency behaviour |
| `roofline.html` | Scatter: intensity vs TFLOPs, roofline overlaid | Why they win (compute vs memory bound) |
| `compiler.html` | Bar: compile time + speedup ratio | Hidden first-run cost |
| `precision.html` | Grouped bar: BF16/FP32 ratio per device | BF16 is free on TPU |
| `sparsity.html` | Bar: dense vs pruned, per device | 2:4 story |
| `architecture.html` | Scatter: FLOPs vs throughput, coloured by family | Architecture-hardware fit |
| `tco.html` | Bar: cost/1k-samples all paths | Real-world TCO |
| `numerics.html` | Scatter: L2 error vs speedup | Precision accuracy tradeoff |
| `mfu.html` | Bar: model FLOP utilisation % per device | Hardware efficiency |

### Layer 2 — Notebook Explorer (`notebooks/explore.ipynb`)
Pandas + Plotly, reads `runs.jsonl`. Pre-built views:
- Roofline with interactive hover
- Batch-size scaling curves (throughput vs bs)
- Latency CDF
- Cross-run anomaly detector
- Claim verifier: "Does the data support Arc X?"

### Layer 3 — TensorBoard + Perfetto
- One profile trace per model per run, stored in `results/run_logs/`
- `tensorboard --logdir=results/run_logs/` for op breakdown
- Perfetto (`ui.perfetto.dev`) for large traces — XLA fusion visualisation
- GPU: torch.profiler traces in `results/run_logs/*/profiles/*.json`

---

## 12. Key Claims and Their Evidence Requirements

| Claim (Learning Arc) | Evidence needed | Tool | Schema field |
|---------------------|----------------|------|--------------|
| EfficientNet slower than ViT despite fewer FLOPs | FLOPs count, MXU%, throughput | flops_counter + system_monitor | flops_per_sample_G, mxu_utilization_pct |
| Mamba faster on GPU, slower on TPU | Throughput comparison; XLA HLO dump showing fallback | tracer | throughput_mean; profile trace |
| BF16 is free on TPU | FP32 vs BF16 latency within noise band | stats.py (need CV<5%) | latency_mean_ms ± std across precisions |
| XLA compile 10–50× slower than torch.compile | compile_time_s on both | lineage + runner | compile_time_s |
| 2:4 sparsity gives 2× GPU speedup | Dense vs 2:4 throughput; SM% | pruning + system_monitor | throughput_mean_samples_sec, sm_utilization_pct |
| Gemma-2B fastest 2B on TPU | Throughput rank across all 2B models on TPU | runner | throughput_mean_samples_sec filtered |
| TPU v5e-1 cost-per-sample competitive with 4090 | cost_per_1k across both | runner | cost_per_1k_samples_usd |
| INT8 preserves accuracy | Output L2 vs FP32 < threshold | numerics.py | output_cosine_sim_vs_fp32 |

---

## 13. Cost Reference

| Scenario | Cost |
|----------|------|
| Single experiment (~2 min) on v5e-1 preemptible | $0.012 |
| `quick` suite (50 min) | $0.30 |
| `full` suite (~8 hrs) | $2.88 |
| Weekly `full` suite for 1 month | ~$12 |
| GCS model cache (~80 GB) | $1.60/month |
| Colab Pro | $9.99/month |
| HF PRO (Inference API) | $9/month |

---

## 14. Gaps Analysis and Remediation

### 14.1 Critical Gaps (would invalidate claims if unaddressed)

**Gap C1 — No actual FLOPs counter**
The roofline scatter (arithmetic intensity vs TFLOPs) has no x-axis without measured FLOPs.
- Claim at risk: "EfficientNet fewer FLOPs but slower on TPU"
- Fix: `observe/flops_counter.py` — JAX: `jax.make_jaxpr()` → parse HLO; PyTorch: `fvcore.nn.FlopCountAnalysis`
- Schema fields: `flops_per_sample_G`, `flops_breakdown_by_op_type`

**Gap C2 — Single-run numbers are not evidence**
One 50-pass block has unknown variance. "TPU is 2.3× faster" from a single run may be within noise.
- Fix: minimum 3 independent cold runs per experiment; report mean ± std; t-test if gap <20%
- Schema fields: `latency_std_ms`, `latency_cv_pct`, `n_independent_runs`
- Rule: CV > 10% triggers `high_variance` flag; claim cannot be made from flagged run

**Gap C3 — Compile cache not explicitly controlled**
XLA compile time varies wildly between cold/warm cache. Inconsistent measurement confounds the "XLA 10–50× slower to compile" claim.
- Fix: explicitly clear XLA persistent cache before each compile measurement; record `compile_cache_hit: bool`
- Schema fields: `compile_cache_hit`, `first_compile_s`, `subsequent_compile_s`

**Gap C4 — No thermal / clock state control**
GPU boost clocks drop under sustained heat. A suite run on 4090 may start at 2.7 GHz, end at 2.4 GHz — an 11% confound.
- Fix: record `gpu_clock_mhz` at start/end; flag `throttle_detected` if delta >5%; enforce 30s idle between experiments
- Schema fields: `gpu_clock_mhz_start`, `gpu_clock_mhz_end`, `gpu_temp_c_start`, `throttle_detected`

**Gap C5 — Hardware utilisation metrics not yet captured**
`mxu_utilization_pct` and `sm_utilization_pct` are in schema but no collection code exists yet.
- Fix: `observe/system_monitor.py` — GPU: `pynvml.nvmlDeviceGetUtilizationRates()` polled every 100ms; TPU: Cloud Monitoring API metric `tpu/container/accelerator/matrix_unit_utilization`
- Without these, roofline is theoretical not measured

### 14.2 Important Gaps (weaken evidence)

**Gap I1 — No LLM prefill vs decode separation**
`model(input_ids)` conflates prompt processing (compute-bound) and token generation (memory-bound). These are different hardware stories.
- Fix: two sub-experiments per LLM: (a) prefill — long prompt, no KV cache; (b) decode — 1 token, warm KV cache
- Schema fields: `inference_mode: prefill | decode | combined`

**Gap I2 — Warm memory cache not controlled**
Repeating the same input warms L2 cache, inflating throughput for memory-bound models.
- Fix: rotate through K=4 distinct random input batches during measurement phase
- Schema field: `input_varied: bool`

**Gap I3 — MoE dynamic routing not handled**
Phi-3.5-MoE and DeepSeek-Coder-V2-Lite use dynamic expert routing — incompatible with XLA static shapes by default.
- Fix: record `active_params_M` vs `total_params_M`; flag if MoE fell back to dense on TPU
- Schema fields: `active_params_M`, `moe_routing_mode: dynamic | static | dense_fallback`

**Gap I4 — Cross-device input seed not enforced**
JAX and PyTorch must generate identical synthetic inputs from the same seed for fair comparison. Currently planned but not enforced in runner.
- Fix: runner generates numpy arrays with fixed seed, converts to JAX or torch tensor as needed

**Gap I5 — XLA op fusion not measured**
Claim "XLA fuses LayerNorm+GeLU+residual" needs a count of fusion groups in the HLO, not just a qualitative statement.
- Fix: `observe/hlo_analyser.py` — parse `jax.xla_computation(model)(dummy)` output; count `fusion{}` nodes and kernel launches
- Schema fields: `xla_fusion_groups`, `xla_kernel_launches`, `cuda_kernel_launches`

**Gap I6 — HF Inference API conflates latency sources**
Serverless API mixes network RTT, queue time, cold-start, and compute. Not comparable to self-hosted.
- Fix: use Dedicated Endpoints (always warm, known hardware); record `hf_endpoint_hardware`; separate cold vs warm request
- Schema fields: `hf_cold_latency_ms`, `hf_warm_latency_ms`, `hf_endpoint_hardware`

**Gap I7 — No power / energy measurement for TCO**
Self-hosted GPU electricity cost is real. RTX 4090 at 450W TDP, $0.12/kWh = $0.054/hr — changes break-even analysis.
- Fix: `pynvml.nvmlDeviceGetPowerUsage()` → watts during run; compute `energy_wh_per_1k_samples`
- Schema fields: `device_power_w`, `energy_wh_per_1k_samples`, `electricity_cost_per_1k_samples_usd`

### 14.3 Nice-to-Have Gaps (add depth and completeness)

**Gap N1 — No full batch-size throughput curve**
Currently: bs=1 (latency) and bs=max (throughput). The curve from bs=1 to bs=max reveals optimal serving batch size.
- Fix: add `bs_sweep_results: [{bs, throughput_mean, peak_memory_gb}, ...]` to schema
- Generates a dedicated chart: "optimal batch size per device per model"

**Gap N2 — No inter-run reproducibility check**
Cannot distinguish hardware noise from bugs without running the same experiment twice and comparing.
- Fix: `repro` suite — runs smoke × 5; reports CV across runs; fails if CV > 5%

**Gap N3 — No compilation cache size tracking**
XLA persistent cache can grow to several GB; slow-disk cache hits can still be slow.
- Fix: record `xla_cache_size_mb`, `xla_cache_load_time_ms`

**Gap N4 — No attention mechanism isolation**
Comparing MHA vs GQA vs MQA across different models conflates architecture changes with attention changes.
- Fix (later stage): synthetic model with swappable attention class; same size, same depth, different attention variant only

### 14.4 Remediation Priority Order

| Priority | Gap | Stage to fix | Effort |
|----------|-----|-------------|--------|
| P0 | C5: hardware utilisation collection | Stage 1 | Medium |
| P0 | C2: multi-run statistics | Stage 1 | Low |
| P0 | C3: compile cache discipline | Stage 1 | Low |
| P1 | C1: FLOPs counter | Stage 3 | High |
| P1 | C4: thermal control | Stage 2 | Low |
| P1 | I1: prefill vs decode | Stage 2 | Medium |
| P2 | I2: input cache control | Stage 2 | Low |
| P2 | I5: XLA fusion measurement | Stage 3 | High |
| P2 | I7: power measurement | Stage 3 | Medium |
| P3 | I3: MoE handling | Stage 6 | Medium |
| P3 | I6: HF API separation | Stage 7 | Low |
| P4 | N1–N4 | Stages 5–9 | Various |

---

## 15. Evidence Chain — Traceability Map

For every claim in the 7 Learning Arcs, the full evidence chain is:

```
Published claim (README / dashboard)
  └─► Chart in dashboard (index.html: claim → chart anchor link)
        └─► Specific run_ids cited in chart tooltip
              └─► results/runs.jsonl rows (filter by run_id)
                    └─► results/run_logs/<run_id>/
                          ├── raw_timings.jsonl     every pass timing (statistical claims)
                          ├── profiles/*.pb          profiler traces (compiler claims)
                          ├── hlo_dump.txt           XLA HLO (fusion claims)
                          ├── memory_timeline.json   memory claims
                          ├── numerics.json          precision / accuracy claims
                          ├── system_state.json      hw utilisation claims
                          └── lineage.json           git SHA, model revision, seed
```

Every claim is reproducible: given the `run_id`, `git_sha`, `hf_model_revision`, and
`input_seed` from `lineage.json`, any experiment can be re-run to verify the result.
