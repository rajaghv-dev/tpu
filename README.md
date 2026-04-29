# TPU × GPU Inference Benchmark

> Rigorous, reproducible inference benchmarks comparing Google TPU and NVIDIA GPU across 75 models, 5 execution paths, and 7 learning arcs.

---

## What This Repo Is

A benchmark harness that answers one question from every angle:

**"Should I run inference on a TPU or a GPU — and why?"**

Not a vague "TPUs are faster" claim. Concrete numbers, per model, per precision, per compiler strategy, across vision / NLP / audio / multimodal — with enough depth to understand *why* the numbers are what they are at the microarchitecture level.

---

## Hardware Landscape

### Local GPUs (zero marginal cost)
| Card | VRAM | Mem BW | Peak BF16 | Compute Gen |
|------|------|--------|-----------|-------------|
| RTX 3080 | 16 GB | 760 GB/s | 119 TFLOPs | Ampere (2:4 sparsity) |
| RTX 4090 | 24 GB | 1008 GB/s | 330 TFLOPs | Ada Lovelace (FP8) |
| **B200 SXM (DGX Dell)** | **192 GB HBM3e** | **4000 GB/s** | **2250 TFLOPs** | **Blackwell (6th-gen TC)** |

### Cloud TPU — Accessible Single-Chip VMs
| TPU | HBM | Mem BW | Peak BF16 | Preemptible | Single-chip VM |
|-----|-----|--------|-----------|-------------|----------------|
| v2 chip | 8 GB | 600 GB/s | 45 TFLOPs | $1.35/hr (8-chip min) | No |
| v3 chip | 16 GB | 900 GB/s | 123 TFLOPs | $2.40/hr (8-chip min) | No |
| v4 chip | 32 GB | 1200 GB/s | 275 TFLOPs | $3.90/hr (8-chip min) | No |
| **v5e-1** | **16 GB** | **820 GB/s** | **394 TFLOPs** | **$0.36/hr** | **Yes** |
| v5p chip | 95 GB | 2765 GB/s | 459 TFLOPs | $9.60/hr (8-chip min) | No |
| **v6e-1 (Trillium)** | **32 GB** | **1640 GB/s** | **918 TFLOPs** | **~$0.75/hr** | **Yes** |

**Primary single-card pairing:** v5e-1 ↔ RTX 3080 (both 16 GB).
**Secondary pairing:** v6e-1 ↔ RTX 4090 (32 GB vs 24 GB).

### Max Model Size Per TPU Chip (Inference, BF16)
| Chip | Comfortable | Tight (bs=1) | INT8 |
|------|-------------|--------------|------|
| v2 (8 GB) | 1.5B | 2.5B | 5B |
| v3 / v5e (16 GB) | 3B | 5B | 10B |
| v4 / v6e (32 GB) | 7B | 10B | 20B |
| v5p (95 GB) | 30B | 38B | 70B |

---

## The 5 Execution Paths

```
            Same pretrained weights (HuggingFace)
                         |
         ┌───────────────┼──────────────────┐
      JAX/Flax        JAX/Flax           PyTorch
         │                │                  │
    ┌────▼────┐     ┌──────▼─────┐    ┌──────┴─────┐     ┌──────────────┐
    │ Path 1  │     │  Path 2    │    │  Path 3    │     │   Path 4     │
    │JAX+TPU  │     │ JAX+GPU    │    │ PyTorch+   │     │ PyTorch+     │
    │  (XLA)  │     │  (CUDA)    │    │   GPU      │     │ torch_xla    │
    └─────────┘     └────────────┘    └────────────┘     └──────────────┘
                                                          (TPU via XLA)

    Path 5: HuggingFace Inference API  (managed, serverless, zero infra)
```

| Comparison | What it isolates |
|-----------|-----------------|
| Path 1 vs 2 | Pure hardware: TPU vs GPU (JAX/XLA held constant) |
| Path 2 vs 3 | Pure framework: JAX vs PyTorch (GPU held constant) |
| Path 1 vs 4 | Pure framework on TPU: JAX vs torch_xla |
| Path 3 vs 4 | Pure compiler: CUDA vs XLA (PyTorch API held constant) |
| Path 1 vs 3 | Real-world: production TPU vs production GPU |
| Path 5 vs all | Managed serving overhead vs self-hosted |

---

## Quick Start

### Run the benchmark (Stage 1 — JAX + TPU, 5 models)

```bash
# Clone and install
git clone https://github.com/rajaghv-dev/tpu && cd tpu
pip install -r requirements.txt        # includes jax[tpu], transformers, pytest

# Smoke test — 1 model (BERT-base), BF16, ~8 min on v5e-1
python benchmarks/harness.py --suite smoke --device tpu

# Quick suite — all 5 Stage 1 models, BF16, ~50 min on v5e-1
python benchmarks/harness.py --suite quick --device tpu

# Preview what would run (no model downloads)
python benchmarks/harness.py --suite quick --device tpu --dry-run

# Single model on local GPU
python benchmarks/harness.py --model bert_base --device gpu
```

Results append to `results/runs.jsonl`. Dashboard at `results/dashboard/index.html`.

### Run tests (no GPU/TPU needed)

```bash
pip install pytest pyyaml numpy
pytest tests/ -v              # 97 tests across stats, lineage, registry, runner, harness
```

### Google Colab Pro

```python
# TPU runtime: Runtime → Change runtime type → TPU
!git clone https://github.com/rajaghv-dev/tpu && cd tpu
import os; os.environ['HF_TOKEN'] = 'your_token'   # for gated models (Gemma etc.)
!pip install -r requirements.txt
!python benchmarks/harness.py --suite smoke --device tpu
```

---

## Benchmark Harness — Stage 1

| File | Purpose |
|------|---------|
| `benchmarks/harness.py` | CLI entry point — suite runner, JSONL writer |
| `benchmarks/runner.py` | Single experiment (9-phase protocol) |
| `models/registry.yaml` | 5 Stage 1 models with full input specs |
| `observe/stats.py` | MAD-based outlier removal, p50/p95/p99, CV check |
| `observe/lineage.py` | Git SHA + package versions + HF model revision |
| `observe/compile_controller.py` | XLA cache clearing, cold + warm compile timing |
| `results/runs.jsonl` | Append-only result log (one JSON per experiment) |
| `results/dashboard/index.html` | Static sortable/filterable table dashboard |
| `tests/` | 97 unit tests (pytest, no GPU required) |

**Stage 1 models:** BERT-base · ViT-B/16 · GPT-2 · Whisper-base · CLIP ViT-B/32

**Stage 1 gaps fixed:** C2 (multi-run statistics with CV<10% check) · C3 (XLA cache cleared before every compile measurement)

**Next:** Stage 2 adds Paths 2+3 (JAX+GPU, PyTorch+GPU), system_monitor.py, 15 models, heatmap dashboard.

---

## Model Registry — 75 Models Across 9 Domains

### Vision — Classification / Feature Extraction (10 models)
| Model | Params | Architecture | Key benchmark story |
|-------|--------|-------------|-------------------|
| ResNet-50 | 86M | Conv + BN + Residual | Conv-heavy baseline |
| ViT-B/16 | 86M | Pure self-attention | Best TPU fit in class |
| ViT-L/16 | 307M | Pure self-attention | Scales well on MXU |
| DINOv2-L | 307M | ViT + self-supervised | SSL features, same compute as ViT-L |
| SigLIP-B/16 | 400M | ViT + sigmoid loss | Google CLIP replacement |
| ConvNeXt-XL | 350M | Depthwise conv (modern) | No attention — pure conv story |
| EfficientNet-B7 | 66M | Depthwise + SE blocks | Few FLOPs, kills MXU — key insight |
| EfficientViT-L3 | 246M | Linear attn + conv | Efficient hybrid |
| SAM-L | 312M | ViT + mask decoder | Segment-anything backbone |
| EVA-02-L | 307M | CLIP-pretrained ViT | Improved ViT baseline |

### Vision — Object Detection (2 models)
| Model | Params | Architecture | Key benchmark story |
|-------|--------|-------------|-------------------|
| DETR-ResNet50 | 41M | Transformer decoder | End-to-end transformer detection |
| RT-DETR-L | 32M | RT detection transformer | Real-time detection transformer |

### Vision — Generative / Diffusion (2 models)
| Model | Params | Architecture | Key benchmark story |
|-------|--------|-------------|-------------------|
| DiT-XL/2 | 675M | Pure transformer diffusion | TPU's best diffusion story — pure matmuls |
| SD-UNet | 860M | U-Net + cross-attention | Conv+attn hybrid — shows XLA fusion on U-Net |

### NLP — Encoders (7 models)
| Model | Params | Architecture | Key benchmark story |
|-------|--------|-------------|-------------------|
| BERT-base | 110M | Transformer encoder | Historical baseline |
| RoBERTa-large | 355M | BERT variant | Better pretraining recipe |
| DeBERTa-v3-large | 400M | Disentangled attention | SOTA encoder |
| ModernBERT-base | 149M | Flash attn + RoPE | Dec 2024 — 8192 context, faster than BERT |
| ModernBERT-large | 395M | Flash attn + RoPE | Latest encoder SOTA |
| BGE-large-en-v1.5 | 335M | BERT-based | Dense retrieval / RAG — huge batch sizes |
| E5-large-v2 | 335M | BERT-based | Universal embedding — memory BW ceiling test |

### NLP — Decoders / LLMs (12 models)
| Model | Params | Architecture | Key benchmark story |
|-------|--------|-------------|-------------------|
| GPT-2 XL | 1.5B | Original transformer | Historical baseline |
| TinyLlama-1.1B | 1.1B | LLaMA + GQA | Efficient small LLM |
| SmolLM2-1.7B | 1.7B | LLaMA arch | HF flagship small model |
| OLMo-2-1B | 1.2B | Fully open arch | Reproducible research baseline |
| Gemma-2B | 2B | MQA + RoPE | Google co-designed for TPU — home turf |
| Gemma-2-2B | 2.6B | GQA + sliding window | Complex attention, TPU-native |
| **RecurrentGemma-2B** | 2B | **RNN-hybrid (RGLRU)** | Same params as Gemma-2B, recurrence vs attention |
| Phi-2 | 2.7B | Transformer | Punches above weight class |
| Phi-3-mini-4k | 3.8B | LLaMA arch | Near 4B ceiling |
| Qwen2.5-3B | 3B | GQA + RoPE | Strong multilingual |
| StarCoder2-3B | 3B | Code + infill | 16k context, non-causal mask |
| CodeGemma-2B | 2B | Gemma backbone | Google code model — pairs with Gemma-2B |

### NLP — Novel Architectures / Non-Transformer (3 models)
| Model | Params | Architecture | Key benchmark story |
|-------|--------|-------------|-------------------|
| **Mamba-2.8B** | 2.8B | **SSM — selective scan** | Custom CUDA kernel; no XLA primitive — biggest GPU advantage |
| **Mamba2-2.7B** | 2.7B | Structured SSM | Improved Mamba |
| **RWKV-4-3B** | 3B | Linear RNN | O(1) memory inference; sequential = bad for systolic arrays |

### Audio (8 models)
| Model | Params | Architecture | Key benchmark story |
|-------|--------|-------------|-------------------|
| Whisper-base | 74M | Conv + enc-dec transformer | Small, fast audio baseline |
| Whisper-medium | 307M | Conv + enc-dec transformer | Balanced |
| Whisper-large-v3 | 1.5B | Conv + enc-dec transformer | Best quality; tests large audio model |
| wav2vec2-large | 317M | Conv feature extractor + transformer | Heavy conv front-end |
| HuBERT-large | 316M | Conv + transformer | SSL audio |
| SeamlessM4T-medium | 1.2B | Multi-task speech | Translation pipeline |
| MMS-1B | 1B | wav2vec2-based | Multilingual — 1000+ languages |
| EnCodec-24kHz | 44M | Conv + LSTM | Neural audio codec — LSTM sequential story |

### Multimodal — Vision-Language (7 models)
| Model | Params | Architecture | Key benchmark story |
|-------|--------|-------------|-------------------|
| CLIP ViT-L/14 | 428M | Dual encoder | Contrastive baseline |
| SigLIP-SO400M | 400M | ViT + sigmoid loss | Google replacement for CLIP |
| moondream2 | 1.86B | SigLIP + Phi | Tiny capable VLM |
| SmolVLM-2B | 2B | SigLIP + SmolLM | Very new (Dec 2024) |
| **PaliGemma-3B** | **3B** | **SigLIP + Gemma** | **Google VLM — all three components Google-built** |
| LLaVA-Phi3 | ~4B | CLIP + Phi-3-mini | Community VLM at 4B ceiling |
| ImageBind | ~1.2B | 6-modality encoder | Audio+image+text+depth — multi-path forward pass |

### Long-Context Variants (2 models)
| Model | Params | Context | Key benchmark story |
|-------|--------|---------|-------------------|
| Phi-3-mini-128k | 3.8B | 128k tokens | Same model as Phi-3-mini-4k — shows quadratic attention wall |
| E5-Mistral-7B | 7B | 4k | LLM-as-embedder — v6e-1 only; decoder used for retrieval |

---

## Experiment Dimensions

### Precision
| Format | TPU | GPU | Key insight |
|--------|-----|-----|-------------|
| FP32 | ✅ | ✅ | Baseline |
| BF16 | ✅ **Native (= FP32 speed)** | ✅ Via autocast (~2× faster) | TPU: free; GPU: a choice |
| INT8 | ✅ v5+ | ✅ Tensor Cores | ~1.3× TPU, ~2× GPU |
| INT4 | ❌ | ✅ bitsandbytes | GPU-exclusive advantage |
| FP8 | ✅ v5e/v6e | ✅ 4090/H100 | Frontier format |

### Compilation Strategy
| Mode | JAX | PyTorch | What you learn |
|------|-----|---------|----------------|
| Eager | Disabled jit | Default | Baseline overhead |
| Standard compiled | `jax.jit` | `torch.compile(default)` | Production mode |
| Max-optimised | `jax.jit` (always) | `torch.compile(max-autotune)` | Peak ceiling |
| CUDA Graphs | — | `make_graphed_callables` | GPU kernel-replay at bs=1 |

### Sparsity / Pruning
| Variant | TPU result | GPU result | Key insight |
|---------|-----------|-----------|-------------|
| Dense | Baseline | Baseline | — |
| Unstructured 50% | ~0% faster | ~0% faster | Zeros still computed |
| Unstructured 90% | Slightly **slower** | Marginally faster | Bad for both |
| 2:4 structured | ❌ No hardware support | **2× faster** (Sparse Tensor Cores) | NVIDIA-exclusive |

---

## The 7 Learning Arcs

Each arc is a self-contained benchmark run that produces one durable insight.

| Arc | Theme | Core insight |
|-----|-------|-------------|
| 1 | **Hardware DNA** | Same model is compute-bound on v5e, memory-bound on 4090 — different chips, different regimes |
| 2 | **The Compiler is the Hardware** | XLA fuses LayerNorm+GeLU+residual into one kernel by default; PyTorch needs torch.compile to match |
| 3 | **Architecture-Hardware Fit** | EfficientNet is slower than ViT on TPU despite fewer FLOPs — depthwise conv starves the systolic array |
| 4 | **The Precision Story** | BF16 is free on TPU (same clock speed as FP32); on GPU it activates Tensor Cores for a 2× gain |
| 5 | **Beyond Transformers** | Mamba is faster than GPT-2 XL on GPU (custom CUDA scan kernel); slower on TPU (no XLA primitive) |
| 6 | **The Sparsity Paradox** | Unstructured pruning hurts runtime on TPU; NVIDIA 2:4 structured sparsity gives GPU a genuine 2× win |
| 7 | **Real-World TCO** | At production scale, preemptible v6e-1 cost-per-sample beats on-demand 4090 once utilisation > 40% |

---

## Per-Experiment Protocol (~1–3 min each)

```
1. Compile/trace   →  1 pass    Record compile_time_s (XLA trace or torch.compile)
2. Warmup          → 20 passes  Discard — kernels and caches stabilise
3. Latency sweep   → 50 passes  batch_size=1  →  p50 / p95 / p99 ms
4. Throughput run  → 50 passes  max batch_size  →  samples/sec
5. Profile         →  5 passes  Full trace  →  op breakdown, roofline data
6. Memory sweep    →  1 pass/bs  bs=1,2,4,8,…  →  max batch before OOM
```

---

## Suite Definitions

| Suite | Models | Variants | Experiments | Wall time | v5e-1 cost |
|-------|--------|----------|-------------|-----------|-----------|
| `smoke` | 1 (BERT-base) | FP32 + BF16 | 4 | ~8 min | $0.05 |
| `quick` | 6 (1/domain) | BF16 only | 24 | ~50 min | $0.30 |
| `domain` | All in one domain | FP32 + BF16 | ~30 | ~60 min | $0.36 |
| `arch` | Novel arches (Mamba, RWKV, DiT, RecurrentGemma) | BF16 | ~20 | ~40 min | $0.24 |
| `full` | All 53 models | All variants | ~800 | ~7 hrs | $2.52 |

Full suite = **$2.52** on preemptible v5e-1. Can be run weekly for ~$10/month.

---

## Results & Dashboard

All runs append to `results/runs.jsonl` (one JSON object per experiment). A static HTML dashboard (`results/dashboard/index.html`) renders comparison charts with Vega-Lite — no server required, works on GitHub Pages.

### Dashboard Views
- **Roofline scatter** — arithmetic intensity vs achieved TFLOPs, hardware roofline overlaid
- **Throughput heatmap** — model × device, colour = samples/sec
- **Latency fan** — bs=1 latency per model, all devices on same chart
- **Compile cost** — XLA trace vs torch.compile, first-run vs amortised
- **Precision speedup** — BF16/FP32 ratio per device (should be ~1.0 on TPU, ~2.0 on GPU)
- **Sparsity impact** — dense vs pruning variants per device
- **TCO calculator** — cost/1k samples across all paths

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Cache model weights (one-time, ~30 min)
```bash
# Downloads all models to GCS for fast VM restarts
HUGGINGFACE_TOKEN=<your-token> ./scripts/cache_models_gcs.sh gs://your-bucket/models
```

### 3. Provision a TPU VM
```bash
./scripts/provision_tpu.sh tpu-bench us-central1-a v5e-1
```

### 4. Run a suite
```bash
# On TPU VM (Path 1 — JAX)
python benchmarks/harness.py --suite=smoke --framework=jax --device=tpu

# On local GPU (Path 3 — PyTorch)
python benchmarks/harness.py --suite=smoke --framework=pytorch --device=gpu

# Via HuggingFace API (Path 5)
python benchmarks/harness.py --suite=smoke --framework=hf_api
```

### 5. Sync results to GitHub
```bash
./scripts/sync_github.sh
```

### 6. View dashboard locally
```bash
cd results/dashboard && python -m http.server 8080
# open http://localhost:8080
```

---

## Google Colab

Colab Pro gives access to TPU v2-8 (8×8 GB chips) and A100/V100 GPUs.
Run any suite from a notebook cell:

```python
# In a Colab cell (TPU runtime selected)
!python benchmarks/harness.py --suite=quick --framework=jax --device=tpu
```

Mount Google Drive to persist model weights between sessions:
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'
```

---

## Repository Layout

```
tpu-gpu-inference-bench/
│
├── 01_hello_tpu/               # Verify TPU setup (JAX)
├── 02_mnist_classification/    # CNN + pmap — first training example
├── 03_resnet_imagenet/         # ResNet-50 with cosine LR + checkpointing
├── 04_bert_finetuning/         # BERT fine-tune on GLUE/SST-2
├── 05_gpt_pretraining/         # GPT-2-scale pre-training from scratch
├── 06_data_pipeline/           # tf.data best-practices, GCS TFRecord
├── 07_custom_training_loop/    # Gradient accumulation, bf16, per-layer LR
├── 08_multi_host/              # Multi-host TPU pod (jax.distributed)
│
├── benchmarks/                 # [IN PROGRESS] Inference benchmark harness
│   ├── harness.py              # CLI entry point
│   ├── runner.py               # 1 experiment → metrics dict
│   ├── models/
│   │   ├── registry.yaml       # 53 models: HF IDs, input specs, bs limits
│   │   ├── jax/                # Flax implementations (Path 1 + 2)
│   │   └── torch/              # PyTorch implementations (Path 3 + 4)
│   ├── variants/               # precision.py, pruning.py, compile.py
│   └── profiler/               # jax_profiler.py, torch_profiler.py, roofline.py
│
├── configs/
│   ├── hardware.yaml           # Peak FLOPs, BW, memory per device
│   └── suites/                 # smoke / quick / domain / arch / full
│
├── results/
│   ├── runs.jsonl              # Append-only results log
│   └── dashboard/              # Static Vega-Lite charts
│
├── scripts/
│   ├── gcloud_setup.sh         # Enable APIs, set GCP project
│   ├── provision_tpu.sh        # Create preemptible TPU VM
│   ├── gcloud_ssh_run.sh       # Run example on remote VM
│   ├── gcloud_upload_data.sh   # Upload data to GCS
│   ├── gcloud_pod_run.sh       # Multi-host pod launch
│   ├── cache_models_gcs.sh     # HuggingFace → GCS (one-time)
│   ├── run_suite.sh            # Run suite + auto-push results
│   ├── sync_github.sh          # Commit + push results/runs.jsonl
│   └── teardown_tpu.sh         # Delete VM (stop billing)
│
├── context.md                  # Full project context and design decisions
├── prompts.md                  # Running log of user prompts
└── requirements.txt
```

---

## Cost Reference

| Scenario | Cost |
|----------|------|
| Single experiment (~2 min) on v5e-1 preemptible | $0.012 |
| `quick` suite (50 min) | $0.30 |
| `full` suite (~7 hrs) | $2.52 |
| Weekly `full` suite for 1 month | ~$10 |
| GCS model cache (~80 GB) | $1.60/month |
| Colab Pro (TPU + GPU access) | $9.99/month |

---

## Status

| Component | Status |
|-----------|--------|
| Training examples (01–08) | Complete |
| gcloud scripts | Complete |
| Benchmark harness | Designing |
| Model registry (53 models) | Designing |
| Profiler + roofline | Designing |
| Results dashboard | Designing |
| HF API path (Path 5) | Designing |

---

## License

MIT
