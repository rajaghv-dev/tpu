# Project Context — TPU × GPU Inference Benchmark

Single-repo reference for hardware, compiler, model inference, and quantization
benchmarking across TPU and GPU. Every claim is evidence-backed.
Last updated: 2026-04-26

---

## 1. Project Goal

Build a rigorous, reproducible inference benchmark that teaches hardware microarchitecture,
compiler behaviour, and model-hardware fit from first principles — with measurable, traceable
evidence for every claim. Covers TPU vs GPU across 75+ models, 5 execution paths,
9 experiment dimensions, and 7 learning arcs from beginner to expert depth.

---

## 2. Hardware Inventory

### 2.1 Local Hardware (zero marginal cost)

| Card | Architecture | VRAM / HBM | Mem BW | Peak BF16 | Peak FP8 | Notes |
|------|-------------|-----------|--------|-----------|----------|-------|
| RTX 3080 | Ampere (GA102) | 16 GB GDDR6X | 760 GB/s | 119 TFLOPs | — | 2:4 structured sparsity (Sparse TC) |
| RTX 4090 | Ada Lovelace (AD102) | 24 GB GDDR6X | 1008 GB/s | 330 TFLOPs | 660 TFLOPs | FP8 Tensor Cores, highest consumer BW |
| **B200 SXM (DGX Dell, 1 card)** | **Blackwell (GB202)** | **192 GB HBM3e** | **4000 GB/s** | **2250 TFLOPs** | **4500 TFLOPs** | **6th-gen TC, NVLink 5.0, 1000W TDP** |

**DGX system details:** Dell PowerEdge server, Blackwell GPU, 256 GB system CPU RAM.
Single-card use for benchmarking. B200 HBM3e at 4 TB/s is 2× A100's bandwidth.
B200 BF16 peak is **6.8× RTX 4090** and **7.2× A100 SXM4**.

**What changes with B200 vs A100 (our earlier assumption):**
- Memory: 192 GB vs 80 GB → can run 70B models in BF16 comfortably (140 GB weights)
- BW: 4 TB/s vs 2 TB/s → memory-bound ops (attention decode, LayerNorm) ~2× faster
- Compute: 2250 vs 312 TFLOPs BF16 → compute-bound ops (FFN matmuls) ~7× faster
- This is the most capable single GPU on the planet today

### 2.2 Cloud TPU — Accessible Single-Chip VMs

| TPU | HBM | Mem BW | Peak BF16 | Preemptible | Single-chip VM |
|-----|-----|--------|-----------|-------------|----------------|
| v2 chip | 8 GB | 600 GB/s | 45 TFLOPs | $1.35/hr (8-chip min) | No |
| v3 chip | 16 GB | 900 GB/s | 123 TFLOPs | $2.40/hr (8-chip min) | No |
| v4 chip | 32 GB | 1200 GB/s | 275 TFLOPs | $3.90/hr (8-chip min) | No |
| **v5e-1** | **16 GB** | **820 GB/s** | **394 TFLOPs** | **$0.36/hr** | **Yes — primary** |
| v5p chip | 95 GB | 2765 GB/s | 459 TFLOPs | $9.60/hr (8-chip min) | No |
| **v6e-1 (Trillium)** | **32 GB** | **1640 GB/s** | **918 TFLOPs** | **~$0.75/hr** | **Yes — secondary** |

**Primary pairing:** v5e-1 (16 GB) ↔ RTX 3080 (16 GB) — identical memory, different compute arch.
**Secondary pairing:** v6e-1 (32 GB) ↔ RTX 4090 (24 GB) — TPU has more memory here.
**Research pairing:** v5e-1 ↔ B200 — David vs Goliath; shows architecture matters more than raw FLOPs.

### 2.3 Max Model Size Per Card (Inference, BF16)

| Card | HBM | Comfortable | Tight (bs=1) | INT8 | FP8 |
|------|-----|-------------|--------------|------|-----|
| v5e chip | 16 GB | 3B | 5B | 10B | 20B |
| v6e chip | 32 GB | 7B | 10B | 20B | 40B |
| RTX 3080 | 16 GB | 3B | 5B | 10B | — |
| RTX 4090 | 24 GB | 5B | 8B | 16B | 32B |
| B200 SXM | 192 GB | **70B** | **96B** | **192B** | **384B** |

### 2.4 Cloud GPU Cost Comparison (Single Card, Inference)

#### GCP (most relevant — same cloud as TPU)
| Instance | GPU | HBM | Mem BW | BF16 TFLOPs | On-demand/hr | Preemptible/hr |
|----------|-----|-----|--------|-------------|-------------|----------------|
| v5e-1 (TPU) | v5e chip | 16 GB | 820 GB/s | 394 | $1.20 | **$0.36** |
| v6e-1 (TPU) | v6e chip | 32 GB | 1640 GB/s | 918 | ~$2.50 | **~$0.75** |
| a3-highgpu-1g | H100 SXM5 | 80 GB | 3350 GB/s | 989 | $3.67 | ~$1.10 |
| a3-megagpu-1g | H100 NVL | 94 GB | 3350 GB/s | 1109 | ~$4.50 | ~$1.35 |
| a4-highgpu-1g | B200 SXM | 192 GB | 4000 GB/s | 2250 | ~$8.00 | ~$2.40 |

#### AWS (per GPU, extracted from multi-GPU instances)
| Instance | GPU | Per-GPU cost (on-demand) | Per-GPU spot |
|----------|-----|--------------------------|--------------|
| p4d.24xlarge ÷ 8 | A100 40GB | ~$4.00/hr | ~$1.50/hr |
| p4de.24xlarge ÷ 8 | A100 80GB | ~$5.00/hr | ~$1.80/hr |
| p5.48xlarge ÷ 8 | H100 SXM5 80GB | ~$12.30/hr | **~$3.50/hr** |
| p5e.48xlarge ÷ 8 | H100 NVL 94GB | ~$14/hr | ~$4.20/hr |
| p6 (Blackwell, rolling out) | B200 192GB | ~$20/hr est. | ~$6/hr est. |

#### Specialist GPU clouds (H100 available now, accessible from India)
| Provider | GPU | Cost/hr | Notes |
|----------|-----|---------|-------|
| Lambda Labs | H100 SXM5 80GB | **$2.49** | Most affordable H100 |
| Lambda Labs | H100 NVL 94GB | $1.99 | Multi-node optimised |
| Lambda Labs | A100 SXM4 80GB | $1.29 | Good for ≤40B BF16 |
| CoreWeave | H100 SXM5 80GB | $2.79 | Low-latency US/EU |
| Vast.ai | H100 (market) | $1.50–$2.50 | Spot-like auction |
| RunPod | H100 SXM5 | $2.49–$3.49 | India-accessible |

#### Cost per 1k samples — single model BERT-base BF16 (estimated)
| Hardware | Throughput | Cost/hr | Cost/1k samples |
|----------|-----------|---------|----------------|
| v5e-1 preemptible | ~5,000/s | $0.36 | **$0.000020** |
| v6e-1 preemptible | ~9,000/s | $0.75 | $0.000023 |
| RTX 3080 (local) | ~3,000/s | $0 (local) | ~$0 |
| H100 SXM5 (Lambda) | ~12,000/s | $2.49 | $0.000058 |
| B200 SXM (local) | ~28,000/s | $0 (local) | ~$0 |
| H100 (GCP preemptible) | ~12,000/s | $1.10 | $0.000025 |

**Key insight:** Local B200 + local RTX cards cost nothing to run. Cloud TPU at $0.36/hr often
beats cloud H100 at $2.49/hr on cost-per-sample despite lower peak TFLOPs — throughput
advantage doesn't fully compensate for price difference.

### 2.5 Access from India — Full Guide

**Can you access all hardware from India? YES, with no legal or technical restrictions.**

| Service | India Region? | Recommended Region | Payment | Notes |
|---------|--------------|-------------------|---------|-------|
| GCP TPU | ❌ No TPU in India | us-central1 (Iowa) | Indian credit card, UPI via Google Pay | $300 free trial for new accounts |
| GCP GPU (H100) | ❌ No H100 in India | us-central1 / europe-west4 | Same | a3-highgpu available in US/EU |
| GCP GPU (A100) | ⚠️ Limited (asia-south1) | asia-south1 or us-central1 | Same | a2 instances in Mumbai |
| AWS GPU | ⚠️ V100 in Mumbai | us-east-1 / us-west-2 for H100 | International card or AWS India | P5 (H100) only in US regions |
| Lambda Labs | ❌ No India DC | US datacenters | International card | Best H100 price; low latency irrelevant for batch |
| RunPod | ❌ No India DC | EU / US | International card | Good for spot-style H100 |
| Colab Pro | ✅ Works from India | GCP backend, auto-assigned | Indian card via Google Pay | Best starting point |

**Regulatory / payment:**
- Foreign currency cloud spend = import of services under FEMA; fully permitted for individuals
- GST (18%) applies if using GST-registered Indian billing; not applicable for personal use with foreign card
- RBI FEMA regulations allow cloud service payments without limit for personal/research use
- Easiest payment: add an international credit/debit card to GCP/AWS; or use Google Pay UPI for GCP credits

**Practical latency:**
- Mumbai → us-central1: ~230ms RTT. Only affects SSH terminal feel, not benchmark accuracy.
- For batch benchmark runs (SSH in, run script, results to GCS): latency is irrelevant.
- Use tmux on the remote VM so network drops don't kill your run.

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

**Direct vs Proxy comparability:** A *direct* comparison varies exactly one dimension while
holding the rest (framework, compiler, model weights, input seed) constant — the result is
attributable to that single variable. A *proxy* comparison varies two things by necessity
(e.g., framework and compiler co-vary because PyTorch+CUDA on GPU vs JAX+XLA on TPU is the
real-world choice), but one dimension is controlled by design or convention so the result is
still meaningful. Use direct pairs for causal claims; use proxy pairs for "real world" claims.

| Comparison pair | Isolates | Comparability |
|----------------|---------|---------------|
| Path 1 vs 2 | Hardware only: TPU vs GPU, JAX/XLA held constant | **Direct** |
| Path 2 vs 3 | Framework: JAX vs PyTorch, GPU and model held constant | **Direct** |
| Path 1 vs 4 | Framework on TPU: JAX/XLA vs torch_xla | **Direct** |
| Path 3 vs 4 | Compiler on TPU: CUDA vs XLA, PyTorch API held constant | **Proxy** (compiler differs but hardware also differs — TPU vs GPU; controlled by using same PT API) |
| Path 1 vs 3 | Real-world: production TPU vs production GPU | **Proxy** (framework + compiler + hardware all differ; controlled by using same model weights + same input seed) |
| Path 5 vs all | Managed serving overhead vs self-hosted | **Proxy** (network + scheduling + autoscaling are bundled; controlled by warm-endpoint + identical model revision) |

---

## 4. Full Model Registry with Benchmark Justification (~75 Models)

Each model is listed with: params, architecture family, what compute pattern it exercises,
what specific TPU vs GPU insight it produces, and what you learn from running it.

### 4.1 Vision — Classification / Feature Extraction (modality: image)

| Model | Params | Architecture | Primary compute | What you learn |
|-------|--------|-------------|----------------|---------------|
| **ResNet-50** | 86M | Conv + BN + Residual | 3×3 conv, 1×1 conv, BN | Baseline. cuDNN selects Winograd/FFT conv algorithms; XLA pads to multiples of 128. Measures how close XLA's algorithm selection is to cuDNN's auto-tuner. |
| **ConvNeXt-XL** | 350M | Depthwise 7×7 + LayerNorm | Depthwise conv, large kernel | Modern pure conv with no attention. 7×7 depthwise is worse for systolic arrays than 3×3 (bigger tiles but still 1-channel depth). LN instead of BN helps XLA fusion. |
| **EfficientNet-B7** | 66M | Depthwise 3×3 + SE blocks | Depthwise conv (serial), SE (2 matmuls) | **Key insight model.** Fewer FLOPs than ResNet-50 but slower on TPU. Depthwise = 1-channel matmul tiles, starves 128×128 MXU. SE attention head = 2 tiny matmuls. Quantifies MXU starvation. |
| **EfficientViT-L3** | 246M | Linear attention + depthwise | Linear attention (O(n)), depthwise | Hybrid: linear attention avoids quadratic scaling but uses elementwise multiply instead of matmul. Tests whether XLA can fuse linear attention ops as well as cuDNN. |
| **ViT-B/16** | 86M | Pure self-attention | QKV projections, attn, MLP | Same param count as ResNet-50, completely different compute. Pure matmul = excellent MXU fit. Compares MXU % (should be >70%) vs SM% on GPU. |
| **ViT-L/16** | 307M | Pure self-attention | Same as ViT-B, larger | Larger matmul tiles → higher MXU efficiency. Tests whether bigger = more efficient per FLOP. |
| **DINOv2-L** | 307M | ViT + self-supervised | Same as ViT-L at inference | Identical compute to ViT-L but trained differently. Demonstrates inference compute is architecture-determined, not training-procedure-determined. |
| **SigLIP-B/16** | 400M | ViT + sigmoid BCE loss | ViT compute + sigmoid output | Google's CLIP replacement. No softmax normalisation across batch. Tests whether removing the cross-batch normalisation changes compute profile. |
| **SAM-L** | 312M | ViT + prompt encoder + mask decoder | ViT encode + 2-stage decode | Two forward passes per inference (image encoder + prompt-conditioned decoder). Tests multi-pass inference overhead and intermediate activation storage. |
| **EVA-02-L** | 307M | CLIP-pretrained ViT | Same as ViT-L | ViT variant with CLIP pretraining. Compute identical; shows that model family matters less than architecture class for hardware benchmarking. |

### 4.2 Vision — Object Detection (modality: image)

| Model | Params | Architecture | What you learn |
|-------|--------|-------------|---------------|
| **DETR-ResNet50** | 41M | CNN backbone + Transformer decoder | End-to-end transformer detection. ResNet extracts features (conv path), transformer decoder attends to all spatial positions (attention path). Tests mixed conv+attention pipeline. |
| **RT-DETR-L** | 32M | Hybrid encoder + transformer | Real-time DETR with efficient encoder. Shows whether architectural efficiency improvements (designed for GPU) translate equally to TPU. |

### 4.3 Vision — Generative / Diffusion (modality: image)

| Model | Params | Architecture | What you learn |
|-------|--------|-------------|---------------|
| **DiT-XL/2** | 675M | Pure transformer (no U-Net) | Diffusion via transformer = pure matmuls. Inference = 50 denoising steps. XLA compile once, run 50× — amortises compile cost. TPU's strongest diffusion story. Compare vs SD-UNet: same quality tier, radically different hardware fit. |
| **Stable Diffusion UNet** | 860M | U-Net + cross-attention | Conv-heavy backbone with attention at bottleneck. Cross-attention between image features and text embeddings = large batched matmul. Shows XLA fusion on skip-connection conv graphs. |

### 4.4 NLP — Encoders (modality: text)

| Model | Params | Architecture | What you learn |
|-------|--------|-------------|---------------|
| **BERT-base** | 110M | Transformer encoder, absolute position | Historical baseline. Bidirectional attention (no causal mask). Standard for NLP latency/throughput reference. seq_len=128 and 512 reveal attention's quadratic scaling. |
| **RoBERTa-large** | 355M | BERT variant, dynamic masking | Same architecture, different training. Confirms that training procedure doesn't affect inference compute profile — architecture determines hardware behaviour. |
| **DeBERTa-v3-large** | 400M | Disentangled attention (content + position separate) | Two separate attention computations per layer (content-to-content, content-to-position). Nearly 2× the attention FLOPs of BERT. Tests whether XLA fuses the two attention streams. |
| **ModernBERT-base/large** | 149M / 395M | Flash attention + RoPE + 8192 ctx | December 2024 model. Flash Attention 2 reduces memory from O(n²) to O(n). Tests how XLA's memory-efficient attention (Splash Attention) competes with GPU's optimised FlashAttention 2 CUDA kernel at long context lengths. |
| **BGE-large-en-v1.5** | 335M | BERT-based encoder | Dense retrieval (RAG). Inference = process 10,000+ documents in large batches. Tests memory bandwidth ceiling: batch sizes of 256–1024 are common here. Shows where TPU's larger static batches shine. |
| **E5-large-v2** | 335M | BERT-based encoder | Universal embedding. Identical compute to BGE. Running both shows run-to-run reproducibility of the benchmark. |
| **nomic-embed-v1.5** | 137M | BERT + RoPE + Matryoshka | Uses RoPE (rotary position) instead of absolute. Matryoshka training means you can truncate output dimension at inference. Tests RoPE computation overhead vs absolute position. |

### 4.5 NLP — Decoders (Autoregressive LLMs) (modality: text)

Decoders have TWO distinct inference modes that must be measured separately:
- **Prefill**: process all prompt tokens simultaneously — compute-bound, high arithmetic intensity
- **Decode**: generate one token per step with KV-cache — memory-bound, low arithmetic intensity

| Model | Params | Architecture | What you learn |
|-------|--------|-------------|---------------|
| **GPT-2 XL** | 1.5B | Original transformer, absolute position | Historical baseline for LLM inference. MHA (all heads have KV). No GQA. Highest KV-cache memory per head count. Shows how KV-cache memory pressure affects max batch on each device. |
| **OPT-2.7B** | 2.7B | GPT-like, absolute position | Meta's open baseline. Pre-dates RoPE/GQA. ALiBi variant available. Comparison point between older and newer architectures. |
| **BLOOM-3B** | 3B | Transformer + ALiBi positional | ALiBi (Attention with Linear Biases) adds a bias to attention logits instead of position embeddings. No separate position lookup step. Tests whether ALiBi's bias addition composes well with XLA fusion. |
| **Falcon-RW-1B** | 1B | Multi-Query Attention (MQA) | MQA: 1 KV head per query head group = 8× less KV-cache memory than MHA. First major model to use MQA. Tests whether reduced KV-cache changes the memory-bound decode profile. |
| **TinyLlama-1.1B** | 1.1B | LLaMA arch, GQA, RoPE | Efficient small LLM. GQA is between MHA and MQA. RoPE is the industry-standard positional encoding since LLaMA 1. |
| **SmolLM2-1.7B** | 1.7B | LLaMA arch | HuggingFace's flagship small model. Very recent training recipe. |
| **OLMo-2-1B** | 1.2B | Fully open including training data | Reproducible research. Identical architecture to others in class; differences show training data matters more than architecture for quality. |
| **Llama-3.2-1B/3B** | 1B/3B | LLaMA-3 arch, GQA, RoPE, long ctx | Meta's production LLMs. Llama-3 uses grouped query attention with 8 groups. Tests whether GQA's memory savings translate to proportional throughput gains. |
| **Gemma-2B** | 2B | MQA, RoPE, no bias in layers | Google-built, TPU co-designed. Layer biases removed (reduces elementwise overhead). MQA reduces KV-cache. Should show highest TPU efficiency among 2B models. |
| **Gemma-2-2B** | 2.6B | GQA, alternating local/global attention, logit soft-capping | More complex than Gemma-2B: alternating window (local=4096 ctx) and global attention. Soft-capping logits via tanh instead of hardmax. Tests XLA fusion on mixed attention types. |
| **RecurrentGemma-2B** | 2B | RGLRU (RNN gates) + local attention | Google's RNN-hybrid. RGLRU = Recurrent Gated Linear Recurrence Unit. Same parameter count as Gemma-2B, different compute graph. Sequential recurrence is anti-pattern for systolic arrays. Tests whether Google's XLA-optimised RGLRU kernel overcomes the structural disadvantage. |
| **Gemma-3-1B / 4B** | 1B/4B | Gemma-3 arch (2025), multimodal-capable | Newest Gemma. Architecture improvements from Google's 2025 research. Best-available Google model for TPU home-turf test. |
| **Phi-1.5 / Phi-2** | 1.3B/2.7B | Standard transformer | Microsoft's textbook-data models. Same architecture family; differences reflect training data quality. Architecture is simple — good baseline for other comparisons. |
| **Phi-3-mini-4k / Phi-3.5-mini-128k** | 3.8B | LLaMA arch, long context | Phi-3.5 with 128k context uses RoPE scaling. Testing seq_len=512 vs 4096 vs 32768 reveals attention's O(n²) wall directly. |
| **Phi-3.5-MoE** | 6.6B total / 2.7B active | Mixture of Experts, 16 experts, 2 active per token | **Critical model.** Dynamic expert routing = variable compute path = incompatible with XLA static shapes. TPU must either: (a) run all 16 experts (6× wasted compute), (b) pad to static routing, or (c) fail. GPU handles dynamic dispatch natively. Quantifies the MoE penalty on TPU. |
| **Qwen2.5-0.5B → 3B** | 500M–3B | LLaMA-like, GQA, RoPE, 128k ctx | Alibaba's model family. Top performance per parameter on many benchmarks (MMLU, GSM8K). Range of sizes tests scaling behaviour: does throughput scale linearly with params, or does memory bandwidth become the bottleneck? |
| **Qwen2.5-Coder-1.5B/3B** | 1.5B/3B | Same as Qwen2.5 + code tokens | Code models have longer average sequence lengths (code has more tokens per idea). Tests whether domain-specific token distributions (longer sequences) change hardware behaviour. |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 1.5B | Qwen arch, reasoning distillation | **Reasoning model.** Produces 500–2000 token chain-of-thought before answering. Inference is 10–20× more decode-heavy than standard LLMs. Tests sustained decode throughput: can TPU maintain tokens/sec over 1000+ token generation? KV-cache grows linearly; memory pressure changes the profile. |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | Same reasoning arch, larger | v6e-1 / B200 only. #1 on most reasoning benchmarks at time of writing (2026). Tests peak decode throughput on the highest-performance hardware. |
| **DeepSeek-Coder-V2-Lite** | 2.3B active / 16B total | MoE, 64 experts, 6 active | Larger MoE than Phi-3.5-MoE. More extreme routing (6/64 experts active = 91% of weights unused per token). Greatest MoE penalty on TPU. GPU advantage should be most visible here. |
| **StableLM-2-1.6B / StableLM-3B** | 1.6B/3B | LLaMA arch | Stability AI open models. Useful comparison baseline against same-size Qwen/Gemma. |
| **MPT-7B** | 7B | ALiBi + FlashAttention | v6e-1 / B200 only. ALiBi with FlashAttention2. Tests 7B model on mid-tier TPU and high-end GPU. |

### 4.6 NLP — Novel Architectures (modality: text)

| Model | Params | Architecture | What you learn |
|-------|--------|-------------|---------------|
| **Mamba-2.8B** | 2.8B | Selective State Space Model (SSM) | **Most important GPU-advantage model.** SSM selective scan has a hand-written Triton/CUDA kernel (Mamba's `selective_scan_cuda`). XLA has no native SSM primitive — falls back to sequential PyLoops. GPU is 3–5× faster. Measures the exact cost of "no XLA kernel" on a production model. |
| **Mamba2-2.7B** | 2.7B | Structured State Space (SSD) | Improved Mamba with structured state matrices. SSD allows larger matrix sizes and better GPU parallelism. Tests if Mamba2's structural improvements help TPU more than Mamba. |
| **RWKV-4-3B** | 3B | Linear RNN (WKV attention) | O(1) memory for inference (no KV-cache growth). Token mixing via a weighted sum, not softmax attention. Sequential by nature. Tests whether O(1) memory advantage justifies sequential compute cost on TPU. |

### 4.7 NLP — Code Models (modality: text/code)

| Model | Params | What you learn |
|-------|--------|---------------|
| **StarCoder2-3B** | 3B | 16k context + fill-in-the-middle (FIM). Infill attention mask is not purely causal — tests non-standard masking on XLA. |
| **CodeGemma-2B** | 2B | Google code model on Gemma backbone. Pairs with Gemma-2B: same architecture, different token distribution. Tests if fine-tuning domain changes hardware profile (it shouldn't, but confirms it). |

### 4.8 Audio (modality: audio)

| Model | Params | Architecture | What you learn |
|-------|--------|-------------|---------------|
| **Whisper-base/medium/large-v3** | 74M/307M/1.5B | Conv feature extractor → Transformer encoder-decoder | Three sizes of same architecture: tests scaling behaviour on audio. Conv frontend (1D conv on mel spectrogram) + full transformer. Long audio = long sequences = attention becomes bottleneck. |
| **wav2vec2-large** | 317M | 1D conv bank → transformer encoder | Heavy conv frontend (7 conv layers). Conv weights are fixed-size filters: different memory access pattern from vision conv. |
| **HuBERT-large** | 316M | Same as wav2vec2 | Self-supervised audio. Identical architecture to wav2vec2-large at inference. Running both confirms reproducibility. |
| **SeamlessM4T-medium** | 1.2B | Multi-task (ASR+MT+TTS) | Cascaded pipeline: speech encoder → text encoder → decoder. Multiple forward passes of different types. Tests multi-model pipeline orchestration overhead. |
| **MMS-1B** | 1B | wav2vec2-based | 1000+ language support. Tests whether large vocab (softmax over 128k tokens) is bottleneck on TPU (matrix multiply at output layer). |
| **EnCodec-24kHz** | 44M | Conv + LSTM + residual VQ | Audio neural codec. LSTM = sequential recurrence. Tests recurrent-sequential-conv hybrid on both hardware types. LSTM is small but sequential; similar story to Mamba but at tiny scale. |

### 4.9 Multimodal — Vision-Language (modality: image+text [+audio for ImageBind])

| Model | Params | Architecture | What you learn |
|-------|--------|-------------|---------------|
| **CLIP ViT-L/14** | 428M | Dual encoder (vision + text) | Two separate forward passes: vision ViT + text transformer. Contrastive loss requires both branches simultaneously. Tests pipelined multi-encoder inference. |
| **SigLIP-SO400M** | 400M | ViT + sigmoid (no softmax across batch) | Google's CLIP replacement. Removes the cross-batch normalisation — each image-text pair scored independently. Enables larger effective batch sizes. Tests whether sigmoid vs softmax output changes TPU throughput. |
| **moondream2** | 1.86B | SigLIP vision encoder + Phi-2 LLM | Small VLM. Vision encoder (conv+attention) → connector (MLP) → language decoder (autoregressive). Tests vision-to-language pipeline latency. |
| **SmolVLM-2B** | 2B | SigLIP + SmolLM-2 | HuggingFace flagship small VLM. Very recent (Dec 2024). Good comparison against moondream2: similar scale, different design choices. |
| **PaliGemma-3B** | 3B | SigLIP-400M vision + Gemma-2B LLM | **Maximum TPU home-turf advantage.** Every component — SigLIP (Google), Gemma (Google), PaliGemma connector (Google) — is designed to run on TPU. Expect highest TPU relative performance. Quantifies what hardware-software co-design is worth. |
| **LLaVA-Phi3** | ~4B | CLIP vision + Phi-3-mini LLM | Community VLM at 4B ceiling. CLIP (OpenAI) + Phi-3 (Microsoft) on Google hardware: no co-design advantage. Comparison with PaliGemma quantifies co-design value. |
| **ImageBind** | ~1.2B | 6-modality encoder (image, text, audio, depth, thermal, IMU) | Six separate encoder branches, then alignment. Tests multi-path forward pass scheduling: can XLA/CUDA parallelise independent encoder branches? |

---

## 5. Experiment Dimensions

### Precision
| Format | TPU | GPU | Key insight |
|--------|-----|-----|-------------|
| FP32 | Yes | Yes | Baseline for all comparisons |
| BF16 | **Native — same clock as FP32** | ~2× via Tensor Cores | Free upgrade on TPU; active decision on GPU |
| FP16 | Limited | Native (Tensor Cores) | Different numerics from BF16 (narrower exponent range) |
| INT8 | v5+ (~1.3×) | Ampere+ Tensor Cores (~2×) | PTQ via `bitsandbytes` (GPU) or `jax-quant` |
| INT4 | Not supported | Ampere+ via `bitsandbytes` / GPTQ | GPU-exclusive; 4× model size reduction |
| FP8 | v5e/v6e native | 4090 / H100 / B200 native | Newest format; 2× over INT8 on supported HW |

### Compilation Strategy
| Mode | JAX path | PyTorch path | Measures |
|------|----------|-------------|---------|
| Eager | `jax.disable_jit()` | Default PyTorch | Pure framework overhead without compiler |
| JIT compiled | `jax.jit` | `torch.compile(default)` | Production mode |
| Max-optimised | `jax.jit` (always) | `torch.compile(max-autotune)` | Peak ceiling — autotuning finds best CUDA kernel |
| CUDA Graphs | N/A | `make_graphed_callables` | Eliminates kernel launch overhead at bs=1 |
| XLA persistent cache | `JAX_COMPILATION_CACHE_DIR` | N/A | Amortise XLA compile across VM restarts |

### Sparsity / Pruning
| Variant | TPU | GPU (Ampere/Ada) | Key insight |
|---------|-----|-----------------|-------------|
| Dense | Baseline | Baseline | — |
| Unstructured 50% | No speedup | No speedup | Zeros computed; no hardware skip |
| Unstructured 90% | Slightly slower | Same | Memory access randomness hurts |
| 2:4 structured | **No hardware support** | **2× Sparse Tensor Core** | NVIDIA-exclusive; TPU has no equivalent |
| Channel pruning | Linear speedup | Linear speedup | Reduces matrix dimensions; both benefit equally |

### LLM Inference Modes (decoders only)
| Mode | Compute profile | Measures |
|------|----------------|---------|
| Prefill (bs=1, prompt=512 tokens) | Compute-bound | Prompt throughput (tokens/sec) |
| Prefill (bs=32) | Mixed | Batch prompt throughput |
| Decode (1 token, KV-cache warm) | Memory-bound | Token generation speed (tokens/sec) |
| Decode (long — 1000+ tokens) | Memory-bound, growing KV | Sustained decode; KV-cache pressure |
| Reasoning decode (2000 tokens) | Extreme memory-bound | DeepSeek-R1 / chain-of-thought profile |

---

## 6. Per-Experiment Protocol (~1–3 min)

```
Phase              Passes    Condition                    Records
──────────────────────────────────────────────────────────────────────────
1. Pre-flight      —         Check thermal state          gpu_temp, clock_mhz
2. Compile         1 pass    Clear XLA cache first        compile_time_s, cache_hit
3. Warmup          20        Discard all timings          kernels stabilise
4. Latency         100       bs=1                         p50/p95/p99/mean/std ms
5. Throughput      100       bs=max_fit                   samples/sec ± std
6. Profiler        10        full trace on               op breakdown, FLOPs/byte
7. Memory sweep    1/bs      bs=1,2,4,8,16,32,...OOM     peak_gb per bs level
8. Numerics        1         FP32 reference first         L2 norm vs FP32 ref
9. Post-flight     —         Check thermal drift          flag if clock dropped >5%
```

---

## 7. Suite Definitions

| Suite | Models | Variants | Experiments | Wall time | v5e-1 cost |
|-------|--------|----------|-------------|-----------|------------|
| `smoke` | 1 (BERT-base) | FP32+BF16, bs=1+max | 4 | ~8 min | $0.05 |
| `quick` | 6 (1/domain) | BF16 | 24 | ~50 min | $0.30 |
| `domain` | All in 1 domain | FP32+BF16 | ~30 | ~60 min | $0.36 |
| `arch` | Novel arches only | BF16 | ~20 | ~40 min | $0.24 |
| `llm` | All decoders | BF16, prefill+decode | ~60 | ~2 hrs | $0.72 |
| `full` | All ~75 models | All variants | ~800 | ~8 hrs | $2.88 |

---

## 8. Results Schema (JSONL)

```jsonc
{
  // --- Identity ---
  "run_id": "uuid-v4",
  "experiment_id": "sha256-of-config",
  "timestamp": "2026-04-25T10:32:00Z",

  // --- Lineage (reproducibility) ---
  "git_sha": "4f8b38e",
  "jax_version": "0.4.25",
  "torch_version": "2.3.0",
  "torch_xla_version": "2.3.0",
  "cuda_version": "12.4",
  "cudnn_version": "8.9.7",
  "tpu_runtime_version": "tpu-vm-base-20240101",
  "hf_model_revision": "sha256-of-model-weights",
  "input_seed": 42,
  "n_independent_runs": 3,
  "environment_hash": "sha256-of-version-string",

  // --- Hardware ---
  "device": "tpu_v5e1|tpu_v6e1|rtx3080|rtx4090|b200|h100",
  "framework": "jax|pytorch|torch_xla|hf_api",
  "path": 1,

  // --- Model ---
  "model": "bert_base",
  "domain": "nlp_encoder",
  "architecture_family": "transformer_encoder",
  "attention_variant": "mha|mqa|gqa|linear|sliding_window",
  "positional_encoding": "absolute|rope|alibi|none",
  "is_moe": false,
  "total_params_M": 110,
  "active_params_M": 110,

  // --- Variant ---
  "precision": "fp32|bf16|fp16|int8|int4|fp8",
  "pruning": "dense|unstructured_50|unstructured_90|structured|2_4",
  "compiled": true,
  "compile_mode": "default|max_autotune|cuda_graphs",
  "inference_mode": "prefill|decode|combined",
  "kv_cache_tokens": 0,

  // --- Input ---
  "batch_size": 32,
  "seq_len": 128,

  // --- Compile metrics ---
  "compile_time_s": 12.4,
  "first_compile_s": 12.4,
  "subsequent_compile_s": 0.3,
  "compile_cache_hit": false,
  "xla_fusion_groups": 47,
  "xla_kernel_launches": 23,
  "cuda_kernel_launches": null,

  // --- Latency (100 passes, bs=1, 3 independent runs) ---
  "latency_mean_ms": 8.4,
  "latency_std_ms": 0.3,
  "latency_cv_pct": 3.6,
  "latency_p50_ms": 8.2,
  "latency_p95_ms": 9.1,
  "latency_p99_ms": 9.8,

  // --- Throughput (100 passes, max batch, 3 runs) ---
  "throughput_mean_samples_sec": 3901,
  "throughput_std_samples_sec": 45,
  "tokens_per_sec": 499328,

  // --- Memory ---
  "peak_memory_gb": 4.2,
  "weight_memory_gb": 0.22,
  "activation_memory_gb": 3.98,
  "kv_cache_memory_gb": 0.0,
  "max_batch_before_oom": 256,

  // --- Compute analysis ---
  "flops_per_sample_G": 22.4,
  "flops_by_op": {"matmul": 18.1, "attention": 3.2, "norm": 0.6, "elementwise": 0.5},
  "arithmetic_intensity_flops_per_byte": 312,
  "achieved_tflops": 87.3,
  "peak_tflops_device": 394,
  "mfu_pct": 22.2,

  // --- Hardware utilisation ---
  "mxu_utilization_pct": 71,
  "sm_utilization_pct": null,
  "memory_bw_utilization_pct": 58,
  "device_power_w": null,
  "energy_wh_per_1k_samples": null,

  // --- Thermal / clock ---
  "gpu_clock_mhz_start": 2520,
  "gpu_clock_mhz_end": 2490,
  "gpu_temp_c_start": 42,
  "throttle_detected": false,

  // --- Numerical correctness ---
  "output_l2_vs_fp32": 0.0012,
  "output_cosine_sim_vs_fp32": 0.9998,
  "max_abs_error_vs_fp32": 0.0043,

  // --- Quality flags ---
  "flags": [],

  // --- Cost ---
  "device_cost_usd_per_hr": 0.36,
  "experiment_cost_usd": 0.012,
  "electricity_cost_per_1k_samples_usd": null,
  "cost_per_1k_samples_usd": 0.000020
}
```

---

## 9. Observability Infrastructure

All gaps from Section 14 are mapped to specific modules here.

### 9.1 `observe/system_monitor.py` — Fixes C4, C5, I7
- **GPU:** `pynvml` → poll every 100ms during measurement: `nvmlDeviceGetUtilizationRates()` (SM%), `nvmlDeviceGetMemoryInfo()`, `nvmlDeviceGetPowerUsage()` (watts), `nvmlDeviceGetTemperature()`, `nvmlDeviceGetClockInfo()` (clock MHz)
- **TPU:** Cloud Monitoring API → `tpu/container/accelerator/matrix_unit_utilization` (MXU%), `tpu/container/accelerator/memory_used` (HBM)
- **Host:** `psutil` → CPU%, RAM, I/O wait
- Writes `system_state.json` with start/end snapshots and per-second averages

### 9.2 `observe/flops_counter.py` — Fixes C1
- **JAX:** `jax.make_jaxpr(model)(dummy_input)` → traverse HLO equations → sum FLOPs per op class (dot_general=matmul, conv_general_dilated=conv, etc.)
- **PyTorch:** `fvcore.nn.FlopCountAnalysis(model, input)` — Meta's battle-tested FLOPs counter
- Outputs: `flops_per_sample_G`, `flops_by_op` dict (matmul, conv, attention, norm, elementwise)

### 9.3 `observe/stats.py` — Fixes C2
- Collects all N pass timings as raw float array
- Runs 3 independent blocks of 100 passes (cold→warm→warm)
- Grubbs outlier test on each block; removes outliers before reporting
- Requires CV < 10% for latency claims; flags `high_variance` otherwise
- Outputs: mean, std, CV, p50/p95/p99 across all valid passes

### 9.4 `observe/compile_controller.py` — Fixes C3
- Before compile measurement: explicitly clears XLA persistent cache (`rm -rf $JAX_COMPILATION_CACHE_DIR/*`)
- Measures `first_compile_s` (cold) and `subsequent_compile_s` (warm)
- Records `compile_cache_hit: bool` for every run
- Also records `torch.compile` timing separately from first forward pass

### 9.5 `observe/memory_profiler.py` — (New depth)
- Peak HBM/VRAM (high watermark)
- Memory timeline: snapshot every 5 steps → detects activation spikes
- Per-component breakdown: weight memory (static) vs activation memory (dynamic) vs KV-cache
- JAX: `jax.profiler.device_memory_profile()` → pprof format
- PyTorch: `torch.cuda.memory_snapshot()` + custom hooks on each module

### 9.6 `observe/numerics.py` — (Accuracy evidence)
- Runs FP32 first as reference, saves output tensor
- For each precision variant: compute L2 norm, cosine similarity, max absolute error vs FP32 reference
- Flags `precision_accuracy_degraded` if cosine sim < 0.99
- For LLMs: measure token-level agreement (% of next-token predictions that match FP32 greedy decode)

### 9.7 `observe/lineage.py` — (Reproducibility)
- `git_sha`: `git rev-parse HEAD`
- `hf_model_revision`: from `model.config._commit_hash` or HF API
- `input_seed`: recorded; same seed used to generate identical synthetic inputs across all paths
- `environment_hash`: SHA256 of `f"{jax.__version__}{torch.__version__}{cuda_version}{driver_version}"`
- All versions via `importlib.metadata.version()`

### 9.8 `observe/tracer.py` — (Deep profiling)
- **JAX:** `jax.profiler.trace(log_dir, create_perfetto_link=True)` → 10 steps → TensorBoard + Perfetto
- **PyTorch GPU:** `torch.profiler.profile(activities=[CPU,CUDA], with_stack=True, with_flops=True, with_modules=True)` → Chrome JSON
- **torch_xla:** `torch_xla.debug.profiler.trace(log_dir)` → same format as JAX
- Output: `results/run_logs/<run_id>/profiles/<model>_<precision>_<path>.pb`

### 9.9 `observe/hlo_analyser.py` — Fixes I5
- `jax.xla_computation(model)(dummy)` → dumps HLO text
- Parses `fusion{}` blocks → counts fusion groups and kernel launches
- Compares: JAX (XLA) kernel launches vs PyTorch eager kernel launches vs `torch.compile` kernel launches
- Outputs: `xla_fusion_groups`, `xla_kernel_launches` in schema

---

## 10. Staged Build Plan (9 Stages)

Each stage lists its deliverables AND its **exit criteria** — what must be true before the
next stage begins. Exit criteria are objective and machine-verifiable wherever possible.

```
Stage 1 — Foundation (1 day) ──────────────────────────────
  New files: benchmarks/harness.py, runner.py
             models/registry.yaml (5 models: BERT, ViT-B, GPT-2, Whisper-base, CLIP)
             observe/lineage.py, observe/stats.py, observe/compile_controller.py
             results/dashboard/index.html (table view only)
  Path: 1 (JAX+TPU) only
  Gaps fixed: C2 (multi-run stats), C3 (compile control)
  Output: first real rows in runs.jsonl; working table dashboard
  Exit criteria:
    - runs.jsonl contains ≥5 rows (one per registry model) on v5e-1
    - CV < 10% on every latency_mean_ms claim (3 independent runs each)
    - lineage.json populated with git_sha + hf_model_revision + input_seed
    - results/dashboard/index.html renders correctly on GitHub Pages
    - `smoke` suite end-to-end passes in <10 min on v5e-1

Stage 2 — Multi-path + GPU (2 days) ────────────────────────
  New files: models/jax/*, models/torch/*
             observe/system_monitor.py
  Paths: Add Path 2 (JAX+GPU) and Path 3 (PyTorch+GPU)
  Models: expand to 15 (add Gemma-2B, ResNet, DINOv2, ModernBERT, PaliGemma)
  Gaps fixed: C4 (thermal control), C5 (hardware utilisation), I1 (prefill/decode), I2 (input rotation)
  Dashboard: throughput heatmap + latency chart
  Exit criteria:
    - Same 15 models execute on Paths 1, 2, 3 with identical input_seed
    - mxu_utilization_pct AND sm_utilization_pct populated for every run
    - Decode/prefill split visible for at least 1 LLM (GPT-2 XL)
    - Thermal drift flagged when clock drops >5% — verified by stress test
    - Heatmap chart shows ≥3 paths × 15 models = 45 cells filled

Stage 3 — Profiler + Roofline (2 days) ─────────────────────
  New files: observe/flops_counter.py, observe/tracer.py
             observe/memory_profiler.py, observe/hlo_analyser.py
             results/dashboard/views/roofline.html
  Gaps fixed: C1 (FLOPs counter), I5 (XLA fusion), I7 (power measurement)
  Adds to schema: flops_per_sample_G, arithmetic_intensity, mfu_pct, xla_fusion_groups
  Exit criteria:
    - flops_per_sample_G populated from BOTH JAX (jaxpr) and PyTorch (fvcore)
    - JAX vs PyTorch FLOPs counts agree within ±5% on shared models
    - hlo_dump.txt + parsed xla_fusion_groups present for every JAX run
    - roofline.html scatter plot has ≥30 points and roofline line is correctly drawn
    - Power readings (W) recorded for every GPU run

Stage 4 — torch_xla Path 4 (1 day) ─────────────────────────
  New files: models/torch/xla_wrapper.py
  Gaps fixed: I6 (HF API control) — start HF API path
  Dashboard: compiler comparison (Path 1 vs 4, Path 3 vs 4)
  Exit criteria:
    - 5 representative models execute on Path 4 (torch_xla on TPU)
    - Path 1 vs Path 4 throughput delta documented per model
    - Compiler chart shows kernel-launch counts side-by-side for Paths 1/3/4

Stage 5 — Novel Architectures (2 days) ─────────────────────
  New files: variants/compile.py
  Models: Mamba, RWKV, RecurrentGemma, DiT — the "aha moment" models
  Dashboard: architecture-hardware fit view, MoE penalty chart
  Exit criteria:
    - Mamba GPU vs TPU ratio measured; matches predicted 3–5× GPU advantage
    - HLO inspection confirms XLA falls back to PyLoop for SSM scan
    - DiT compile-once-run-50× amortisation visible in compile_time vs total_runtime
    - architecture.html populated with all 5 architectural classes

Stage 6 — Precision + Quantization (2 days) ────────────────
  New files: variants/precision.py (INT8, FP8, INT4 GPU)
             observe/numerics.py
  Models: add Qwen, DeepSeek, Phi-3.5-MoE, Gemma-3
  Gaps fixed: I3 (MoE handling)
  Dashboard: precision speedup, numerical accuracy scatter
  Exit criteria:
    - Every precision variant has output_cosine_sim_vs_fp32 recorded
    - INT8/INT4/FP8 speedup measured on at least 5 models
    - MoE active_params_M correctly recorded; routing mode logged
    - Numerical accuracy scatter shows expected precision/speed tradeoff curve

Stage 7 — HF Inference API Path 5 (1 day) ──────────────────
  New files: paths/hf_api.py
  Dashboard: TCO calculator; Path 5 latency breakdown
  Exit criteria:
    - HF API latency split into network + queue + compute components
    - Warm vs cold endpoint latencies separately recorded
    - tco.html produces correct cost/1k-samples for all 5 paths

Stage 8 — Sparsity + Pruning (1 day) ───────────────────────
  New files: variants/pruning.py
  Dashboard: sparsity impact; 2:4 GPU advantage chart
  Exit criteria:
    - 2:4 sparsity GPU speedup measured ≥1.7× on at least 3 models
    - TPU dense vs 2:4 baseline shows no speedup (confirms HW story)
    - sparsity.html chart populated for ≥5 models × 4 sparsity variants

Stage 9 — Full Registry + Automation (ongoing) ─────────────
  New files: .github/workflows/bench.yml (scheduled weekly run)
  Models: all ~75 models
  Dashboard: full interactive explorer; claim → evidence link map
  Exit criteria:
    - All 75 models in registry have ≥1 row in runs.jsonl
    - Weekly CI run executes `quick` suite without manual intervention
    - Every claim in Section 12 links to specific run_ids that evidence it
    - Dashboard claim-verifier produces a green check on every Section-12 claim
```

---

## 11. Visualisation Plan

### Layer 1 — Static GitHub Pages Dashboard
| View | Chart type | Claim it evidences |
|------|-----------|-------------------|
| `throughput.html` | Heatmap: model × device | Who wins at what |
| `latency.html` | Box plot + CDF | Tail latency and variance |
| `roofline.html` | Scatter: intensity vs TFLOPs, roofline overlaid | Why they win |
| `compiler.html` | Bar: compile time + kernel launches + fusion groups | Compiler story |
| `precision.html` | Grouped bar: speedup ratio per device | BF16 free on TPU |
| `sparsity.html` | Bar: dense vs pruned per device | 2:4 story |
| `architecture.html` | Scatter: FLOPs vs throughput, coloured by family | Architecture-hardware fit |
| `tco.html` | Bar: cost/1k-samples all paths | Real-world cost |
| `numerics.html` | Scatter: L2 error vs speedup | Precision accuracy tradeoff |
| `moe.html` | Bar: dense vs MoE penalty per device | MoE routing cost |
| `bs_sweep.html` | Line: throughput vs batch size | Optimal serving batch |

### Layer 2 — Jupyter Notebook Explorer
`notebooks/explore.ipynb` — Pandas + Plotly; reads `runs.jsonl` directly.
Pre-built analyses: roofline, latency CDF, batch-size scaling, cross-run reproducibility check, claim verifier.

### Layer 3 — TensorBoard + Perfetto
Per-model profiler traces. XLA op fusion visualisation. GPU kernel timeline.

---

## 12. Key Claims and Evidence Requirements

| Claim | Evidence | Tools | Schema field |
|-------|---------|-------|--------------|
| EfficientNet slower than ViT despite fewer FLOPs | FLOPs count + MXU% | flops_counter + system_monitor | `flops_per_sample_G`, `mxu_utilization_pct` |
| Mamba 3–5× faster on GPU than TPU | Throughput comparison + XLA fallback in HLO | tracer + hlo_analyser | `throughput_mean`, `xla_fusion_groups` |
| BF16 is free on TPU (within noise) | FP32 vs BF16 latency, CV<5%, n=3 runs | stats + compile_controller | `latency_mean_ms ± std` across precisions |
| XLA compile 10–50× slower first call | `first_compile_s` on both paths | compile_controller | `first_compile_s`, `subsequent_compile_s` |
| 2:4 sparsity gives 2× GPU speedup | Dense vs 2:4 throughput + SM% | pruning + system_monitor | `throughput_mean`, `sm_utilization_pct` |
| Gemma-2B fastest 2B model on TPU | Throughput rank on TPU, all 2B models | runner | `throughput_mean_samples_sec` |
| PaliGemma shows co-design advantage | PaliGemma vs LLaVA-Phi3 on TPU vs GPU | runner | `throughput_mean`, `mfu_pct` |
| MoE models have higher TPU penalty | Dense vs MoE throughput, `moe_routing_mode` | runner + system_monitor | `active_params_M`, `throughput_mean` |
| TPU v5e cost-per-sample competitive with H100 | cost/1k samples all devices | runner | `cost_per_1k_samples_usd` |
| INT8 preserves accuracy | Output cosine sim vs FP32 > 0.99 | numerics | `output_cosine_sim_vs_fp32` |
| B200 memory-bound advantage: 2× A100 | BW utilisation % at peak decode | system_monitor + roofline | `memory_bw_utilization_pct` |

---

## 13. Gaps — Status After Remediation

All 15 gaps from the previous analysis are now assigned to specific modules and stages.

| Gap | Severity | Status | Fixed by | Stage |
|-----|----------|--------|---------|-------|
| C1: No FLOPs counter | Critical | ✅ Addressed | `observe/flops_counter.py` | 3 |
| C2: Single-run statistics | Critical | ✅ Addressed | `observe/stats.py` (n=3 runs, CV check) | 1 |
| C3: Compile cache not controlled | Critical | ✅ Addressed | `observe/compile_controller.py` | 1 |
| C4: Thermal/clock not controlled | Critical | ✅ Addressed | `observe/system_monitor.py` | 2 |
| C5: Hardware utilisation not captured | Critical | ✅ Addressed | `observe/system_monitor.py` | 2 |
| I1: Prefill vs decode not separated | Important | ✅ Addressed | `inference_mode` schema field + runner | 2 |
| I2: Memory cache warm/cold not controlled | Important | ✅ Addressed | Input rotation (K=4 batches) in runner | 2 |
| I3: MoE dynamic routing | Important | ✅ Addressed | `active_params_M`, `moe_routing_mode` schema | 6 |
| I4: Cross-device input seed | Important | ✅ Addressed | Numpy-first generation in runner | 1 |
| I5: XLA fusion not measured | Important | ✅ Addressed | `observe/hlo_analyser.py` | 3 |
| I6: HF API conflates latency sources | Important | ✅ Addressed | Dedicated endpoints + warm/cold split | 7 |
| I7: No power measurement | Important | ✅ Addressed | `pynvml.nvmlDeviceGetPowerUsage()` | 3 |
| N1: No batch-size sweep curve | Nice-to-have | ✅ Addressed | `bs_sweep_results` + `bs_sweep.html` | 2 |
| N2: No reproducibility check suite | Nice-to-have | ✅ Addressed | `repro` suite (smoke × 5) | 1 |
| N3: Compile cache size not tracked | Nice-to-have | ✅ Addressed | `xla_cache_size_mb` in schema | 3 |

---

## 14. Evidence Chain — Traceability Map

```
Claim in README / dashboard
  └─► Chart view (index.html: claim text → anchor link)
        └─► Run IDs cited in chart tooltip / data table
              └─► results/runs.jsonl rows (filter by run_id)
                    └─► results/run_logs/<run_id>/
                          ├── raw_timings.jsonl     (statistical claims)
                          ├── profiles/*.pb          (compiler / op claims)
                          ├── hlo_dump.txt           (XLA fusion claims)
                          ├── memory_timeline.json   (memory claims)
                          ├── system_state.json      (hw utilisation claims)
                          ├── numerics.json          (precision claims)
                          └── lineage.json           (git SHA, model hash, seed)
```

Given `run_id` + `git_sha` + `hf_model_revision` + `input_seed`, any experiment can be
re-run by anyone on the same hardware class to independently verify results.

---

## 15. Cost Reference

| Scenario | Cost |
|----------|------|
| Single experiment on v5e-1 preemptible (~2 min) | $0.012 |
| `quick` suite (50 min) on v5e-1 | $0.30 |
| `full` suite (~8 hrs) on v5e-1 | $2.88 |
| `full` suite on H100 GCP preemptible | ~$8.80 |
| `full` suite on H100 Lambda Labs | ~$19.92 |
| Weekly `full` suite for 1 month (v5e-1) | ~$12 |
| GCS model cache (~80 GB) | $1.60/month |
| Colab Pro | $9.99/month |
| HF PRO API | $9/month |
| B200 local = $0/hr for experiments | $0 |

---

## 16. Artifact Catalog

The repo is a multi-artifact knowledge base. Each artifact below has a clear purpose,
owner, and generation trigger. Treat the catalog as the source of truth for "what
this repo produces."

| Artifact | Purpose | Contents | Repo Path | Generated By |
|----------|---------|----------|-----------|--------------|
| **runs.jsonl** | Canonical results table; one row per experiment | Full schema (Section 8): identity, lineage, hardware, model, variant, latency/throughput/memory/compute/utilisation/numerics/cost | `results/runs.jsonl` | `runner.py` after each experiment finishes |
| **run_logs/** | Per-run deep evidence directory | Sub-files below; one directory per `run_id` | `results/run_logs/<run_id>/` | `runner.py` (creates dir at start of each run) |
| **raw_timings.jsonl** | Every individual pass timing for statistical re-analysis | Float ms per pass; block index; outlier flag | `results/run_logs/<run_id>/raw_timings.jsonl` | `observe/stats.py` |
| **profiles/** | Profiler traces for compiler/op-level inspection | Perfetto / Chrome JSON / TensorBoard .pb files | `results/run_logs/<run_id>/profiles/<model>_<precision>_<path>.pb` | `observe/tracer.py` |
| **hlo_dump.txt** | XLA HLO text for fusion / kernel-launch analysis | HLO IR + parsed fusion groups summary | `results/run_logs/<run_id>/hlo_dump.txt` | `observe/hlo_analyser.py` |
| **memory_timeline.json** | Per-step memory snapshots during a run | Time-series of HBM/VRAM use, peak, weight vs activation vs KV-cache split | `results/run_logs/<run_id>/memory_timeline.json` | `observe/memory_profiler.py` |
| **system_state.json** | Hardware utilisation, thermal, power, clock | Pre/post snapshots + per-second averages of MXU%/SM%/BW%/W/MHz/°C | `results/run_logs/<run_id>/system_state.json` | `observe/system_monitor.py` |
| **numerics.json** | Numerical correctness vs FP32 reference | L2, cosine sim, max-abs error, token-level agreement (LLMs) | `results/run_logs/<run_id>/numerics.json` | `observe/numerics.py` |
| **lineage.json** | Reproducibility metadata | git_sha, jax/torch/cuda/cudnn/tpu_runtime versions, hf_model_revision, input_seed, environment_hash | `results/run_logs/<run_id>/lineage.json` | `observe/lineage.py` |
| **dashboard HTML** | Public-facing claim-evidence views | All views from Section 11 (throughput, latency, roofline, compiler, precision, sparsity, architecture, tco, numerics, moe, bs_sweep) | `results/dashboard/index.html` + `results/dashboard/views/*.html` | `dashboard/build.py` (post-suite) |
| **explore notebook** | Interactive Pandas+Plotly exploration over runs.jsonl | Pre-built analyses: roofline, latency CDF, bs scaling, repro check, claim verifier | `notebooks/explore.ipynb` | Hand-authored; updated after schema changes |
| **model registry** | Single source of truth for model list + metadata | YAML: model id, family, modality, size, hf revision, expected memory, suite tags | `models/registry.yaml` | Hand-authored; CI validates against runs.jsonl |
| **observe modules** | Reusable observability library | system_monitor.py, flops_counter.py, stats.py, compile_controller.py, memory_profiler.py, numerics.py, lineage.py, tracer.py, hlo_analyser.py | `observe/` | Hand-authored; covered by unit tests |
| **suites** | Named experiment-set definitions | YAML for smoke/quick/domain/arch/llm/full/repro | `suites/*.yaml` | Hand-authored |
| **GH Actions workflow** | Scheduled CI benchmark runs | Weekly `quick` suite on v5e-1 preemptible; uploads runs.jsonl + dashboard | `.github/workflows/bench.yml` | Hand-authored (Stage 9) |
| **lesson plan / session docs** | Continuity across sessions | LESSON_PLAN.md, SESSION.md, MEMORY.md (project-side), prompts.md | repo root | Hand-authored; updated each session |

---

## 17. Colab Pro + HF Workflow

Colab Pro ($9.99/mo) + HF PRO ($9/mo) = the cheapest practical path to actual TPU and
GPU runs without a cloud setup. This section is the operating manual.

### 17.1 Runtime selection (Colab Pro)
- **Runtime menu → Change runtime type:** TPU v2-8 (free with Pro), A100 (Pro budget unit cost), L4 (cheaper alternative), T4 (free). v2-8 is 8 chips of v2 — for single-chip semantics, use only `jax.devices()[0]`.
- For our matrix:
  - **TPU v2-8**: Path 1 surrogate (older arch, but exercises XLA + TPU pod attach)
  - **A100 40GB**: Path 2/3 surrogate for cloud-class GPU
  - **T4**: cheap smoke-test target; useful for code correctness only
- Colab Pro+ unlocks longer runtimes and better GPU priority — worth it if you hit session caps weekly.

### 17.2 HF token setup for gated models
Required for: Gemma family, PaliGemma, LLaMA-3, some DeepSeek, MMS, SeamlessM4T.

```python
# Once per Colab session:
from huggingface_hub import login
from google.colab import userdata
login(token=userdata.get('HF_TOKEN'))   # store HF_TOKEN once in Colab secrets
```

Steps:
1. Create token at https://huggingface.co/settings/tokens (`read` scope is enough for gated models you've accepted).
2. Accept each gated model's license on its HF page (one-time, per model family).
3. Add the token to Colab Secrets (left sidebar key icon) as `HF_TOKEN`.
4. Same token works for HF Inference API (Path 5).

### 17.3 Running suites from Colab (CLI commands)
Colab notebooks can shell out to bash with `!`. The runner is CLI-driven:

```bash
# Clone + install
!git clone https://github.com/rajaghv-dev/tpu /content/tpu
%cd /content/tpu
!pip install -q -r requirements.txt

# Smoke test (≤10 min — fits well under Colab cell timeouts)
!python -m benchmarks.runner --suite smoke --device auto --out results/runs.jsonl

# Quick suite (~50 min — fits in a single Colab Pro session)
!python -m benchmarks.runner --suite quick --device auto

# Single-experiment debugging
!python -m benchmarks.runner --model bert_base --precision bf16 --bs 32 --device tpu
```

`--device auto` detects TPU vs GPU at runtime via `jax.devices()` / `torch.cuda.is_available()`.

### 17.4 GCS model cache setup in Colab
HF model downloads dominate first-run cost. Cache to GCS once, mount thereafter.

```bash
# One-time: authenticate Colab to GCP
from google.colab import auth; auth.authenticate_user()

# Mount your GCS bucket (you create it once in GCP console, ~$1.60/month for 80 GB)
!gcsfuse --implicit-dirs my-tpu-bench-cache /content/hf_cache

# Point HF at it
import os
os.environ['HF_HOME'] = '/content/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/content/hf_cache/transformers'
```

After the first run downloads weights, subsequent Colab sessions reuse them — saving
~10–60 min of download time per model on slow days.

### 17.5 Session limits and what to do about them
Colab Pro caveats:
- Sessions die after **~12 hours of total** runtime, or **~90 min idle**.
- Disconnects on tab close — your kernel survives ~30 min, then dies.
- A100/L4 access is **not guaranteed**; falls back to T4 if GPUs are saturated.
- No native tmux. The closest equivalents:
  - **Always write to GCS, not local disk.** `runs.jsonl` and `run_logs/` live in the mounted bucket so a disconnect loses nothing.
  - **Make the runner resumable.** `runner.py` checks runs.jsonl for completed `(model, precision, bs, device)` tuples and skips them.
  - **Suite the work.** Run `quick` (50 min) per session, not `full` (8 hrs).
  - **Use `nohup`-style backgrounding inside the cell** so a tab close doesn't immediately kill it: `!nohup python -m benchmarks.runner ... > /content/hf_cache/run.log 2>&1 &` then poll the log.
- For real long runs, switch to a GCP TPU VM with `tmux`. Colab Pro is for development and `quick` suites.

### 17.6 HF Inference API setup (Path 5)
HF PRO ($9/mo) gives higher rate limits + access to dedicated endpoints.

When to use HF API vs self-hosted:
- **Use HF API when:** you want to measure managed-serving overhead (Path 5's whole point), the model is in HF's hosted catalogue, or you want zero-setup access to a model you can't fit locally.
- **Use self-hosted when:** you need precise compile-time control, custom precision/quantization, deterministic batch boundaries, or any of Path 1–4's level of profiler depth.

```python
from huggingface_hub import InferenceClient
client = InferenceClient(model="google/gemma-2b", token=userdata.get('HF_TOKEN'))
out = client.text_generation("Hello world", max_new_tokens=64)
```

For latency decomposition (Stage 7 requirement), wrap calls to capture network + queue + compute:
- Cold endpoint: first call after idle — measures cold-start.
- Warm endpoint: 2nd–Nth call — measures steady-state inference + network.
- Use HF Dedicated Endpoints (paid, separate from PRO subscription) for stable warm latency
  numbers without scheduling jitter.

### 17.7 Session continuity workflow
At session start: read `MEMORY.md` and `SESSION.md` (per project memory rules).
At session end: append the day's run_ids and learnings to `SESSION.md`; commit `runs.jsonl`.

---

## 18. Open Questions

Genuinely unresolved as of 2026-04-26. Each question has a specific experiment that
would answer it; results land in `runs.jsonl` and update this section when known.

1. **Will XLA's Pallas/Splash Attention close the FlashAttention2 gap at long context (>8k tokens)?**
   - Test plan: ModernBERT-large + Phi-3.5-mini-128k at seq_len ∈ {2048, 8192, 32768} on Path 1 vs Path 3.
   - Resolves: whether TPU is competitive on long-context inference, or whether the FA2 CUDA kernel remains structurally ahead.

2. **Can MoE routing be made static-shape-compatible in JAX without full expert materialisation?**
   - Test plan: Phi-3.5-MoE and DeepSeek-Coder-V2-Lite on Path 1 with three routing strategies (full materialisation, padded top-k, capacity-factor variant).
   - Resolves: how much of the MoE penalty on TPU is fundamental vs solvable with smarter routing kernels.

3. **Does B200's FP8 give >2× over BF16 on memory-bound decode, or does bandwidth ceiling dominate?**
   - Test plan: Llama-3.2-3B + DeepSeek-R1-Distill-Qwen-7B in BF16 vs FP8 on B200 at decode (1 token, KV warm).
   - Resolves: whether FP8's compute advantage matters for decode (memory-bound) or only for prefill (compute-bound).

4. **At what batch size does v6e-1 TPU match H100 SXM5 on BERT throughput?**
   - Test plan: BERT-base bs sweep {1, 2, 4, ..., 1024} on v6e-1 vs H100. Crossover point becomes the practical "TPU-wins-from-here" line for this class.
   - Resolves: whether TPU's higher-batch advantage starts at bs=128 (folklore) or earlier/later in reality.

5. **Is the RWKV sequential compute disadvantage on TPU worse than Mamba, or does WKV's simpler recurrence help?**
   - Test plan: RWKV-4-3B vs Mamba-2.8B on Path 1 (TPU) vs Path 3 (GPU); compare relative slowdown ratios.
   - Resolves: whether recurrence simplicity (RWKV) compensates for lacking a custom CUDA kernel relative to Mamba's selective_scan_cuda.

6. **(Open, lower priority)** Does PaliGemma's TPU advantage survive when you swap its connector to a non-Google MLP? — isolates "co-design" from "good architecture."

7. **(Open, lower priority)** Do Colab Pro v2-8 results predict v5e-1 results within 20%? — would let us use Colab as a cheap pre-flight check before spending v5e-1 hours.
