# Lesson Plan — From Essentials to Expert
## TPU × GPU Inference Benchmarking

A structured learning path tied entirely to this repository.
Uses **Google Colab Pro** for cloud TPU/GPU access and **HuggingFace paid account** for models and serving.
No code to write yet — understand first, then experiment.

---

## How to Use This Plan

Each module has:
- **What you will understand** — the outcome
- **Key terms** — vocabulary to internalize before moving on
- **Repo touchpoints** — which files to read in this repo
- **Colab Pro activity** — what to run or observe in Colab
- **HuggingFace activity** — what to explore on HF
- **Self-check** — questions you should be able to answer before moving to the next module

Work at your own pace. Modules 1–4 are essentials (days 1–3).
Modules 5–9 are intermediate (week 2). Modules 10–15 are advanced/expert (weeks 3–6).

---

## Module 0 — How This Repo Is Organised and Why

**What you will understand:**
The reasoning behind every structural decision in this repo, so you can navigate it intelligently rather than treating it as a black box.

### 0.1 The Five Knowledge Documents

This repo has five strategic reference documents. Read them in this order when starting:

| Document | What it answers | When to read |
|----------|----------------|--------------|
| `MEMORY.md` | Dense 3-minute summary of everything: hardware, paths, models, next task | **First thing every session** |
| `SESSION.md` | Full project state: decisions made, files built, what's next | After MEMORY.md |
| `DECISIONS.md` | Why every architectural choice was made (13 ADRs) | When you wonder "why did we choose X?" |
| `RISKS.md` | What can go wrong and how to prevent it (25+ risks) | Before starting a new stage |
| `QUESTIONS.md` | Open research questions with test plans (23 questions) | When designing an experiment |
| `RECOMMENDATIONS.md` | Prioritized actions in 3 tiers | When deciding what to do next |

### 0.2 The 13 Architectural Decisions (ADRs)

These choices are locked — they were made deliberately. Understanding *why* they were made helps you understand the entire repo design:

| ADR | Decision | The short reason |
|-----|----------|-----------------|
| ADR-001 | Inference-only | Training doesn't fit 1–3 min budget; inference teaches the same hardware concepts |
| ADR-002 | JAX AND PyTorch | Compiler comparison is a primary goal; you can't compare XLA vs Inductor with one framework |
| ADR-003 | v5e-1 preemptible primary | $0.36/hr, 16GB matches RTX 3080, single-chip VM, no pod management |
| ADR-004 | Synthetic inputs (seed=42) | No data download, full reproducibility, hardware-only comparison |
| ADR-005 | HF pretrained weights | Real weights give realistic compute patterns; random init is misleading |
| ADR-006 | GCS model cache | Download once to GCS; reuse across preemptible VM restarts; free egress within us-central1 |
| ADR-007 | Append-only JSONL | Simple, grep-able, version-controllable, no schema migrations |
| ADR-008 | Static HTML + Vega-Lite | Zero server cost, works on GitHub Pages, no build step |
| ADR-009 | n=3 runs, CV<10% | Evidence-grade measurements; distinguishes real differences from noise |
| ADR-010 | Sequential single experiments | Colab-compatible; no parallel complexity; 1–3 min per run |
| ADR-011 | 9-stage incremental build | Each stage informed by results from previous; avoids building wrong thing |
| ADR-012 | 75 models, ≤4B params | v5e-1 has 16GB; 4B BF16 = ~8GB weights; leaves room for KV-cache |
| ADR-013 | MoE/SSM included | Maximum learning value despite XLA challenges; Stage 5+ |

See `DECISIONS.md` for full rationale, alternatives considered, risks, and revisit triggers.

### 0.3 The Framework Hierarchy

When working with this repo, always know what layer you're at:

```
You (Python, prompts.md)
    ↓
Framework (JAX or PyTorch)          ← Examples 01–08, benchmarks/
    ↓
Compiler (XLA or Inductor/CUDA)     ← observe/compile_controller.py, hlo_analyser.py
    ↓
Hardware (TPU or GPU)               ← context.md Section 2, DECISIONS.md ADR-003
    ↓
Results (runs.jsonl, dashboard)     ← results/, QUESTIONS.md, RECOMMENDATIONS.md
```

A confusion about "why is this slow?" almost always means you're reasoning at the wrong layer. Is it the Python code? The framework tracing? The compiler? The hardware? Each has different diagnostic tools.

### 0.4 How to Use prompts.md

`prompts.md` is a complete history of how this repo was built. It serves three purposes:
1. **Raw prompts** (bottom section): verbatim record of what was asked — useful to understand the original intent
2. **Improved prompts** (top section): well-engineered versions — use these as templates when asking similar questions
3. **Standing Instructions** (very top): rules that apply to every session automatically

When you start a new session and want to pick up where we left off, the standing instructions eliminate the need to repeat setup context.

### 0.5 Terminology Cross-Reference

Quick lookup — which module defines each term:

| Term | Module |
|------|--------|
| Roofline, arithmetic intensity, ridge point | Module 3 |
| JAX, jit, pmap, XLA, HLO | Module 4 |
| Operator fusion, kernel, cuDNN | Module 4, 8 |
| MFU, CV, throughput, latency, OOM | Module 7 |
| Profiler trace, Perfetto, fusion groups | Module 8 |
| MXU, Tensor Core, systolic array | Module 2 |
| MHA, GQA, MQA, KV-cache, RoPE | Module 9 |
| BF16, INT8, FP8, GPTQ, AWQ | Module 10 |
| Prefill, decode, TTFT, TPOT | Module 11 |
| 2:4 sparsity, Sparse Tensor Cores | Module 12 |
| Lineage, reproducibility, evidence chain | Module 13 |
| TCO, cost/1k samples, break-even | Module 14 |

**Self-check (Module 0):**
- Why did we choose JAX AND PyTorch instead of just one? (See ADR-002)
- What is the difference between `DECISIONS.md` and `RECOMMENDATIONS.md`?
- If someone asks "why is inference only?", which ADR number do you reference?
- What is the first thing to read at the start of every session?

---

## Prerequisites (Before Module 1)

No hardware or ML knowledge needed, but you should be comfortable with:
- Python basics (functions, loops, dicts, lists)
- Running a cell in Google Colab
- Uploading and reading a file
- Understanding what a function call does without knowing its internals

If you can run `print("hello")` in a Colab cell and understand what happened, you are ready.

---

## Module 1 — Your Environment: Colab Pro + HuggingFace

**What you will understand:**
How your two primary tools work, what they give you access to, and how they connect to this repo.

### 1.1 Google Colab Pro — What It Is

Colab is a cloud notebook that runs on Google's servers. You write Python in browser cells.
Colab Pro ($9.99/month) upgrades the free tier in four important ways:

| Feature | Free Colab | Colab Pro |
|---------|-----------|-----------|
| GPU | T4 (16 GB) | A100 (40 GB) or V100 (16 GB) — priority |
| TPU | v2-8 (limited, queued) | v2-8 or v3-8 (priority access) |
| Session length | ~1.5 hrs idle cut | Up to 24 hrs |
| Compute units | ~100 units/month | ~500 units/month |
| RAM | 12 GB | 25 GB |

For this repo's `smoke` and `quick` suites: Colab Pro is sufficient.
For `full` suite (8 hrs): use Cloud TPU VM or local hardware.

**How to select TPU runtime in Colab:**
Runtime → Change runtime type → Hardware accelerator → TPU

**How to select GPU runtime:**
Runtime → Change runtime type → Hardware accelerator → A100 GPU (or T4)

You will switch between these two runtimes deliberately — one for TPU experiments, one for GPU.

### 1.2 HuggingFace Paid Account — What It Gives You

HuggingFace is the primary source of all 75 models in this repo.

| Feature | Free HF | HF PRO ($9/mo) |
|---------|---------|----------------|
| Model downloads | Unlimited public | Unlimited |
| Gated models (Gemma, LLaMA, PaliGemma) | Request access (approved manually) | Same — request once |
| Inference API calls | 1,000/day rate-limited | 10,000/day, higher rate |
| Dedicated Endpoints | Not available | Available (pay-per-hour) |
| Private models/datasets | Limited | Unlimited private repos |
| HF Spaces (GPU-backed) | Limited | More compute |

**Gated models in this repo that require HF access approval:**
Gemma-2B, Gemma-2-2B, Gemma-3-1B/4B, RecurrentGemma-2B, PaliGemma-3B, CodeGemma-2B,
Llama-3.2-1B/3B, Gemma-2-2B-IT. Apply at each model's HF page — usually approved in minutes.

**Your HF token:** Go to hf.co → Settings → Access Tokens → New token (read access).
This token is set as `HF_TOKEN` environment variable in Colab before any model download.

### 1.3 How This Repo Connects to Both

```
This Repo (github.com/rajaghv-dev/tpu)
    │
    ├── Clone into Colab: !git clone https://github.com/rajaghv-dev/tpu
    │
    ├── Models come from HuggingFace Hub (via transformers library)
    │
    ├── TPU experiments: select TPU runtime → run JAX examples
    │
    ├── GPU experiments: select GPU runtime → run PyTorch examples
    │
    └── Results sync back: !git push (after committing)
```

**Key terms for this module:**
- **Runtime** — the hardware backend Colab connects your notebook to (CPU / GPU / TPU)
- **Compute units** — Colab Pro's credit system; TPU costs more units per hour than CPU
- **Gated model** — a model that requires HF account approval before downloading
- **Access token** — a secret key that proves your HF identity to download models
- **HF Hub** — HuggingFace's model repository (like GitHub but for ML models)

**Repo touchpoints:** `README.md` sections "Quick Start" and "Google Colab"

**Colab Pro activity:**
Open a new Colab notebook. Set runtime to TPU. In a cell, run:
`import jax; print(jax.devices())`
You should see a list of 8 TPU devices. This confirms your TPU runtime is working.

**HuggingFace activity:**
Log in at hf.co. Go to Settings → Access Tokens. Create a token. Navigate to
`google/gemma-2b` model page. Click "Acknowledge license" to request access.

**Self-check:**
- What is the difference between a Colab GPU runtime and TPU runtime?
- Why do some models require HF access approval?
- How many TPU chips does a Colab Pro v2-8 runtime give you?

---

## Module 2 — What Is a TPU? What Is a GPU? (Hardware Essentials)

**What you will understand:**
What these chips are physically, why they were built, and what problems they solve differently.

### 2.1 The GPU

A GPU was originally built to render 3D graphics — millions of triangles per frame, each
calculated independently. This requires thousands of small parallel processors that all do the
same arithmetic simultaneously. NVIDIA repurposed this for deep learning.

Key components:
- **CUDA Cores** — general-purpose floating point units; thousands of them; run in parallel
- **Tensor Cores** — specialist matrix multiply units added in 2017 (Volta architecture);
  operate on 4×4 matrix tiles natively; each Tensor Core does a 4×4 matmul in one clock cycle
- **HBM / GDDR6X** — the memory on the GPU chip; HBM (High Bandwidth Memory) uses stacked dies
  for extreme bandwidth; GDDR6X is cheaper, slightly less bandwidth

Your GPUs:
- RTX 3080: 8704 CUDA cores, 272 Tensor Cores (3rd gen), 16 GB GDDR6X
- RTX 4090: 16,384 CUDA cores, 512 Tensor Cores (4th gen), 24 GB GDDR6X
- B200 (DGX): 10,752 CUDA cores, 6th-gen Tensor Cores, 192 GB HBM3e

### 2.2 The TPU

Google built TPUs in 2016 specifically and exclusively for matrix multiplication — the
dominant operation in neural networks. They did not start from graphics rendering.

Key component: **MXU (Matrix Multiply Unit)** — a 128×128 systolic array.
A systolic array is a grid of multiply-accumulate units where data flows through like a wave —
one cell computes, passes its result to the neighbour, which computes next. This is extremely
efficient for large matrix multiplies but inflexible for other operations.

- No general-purpose CUDA cores
- No dynamic dispatch — the computation graph is fixed at compile time
- Memory: HBM (same as GPU HBM, different capacity and bandwidth per generation)

### 2.3 The Fundamental Difference

| Dimension | GPU | TPU |
|-----------|-----|-----|
| Designed for | Parallel tasks (graphics → ML) | Matrix multiply only |
| Flexibility | High — can run anything | Low — XLA compiler required |
| When it excels | Irregular compute, custom ops | Regular large matmuls |
| Programming model | CUDA kernels + cuDNN | XLA (JAX) or torch_xla |
| Debugging | Rich tooling (Nsight, nvprof) | Limited tooling |

**Key terms for this module:**
- **CUDA** — NVIDIA's parallel computing platform; what GPU programs are written in
- **Tensor Core** — NVIDIA's specialist matrix multiply hardware
- **MXU** — TPU's matrix multiply unit (systolic array)
- **Systolic array** — a 2D grid of compute cells where data flows in waves; rigid but fast
- **HBM (High Bandwidth Memory)** — stacked memory dies with very high bandwidth
- **GDDR6X** — consumer GPU memory; lower cost, slightly lower bandwidth than HBM
- **TFLOPs** — trillion floating-point operations per second; raw compute speed
- **Memory bandwidth (GB/s)** — how fast data moves from memory to compute units

**Repo touchpoints:** `01_hello_tpu/hello_tpu.py` and `01_hello_tpu/README.md`

**Colab Pro activity (TPU runtime):**
In a cell: `import jax; print(jax.device_count(), jax.local_devices())`
Look at the device descriptions — notice "TpuDevice" with coords and core numbers.
These are 8 separate chips on a v2-8 or v3-8 slice.

**Colab Pro activity (GPU runtime):**
Switch to GPU runtime. In a cell:
`import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))`
Read the total memory, multiprocessor count, and clock rate.

**Self-check:**
- What is a systolic array and why is it fast for matrix multiply but slow for other ops?
- Your RTX 4090 has 330 BF16 TFLOPs. Your v6e-1 TPU has 918 BF16 TFLOPs. Does that mean the TPU is always 2.8× faster? Why or why not?
- What does "memory bandwidth" limit in practice?

---

## Module 3 — Memory, Bandwidth, and the Roofline Model

**What you will understand:**
The single most important mental model for interpreting every result in this repo.

### 3.1 The Two Ceilings

Every computation is bounded by one of two limits:

**Ceiling 1 — Compute-bound:** You have more data ready to process than your compute units
can handle. The chip is 100% busy calculating. Adding memory bandwidth would not help.
*Example: multiplying two 4096×4096 matrices — billions of MACs, small data.*

**Ceiling 2 — Memory-bound:** Your compute units are starving — they finish each calculation
and have to wait for the next data to arrive from memory. Adding more FLOPs would not help.
*Example: adding a bias vector to every element of a large activation — trivial arithmetic,
lots of memory reads.*

### 3.2 Arithmetic Intensity

Arithmetic intensity = FLOPs ÷ bytes read from memory (FLOPs/byte).

- High intensity (e.g. 300+ FLOPs/byte): compute-bound
- Low intensity (e.g. 10 FLOPs/byte): memory-bound

Different operations have very different intensities:
| Operation | Typical Intensity | Bound by |
|-----------|-----------------|---------|
| Large matrix multiply (GEMM) | 100–1000 FLOPs/byte | Compute |
| Self-attention (long seq) | 50–200 FLOPs/byte | Memory or compute |
| Depthwise convolution | 1–5 FLOPs/byte | Memory |
| LayerNorm, GeLU, softmax | 2–10 FLOPs/byte | Memory |
| Embedding lookup | <1 FLOPs/byte | Memory |

### 3.3 The Roofline

Plot your chip's peak FLOPs on the Y-axis and peak bandwidth on the X-axis.
Draw two lines: a horizontal line at peak FLOPs (compute ceiling) and a diagonal line
representing memory bandwidth × intensity (memory ceiling).
Where they intersect is the "ridge point" — the minimum arithmetic intensity needed to be compute-bound.

**Your chips' ridge points:**
| Chip | Peak BF16 TFLOPs | Peak BW (GB/s) | Ridge point (FLOPs/byte) |
|------|-----------------|---------------|--------------------------|
| RTX 3080 | 119 | 760 | 156 |
| RTX 4090 | 330 | 1008 | 327 |
| B200 SXM | 2250 | 4000 | 562 |
| TPU v5e | 394 | 820 | 480 |
| TPU v6e | 918 | 1640 | 560 |

**The insight:** BERT-base has arithmetic intensity ~300 FLOPs/byte.
- On RTX 3080 (ridge = 156): compute-bound → adding bandwidth does not help; adding FLOPs does
- On RTX 4090 (ridge = 327): just below the ridge → borderline; sensitive to both
- On TPU v5e (ridge = 480): memory-bound → adding FLOPs does not help; adding bandwidth does

The same model is in a different regime on each chip. This is why results differ even when
on-paper FLOPs are similar.

**Key terms for this module:**
- **Roofline model** — the two-ceiling performance model visualised as two intersecting lines
- **Arithmetic intensity** — FLOPs per byte of memory traffic (FLOPs/byte)
- **Ridge point** — the intensity where a chip transitions from memory-bound to compute-bound
- **Memory-bound** — performance limited by how fast data can be moved, not computed
- **Compute-bound** — performance limited by how fast arithmetic can be done, not data moved
- **GEMM** — General Matrix Multiply; the dominant operation in transformers
- **MACs** — Multiply-Accumulate operations; the basic unit of neural network arithmetic

**Repo touchpoints:** `context.md` Section 12 (Key Claims table), `results/dashboard/roofline.html`

**Colab Pro activity:**
In either runtime, after running a benchmark, look at the `arithmetic_intensity` field in
`results/runs.jsonl`. Compare it to the ridge point of the hardware it ran on.
If intensity < ridge: memory-bound. If intensity > ridge: compute-bound.

**Self-check:**
- EfficientNet-B7 has ~3 FLOPs/byte. What does the roofline predict about its performance on TPU v5e?
- ViT-B/16 has ~150 FLOPs/byte. Is it memory-bound or compute-bound on the RTX 4090?
- If you made the B200 twice as fast at compute but kept memory bandwidth the same, would BERT-base get faster? Why or why not?

---

## Module 4 — The Software Stack (JAX, PyTorch, XLA, CUDA)

**What you will understand:**
What each layer of the software stack does and why there are 5 execution paths in this repo.

### 4.1 Layers from Model to Silicon

```
Your Python code (model definition)
        ↓
Framework (JAX or PyTorch) — defines operations abstractly
        ↓
Compiler (XLA or CUDA/cuDNN) — translates operations to hardware instructions
        ↓
Hardware (TPU or GPU) — runs instructions
```

### 4.2 JAX

JAX is Google's numerical computing library. Key properties:
- **Functional** — no mutation; arrays are immutable; functions must be pure
- **`jax.jit`** — traces your function once, compiles it via XLA, then runs the compiled version
- **`jax.vmap`** — automatically vectorises a function over a batch dimension
- **`jax.pmap`** — parallelises a function across multiple devices (TPU chips)
- **Native on TPU** — JAX's compiler backend is XLA, which is what TPUs run natively
- **Works on GPU too** — XLA also targets CUDA; same JAX code runs on both

### 4.3 PyTorch

PyTorch is Meta/community's ML framework. Key properties:
- **Imperative** — operations run immediately when called (eager mode)
- **`torch.compile`** — optional compiler pass added in PyTorch 2.0; generates optimised CUDA
- **Ecosystem standard** — most HuggingFace models are PyTorch-first
- **torch_xla** — a bridge: PyTorch API but XLA backend; runs PyTorch on TPU

### 4.4 XLA (Accelerated Linear Algebra)

XLA is Google's compiler. It:
- Takes a computation graph (HLO — High Level Operations)
- Performs **operator fusion** — combines multiple ops into one kernel (e.g. LayerNorm + GeLU = 1 kernel)
- **Requires static shapes** — all tensor dimensions must be known at compile time
- Compiles once and caches — subsequent runs skip compilation
- Targets both TPU and GPU

XLA's requirement for static shapes is the key constraint:
- It enables aggressive optimisation (padding, tiling, fusion)
- It breaks MoE models (dynamic expert routing = dynamic shapes)
- It means your batch size must be fixed before compilation

### 4.5 CUDA and cuDNN

CUDA is NVIDIA's programming platform for GPUs. cuDNN is NVIDIA's library of optimised
neural network primitives (convolution, attention, batch norm, etc.).

- **cuDNN auto-tuner** — on first run, tries multiple algorithms for each operation
  (e.g. 6 different convolution algorithms) and picks the fastest for your hardware
- **Custom CUDA kernels** — hand-written GPU programs for specific operations
  (e.g. Mamba's `selective_scan_cuda`, FlashAttention 2)
- **CUDA Graphs** — records a sequence of kernel launches and replays them without
  Python overhead; critical for low-latency bs=1 inference

### 4.6 The 5 Paths — Why They Exist

| Path | Framework | Compiler | Hardware | What it isolates |
|------|-----------|---------|---------|-----------------|
| 1 | JAX | XLA | TPU | — |
| 2 | JAX | XLA (CUDA) | GPU | Hardware only (XLA constant) |
| 3 | PyTorch | CUDA/cuDNN | GPU | Framework + compiler on GPU |
| 4 | torch_xla | XLA | TPU | Framework only (XLA constant) |
| 5 | HTTP/REST | HF-managed | Unknown | Managed serving overhead |

Path 2 vs Path 3 on the same GPU tells you how much the framework choice matters.
Path 1 vs Path 4 on the same TPU tells you how much JAX vs PyTorch API matters.

**Key terms for this module:**
- **JIT (Just-In-Time compilation)** — compiling code at runtime the first time it is called
- **Eager mode** — running operations immediately without compiling; default in PyTorch
- **Operator fusion** — combining multiple operations into one GPU/TPU kernel; reduces memory traffic
- **HLO (High Level Operations)** — XLA's intermediate representation of a computation
- **Static shapes** — all tensor dimensions fixed before compilation (XLA requirement)
- **Dynamic shapes** — tensor dimensions can change between calls (PyTorch's default)
- **Kernel** — a single function that runs on a GPU/TPU; the unit of work
- **cuDNN** — NVIDIA's library of pre-optimised neural network operations
- **torch_xla** — PyTorch library that routes operations through XLA instead of CUDA

**Repo touchpoints:** `05_gpt_pretraining/train.py` (jax.jit usage),
`07_custom_training_loop/train.py` (gradient accumulation + jit),
`context.md` Section 3 (Execution Paths)

**Colab Pro activity (TPU runtime):**
Open `01_hello_tpu/hello_tpu.py`. Read the `jit_fn` function. Notice `@jax.jit`.
Ask yourself: what happens the first time it is called? What happens the second time?

**HuggingFace activity:**
Go to hf.co. Search for `bert-base-uncased`. On the model page, click "Files and versions".
You will see `pytorch_model.bin` (PyTorch weights) and sometimes `flax_model.msgpack`
(JAX/Flax weights). Both are the same model — different serialisation formats for different frameworks.
This is how Path 1 (JAX) and Path 3 (PyTorch) load the same weights.

**Self-check:**
- What happens if you change the batch size in the middle of a JAX jitted function? Why?
- Why does PyTorch need `torch.compile` to get performance close to XLA?
- If Mamba requires a custom CUDA kernel, what happens when you run it through XLA (Path 1 on TPU)?

---

## Module 5 — Reading This Repo: Examples 01–08

**What you will understand:**
What each of the 8 training examples in this repo teaches, and how they build on each other.

### The Training Examples Ladder

| Example | Core concept | Read to understand |
|---------|-------------|-------------------|
| `01_hello_tpu` | Device enumeration, jit, pmap | How JAX sees the TPU hardware |
| `02_mnist_classification` | CNN + `pmap` data parallelism | How to split a batch across 8 TPU chips |
| `03_resnet_imagenet` | Cosine LR, checkpointing | Full training infrastructure |
| `04_bert_finetuning` | HuggingFace Flax + fine-tuning | How HF models work in JAX |
| `05_gpt_pretraining` | Causal attention, LR schedule | Building a decoder from scratch |
| `06_data_pipeline` | `tf.data`, GCS, prefetching | Why data starvation kills TPU throughput |
| `07_custom_training_loop` | Gradient accumulation, bf16 | Manual control over the training loop |
| `08_multi_host` | `jax.distributed`, pod training | Scaling across multiple TPU VMs |

### Why Training Examples Matter for Inference Benchmarking

Even though this repo focuses on inference, training examples teach concepts that directly
apply to inference understanding:

- `02_mnist_classification`: `pmap` shards the batch across devices — same pattern used in inference Path 1
- `06_data_pipeline`: `tf.data` prefetching prevents the TPU from starving — same problem exists in inference input pipelines
- `07_custom_training_loop`: bf16 casting — the same precision story applies to inference
- `08_multi_host`: understanding `jax.distributed` prepares you for multi-chip inference

**Repo touchpoints:** All `0X_*/` directories

**Colab Pro activity (TPU runtime):**
Clone the repo in a Colab cell. Navigate to `01_hello_tpu/`. Run `hello_tpu.py`.
Read its output carefully. Every line corresponds to a concept from Module 2.

**Colab Pro activity (GPU runtime):**
Switch to GPU runtime. Re-run the same file. It will error because `jax` on a GPU runtime
in Colab requires a different setup than TPU. This error is informative — it tells you that
JAX needs to know what backend to target.

**HuggingFace activity:**
Open `04_bert_finetuning/train.py`. Find the line `AutoTokenizer.from_pretrained("bert-base-uncased")`.
On HF, search for `bert-base-uncased`. Click "Files and versions". 
Understand that `from_pretrained` downloads those files to your Colab session's `/root/.cache/huggingface/`.

**Self-check:**
- In `02_mnist_classification`, why is `drop_remainder=True` set on the dataset batch call?
- In `06_data_pipeline`, what does `prefetch(tf.data.AUTOTUNE)` do and why is it critical for TPU?
- In `07_custom_training_loop`, why do gradients get cast to float32 even when running in bf16?

---

## Module 6 — HuggingFace Deep Dive: Models, Tokenizers, Pipelines

**What you will understand:**
How HuggingFace organises models, how to load them for both JAX and PyTorch, and how
your paid account unlocks gated models used in this repo's benchmark suite.

### 6.1 The HuggingFace Model Hub

Every model in this repo's registry (75 models, see `context.md` Section 4) lives on HF Hub.
Each model page has:
- **Model card** — explains architecture, training data, intended use, limitations
- **Files and versions** — the actual weight files, config, tokenizer
- **Discussions** — known issues, community tips
- **Spaces** — interactive demos

### 6.2 Loading Models: Two Paths

**Path 1 (JAX): FlaxAutoModel**
```
FlaxAutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```
Downloads Flax weights (`flax_model.msgpack`) if available, otherwise converts from PyTorch.
Returns a Flax module whose `params` dict is a standard Python dict of JAX arrays.

**Path 3 (PyTorch): AutoModel**
```
AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```
Downloads PyTorch weights (`pytorch_model.bin` or `model.safetensors`).
Returns a `nn.Module` you can call with `.forward()` or directly.

Both load the same weights — just different formats. This is why Path 1 and Path 3 results
are comparable: same numbers, different execution engine.

### 6.3 Gated Models — Your HF Pro Account Unlocks These

These models in the benchmark suite require HF access:
- **Gemma-2B/3B/4B** (`google/gemma-*`) — Google's TPU-co-designed models; critical for the "home turf" benchmark
- **PaliGemma-3B** (`google/paligemma-*`) — Google's VLM; the strongest TPU advantage model
- **RecurrentGemma-2B** (`google/recurrentgemma-2b`) — RNN vs transformer comparison
- **CodeGemma-2B** (`google/codegemma-2b`) — code domain variant
- **Llama-3.2-1B/3B** (`meta-llama/Llama-3.2-*`) — Meta's models

To request: go to each model page → "Gated model" banner → click to accept license.
Your HF token then allows download automatically.

### 6.4 HuggingFace Inference API — Path 5 in This Repo

With HF PRO, you get access to the Inference API — send a JSON request, get a model prediction.
No VM, no GPU management, no JAX or PyTorch setup. Pure HTTP.

**Serverless Inference:** Shared pool, variable latency, free tier + PRO tier rate limits.
**Dedicated Endpoints:** You pay by the hour for a dedicated GPU (A10G, A100, etc.).
Always warm — no cold-start delays. You choose the hardware.

This is Path 5 in the benchmark — it answers: "What does managed serving cost you vs self-hosted?"

### 6.5 Caching Models in Colab

Colab resets its disk every session. Each time you start a new session, models re-download.
With 75 models averaging 3 GB each, that is 225 GB of downloads — impractical.

Two solutions:
- **Google Drive mount**: persist the HF cache (`/root/.cache/huggingface/`) to Drive between sessions
- **GCS bucket** (for Cloud TPU VMs): `gsutil cp` the cache to GCS once; download to each VM in ~30 seconds

With Colab Pro and Drive, you can keep frequently used models (BERT, ViT, GPT-2, Gemma-2B)
cached across sessions without re-downloading.

**Key terms for this module:**
- **FlaxAutoModel / AutoModel** — HuggingFace's automatic class selection based on model type
- **safetensors** — a safe, fast format for model weights (replacing `.bin`); preferred on HF
- **tokenizer** — converts raw text to integer token IDs that the model expects
- **model card** — documentation page for a model on HF; read this for every new model
- **gated model** — requires license agreement before download
- **Dedicated Endpoint** — a paid HF service: your own GPU, always warm, per-hour billing
- **HF cache** — local directory where downloaded models are stored; `~/.cache/huggingface/`

**Repo touchpoints:** `04_bert_finetuning/train.py`, `context.md` Section 4.5 (decoder justifications),
`context.md` Section 2.5 (HF integration)

**HuggingFace activity:**
Navigate to `google/gemma-2b`. Read the model card fully. Note: architecture details,
context length (8192), training data size. Then navigate to "Files and versions" — 
note the `config.json` (architecture description) and the `.safetensors` files (weights).
This JSON file is what `from_pretrained` reads to build the model architecture before loading weights.

**Colab Pro activity:**
In Colab, set an environment variable:
`import os; os.environ['HF_TOKEN'] = 'your_token_here'`
Then (with Gemma access approved): `from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("google/gemma-2b")`
Watch the download progress. Note the cache directory path in the output.

**Self-check:**
- What is the difference between `FlaxAutoModel` and `AutoModel`?
- If a Gemma-2B model has 2 billion parameters in BF16, how many gigabytes is it on disk?
- What is a Dedicated Endpoint and when would you use it instead of Serverless Inference?

---

## Module 7 — Running Your First Benchmark (Smoke Suite)

**What you will understand:**
How the benchmark harness works, what a single experiment looks like end-to-end,
and how to read the output.

### 7.1 The Harness Structure

```
benchmarks/harness.py   ←  you call this; it parses arguments
        ↓
benchmarks/runner.py    ←  runs one experiment; returns a metrics dict
        ↓
observe/*.py            ←  collects hardware metrics, stats, lineage
        ↓
results/runs.jsonl      ←  one JSON line appended per experiment
```

### 7.2 The 9 Phases of One Experiment

Refer to `context.md` Section 6 for the full table. Key phases:

1. **Pre-flight**: record GPU temperature and clock speed before starting
2. **Compile**: run one forward pass; record how long XLA/torch.compile takes
3. **Warmup**: 20 passes discarded; CUDA kernels and XLA caches stabilise
4. **Latency (bs=1)**: 100 passes; record p50, p95, p99 milliseconds
5. **Throughput (bs=max)**: 100 passes at largest batch that fits; record samples/sec
6. **Profiler**: 10 passes with full tracing enabled; expensive but thorough
7. **Memory sweep**: increase batch size from 1 until OOM; record limit
8. **Numerics**: compare output to FP32 reference; record accuracy delta
9. **Post-flight**: check if clocks dropped (thermal throttling)

### 7.3 Reading a Result Row

A single row in `results/runs.jsonl` tells you everything about one experiment.
Key fields to understand first:

| Field | What it means |
|-------|--------------|
| `compile_time_s` | How long the compiler took on the first call |
| `latency_p50_ms` | Half of all passes were faster than this |
| `latency_p99_ms` | Only 1% of passes were slower than this |
| `latency_cv_pct` | Coefficient of variation — how stable the results are (lower = more reliable) |
| `throughput_mean_samples_sec` | How many inputs processed per second at max batch |
| `peak_memory_gb` | How much HBM/VRAM was used at peak |
| `mfu_pct` | Model FLOP Utilisation — % of the chip's theoretical FLOPs you actually used |
| `arithmetic_intensity` | FLOPs/byte — tells you where on the roofline this model sits |
| `output_cosine_sim_vs_fp32` | How close BF16/INT8 output is to FP32 (1.0 = identical) |
| `flags` | Any automated warnings (e.g. `["high_variance", "throttle_detected"]`) |

### 7.4 The Smoke Suite on Colab Pro

The `smoke` suite runs one model (BERT-base) with two precision variants.
It takes ~8 minutes on Colab Pro TPU or GPU.
After it runs, you have 4 rows in `runs.jsonl` — your first real data.

**Key terms for this module:**
- **p50 / p95 / p99 latency** — percentile latencies; p99 tells you worst-case behaviour
- **coefficient of variation (CV)** — standard deviation ÷ mean × 100%; measures noise in measurements
- **MFU (Model FLOP Utilisation)** — achieved TFLOPs ÷ peak TFLOPs × 100%; the ultimate efficiency metric
- **throughput** — inputs processed per unit time; measured at maximum batch size
- **latency** — time for one request to complete; measured at batch size = 1
- **OOM (Out of Memory)** — when the model + activations exceed the chip's memory; the batch size limit

**Repo touchpoints:** `benchmarks/harness.py`, `benchmarks/runner.py`, `context.md` Section 6 and 8

**Colab Pro activity (TPU runtime):**
After the harness is built in Stage 1, run:
`!python benchmarks/harness.py --suite=smoke --framework=jax --device=tpu`
Observe: compilation takes 10–45 seconds. Then warmup. Then latency. Note the difference.

**Colab Pro activity (GPU runtime):**
Switch to GPU runtime. Run the same command with `--device=gpu --framework=pytorch`.
Compare `compile_time_s`: torch.compile is faster than XLA compile.
Compare `latency_p50_ms`: which is lower?

**Self-check:**
- Why is `latency_cv_pct` an important quality check?
- If MFU is 15%, what does that mean about the chip's efficiency?
- Why do we run 20 warmup passes and discard them?

---

## Module 8 — Compiler Deep Dive: XLA vs CUDA

**What you will understand:**
What happens inside the compiler, why it matters as much as the hardware, and how to read
profiler traces to see the compiler's decisions.

### 8.1 Operator Fusion — The Core Compiler Job

Consider LayerNorm followed by GeLU activation followed by a residual addition.
Naively, each is a separate kernel:
1. Read activations from HBM → compute LayerNorm → write to HBM
2. Read from HBM → compute GeLU → write to HBM
3. Read from HBM → compute addition → write to HBM

That is 6 HBM round-trips for 3 elementwise operations. All three ops are memory-bound
(low arithmetic intensity). Each HBM access is the bottleneck.

A fusing compiler combines them into one kernel:
1. Read activations once → compute LayerNorm + GeLU + addition → write once

That is 2 HBM round-trips instead of 6 — 3× less memory traffic for these ops.

**XLA fuses aggressively by default.** Every `jax.jit` call triggers fusion analysis.
**PyTorch eager mode fuses nothing.** `torch.compile` fuses but requires explicit opt-in.

This is why JAX code is often faster than equivalent PyTorch eager code even on the same hardware —
not because JAX is cleverer, but because XLA always compiles and fuses.

### 8.2 XLA Compilation Phases

When you call a `jax.jit` function for the first time:

1. **Tracing**: JAX runs your Python function with abstract (shape-only) values;
   records every operation into an HLO program. ~0.1s
2. **HLO optimisation**: XLA analyses the HLO; applies fusion, constant folding, layout optimisation. ~1–5s
3. **Code generation**: XLA generates machine code for the target (TPU instructions or PTX for GPU). ~5–40s
4. **Caching**: compiled binary is saved to disk (`JAX_COMPILATION_CACHE_DIR`). Next run: ~0.3s

On first call, this takes 12–45 seconds. After that, it runs at full speed.

### 8.3 Static Shape Constraint — Why It Matters

XLA compiles for a specific input shape. If you change the batch size, it recompiles.
For the benchmark: this is why every suite uses fixed batch sizes for each experiment.
For production serving: this is why TPU serving often uses fixed batch size = 8 or 16,
with padding for smaller requests — to avoid recompilation.

### 8.4 `torch.compile` — PyTorch's Answer

PyTorch 2.0 added `torch.compile`, which:
- Traces your function using TorchDynamo (captures the computation graph)
- Compiles via TorchInductor to optimised CUDA code (or XLA if using torch_xla)
- Handles dynamic shapes better than XLA (can recompile for different shapes quickly)
- Compilation is faster than XLA (~5–20s vs 12–45s first call)
- Final performance is often slightly below XLA on TPU, comparable on GPU

`max-autotune` mode: TorchInductor tries multiple kernel implementations and benchmarks them.
Compilation takes longer (minutes) but produces the fastest possible GPU code.

### 8.5 Reading a Profiler Trace

After running an experiment with profiling enabled, you get a `.pb` trace file.
Open it in TensorBoard (`tensorboard --logdir=results/run_logs/`) or Perfetto (`ui.perfetto.dev`).

**What to look for:**
- **Fused ops**: look for single kernel spans that cover multiple logical operations (e.g. `fused_LayerNorm_GeLU_add`)
- **Gaps**: empty space between kernels means the GPU/TPU was idle — waiting for data or launch overhead
- **Long kernels**: the biggest kernel spans are your performance bottleneck
- **Memory copy events**: data moving between host and device; should be minimal during inference

**Key terms for this module:**
- **HLO (High Level Operations)** — XLA's computation graph format; human-readable text
- **Kernel fusion** — combining multiple ops into one hardware kernel; reduces memory traffic
- **Tracing** — running a function with abstract values to capture its computation graph
- **Recompilation** — XLA compiling again because input shape changed; expensive
- **TorchDynamo** — PyTorch's graph capture system for `torch.compile`
- **TorchInductor** — PyTorch's code generation backend; generates optimised CUDA or C++
- **Profiler trace** — a timeline recording of which kernels ran, when, and how long
- **Perfetto** — Google's open-source trace viewer (use at `ui.perfetto.dev`)

**Repo touchpoints:** `observe/tracer.py`, `observe/hlo_analyser.py`,
`context.md` Section 9.8 and 9.9

**HuggingFace activity:**
On HF, search for `bert-base-uncased`. Go to "Files and versions".
Download `config.json` and open it. Read `num_hidden_layers`, `hidden_size`, `num_attention_heads`.
Calculate: how many matrix multiplications happen in one forward pass?
(Answer: 4 per layer for attention Q/K/V/O, 2 per layer for FFN = 6 per layer × 12 layers = 72 matmuls)

**Self-check:**
- LayerNorm has arithmetic intensity ~5 FLOPs/byte. After fusion with GeLU (also ~5 FLOPs/byte),
  what happens to the combined intensity? Is the fused op faster? Why?
- If you change batch size from 32 to 31 in JAX, what happens? How would you work around this?
- When would you use `torch.compile(max-autotune)` vs `torch.compile(default)`?

---

## Module 9 — Model Architecture Classes and Hardware Fit

**What you will understand:**
Why some model architectures run faster on TPU and others on GPU — from the silicon perspective.

### 9.1 The Standard Transformer Block

All BERT, GPT, Gemma, Qwen, Phi models use this core block:

```
Input
  ↓
LayerNorm                         ← elementwise; memory-bound
  ↓
Q, K, V projections               ← 3 large matmuls; compute-bound if batch large
  ↓
Attention (Q×K^T, softmax, ×V)   ← matmul + softmax; quadratic in seq_len
  ↓
Output projection                 ← 1 large matmul
  ↓
Residual add                      ← elementwise; memory-bound
  ↓
LayerNorm                         ← elementwise; memory-bound
  ↓
FFN: up-projection (matmul)
     GeLU / SwiGLU                ← elementwise; memory-bound
     down-projection (matmul)    ← compute-bound
  ↓
Residual add                      ← elementwise; memory-bound
```

The compute-bound parts (large matmuls) suit both MXU and Tensor Cores well.
The memory-bound parts (LayerNorm, GeLU, residual) get fused by XLA — free on TPU.
Net result: transformers are the architectures both chips were designed for.

### 9.2 Convolution — Standard vs Depthwise

**Standard 3×3 convolution (ResNet):**
Output[x,y,c_out] = Σ over (3×3 patch × c_in inputs)
For a 64-channel conv: 64×64×9 = 36,864 MACs per output position.
This is a large matrix multiply in disguise — excellent MXU/Tensor Core fit.

**Depthwise 3×3 convolution (EfficientNet):**
Output[x,y,c] = Σ over (3×3 patch × 1 input channel only)
For a 64-channel depthwise: 64×9 = 576 MACs per output position.
Each channel is computed independently — cannot be expressed as one large matmul.
Presented to the MXU as 64 separate 1×9 matrix multiplications — 128× fewer FLOPs per tile.
MXU sits mostly idle. This is why EfficientNet is slow on TPU despite fewer total FLOPs.

### 9.3 Attention Variants and Memory

All attention variants compute the same thing mathematically but with different numbers of
K and V heads, which changes memory bandwidth requirements:

| Variant | KV heads | KV-cache size | Use case |
|---------|---------|--------------|---------|
| MHA (Multi-Head Attention) | Same as Q heads (e.g. 32) | Large | BERT, GPT-2, original LLaMA |
| GQA (Grouped Query) | Q_heads ÷ group (e.g. 8 KV for 32 Q) | Medium | LLaMA-3, Gemma-2, Qwen2.5 |
| MQA (Multi-Query) | Always 1 KV head | Tiny | Falcon, Gemma-2B |

More KV heads = more parameters = more memory bandwidth during decode.
MQA cuts KV-cache by 32× vs MHA — decode becomes faster but quality can drop.
The benchmark measures this directly: MQA models should decode faster per token.

### 9.4 Non-Transformer Architectures — The Hardware Mismatch

**State Space Models (Mamba):**
Mamba processes sequences by maintaining a hidden state that gets updated each token.
The update is: h_t = A × h_{t-1} + B × x_t (a recurrence).
This is sequential — each step depends on the previous one.
GPU: has a hand-written CUDA kernel that processes the recurrence in a parallel scan.
TPU: no native XLA primitive for parallel scan; falls back to sequential loop — 3–5× slower.

**Mixture of Experts (Phi-3.5-MoE, DeepSeek-Coder-V2-Lite):**
Each token is routed to 2 out of 16 (or 6 out of 64) expert FFN layers.
Which experts activate depends on the input — dynamic per token.
GPU: `torch.gather` and `torch.scatter` handle dynamic indexing efficiently.
TPU: XLA requires static shapes; dynamic expert selection requires padding to worst case
or falling back to running all experts (destroying the efficiency gain).

### 9.5 Positional Encodings — A Small but Measurable Difference

| Encoding | How it works | Compute cost |
|----------|-------------|-------------|
| Absolute (BERT) | Lookup a fixed embedding per position | One embedding table lookup |
| RoPE (LLaMA, Gemma) | Rotate Q and K vectors by angle proportional to position | 2 sin/cos + 2 multiply per head per layer |
| ALiBi (BLOOM, MPT) | Add a linear bias to attention scores | 1 addition per attention score |
| None (some ViTs) | No positional information | Free |

RoPE is now standard for most modern LLMs. Its sin/cos computation is elementwise
(memory-bound) and fuses with the Q/K computation in XLA.
ALiBi requires adding a bias to the attention matrix — fits differently in fused attention kernels.

**Key terms for this module:**
- **FFN (Feed-Forward Network)** — the 2-matmul block after attention in each transformer layer
- **SwiGLU** — a gated variant of GeLU activation used in LLaMA, Gemma; requires 3 matmuls instead of 2 in the FFN
- **KV-cache** — cached key and value tensors from previous tokens; enables efficient autoregressive decoding
- **Parallel scan** — a technique that computes recurrent operations in O(log n) parallel steps instead of O(n) sequential steps; requires special hardware support
- **Expert** — in MoE, one of the parallel FFN alternatives; only a subset are activated per token
- **RoPE (Rotary Position Embedding)** — encodes position by rotating Q and K in complex space

**Repo touchpoints:** `context.md` Section 4 (full model registry with justifications),
models `05_gpt_pretraining/model.py` (transformer block), `02_mnist_classification` (CNN)

**HuggingFace activity:**
Go to `mistralai/Mistral-7B-v0.3` model page. Read the model card section on "Sliding Window Attention".
Mistral uses a 4096-token local window instead of full 32k attention. Understand why: O(n²) becomes O(n×w).
Now go to `google/gemma-2-2b`. Read about its alternating local/global attention layers.
Why might this design be better than always-local or always-global?

**Self-check:**
- Why does depthwise convolution have lower arithmetic intensity than standard convolution?
- In GQA with 4 KV heads and 32 Q heads, how many unique K/V matrices are computed per layer?
- Why can Mamba run in O(1) memory during inference but a transformer requires O(n) memory?

---

## Module 10 — Precision and Quantization Deep Dive

**What you will understand:**
What BF16, INT8, and INT4 mean numerically, how they affect hardware performance,
and how to verify whether quantization preserves model quality.

### 10.1 Floating Point Formats

Every number in a neural network is stored as bits. The format determines:
- **Precision**: how many significant digits you can represent
- **Range**: the largest and smallest numbers representable
- **Hardware support**: which chips have native hardware for this format

| Format | Total bits | Sign | Exponent | Mantissa | Range | Notes |
|--------|-----------|------|----------|----------|-------|-------|
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | Standard; always supported |
| FP16 | 16 | 1 | 5 | 10 | ±6.5×10⁴ | Narrow range; can overflow |
| BF16 | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | Same range as FP32; lower precision |
| FP8 | 8 | 1 | 4/5 | 3/2 | varies | Newest; hardware in B200/v5e/v6e |
| INT8 | 8 | 1 | — | 7 | -128 to 127 | Integer, not float; requires calibration |
| INT4 | 4 | 1 | — | 3 | -8 to 7 | Extreme compression; GPU only |

**Why BF16 is preferred over FP16 for LLMs:**
BF16 has the same exponent range as FP32 (8 bits) — it can represent the same magnitude of numbers.
FP16 has only 5 exponent bits — large gradients or activations can overflow to infinity.
BF16 was designed specifically to be a safe FP32 drop-in replacement for deep learning.

**Why BF16 is free on TPU:**
The TPU MXU hardware runs both FP32 and BF16 at the same clock speed and the same number of
operations per cycle. Using BF16 halves the memory needed but does not speed up compute.
The benefit is that you can fit 2× more data in HBM — larger batches, larger models.

**Why BF16 is a speedup on GPU:**
Tensor Cores have separate hardware for FP32 and BF16/FP16. The BF16 Tensor Core is 2× wider —
it performs 2× more operations per cycle than the FP32 path. So BF16 on GPU is both
smaller in memory AND faster to compute.

### 10.2 INT8 Quantization

INT8 quantization replaces 32-bit floating point weights (and sometimes activations)
with 8-bit integers. This requires a calibration step:
- Run the model on a small dataset
- Record the min and max values of each weight tensor
- Map the float range [min, max] to integer range [-128, 127]

At inference: multiply the integer by a scale factor to recover approximate float values.

**Post-Training Quantization (PTQ):** quantize after training is done; fast but may lose accuracy.
**Quantization-Aware Training (QAT):** simulate quantization during training; better accuracy.

For this repo: we use PTQ only (inference focus). The `observe/numerics.py` module
measures whether quantization preserves accuracy by comparing INT8 outputs to FP32 outputs.

### 10.3 GPTQ and AWQ — Advanced INT4 for LLMs

For 4-bit quantization, naive PTQ loses too much accuracy. Two better approaches:
- **GPTQ** (2022): uses second-order gradient information to minimise quantization error per layer
- **AWQ** (2023): scales weights before quantization based on activation magnitudes; preserves outliers

Both require a calibration dataset and produce INT4 weights that can be loaded directly.
`bitsandbytes` library (GPU) and `optimum` library (HF) support both.
These are GPU-only currently — TPU does not support INT4 inference natively.

### 10.4 FP8 — The New Frontier

FP8 has two variants: E4M3 (4-bit exponent, 3-bit mantissa — higher precision) and
E5M2 (5-bit exponent, 2-bit mantissa — higher range).

FP8 hardware exists on: B200, H100, RTX 4090, TPU v5e and v6e.
In practice: FP8 inference is 2× faster than INT8 on supported hardware.
Still maturing — not all models or frameworks support it cleanly yet.

### 10.5 How the Benchmark Measures This

For each model, the benchmark runs:
1. FP32 → records all outputs, saves as reference
2. BF16 → compares to FP32; records `output_cosine_sim_vs_fp32`
3. INT8 → compares to FP32; flags if cosine sim < 0.99
4. FP8 (if hardware supports) → same

A result like `cosine_sim = 0.9998` means BF16 is nearly identical to FP32.
`cosine_sim = 0.97` for INT8 means visible quality degradation on this model.

**Key terms for this module:**
- **Quantization** — reducing the numerical precision of weights/activations to save memory and speed inference
- **Scale factor** — the multiplier that maps integer values back to approximate float values
- **Calibration** — running the model on sample data to determine appropriate scale factors
- **Outliers** — extreme activation values that cause poor quantization if not handled
- **GPTQ** — post-training quantization method using Hessian information
- **AWQ** — activation-aware weight quantization; preserves salient weights
- **bitsandbytes** — Python library for efficient INT8/INT4 inference on GPU
- **cosine similarity** — measure of angle between two vectors; 1.0 = identical direction; used to compare outputs

**Repo touchpoints:** `context.md` Section 5 (Precision table), Section 9.6 (numerics.py),
`variants/precision.py` (when built in Stage 6)

**HuggingFace activity:**
Search for `Qwen/Qwen2.5-3B` on HF. Then search for `Qwen2.5-3B-GPTQ`. Compare file sizes:
the GPTQ version should be ~4× smaller. Read the GPTQ model card for quantization details.

**Self-check:**
- Why can BF16 represent the number 1×10³⁰ but FP16 cannot?
- If INT8 quantization cuts model size by 4× (vs FP32), why is the speedup on GPU only 2× (not 4×)?
- What does a cosine similarity of 0.95 between INT8 and FP32 outputs imply about model quality?

---

## Module 11 — LLM Inference: Prefill, Decode, and KV-Cache

**What you will understand:**
Why running a language model for inference is not one problem but two — and why they have
completely different performance characteristics on TPU vs GPU.

### 11.1 Two Phases of LLM Inference

**Phase 1 — Prefill (Prompt Processing):**
Input: full prompt tokens [t₁, t₂, ..., tₙ] simultaneously.
The model processes all n tokens in parallel through all layers.
Output: the probability distribution for token n+1, and the KV-cache for all n positions.
Compute profile: large batch × seq_len → compute-bound (high arithmetic intensity).
This is what happens when you send a long prompt to ChatGPT and wait.

**Phase 2 — Decode (Token Generation):**
Input: one new token at a time (the most recently generated token).
The model uses the KV-cache from prefill (+ previous decode steps) to attend back.
Output: one probability distribution → sample or argmax for next token.
Compute profile: 1 token per step → memory-bound (tiny matmuls, large KV-cache reads).
This is what happens during streaming — each token appearing one by one.

### 11.2 Why Decode Is So Different

In decode mode, you run a full forward pass for just 1 new token.
The Q×K^T attention reads the entire KV-cache (all previous tokens) from HBM.
For a 1B parameter model at context length 1000:
- Weight reads: ~2 GB (FP16)
- KV-cache reads: ~0.5 GB (grows with context)
- Arithmetic: 1000 × attention operations per layer

This is extreme memory bandwidth bound. Arithmetic intensity ≈ 1–10 FLOPs/byte.
All chips are operating far to the left of their roofline ridge point.

**Implication:** at decode time, memory bandwidth is everything.
- B200 at 4000 GB/s bandwidth dominates A100 at 2000 GB/s → 2× faster decode
- TPU v6e at 1640 GB/s is competitive with H100 at 3350 GB/s? Not quite — H100 wins decode.

### 11.3 KV-Cache Memory Growth

As you generate tokens, the KV-cache grows:
KV-cache size = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element

For Gemma-2B (18 layers, 1 KV head per layer group, 256 head_dim, BF16):
At 1000 tokens: 2 × 18 × 1 × 256 × 1000 × 2 bytes = ~18 MB (small, due to MQA)

For GPT-2 XL (48 layers, 25 KV heads, 64 head_dim, BF16):
At 1000 tokens: 2 × 48 × 25 × 64 × 1000 × 2 bytes = ~307 MB

This is why MQA and GQA exist: they reduce the KV-cache, enabling larger batches
or longer contexts for the same memory.

### 11.4 DeepSeek-R1: The Reasoning Decode Problem

Reasoning models (DeepSeek-R1-Distill-Qwen-1.5B) produce 500–2000 tokens of
chain-of-thought before giving a final answer. This means:
- Decode phase is 10–20× longer than a standard chat response
- KV-cache grows to 2000+ tokens
- Sustained memory bandwidth is the entire story
- Time-to-first-token (TTFT): how long to process the prompt
- Time-per-output-token (TPOT): how fast each generated token appears

The benchmark measures both TTFT and TPOT separately for all decoder models.
For R1 models, the TPOT × 2000 = total response latency that matters most.

### 11.5 Batched Decode — Throughput Mode

If you serve many users simultaneously, you can batch their decode steps:
run 32 users' decode steps in one forward pass instead of 32 sequential passes.

This dramatically increases arithmetic intensity (from ~5 to ~50+ FLOPs/byte),
making batched decode more compute-bound. TPU's higher FLOPs advantage starts to show.

The benchmark's throughput measurement (max batch, high samples/sec) corresponds to this scenario.

**Key terms for this module:**
- **Prefill** — processing all prompt tokens simultaneously; compute-bound
- **Decode** — generating one token per step using KV-cache; memory-bound
- **KV-cache** — stored Key and Value tensors from all previous positions; grows with context length
- **TTFT (Time to First Token)** — latency until the first generated token appears; dominated by prefill
- **TPOT (Time Per Output Token)** — speed of each subsequent token; dominated by decode bandwidth
- **Tokens/sec** — throughput measure for language generation; the standard LLM serving metric
- **Context length** — maximum number of tokens the model can attend to; determines max KV-cache size

**Repo touchpoints:** `context.md` Section 5 (LLM Inference Modes), Section 4.5 (decoder model justifications)

**HuggingFace activity:**
Go to `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Click "Model card". Read the section
on chain-of-thought reasoning and example outputs. Notice the long reasoning traces before
final answers. Count how many tokens a typical reasoning chain contains.

**Self-check:**
- Why is batch_size=1 latency dominated by memory bandwidth rather than compute?
- For a model with MQA (1 KV head), how does KV-cache size compare to a model with MHA (32 KV heads) at the same context length and model depth?
- If you double the context length, by how much does the prefill compute increase?

---

## Module 12 — Sparsity and Pruning: What They Can and Cannot Do

**What you will understand:**
Why removing weights from a model does not always make it faster, and when it does.

### 12.1 Unstructured Pruning — The Illusion of Speedup

Set 90% of weight values to exactly zero. The model now has 90% fewer non-zero parameters.
In theory: 10× less compute. In practice: 0–2% speedup on both GPU and TPU.

Why? Because zero multiplications are not skipped by hardware.
The GPU/TPU loads the full weight matrix from memory and multiplies everything, including zeros.
The savings require hardware that can detect and skip zero-multiplication — neither standard
GPU Tensor Cores nor TPU MXU do this.

Unstructured sparse patterns also destroy cache locality — irregular memory access is slower.
High sparsity can make the model slower than dense, not faster.

### 12.2 Structured Pruning — Actual Speedup

Instead of zeroing individual weights, remove entire rows or columns from weight matrices.
A weight matrix that was 512×512 becomes 512×384 after channel pruning.
Now every matrix multiply is genuinely smaller. Both TPU and GPU see linear speedup.

The cost: you lose model capacity. Quality drops more than with unstructured pruning at the
same compression ratio. This is the tradeoff the benchmark measures.

### 12.3 NVIDIA's 2:4 Structured Sparsity — GPU-Exclusive

Ampere (RTX 3080), Ada Lovelace (RTX 4090), and Blackwell (B200) GPUs support
a hardware pattern: in every group of 4 consecutive weights, exactly 2 must be zero.
This 2:4 pattern is stored in a compressed format — the Sparse Tensor Core runs the
non-zero 2 values only, effectively doubling throughput for the weight matrix.

Result: a 2:4-sparse model runs at exactly 2× the throughput of its dense counterpart.
This requires:
- Fine-tuning the model to enforce the 2:4 pattern (not just post-hoc pruning)
- Using `apex.contrib.sparsity` or NVIDIA's sparse training tools
- The 2:4 constraint limits quality compared to unstructured pruning at same sparsity

TPU has no equivalent. The MXU has no mechanism to skip pairs of zeros.
This is the one structural hardware advantage NVIDIA has over TPU at the silicon level.

**Key terms for this module:**
- **Pruning** — removing weights from a model to reduce size and (potentially) computation
- **Unstructured sparsity** — zeros scattered randomly through weight matrices; no hardware benefit
- **Structured sparsity** — removing entire rows/columns; reduces matrix dimensions; real speedup
- **2:4 sparsity** — NVIDIA hardware pattern: 2 zeros in every group of 4 weights; 2× Tensor Core speedup
- **Sparse Tensor Cores** — hardware in Ampere/Ada/Blackwell that processes 2:4 patterns at 2× speed
- **Magnitude pruning** — zeroing the smallest-magnitude weights; simplest form of unstructured pruning

**Repo touchpoints:** `context.md` Section 5 (Sparsity table), `variants/pruning.py` (Stage 8)

**Self-check:**
- A model has 90% unstructured sparsity. Why does it not run 10× faster?
- What does the benchmark need to measure to prove the 2:4 sparsity speedup on RTX 4090?
- Why can't TPU benefit from 2:4 sparsity even in principle?

---

## Module 13 — Full Observability: Reading Evidence for Every Claim

**What you will understand:**
How to use every piece of evidence this repo generates to verify or challenge any claim.

### 13.1 The Evidence Chain

For any claim (e.g. "Gemma-2B is the fastest 2B model on TPU"), follow this chain:

```
Claim in README
  → Chart in dashboard (throughput.html, filtered to 2B models, TPU device)
    → Specific run_id values shown in chart tooltip
      → results/runs.jsonl, grep for those run_ids
        → results/run_logs/<run_id>/
          → raw_timings.jsonl  (100 individual pass timings)
          → system_state.json  (MXU% during those passes)
          → lineage.json       (git SHA, model revision, input seed)
```

Every number in a chart has a full chain of evidence behind it.
If you disagree with a result, you can take the lineage.json and reproduce the exact experiment.

### 13.2 Hardware Utilisation — The Honest Number

`mfu_pct` (Model FLOP Utilisation) = achieved TFLOPs ÷ peak TFLOPs × 100%.

A TPU v5e with MFU = 70% for a ViT-L model means:
- The chip is running at 70% of its 394 TFLOPs peak
- 30% is lost to: memory wait time, kernel launch overhead, unoptimised ops, compiler gaps
- An MFU of 70% is excellent for a large transformer

An MFU of 15% for EfficientNet on TPU v5e means:
- The depthwise conv is so poorly suited to the MXU that 85% of compute capacity is wasted
- Even though EfficientNet has fewer total FLOPs, it uses a smaller fraction of the chip's capability

MFU is the honest metric. "Peak TFLOPs" from a spec sheet is only achieved on perfectly
optimised code with ideal problem sizes. Real models land at 20–75% MFU.

### 13.3 Statistical Validity — When to Trust a Number

A result from one 50-pass measurement is not trustworthy. The benchmark enforces:
- n=3 independent runs (separate measurement blocks)
- CV (coefficient of variation) < 10% to accept the result
- If CV > 10%: the `high_variance` flag is set and the claim cannot be made

Before trusting any comparison ("A is 2× faster than B"), check:
- Are both measurements from runs without `high_variance` or `throttle_detected` flags?
- Is the gap larger than the combined standard deviation of both measurements?
- Were both measured at the same temperature and clock state?

### 13.4 Using TensorBoard + Perfetto for Compiler Claims

The claim "XLA fuses LayerNorm + GeLU into one kernel" is verified by:
1. Open the JAX profiler trace in TensorBoard or Perfetto
2. Look for a single kernel span labelled `fused_*` that covers both ops
3. Compare to the PyTorch eager trace where they appear as two separate kernels
4. Count: JAX XLA → 1 kernel; PyTorch eager → 2 kernels; `torch.compile` → 1 kernel (after compile)

The `hlo_dump.txt` file is the XLA computation graph in text form.
Each `fusion {}` block in this file is one fused kernel on hardware.
Counting fusion groups before and after `torch.compile` or `jax.jit` tells you the compiler's work.

**Key terms for this module:**
- **MFU (Model FLOP Utilisation)** — fraction of peak compute actually used; the real efficiency number
- **Coefficient of variation (CV)** — noise in measurements; must be < 10% for trustworthy results
- **Lineage** — the complete record of what code, weights, inputs, and hardware produced a result
- **Reproducibility** — ability to re-run an experiment and get the same result within noise bounds
- **`hlo_dump.txt`** — XLA's text representation of the computation graph; shows fusion decisions

**Repo touchpoints:** `observe/` directory (all modules), `context.md` Section 14 (Evidence Chain)

**Colab Pro activity:**
After running any benchmark, open `results/run_logs/<run_id>/system_state.json`.
Read the `mxu_utilization_pct` field. Look at `raw_timings.jsonl` and calculate the standard
deviation yourself. Verify it matches `latency_std_ms` in `runs.jsonl`.

**Self-check:**
- If a benchmark result has CV = 15%, why should you not trust a "2× speedup" claim from it?
- What does MFU = 35% tell you about a model on a given chip?
- How would you verify the claim "BF16 is free on TPU" using only `runs.jsonl`?

---

## Module 14 — Total Cost of Ownership: The Real Decision

**What you will understand:**
How to combine all benchmark metrics into a decision framework for hardware selection.

### 14.1 Cost Per 1k Samples — The Normalised Metric

Raw throughput comparisons are misleading without normalising by cost.
The benchmark records `cost_per_1k_samples_usd` for every experiment.

For cloud hardware: cost/hr × (1 hr ÷ throughput × 1000 samples).
For local hardware: electricity cost/hr × (1 hr ÷ throughput × 1000 samples).
  (B200 at 1000W, $0.12/kWh = $0.12/hr electricity)

Example for BERT-base BF16:
| Hardware | Throughput | Cost/hr | Cost/1k samples |
|---------|-----------|---------|----------------|
| v5e-1 preemptible | 5,000 samples/s | $0.36 | $0.000020 |
| RTX 4090 (local electricity) | 8,000 samples/s | $0.12 | $0.0000042 |
| H100 Lambda Labs | 15,000 samples/s | $2.49 | $0.000046 |
| B200 local (electricity) | 40,000 samples/s | $0.12 | $0.00000083 |

Your B200 locally is the cheapest by far at scale. But it is also fixed infrastructure —
if your workload is intermittent, you pay for it whether it runs or not.
Cloud TPU preemptible only costs money when running.

### 14.2 Utilisation Rate — The Break-Even

If your B200 runs at 10% utilisation (only busy 2.4 hrs/day), the amortised cost
over a 3-year server life becomes:
~$50,000 hardware ÷ (3 years × 365 days × 24 hrs × 10%) ÷ 8760 hrs
≈ $6/hr effective cost — more expensive than H100 Lambda Labs at $2.49/hr.

At 80% utilisation: $0.75/hr effective — cheaper than any cloud option.

The benchmark's TCO calculator (dashboard `tco.html`) lets you input:
- Utilisation rate
- Hardware amortisation period
- Electricity cost (India: ~$0.07/kWh average)
- Cloud alternative pricing

### 14.3 India-Specific Considerations

Electricity cost in India (~$0.07/kWh) is lower than the US average ($0.12/kWh).
This reduces local GPU running costs by ~42% vs US benchmark numbers.

At high utilisation (>60%), your local B200 and RTX 4090 are almost certainly
cheaper per experiment than any cloud alternative — especially at India electricity rates.

Cloud TPU is still valuable for:
- Experiments that exceed your local GPU memory (v5p for 70B+ models)
- Experiments where you need a specific TPU generation (e.g. v5e vs v3 comparison)
- Reproducible cloud experiments that others can replicate (consistent hardware)

**Key terms for this module:**
- **Cost per 1k samples** — the normalised cost efficiency metric; accounts for both throughput and price
- **Total Cost of Ownership (TCO)** — total cost including hardware amortisation, electricity, and operations
- **Utilisation rate** — fraction of time the hardware is actively running workloads
- **Amortisation** — spreading the one-time hardware purchase cost over its useful lifetime
- **Break-even analysis** — finding the utilisation rate where self-hosted equals cloud cost

**Repo touchpoints:** `context.md` Section 15 (Cost Reference), `results/dashboard/tco.html`

**Self-check:**
- If your RTX 4090 achieves 4× higher throughput than v5e-1, but costs 0× per hour (local electricity ~$0), should you ever use v5e-1? When?
- At what utilisation rate does your B200 (assume $50,000 over 3 years) break even with H100 at $2.49/hr?
- For an Indian researcher running 2 hrs/day of experiments, which hardware is cheapest per sample?

---

## Module 15 — Expert: Connecting Everything — Your Research Workflow

**What you will understand:**
How to design, run, interpret, and communicate a complete hardware comparison study
using this repo as your primary tool.

### 15.1 Research Question Design

A good benchmark research question has:
- A specific claim to test ("TPU v5e-1 matches H100 on transformer inference cost-per-sample")
- A specific controlled variable ("model architecture: transformer encoder only")
- A specific metric ("cost_per_1k_samples_usd at bs=max, BF16")
- A null hypothesis to disprove ("no difference between devices")

Poorly defined: "Is TPU faster than GPU?"
Well defined: "For BERT-base BF16 at batch_size=64, seq_len=128, does TPU v5e-1
(preemptible) achieve lower cost_per_1k_samples than H100 SXM5 (Lambda Labs)?
What is the 95% confidence interval of the difference?"

### 15.2 Planning an Experiment Set

For any research question, choose:
1. **Models**: which subset of the 75-model registry tests your hypothesis?
2. **Variants**: which precision / compile / batch sizes are relevant?
3. **Devices**: which hardware paths (1–5) isolate the variable you care about?
4. **Statistics**: how many independent runs? (n=3 minimum; n=5 for publication-quality)
5. **Suite**: which pre-built suite covers your needs, or do you need a custom YAML?

### 15.3 Running on Colab Pro — Practical Workflow

For a complete TPU vs GPU comparison using only Colab Pro:

**Step 1:** Open one Colab tab, select TPU runtime.
Clone the repo: `!git clone https://github.com/rajaghv-dev/tpu`
Set `HF_TOKEN`. Run: `!python benchmarks/harness.py --suite=quick --framework=jax --device=tpu`
Results go to `results/runs.jsonl`.

**Step 2:** Open a second Colab tab, select GPU runtime (A100).
Clone the repo again (different session, different hardware).
Run: `!python benchmarks/harness.py --suite=quick --framework=pytorch --device=gpu`

**Step 3:** Download both `runs.jsonl` files to your local machine or Drive.
Merge them: `cat tpu_runs.jsonl gpu_runs.jsonl > combined_runs.jsonl`
Open `notebooks/explore.ipynb` with the combined file to compare.

**Step 4:** Push results to GitHub:
`!git add results/runs.jsonl && git commit -m "quick suite TPU v2-8 + Colab A100" && git push`

The GitHub Pages dashboard at `https://rajaghv-dev.github.io/tpu/` then shows updated charts.

### 15.4 Interpreting Multi-Model Results

When you have 24 experiments from the quick suite:
- Sort by `mfu_pct` (descending): which models use the chip best?
- Filter to `device=tpu_v*` and plot `throughput_mean` vs `flops_per_sample_G`:
  should be roughly linear for transformer models; outliers are poorly-fit architectures
- Filter to `flags contains throttle_detected`: discard those measurements; rerun at lower batch size
- Compare `arithmetic_intensity` to the device's ridge point: models above ridge = compute-bound

### 15.5 Making Claims You Can Defend

For every claim you make from the data:
1. State the exact conditions: model, precision, batch size, hardware, framework version
2. Report mean ± std, not just mean: "8.4 ± 0.3 ms" not "8.4 ms"
3. Report n: "n=3 independent runs"
4. Cite the run_ids from `runs.jsonl` that support the claim
5. Include the lineage: git SHA, model revision, input seed
6. State what would falsify the claim: "This claim would not hold if MXU utilisation dropped below 50%"

This is the difference between benchmark marketing and benchmark science.

### 15.6 What You Can Contribute Back to This Repo

Once you have run experiments and understand the results:
- Add a new model to `models/registry.yaml` with its HF ID and input spec
- Add a new benchmark insight to the dashboard as a new chart view
- Add a new "learning arc" to the README if you discover an insight not already documented
- Open an issue if a claim in the README does not match your experimental results

The repo is designed to grow from evidence, not assumptions.

---

## Learning Path Summary

```
Week 1  ─── Modules 1–4:   Environment, hardware basics, roofline, software stack
Week 2  ─── Modules 5–7:   Repo examples, HuggingFace deep dive, first benchmark run
Week 3  ─── Modules 8–9:   Compiler internals, architecture classes
Week 4  ─── Modules 10–11: Quantization, LLM prefill vs decode
Week 5  ─── Modules 12–13: Sparsity, full observability and evidence reading
Week 6  ─── Module 14–15:  TCO analysis, research workflow, contributing back
```

---

## Quick Reference: Key Terms by Module

| Module | Terms |
|--------|-------|
| 1 | Runtime, compute units, gated model, access token, HF Hub |
| 2 | CUDA core, Tensor Core, MXU, systolic array, HBM, TFLOPs |
| 3 | Roofline, arithmetic intensity, ridge point, memory-bound, compute-bound, GEMM |
| 4 | JIT, eager mode, kernel fusion, HLO, static shapes, cuDNN, torch_xla |
| 5 | pmap, prefetch, drop_remainder, bf16 casting |
| 6 | FlaxAutoModel, safetensors, tokenizer, gated model, Dedicated Endpoint, HF cache |
| 7 | p50/p95/p99 latency, CV, MFU, throughput, OOM |
| 8 | Operator fusion, XLA compilation phases, TorchDynamo, TorchInductor, Perfetto |
| 9 | FFN, SwiGLU, KV-cache, MHA/GQA/MQA, RoPE, ALiBi, parallel scan, MoE |
| 10 | BF16, FP8, INT8, INT4, PTQ, QAT, GPTQ, AWQ, bitsandbytes, cosine similarity |
| 11 | Prefill, decode, TTFT, TPOT, tokens/sec, context length |
| 12 | Unstructured/structured sparsity, 2:4 sparsity, Sparse Tensor Cores, magnitude pruning |
| 13 | MFU, CV, lineage, reproducibility, hlo_dump |
| 14 | Cost/1k samples, TCO, utilisation rate, amortisation, break-even |
| 15 | Research question design, controlled variable, null hypothesis, confidence interval |

---

*This lesson plan is a living document. As new experiments are run and new insights are added
to the repo, new modules will be added here. Check `prompts.md` for the history of how
the repo's scope evolved.*
