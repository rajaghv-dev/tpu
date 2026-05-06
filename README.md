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

### One-shot pipeline (recommended) — staged scripts in `scripts/`

```bash
# Clone the repo, install gcloud, authenticate.
git clone https://github.com/rajaghv-dev/tpu && cd tpu

# 1. Local + cloud preflight (free, read-only).
./scripts/00_validate_local.sh
./scripts/01_validate_gcp.sh
./scripts/91_predict_cost.sh smoke tpu_v5litepod_1   # forecast: ~$0.05

# 2. Run the entire pipeline (provision → install → smoke → teardown).
./scripts/run_all.sh --suite smoke
# Or: --suite quick (50 min, ~$0.30) — and --keep-tpu / --no-gcs / --from N to customise.
```

The orchestrator picks a US zone with v5e-1 spot capacity (multi-zone fallback),
deploys the repo, installs `jax[tpu]`, runs the harness inside `tmux`, pulls
results, and tears down. See `scripts/README.md` for the full stage list and
`scripts/lib/config.sh` for every overridable variable.

### Manual harness invocation

```bash
pip install -r requirements.txt        # includes jax[tpu], transformers, pytest

python benchmarks/harness.py --suite smoke --device tpu
python benchmarks/harness.py --suite quick --device tpu
python benchmarks/harness.py --suite quick --device tpu --dry-run    # plan only
python benchmarks/harness.py --model bert_base --device gpu          # single model
```

### Cost monitoring

```bash
./scripts/90_status.sh                             # current burn rate ($/hr)
./scripts/91_predict_cost.sh quick tpu_v5litepod_1 # forecast a planned run
./scripts/92_idle_check.sh                         # flag VMs running >2h
./scripts/71_verify_teardown.sh                    # confirm $0/hr after teardown
```

Results append to `results/runs.jsonl`. Dashboard at `results/dashboard/index.html`.
Run `python scripts/render_results.py` to regenerate `results/RESULTS.md`
(top-level summary + per-probe coverage) and per-run `REPORT.md` files.

### Observability (optional)

```python
from observe.probe import set_active_probes
from observe.timing_probe import TimingProbe
from observe.memory_probe import MemoryProbe
from observe.otel_probe import OTelProbe

set_active_probes([TimingProbe(), MemoryProbe(), OTelProbe()])  # before harness run
```

```bash
# OTel exporter (probe is a no-op unless these are set):
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=tpu-bench
```

### Run tests (no GPU/TPU needed)

```bash
pip install pytest pyyaml numpy
pytest tests/ -v              # ~180 tests (probe tests + harness tests)
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
| `tests/` | ~180 unit tests (pytest, no GPU required) |

**Stage 1 models:** BERT-base · ViT-B/16 · GPT-2 · Whisper-base · CLIP ViT-B/32

**Stage 1 gaps fixed:** C2 (multi-run statistics with CV<10% check) · C3 (XLA cache cleared before every compile measurement)

**Next:** Stage 2 adds Paths 2+3 (JAX+GPU, PyTorch+GPU), system_monitor.py, 15 models, heatmap dashboard.

---

## Probes & Observability

A pluggable probe layer sits alongside the harness. Probes implement the
`Probe` ABC in `observe/probe.py` and opt into any subset of six lifecycle
hooks: `before_run`, `after_run`, `before_phase`, `after_phase`, `on_error`,
`write_log`. The runner fans events out to every registered probe; failures
in one probe are isolated and never break the run. Register them via
`set_active_probes([...])` before invoking the harness.

| Probe | What it captures | Optional dep |
|-------|------------------|--------------|
| `TimingProbe` | Wall-clock per phase + run total | — |
| `MemoryProbe` | psutil RSS/VMS snapshots at phase boundaries | `psutil` |
| `InputFingerprintProbe` | SHA-256 of synthetic inputs (reproducibility) | — |
| `HloDumpProbe` | Sets `XLA_FLAGS` for HLO IR dump; parses summary stats | — |
| `JaxProfilerProbe` | Wraps the latency phase with `jax.profiler.start_trace` | `jax` |
| `CloudMonitoringProbe` | 1 Hz polling of GCP TPU metrics (mxu_util, mem_util, mem_bw_util) | `google-cloud-monitoring` |
| `OTelProbe` | OpenTelemetry spans for runs/phases + histograms (latency, throughput, cost) | `opentelemetry-sdk` |

Grafana dashboards (importable JSON) live at `results/dashboard/grafana/` —
roofline, MXU heatmap, latency violins, failures, cost. See that
directory's README for the data-source wiring.

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
Phase              Passes    Purpose                                         Stage built
─────────────────────────────────────────────────────────────────────────────────────────
1. Pre-flight      —         Verify device reachable; record thermal state   1
2. Compile         1         Clear XLA cache; time cold + warm compilation   1
3. Warmup          20        Discard — kernels and caches stabilise          1
4. Latency         3 × 100   bs=1  →  p50 / p95 / p99 ms, CV %             1
5. Throughput      3 × 100   bs=max  →  samples/sec ± std                   1
6. Profiler        10        Full trace  →  op breakdown, fusion groups      3
7. Memory sweep    1/bs      bs=1,2,4,8,…  →  max batch before OOM         3
8. Numerics        1         Compare BF16/INT8 output to FP32 ref           6
9. Post-flight     —         Verify device still responds; detect throttle   1
```

Stage 1 implements phases 1–5 and 9. Phases 6–8 added in Stages 3 and 6.

---

## Suite Definitions

| Suite | Models | Variants | Experiments | Wall time | v5e-1 cost |
|-------|--------|----------|-------------|-----------|-----------|
| `smoke` | 1 (BERT-base) | BF16 | 1 | ~8 min | $0.05 |
| `quick` | 5 (1 per domain) | BF16 | 5 | ~50 min | $0.30 |
| `domain` | All in one domain | FP32 + BF16 | ~30 | ~60 min | $0.36 |
| `arch` | Novel arches (Mamba, RWKV, DiT, RecurrentGemma) | BF16 | ~20 | ~40 min | $0.24 |
| `llm` | All decoders | BF16, prefill + decode | ~60 | ~2 hrs | $0.72 |
| `full` | All 75 models | All variants | ~800 | ~8 hrs | $2.88 |

Full suite = **$2.88** on preemptible v5e-1. Can be run weekly for ~$10/month.

*Stage 1 (built): `smoke` and `quick`. Remaining suites added in Stages 2–9.*

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
tpu/                                  (github.com/rajaghv-dev/tpu)
│
├── 01_hello_tpu/                     hello_tpu.py · README.md
├── 02_mnist_classification/          train.py · README.md
├── 03_resnet_imagenet/               model.py · train.py · README.md
├── 04_bert_finetuning/               train.py · README.md
├── 05_gpt_pretraining/               model.py · train.py · README.md
├── 06_data_pipeline/                 pipeline.py · README.md
├── 07_custom_training_loop/          train.py · README.md
├── 08_multi_host/                    train.py · README.md
│
├── benchmarks/                       ✅ Stage 1 complete
│   ├── harness.py                    CLI: --suite smoke/quick, --model, --dry-run
│   └── runner.py                     9-phase experiment runner (Path 1: JAX + XLA)
│
├── models/
│   └── registry.yaml                 ✅ 5 Stage 1 models (BERT · ViT · GPT-2 · Whisper · CLIP)
│
├── observe/                          ✅ Stage 1 complete + probe.py + 7 probe modules
│   ├── stats.py                      MAD outlier removal · p50/p95/p99 · CV<10% check
│   ├── lineage.py                    git SHA · package versions · HF model revision
│   ├── compile_controller.py         XLA cache clear · cold + warm compile timing
│   ├── probe.py                      Probe ABC · registry · lifecycle fan-out
│   ├── timing_probe.py               Wall-clock per phase + run total
│   ├── memory_probe.py               psutil RSS/VMS at phase boundaries
│   ├── input_fingerprint.py          SHA-256 of synthetic inputs
│   ├── hlo_dump_probe.py             XLA_FLAGS HLO dump + summary parser
│   ├── jax_profiler_probe.py         jax.profiler trace around latency phase
│   ├── cloud_monitoring_probe.py     1 Hz GCP TPU metrics (mxu/mem/bw util)
│   └── otel_probe.py                 OpenTelemetry spans + histograms
│
├── results/
│   ├── runs.jsonl                    Append-only benchmark results (one JSON line/experiment)
│   └── dashboard/
│       ├── index.html                ✅ Static sortable/filterable table dashboard
│       └── grafana/                  ✅ Importable dashboards (roofline · mxu_heatmap ·
│                                        latency_violins · failures · cost · README)
│
├── tests/                            ✅ ~180 unit tests (no JAX/GPU required)
│   ├── conftest.py
│   ├── test_stats.py
│   ├── test_lineage.py
│   ├── test_compile_controller.py
│   ├── test_registry.py
│   ├── test_runner.py
│   ├── test_harness.py
│   ├── test_app_probes.py
│   ├── test_compiler_probes.py
│   ├── test_cloud_monitoring_probe.py
│   ├── test_otel_probe.py
│   └── test_render_results.py
│
├── scripts/                          ✅ Staged pipeline (00 → 71) + cost monitors (90/91/92)
│   ├── lib/
│   │   ├── common.sh                 Shared logging/error trap/state helpers
│   │   └── config.sh                 Defaults: TPU_NAME, ZONES, GCS_BUCKET, prices
│   ├── README.md                     Stage table + happy path + edge cases
│   ├── run_all.sh                    Master orchestrator (--suite, --from, --keep-tpu, --dry-run)
│   ├── render_results.py             Generate results/RESULTS.md + per-run REPORT.md
│   ├── 00_validate_local.sh          Local preflight (bash, gcloud, python3)
│   ├── 01_validate_gcp.sh            GCP preflight (billing, APIs, IAM, quota)
│   ├── 02_validate_bucket.sh         GCS bucket exists + R/W probe
│   ├── 03_validate_hf.sh             HF_TOKEN valid (optional, for gated models)
│   ├── 10_setup_bucket.sh            Create gs://rajaghv-tpu-cache (idempotent)
│   ├── 11_setup_budget.sh            $5/mo budget alert (idempotent, R8)
│   ├── 20_provision_tpu.sh           v5e-1 spot, multi-zone fallback
│   ├── 21_wait_tpu_ready.sh          Poll until SSH reachable
│   ├── 30_deploy_repo.sh             Tar + scp repo to VM
│   ├── 31_install_deps.sh            pip install jax[tpu] + transformers
│   ├── 32_mount_gcs.sh               gcsfuse + HF_HOME + JAX_COMPILATION_CACHE_DIR
│   ├── 40_verify_jax.sh              Confirm jax.devices() shows TPU
│   ├── 41_run_pytests.sh             pytest tests/ on VM (~180 tests)
│   ├── 42_dry_run.sh                 Harness --dry-run plan
│   ├── 50_run_smoke.sh               Smoke suite (1 model, ~8 min, tmux)
│   ├── 51_run_quick.sh               Quick suite (5 models, ~50 min, tmux)
│   ├── 60_pull_results.sh            scp runs.jsonl + run_logs/ back
│   ├── 70_teardown_tpu.sh            Delete VM (stops billing)
│   ├── 71_verify_teardown.sh         Confirm $0/hr — no leftover resources
│   ├── 90_status.sh                  Current burn rate (+ MTD if BQ export configured)
│   ├── 91_predict_cost.sh            Forecast cost of <suite> on <device>
│   ├── 92_idle_check.sh              Flag VMs running >2h ("possibly forgotten")
│   ├── gcloud_setup.sh               (legacy) Enable GCP APIs
│   ├── provision_tpu.sh              (legacy, preserved) — see 20_provision_tpu.sh
│   ├── teardown_tpu.sh               (legacy, preserved) — see 70_teardown_tpu.sh
│   ├── gcloud_ssh_run.sh             (utility) SSH + run script on remote VM
│   ├── gcloud_pod_run.sh             (utility) Multi-host TPU pod launch
│   └── gcloud_upload_data.sh         (utility) Upload data to GCS
│
├── README.md                         This file
├── MEMORY.md                         Fast session startup reference (read first)
├── SESSION.md                        Complete project state + decisions (read second)
├── DECISIONS.md                      13 ADRs — locked architectural decisions
├── RISKS.md                          25+ risks with mitigations and contingencies
├── QUESTIONS.md                      23 open research questions with test plans
├── RECOMMENDATIONS.md                Prioritised actions in 3 tiers
├── LESSON_PLAN.md                    15-module beginner→expert curriculum
├── context.md                        Full project context (700+ lines)
├── prompts.md                        All session prompts + standing instructions
└── requirements.txt
```

**Stage 2 will add:** `observe/system_monitor.py`, `models/jax/`, `models/torch/`, `results/dashboard/throughput.html`

---

## Cost Reference

| Scenario | Cost |
|----------|------|
| Single experiment (~2 min) on v5e-1 preemptible | $0.012 |
| `smoke` suite (8 min, 1 model) | $0.05 |
| `quick` suite (50 min, 5 models) | $0.30 |
| `full` suite (~8 hrs, 75 models) | $2.88 |
| Weekly `full` suite for 1 month | ~$12 |
| GCS model cache (~100 GB at target) | $2.00/month |
| Colab Pro (TPU + GPU access) | $9.99/month |
| Local RTX 3080 / 4090 / B200 | electricity only (~$0.07/kWh in India) |

### Cost guardrails

- **Forecast first:** `./scripts/91_predict_cost.sh <suite> <device>` prints
  estimated wall-time + USD before you provision.
- **Burn check while running:** `./scripts/90_status.sh` totals the current
  hourly rate from live resources (no BQ export needed). With
  `BILLING_BQ_TABLE` set it also queries month-to-date.
- **Forgotten-VM check:** `./scripts/92_idle_check.sh` flags VMs/TPUs running
  >2h. Add to cron daily for cheap insurance against the $126/week
  "left a v6e on overnight" failure mode.
- **Budget alert:** `./scripts/11_setup_budget.sh` creates a $5/month soft
  alert (50/90/100% email thresholds). Manual setup if your account lacks
  `roles/billing.user` on the billing account.
- **Teardown verification:** `./scripts/71_verify_teardown.sh` lists every
  remaining billable resource (TPUs, VMs, idle reserved IPs, orphan disks)
  so you know the bill is $0/hr.

### Region note (ADR-006 vs current GCP availability)

ADR-003 + ADR-006 specify `us-central1` for both the TPU and the GCS model
cache (free intra-region reads). As of 2026-05, GCP no longer offers
`v5litepod-1` capacity in `us-central1`. The default zone list in
`scripts/lib/config.sh` is `us-east5-{a,b,c}` → `us-west4-{a,b}` →
`us-west1-c`. The bucket region defaults to `us-central1` per the ADR; this
is a legitimate revisit point — see the comment block at the top of
`scripts/lib/config.sh` for the trade-off.

---

## Status

| Component | Status |
|-----------|--------|
| Training examples (01–08) | ✅ Complete |
| gcloud scripts | ✅ Complete |
| Benchmark harness — Stage 1 (Path 1: JAX + XLA) | ✅ Complete (2026-04-29) |
| Model registry — 5 Stage 1 models | ✅ Complete (2026-04-29) |
| Observe: stats, lineage, compile_controller | ✅ Complete (2026-04-29) |
| Results dashboard — table view | ✅ Complete (2026-04-29) |
| Unit tests (~180, no GPU required) | ✅ Complete (2026-04-29) |
| Probe-based observability layer | ✅ Complete (2026-05-06) |
| Grafana dashboards (importable) | ✅ Complete (2026-05-06) |
| Multi-path: Paths 2+3 (JAX+GPU, PyTorch+GPU) | Stage 2 |
| System monitor (GPU SM%, MXU%, power, thermals) | Stage 2 |
| Profiler + FLOPs counter + roofline | Stage 3 |
| torch_xla Path 4 | Stage 4 |
| Novel architectures (Mamba, RWKV, DiT) | Stage 5 |
| Precision + quantization (INT8, FP8, INT4) | Stage 6 |
| HF Inference API path (Path 5) | Stage 7 |
| Sparsity + pruning (2:4 structured) | Stage 8 |
| Full 75-model registry + GitHub Actions CI | Stage 9 |

---

## License

MIT
