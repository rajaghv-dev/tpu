# Session State — TPU × GPU Benchmark Repo

Read this file at the start of every new session before doing anything else.
It contains the complete state of the project, decisions made, and what comes next.

---

## Identity

| Item | Value |
|------|-------|
| Repo | https://github.com/rajaghv-dev/tpu |
| Local path | `/home/raja/tpu` |
| GitHub user | `rajaghv-dev` (email: rajaghv.dev@gmail.com) |
| GitHub auth | `gh` CLI authenticated, HTTPS protocol |
| Branch | `main` |
| Last commit | `e688c67` — Fix DGX→Blackwell B200, India access, H100/B200 costs, model justifications |
| Shell | WSL2 (Linux on Windows), bash, /home/raja/tpu |

---

## What This Repo Is

Single repo to understand and benchmark inference across:
- **Hardware:** Google TPU (v2/v3/v4/v5e/v5p/v6e) vs NVIDIA GPU (RTX 3080, RTX 4090, DGX B200)
- **Frameworks:** JAX/Flax, PyTorch, torch_xla, HuggingFace Inference API (5 execution paths)
- **Models:** ~75 models across vision, NLP (encoder/decoder/SSM/MoE), audio, multimodal
- **Dimensions:** precision (FP32/BF16/INT8/INT4/FP8), compilation, sparsity, LLM decode modes
- **Goal:** evidence-backed claims with full traceability (profiler traces, lineage, statistics)

---

## User Profile

| Item | Detail |
|------|--------|
| Name | Raja |
| Location | India |
| Hardware | RTX 3080 (16GB), RTX 4090 (24GB), DGX Dell B200 Blackwell (192GB HBM3e, 256GB system RAM) |
| Cloud | Google Colab Pro, GCP personal account (rajaghv@gmail.com), $300 free credits |
| HuggingFace | Paid PRO account — gated model access (Gemma, PaliGemma, LLaMA) + Inference API |
| Expertise | Learning from essentials to expert; wants depth across hardware/compiler/model |
| Style | Prefers discussion before coding; wants MD documentation files; no fluff |

---

## Current Repo File Structure

```
/home/raja/tpu/
├── 01_hello_tpu/           hello_tpu.py + README.md
├── 02_mnist_classification/ train.py + README.md
├── 03_resnet_imagenet/      model.py + train.py + README.md
├── 04_bert_finetuning/      train.py + README.md
├── 05_gpt_pretraining/      model.py + train.py + README.md
├── 06_data_pipeline/        pipeline.py + README.md
├── 07_custom_training_loop/ train.py + README.md
├── 08_multi_host/           train.py + README.md
├── scripts/
│   ├── gcloud_setup.sh
│   ├── provision_tpu.sh
│   ├── gcloud_ssh_run.sh
│   ├── gcloud_upload_data.sh
│   ├── gcloud_pod_run.sh
│   └── teardown_tpu.sh
├── README.md               Full landing page (hardware, 5 paths, 75 models, 7 arcs, suites, costs)
├── LESSON_PLAN.md          15-module beginner→expert curriculum (just created)
├── context.md              Full project context (15 sections; all design decisions)
├── prompts.md              All user prompts P1–P21
├── SESSION.md              This file — session continuity
├── MEMORY.md               Key facts for fast session startup
└── requirements.txt
```

**Not yet built (benchmarks/ harness):** Stage 1 of 9 is next coding task.

---

## Committed but Not Yet Pushed in This Session

At time of writing: LESSON_PLAN.md, SESSION.md, MEMORY.md, prompts.md updates are
staged but not committed. The push command is at the end of this session's work.

---

## The 5 Execution Paths

| Path | Framework | Compiler | Hardware |
|------|-----------|---------|---------|
| 1 | JAX/Flax | XLA | TPU |
| 2 | JAX/Flax | XLA (CUDA) | GPU |
| 3 | PyTorch | CUDA/cuDNN | GPU |
| 4 | PyTorch + torch_xla | XLA | TPU |
| 5 | HTTP | HF-managed | Unknown |

---

## Staged Build Plan — Current Status

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Foundation: harness.py, runner.py, 5 models, Path 1, table dashboard | **NOT STARTED** |
| 2 | Multi-path: Paths 2+3, system_monitor, 15 models, heatmap dashboard | Not started |
| 3 | Profiler + Roofline: flops_counter, tracer, hlo_analyser | Not started |
| 4 | torch_xla: Path 4 | Not started |
| 5 | Novel architectures: Mamba, RWKV, DiT | Not started |
| 6 | Precision + Quantization: INT8, FP8, numerics | Not started |
| 7 | HF Inference API: Path 5 | Not started |
| 8 | Sparsity + Pruning | Not started |
| 9 | Full registry + GitHub Actions automation | Not started |

**Next coding session starts at Stage 1.**

---

## Key Decisions Made (do not re-discuss)

| Decision | Choice | Reason |
|----------|--------|--------|
| Framework | JAX + PyTorch (both) | JAX for clean hardware comparison; PyTorch for real-world GPU parity |
| TPU target | v5e-1 preemptible ($0.36/hr) primary; v6e-1 secondary | Single-chip VMs; cost-effective |
| Model scope | ~75 models; up to 4B params on v5e-1 BF16 | v5e-1 has 16GB; 4B BF16 = ~8GB weights |
| Input data | Synthetic random (numpy seed = 42) | No data download; hardware-only comparison |
| Model weights | HuggingFace pretrained | Real weights give realistic compute patterns |
| Model cache | GCS bucket (rajaghv@gmail.com) | Download once; reuse across VM restarts |
| Results format | Append-only JSONL | Simple, grep-able, version-controllable |
| Dashboard | Static HTML + Vega-Lite (GitHub Pages) | No server; zero cost; shareable |
| Run mode | Sequential, 1–3 min per experiment | Colab-compatible; no parallel complexity |
| Statistics | n=3 independent runs; CV < 10% threshold | Evidence-grade measurements |
| Inference only | Yes — no training benchmarks in harness | Cleaner; faster experiments |

---

## Hardware Reference (Quick)

| Card | Memory | BW | BF16 TFLOPs | Notes |
|------|--------|-----|-------------|-------|
| RTX 3080 | 16GB GDDR6X | 760 GB/s | 119 | Local |
| RTX 4090 | 24GB GDDR6X | 1008 GB/s | 330 | Local; FP8 |
| DGX B200 | 192GB HBM3e | 4000 GB/s | 2250 | Local; Blackwell; 1000W TDP |
| TPU v5e-1 | 16GB HBM | 820 GB/s | 394 | $0.36/hr preemptible |
| TPU v6e-1 | 32GB HBM | 1640 GB/s | 918 | ~$0.75/hr preemptible |

---

## All 15 Gaps — Status

All 15 gaps identified and assigned. See `context.md` Section 13 for full table.
Short summary: C1–C5 (critical) addressed in Stages 1–3; I1–I7 (important) in Stages 1–7; N1–N4 in later stages.

---

## How to Push to GitHub

```bash
git add -A
git commit -m "message"
git push origin main
```

`gh` CLI is authenticated as `rajaghv-dev`. No password prompt needed.

---

## What to Do at Session Start

1. Read MEMORY.md (2 min) for the dense summary
2. Read this SESSION.md file (5 min) for complete state
3. Check `git log --oneline` to see what was last committed
4. Check `git status` to see if anything is uncommitted
5. Ask the user what they want to do today, then proceed

Do NOT re-read context.md from scratch every session — it is 700+ lines.
Use MEMORY.md as the quick reference; context.md as the deep reference.
