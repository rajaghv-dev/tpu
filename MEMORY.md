# MEMORY — Fast Session Startup Reference

Read this first. Dense, no prose. Everything needed to resume work in under 3 minutes.

---

## REPO

URL: https://github.com/rajaghv-dev/tpu
Local: /home/raja/tpu | Branch: main | Auth: gh CLI as rajaghv-dev
Push: `git add -A && git commit -m "msg" && git push origin main`

---

## USER

Raja, India. Beginner-to-expert learning journey.
Hardware: RTX 3080 (16GB) + RTX 4090 (24GB) + DGX Dell Blackwell B200 (192GB HBM3e, 256GB system RAM).
Cloud: Colab Pro + GCP personal (rajaghv@gmail.com, $300 free credits).
HF: PRO paid — gated access (Gemma, PaliGemma, LLaMA) + Inference API.
Style: discuss before coding. MD files for documentation. No fluff.

---

## WHAT THE REPO DOES

Inference benchmark: TPU vs GPU. 75 models. 5 execution paths. 9 experiment dimensions.
Evidence-backed. Full traceability. Beginner→expert curriculum included.
Training examples 01–08 exist and are complete. Benchmark harness does NOT exist yet.

---

## 5 PATHS

1=JAX+TPU(XLA)  2=JAX+GPU(XLA-CUDA)  3=PyTorch+GPU(CUDA)  4=PyTorch+torch_xla+TPU  5=HF API
Pairs: 1v2=hardware, 2v3=framework, 1v4=framework-on-TPU, 3v4=compiler-on-TPU, 1v3=real-world

---

## KEY HARDWARE NUMBERS

RTX 3080:  16GB GDDR6X, 760 GB/s,  119 BF16 TFLOPs
RTX 4090:  24GB GDDR6X, 1008 GB/s, 330 BF16 TFLOPs, FP8
B200 SXM:  192GB HBM3e, 4000 GB/s, 2250 BF16 TFLOPs, FP8, 1000W (Blackwell)
v5e-1 TPU: 16GB HBM,    820 GB/s,  394 BF16 TFLOPs, $0.36/hr preemptible
v6e-1 TPU: 32GB HBM,    1640 GB/s, 918 BF16 TFLOPs, ~$0.75/hr preemptible

Ridge points (BF16 FLOPs/byte): 3080=156, 4090=327, B200=562, v5e=480, v6e=560

Primary pairing: v5e-1 ↔ RTX 3080 (both 16GB)
Secondary pairing: v6e-1 ↔ RTX 4090 (32 vs 24GB)

---

## MAX MODEL SIZE (BF16 inference, single card)

v5e/3080 (16GB): comfortable 3B, tight 5B, INT8 10B
v6e/4090 (32GB): comfortable 7B, tight 10B, INT8 20B
B200 (192GB): comfortable 70B, tight 96B, INT8 192B

---

## CLOUD COSTS (single GPU, preemptible/spot)

v5e-1: $0.36/hr | v6e-1: $0.75/hr
H100 GCP: $1.10/hr | H100 Lambda Labs: $2.49/hr
B200 GCP: ~$2.40/hr est. | B200 local: electricity only (~$0.12/hr at 1000W)

India electricity: ~$0.07/kWh (cheaper than US $0.12). Break-even for local B200 at ~60% utilisation.
GCP/AWS/Lambda fully accessible from India. No legal restrictions. International card or UPI for GCP.
TPU NOT available in India GCP regions — use us-central1.

---

## MODEL REGISTRY SUMMARY (75 models, see context.md §4 for full detail)

Vision (14): ResNet-50, ConvNeXt-XL, EfficientNet-B7, EfficientViT-L3, ViT-B/16, ViT-L/16,
             DINOv2-L, SigLIP-B/16, SAM-L, EVA-02-L, DETR-ResNet50, RT-DETR-L, DiT-XL/2, SD-UNet

NLP Encoders (8): BERT-base, RoBERTa-large, DeBERTa-v3-large, ModernBERT-base/large,
                  BGE-large-en-v1.5, E5-large-v2, nomic-embed-v1.5

NLP Decoders (20+): GPT-2 XL, OPT-2.7B, BLOOM-3B, Falcon-RW-1B, TinyLlama, SmolLM2-1.7B,
  OLMo-2-1B, Llama-3.2-1B/3B, Gemma-2B/Gemma-2-2B/Gemma-2-2B-IT/Gemma-3-1B/Gemma-3-4B,
  RecurrentGemma-2B, Phi-1.5/2/3-mini/3.5-mini/3.5-MoE,
  Qwen2.5-0.5B/1.5B/3B/Coder-1.5B/Coder-3B,
  DeepSeek-R1-Distill-Qwen-1.5B/7B, DeepSeek-Coder-V2-Lite, StableLM-2-1.6B/3B, MPT-7B

Novel arch (3): Mamba-2.8B, Mamba2-2.7B, RWKV-4-3B
Code (2): StarCoder2-3B, CodeGemma-2B
Audio (8): Whisper-base/medium/large-v3, wav2vec2-large, HuBERT-large, SeamlessM4T-medium,
           MMS-1B, EnCodec-24kHz
Multimodal (7): CLIP ViT-L/14, SigLIP-SO400M, moondream2, SmolVLM-2B, PaliGemma-3B,
                LLaVA-Phi3, ImageBind

GATED (need HF token): Gemma-*, PaliGemma-*, CodeGemma-*, RecurrentGemma-*, Llama-3.2-*

---

## KEY ARCHITECTURAL INSIGHTS (the "aha moments")

1. EfficientNet < ViT throughput on TPU despite fewer FLOPs → depthwise conv starves MXU
2. Mamba: GPU wins 3-5× → custom CUDA selective_scan kernel; no XLA native primitive
3. BF16 is free on TPU (same speed as FP32); 2× faster on GPU
4. XLA first-compile: 12–45s; torch.compile: 5–20s; subsequent runs equally fast
5. 2:4 structured sparsity: 2× GPU speedup (Sparse TC); NO TPU equivalent
6. PaliGemma-3B: Google-designed model + compiler + hardware → maximum TPU advantage
7. MoE (Phi-3.5-MoE, DeepSeek-Coder-V2-Lite): dynamic routing breaks XLA static shapes
8. LLM prefill = compute-bound; LLM decode = memory-bound → different winning hardware
9. DeepSeek-R1: 2000-token reasoning chains → extreme decode-bound; BW is everything
10. DiT-XL/2: pure transformer diffusion → best TPU diffusion model (pure matmuls)

---

## EXPERIMENT PROTOCOL (1-3 min per run)

1. Pre-flight: record temp/clock
2. Compile: 1 pass, clear XLA cache first
3. Warmup: 20 passes (discard)
4. Latency: 100 passes @ bs=1 → p50/p95/p99
5. Throughput: 100 passes @ bs=max
6. Profiler: 10 passes (full trace)
7. Memory sweep: bs=1,2,4,... until OOM
8. Numerics: compare to FP32 reference
9. Post-flight: check throttle

Stats requirement: n=3 independent runs; CV < 10%; flag if throttle detected.

---

## RESULTS SCHEMA (key fields only)

run_id, git_sha, hf_model_revision, input_seed (lineage)
device, framework, path, model, precision, pruning, compiled (identity)
compile_time_s, first_compile_s, subsequent_compile_s, compile_cache_hit
latency_p50_ms, latency_p99_ms, latency_cv_pct, latency_std_ms
throughput_mean_samples_sec, tokens_per_sec
peak_memory_gb, max_batch_before_oom
flops_per_sample_G, arithmetic_intensity, mfu_pct
mxu_utilization_pct (TPU), sm_utilization_pct (GPU)
device_power_w, energy_wh_per_1k_samples
output_cosine_sim_vs_fp32 (accuracy)
flags: ["high_variance", "throttle_detected", "near_oom", "compile_slow"]
cost_per_1k_samples_usd

---

## OBSERVABILITY MODULES (all gaps addressed)

observe/system_monitor.py  → pynvml (GPU SM%, power, temp, clock) + Cloud Monitoring (TPU MXU%)
observe/flops_counter.py   → jax.make_jaxpr → HLO FLOPs; fvcore for PyTorch
observe/stats.py           → n=3 runs, Grubbs outlier test, CV check
observe/compile_controller.py → clear XLA cache; measure first vs subsequent compile
observe/memory_profiler.py → timeline + per-component breakdown
observe/numerics.py        → L2 norm + cosine sim vs FP32 reference
observe/lineage.py         → git SHA + HF revision + seed + env hash
observe/tracer.py          → jax.profiler / torch.profiler / torch_xla.profiler
observe/hlo_analyser.py    → parse XLA HLO → count fusion groups, kernel launches

---

## 15 GAPS — ALL ADDRESSED

C1(FLOPs counter)→Stage3, C2(multi-run stats)→Stage1, C3(compile cache)→Stage1,
C4(thermal)→Stage2, C5(hw utilisation)→Stage2, I1(prefill/decode)→Stage2,
I2(input rotation)→Stage2, I3(MoE handling)→Stage6, I4(input seed)→Stage1,
I5(XLA fusion)→Stage3, I6(HF API split)→Stage7, I7(power)→Stage3,
N1(bs sweep)→Stage2, N2(repro suite)→Stage1, N3(cache size)→Stage3

---

## STAGED BUILD PLAN — STATUS

Stage 1: Foundation (harness.py, runner.py, 5 models, Path 1, table dashboard) — NOT STARTED ← NEXT
Stage 2–9: Not started. See context.md §10 for full descriptions.

---

## SUITE DEFINITIONS

smoke: 1 model, FP32+BF16, ~8min, $0.05
quick: 6 models (1/domain), BF16, ~50min, $0.30
domain: all in 1 domain, FP32+BF16, ~60min, $0.36
arch: novel arches, BF16, ~40min, $0.24
llm: all decoders, BF16 prefill+decode, ~2hrs, $0.72
full: all 75 models, all variants, ~8hrs, $2.88

---

## LESSON PLAN — 15 MODULES (in LESSON_PLAN.md)

M1: Colab Pro + HF setup
M2: GPU vs TPU hardware
M3: Memory + Roofline model
M4: JAX/PyTorch/XLA/CUDA software stack
M5: Repo examples 01-08
M6: HuggingFace deep dive
M7: Running first benchmark (smoke suite)
M8: Compiler deep dive (XLA vs CUDA)
M9: Model architecture classes + hardware fit
M10: Precision + Quantization
M11: LLM prefill vs decode + KV-cache
M12: Sparsity + Pruning
M13: Full observability + evidence reading
M14: TCO analysis
M15: Expert research workflow

---

## EVIDENCE CHAIN (how to verify any claim)

Claim → Chart (dashboard) → run_ids → runs.jsonl rows → run_logs/<run_id>/
  ├── raw_timings.jsonl (statistical claims)
  ├── profiles/*.pb (compiler claims; open in TensorBoard or Perfetto)
  ├── hlo_dump.txt (XLA fusion claims)
  ├── memory_timeline.json (memory claims)
  ├── system_state.json (hw utilisation claims)
  ├── numerics.json (precision claims)
  └── lineage.json (git SHA, model hash, input seed)

---

## COLAB PRO PRACTICAL NOTES

- TPU runtime: Runtime → Change runtime type → TPU → v2-8 or v3-8
- GPU runtime: Runtime → Change runtime type → A100 or T4
- HF token: `import os; os.environ['HF_TOKEN'] = 'hf_...'`
- Mount Drive for model cache: `from google.colab import drive; drive.mount('/content/drive')`
- Run suite: `!python benchmarks/harness.py --suite=quick --framework=jax --device=tpu`
- Pro session: up to 24 hrs. Pro+ needed for background execution.
- For full suite (8hrs): use Cloud TPU VM, not Colab.

---

## INDIA ACCESS SUMMARY

GCP: accessible; TPU only in US/EU regions; Indian card or UPI accepted; $300 free trial
AWS: H100 only in US regions; Mumbai has V100/A100 limited
Lambda Labs: $2.49/hr H100; no India DC; accessible; international card
No legal/regulatory restrictions for personal cloud research use from India
