# Open Questions

> **Format.** Each question states a precise, testable hypothesis where possible, plus why it matters, how to answer it in this repo, an expected answer with reasoning, and a difficulty rating.
> **Difficulty scale.** Easy = answerable from smoke suite (5 models, ≤1 hour). Medium = needs domain suite (Stage 2–3, ≤8 hours). Hard = requires Stage 3+ infrastructure or non-trivial coding.
> **Last reviewed.** 2026-04-26.

---

## Hardware & memory

### Q1. At what batch size does v6e-1 match H100 SXM5 on BERT-base BF16 throughput?
- **Why it matters.** Tells us where TPU stops being memory-bound and starts saturating its (smaller) systolic array, vs the GPU reaching its (much larger) Tensor-Core peak. Decides "buy v6e or rent H100" advice.
- **How to answer it.** Run BERT-base (110 M, BF16) at batch ∈ {1, 8, 32, 128, 512, 2048, 8192} on v6e-1 (32 GB HBM, 1.64 TB/s, 918 TFLOPs) and on H100 SXM5 spot (80 GB HBM, 3.35 TB/s, 1979 TFLOPs). Find the cross-over batch where samples/sec are within 5%.
- **Expected answer.** v6e-1 wins at very small batches (memory-bandwidth ratio favours TPU per dollar) and loses at very large batches. Cross-over likely around batch 256–512 because H100's 2.16× higher TFLOPs need full utilization to dominate, which BERT-base reaches around there. Pure throughput champion at batch ≥1024 = H100; throughput-per-dollar champion at any batch under spot pricing = v6e-1.
- **Difficulty.** Medium.

### Q2. Does B200's 4 TB/s bandwidth give 2× over H100 on memory-bound decode?
- **Why it matters.** Decode latency in LLM serving is bandwidth-bound (each token reads all weights). The B200-vs-H100 bandwidth ratio (4.0 / 3.35 = 1.19×) sets a hard ceiling on the speedup, regardless of TFLOPs.
- **How to answer it.** Run GPT-2-XL (1.5 B, BF16) decode-only at batch=1, 512 tokens. Measure tokens/sec on local B200 and on H100 spot. Compute the ratio.
- **Expected answer.** ~1.15–1.20×, not 2×. The bandwidth ratio caps the gain; B200's higher TFLOPs don't help a memory-bound workload. If we observe >1.3×, we should look for non-bandwidth contributions (better cache hierarchy, FP8 path).
- **Difficulty.** Medium.

### Q3. What is the actual MXU utilization for ResNet-50 at batch=1024 on v5e-1?
- **Why it matters.** The MXU is the 128×128 systolic array; convs need to be reshaped into matmul patterns that fit. ResNet-50 at large batch should saturate it. Tells us whether v5e-1's 394 BF16 TFLOPs are reachable by a "well-behaved" model.
- **How to answer it.** Run ResNet-50 forward at batch=1024 on v5e-1. Use `tpu/duty_cycle_percent` Cloud Monitoring metric and per-op profiler trace. Compare achieved TFLOPs vs peak.
- **Expected answer.** 60–75% of peak (~240–290 TFLOPs achieved). True peak is rare on real models because depthwise (none in plain ResNet) and 1×1 transitions don't saturate the MXU. If we observe <50% the model padding is wrong and worth investigating.
- **Difficulty.** Easy.

### Q4. At what sequence length does BERT attention become memory-bound on v5e-1?
- **Why it matters.** Attention is O(seq²) compute, O(seq²) memory traffic — the ratio is constant in seq, BUT activation memory grows quadratically and starts spilling out of cache. Knowing the cross-over sets KV-cache budgets.
- **How to answer it.** BERT-base BF16, fixed batch=8, sweep seq ∈ {128, 256, 512, 1024, 2048, 4096}. Compute arithmetic intensity per cell. Observe where the achieved-TFLOPs curve flattens and the achieved-bandwidth curve hits the roof.
- **Expected answer.** Cross-over around seq=1024–2048. Below seq=512 attention is compute-bound; above seq=4096 it's strongly memory-bound. v5e-1's 16 GB HBM caps the experiment around seq=8192 at batch=8 anyway.
- **Difficulty.** Medium.

### Q5. How much KV-cache memory does GPT-2-XL use at decode step 1000?
- **Why it matters.** Practical maximum context length on v5e-1 / RTX 4090 — the bound that decides "can I run this model with a 1024-token prompt + 1024-token generation".
- **How to answer it.** Compute formula: `2 (K+V) × n_layers × n_heads × head_dim × seq × dtype_bytes × batch`. For GPT-2-XL: 2 × 48 × 25 × 64 × 1000 × 2 × 1 = 384 MB at batch=1. Validate by reading `jax.live_arrays()` HBM usage at step 1000 of an actual decode run.
- **Expected answer.** ~384 MB at batch=1. Scales linearly with batch — at batch=32 we're at ~12 GB which is past v5e-1 ceiling. The takeaway: KV cache is the binding constraint for long-context decoding on small-HBM accelerators, not weights.
- **Difficulty.** Easy.

---

## Compiler & XLA

### Q6. Will XLA's Pallas/Splash Attention close the FlashAttention2 CUDA kernel gap at seq>8k?
- **Why it matters.** A primary reason "TPU loses on long-context attention" is the lack of an XLA equivalent to FlashAttention2's IO-aware tiling. If Pallas (XLA's GPU-style kernel-DSL extension) closes this, the calculus shifts.
- **How to answer it.** Run BERT-base attention layer at seq=16384 on v5e-1 with (a) standard XLA, (b) Pallas Splash Attention if available in the JAX version we pin. Compare TFLOPs achieved.
- **Expected answer.** Pallas closes most of the gap (within 20%) but not all. FlashAttention2's CUDA-specific IO tiling exploits SM shared memory in ways XLA's general lowering cannot match. Practical recommendation: long-context attention prefers GPU+FlashAttention until Pallas matures further.
- **Difficulty.** Hard.

### Q7. What is the exact XLA fallback for Mamba's selective_scan — sequential PyLoop or something else?
- **Why it matters.** Decides whether the gap is "missing kernel" (closeable) or "fundamentally serial recurrence on a parallel device" (intrinsic).
- **How to answer it.** Dump XLA HLO for `mamba_block.selective_scan(...)` via `XLA_FLAGS=--xla_dump_to=/tmp/hlo`. Inspect the lowered ops — look for `while`/`scan` primitives.
- **Expected answer.** XLA emits a `lax.scan` (sequential while loop in HLO), no parallel scan reduction. The associativity that would let it parallelize is broken by the input-dependent gating — the operation isn't a parallel-scannable monoid as written. Conclusion: this is intrinsic, not a missing kernel.
- **Difficulty.** Medium.

### Q8. At what conv kernel size does XLA's pad-to-128 cost exceed cuDNN's algorithm selection overhead?
- **Why it matters.** TPU MXU dimensions are 128-multiples; XLA pads convs to fit. cuDNN picks among ~10 algorithms. Both costs are real but different in shape.
- **How to answer it.** Run a parameterised conv (single layer, batch=64, channels=64) with kernel ∈ {1, 3, 5, 7, 11} on v5e-1 and RTX 4090. Plot achieved-TFLOPs vs kernel.
- **Expected answer.** TPU pads to 128 regardless of kernel — overhead is fixed in compute but a bigger fraction at small kernel. cuDNN picks well at common sizes (3, 5) and is slower at unusual sizes (11). Cross-over likely around kernel=7. Smaller kernels favour GPU (cuDNN picks the right algo); larger kernels and unusual sizes favour TPU (XLA's regularity wins).
- **Difficulty.** Medium.

### Q9. Does `torch.compile(max-autotune)` beat `jax.jit` on ResNet-50 on GPU?
- **Why it matters.** Direct compiler shoot-out on identical model+device. If Inductor's max-autotune (which spends time generating Triton kernels) wins, JAX-on-GPU is throughput-leaving-table.
- **How to answer it.** ResNet-50 BF16, batch=128, RTX 4090. Path A: PyTorch eager + `torch.compile(mode='max-autotune')`. Path B: JAX/Flax + `jax.jit`. Measure steady-state throughput, n=3.
- **Expected answer.** Within 5% of each other — both lower to similar Tensor-Core kernels. Inductor's max-autotune may edge ahead at unusual shapes (it generates per-shape kernels) and lose at standard ones (XLA's library calls equally well-tuned). If gap exceeds 10%, deeper investigation warranted.
- **Difficulty.** Easy.

### Q10. Can MoE routing be made static-shape-compatible in JAX without running all experts?
- **Why it matters.** If yes, MoE on TPU is viable; if no, MoE is GPU-only. Settles a real architectural debate for personal-scale deployment.
- **How to answer it.** Implement capacity-factor routing: each expert gets a fixed slot count (e.g., 1.25× tokens/n_experts), tokens beyond capacity are dropped. Measure throughput on Phi-3.5-MoE vs running all experts (the dense fallback) vs the original PyTorch dynamic-routing baseline.
- **Expected answer.** Yes — capacity-factor routing is static-shape-compatible AND retains ~70–90% of MoE benefit (some quality loss from dropped tokens, ~30% throughput cost from unused capacity). This is the standard production approach (used in Switch-Transformer, Mixtral). The dense fallback is the worst of both worlds.
- **Difficulty.** Hard.

---

## Model architecture

### Q11. Does RecurrentGemma-2B outperform Gemma-2B on TPU decode despite the sequential RGLRU?
- **Why it matters.** RecurrentGemma replaces attention with a Linear Recurrent Unit — O(1) per-token state vs O(seq) for attention. On long contexts the RGLRU should win; on short contexts attention's parallelism wins.
- **How to answer it.** Both at batch=1, decode 1000 tokens. v5e-1, BF16. Measure tokens/sec.
- **Expected answer.** RecurrentGemma wins past seq~512–1024 (its O(1) state pays off) and loses at very short generation (the recurrence doesn't get to amortize its sequential cost). On TPU specifically, the RGLRU may be poorly served by XLA (similar to Mamba) — could be slower than expected.
- **Difficulty.** Medium.

### Q12. Is RWKV-4-3B's sequential WKV recurrence worse than Mamba's on TPU?
- **Why it matters.** Both are SSM-style — both face the "no XLA primitive for selective_scan" issue. Comparing them isolates implementation maturity from architectural choice.
- **How to answer it.** RWKV-4-3B and Mamba-2.8B both at batch=1, decode 512 tokens. v5e-1.
- **Expected answer.** Both poorly served on TPU. RWKV may be slightly better because its recurrence is simpler (no input-dependent gating, more amenable to XLA fusion). Both lose to attention transformers of similar size on TPU; both win on GPU with custom kernels.
- **Difficulty.** Medium.

### Q13. What is PaliGemma's actual co-design advantage vs LLaVA-Phi3 on TPU?
- **Why it matters.** PaliGemma was designed by Google with TPU-aware shapes. LLaVA-Phi3 was bolted-together (pretrained Phi-3 + CLIP). Tests whether co-design matters in practice or is marketing.
- **How to answer it.** Both at batch=8, single image + 64-token prompt + 64-token generation. v5e-1, BF16. Measure end-to-end latency.
- **Expected answer.** PaliGemma 1.5–3× faster on TPU. Vision projector dimensions chosen as 128-multiples; Gemma backbone tuned for TPU MXU; KV-cache layout pre-optimised. LLaVA-Phi3 has odd intermediate dimensions that XLA pads. On GPU the gap shrinks dramatically (cuDNN doesn't care about 128-multiples).
- **Difficulty.** Medium.

### Q14. Do GQA models achieve proportional KV-cache reduction in practice on our hardware?
- **Why it matters.** Llama-3 advertises GQA-8 → 8× KV-cache reduction. Reality on actual hardware may be less due to alignment padding and cache-line effects.
- **How to answer it.** Run Llama-3-8B-INT4 (with GQA) and a hypothetical full-attention reference (computed via formula since no full-attention 8B exists). Measure HBM at decode step 1000.
- **Expected answer.** ~7× reduction observed (slightly less than 8×) due to head-group alignment overhead. Claim holds within 10–15%. The 8× headline is a clean theoretical number; real hardware delivers ~85–90% of it.
- **Difficulty.** Medium.

### Q15. Does DeepSeek-R1's reasoning chain (2000-token outputs) shift the memory-bound profile significantly?
- **Why it matters.** Reasoning models generate far more tokens per query. If decode is memory-bound, longer chains amplify the bandwidth bottleneck — a different regime from chat models.
- **How to answer it.** Run DeepSeek-R1 at batch=1, generate 2000 tokens. Compare arithmetic intensity at step 100 vs step 1500 (KV cache larger at step 1500). Place both on roofline.
- **Expected answer.** Yes — at step 1500, KV cache reads dominate weight reads (~3× the bytes), pushing arithmetic intensity even lower (further memory-bound). Throughput drops as the chain grows. This is a generic effect, not DeepSeek-specific, but reasoning models make it operationally relevant.
- **Difficulty.** Hard.

---

## Measurement & statistics

### Q16. Is n=3 with Grubbs test sufficient to detect 10% throughput differences reliably?
- **Why it matters.** Validates ADR-009. If n=3 has 30% type-II error rate at this effect size, we're missing real differences.
- **How to answer it.** Synthetic Monte Carlo: generate two distributions with mean 100 and 110, std 5, sample n=3 from each, run two-sample t-test, repeat 10 000 times, compute power. Repeat for std 10 (matches our CV ceiling).
- **Expected answer.** At std=5 (CV~5%), power for detecting 10% difference at n=3 is ~50% (insufficient — need n=5+). At std=10 (CV~10%), power drops to ~25%. Implication: n=3 reliably detects differences only at ≥20% effect size. Smaller differences need n=5 or pooling across batches.
- **Difficulty.** Easy.

### Q17. How much does thermal state at run start affect CV?
- **Why it matters.** If a "cold" first run is systematically slower, CV is inflated and real differences are masked.
- **How to answer it.** Single model (ResNet-50), n=10 consecutive runs with no cool-down. Plot run_idx vs throughput. Compute CV with/without first run.
- **Expected answer.** First run 5–10% slower than runs 5+ on B200 (cache warming, clock ramp). On v5e-1 the effect is smaller (cloud TPUs run at fixed clock). Recommendation: ≥2 warmup runs before measured runs (P27).
- **Difficulty.** Easy.

### Q18. Does clearing the XLA cache between runs add measurable variance vs within-run variance?
- **Why it matters.** Decides whether "cold runs" should be reported alongside "warm runs" or only warm. Clearing cache adds a real variability source.
- **How to answer it.** Same model, n=5 with cache-clear-each-time, n=5 without. Compare CV.
- **Expected answer.** Cache-clear runs have CV 2–3× higher because compile time itself has variance (XLA scheduling). Warm-run CV is the "true" hardware noise. Reported numbers are warm-run; cold/compile is reported separately.
- **Difficulty.** Easy.

### Q19. What is the baseline CV for a well-behaved model (ViT-B/16) on v5e-1?
- **Why it matters.** Sets the noise floor — anything below this CV is structural, anything above is measurement-related.
- **How to answer it.** ViT-B/16, batch=64, BF16, n=20 measured runs (after warmup). Compute CV.
- **Expected answer.** 1–3%. ViT is a clean rectangular model, no dynamic shapes, hits MXU well. Anything we measure with CV>3% has a hardware/measurement reason worth investigating. This is the calibration constant for the whole project.
- **Difficulty.** Easy.

---

## Infrastructure & workflow

### Q20. Can Colab Pro sustain a 50-minute quick suite without session timeout?
- **Why it matters.** Colab is the cheapest TPU access path. If it can't sustain the quick suite, it's smoke-only.
- **How to answer it.** Run quick suite (15 models × 3 batches × 3 runs ≈ 50 min) on a Colab Pro v2-8 TPU runtime. Measure completion rate over 5 attempts.
- **Expected answer.** ~80% success. Failures from network drops, not session timeouts (12 h cap is plenty). Daily TPU minute cap (~3 h) limits us to one attempt per day. Verdict: yes for quick suite, with a resume-on-disconnect strategy required.
- **Difficulty.** Easy.

### Q21. What is the GCS read throughput from us-central1 when initiated from India?
- **Why it matters.** Decides whether "in-region read" means TPU only or also IST-laptop debug. Affects how we structure the workflow.
- **How to answer it.** From India laptop, `gsutil cp gs://rajaghv-tpu-cache/test_1gb.bin /tmp/`. Measure throughput. Repeat from us-central1 VM.
- **Expected answer.** India-laptop: ~10–30 MB/s (TCP-distance limited). us-central1 VM: ~500–1000 MB/s. Recommendation: never read large objects from laptop; always SSH to a VM.
- **Difficulty.** Easy.

### Q22. How large will runs.jsonl grow for a full suite (800 experiments)?
- **Why it matters.** Sets the shard cadence. Affects dashboard render performance and git diff usability.
- **How to answer it.** Generate one full suite run; measure file size; multiply by expected weekly cadence.
- **Expected answer.** ~2 KB/row × 800 rows × 3 repeats = 4.8 MB per full suite. Annual: 250 MB if we store every weekly run, or ~25 MB if we keep only aggregates. Recommendation: shard yearly; in-repo file caps at 50 MB/year (R-I04).
- **Difficulty.** Easy.

### Q23. Will GitHub Pages render a Vega-Lite dashboard with 800 rows without lag?
- **Why it matters.** If 800 rows lags, we have to pre-aggregate before shipping to the dashboard, which loses drill-down ability.
- **How to answer it.** Generate a synthetic 800-row JSONL, render the dashboard locally and on GitHub Pages, measure first-render time and filter-interaction time.
- **Expected answer.** Yes for table view (~200 ms first render, snappy filters). Borderline for chart view at >2000 rows (Vega-Lite's renderer struggles). Recommendation: keep table view direct; aggregate to per-(model,path) rows for charts (~75 rows headline).
- **Difficulty.** Easy.

---