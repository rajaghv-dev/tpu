# Stage 1 Interpretation

> **Closes R19** (RECOMMENDATIONS.md). One paragraph per `runs.jsonl` row; one synthesis paragraph at the end. Read before extending to Path 2.
> **As of:** 2026-05-10. Run count: 2 (1 failed, 1 success).

---

## Run 1 — `a0020844-…` (FAILED, model_load)

| Field | Value |
|---|---|
| Status | failed |
| Phase | `model_load` |
| Category | `other` (mis-classified — see below) |
| Model / precision | bert_base · bf16 |
| Device | tpu (v5e-1, us-west4-a) |
| transformers | **5.8.0** |
| Exception | `ImportError: cannot import name 'FlaxAutoModelForSequenceClassification'` |

**What went wrong.** `transformers` 5.x removed every Flax symbol; the runner imports `FlaxAutoModelForSequenceClassification` unconditionally inside `_load_flax_model`. The TPU VM had picked up the latest `transformers` from PyPI default-resolution despite our intent to pin `4.44.2`. **R9 (pin every dependency)** had not yet been applied at the VM provisioning step — `requirements.stage1.lock.txt` exists in-repo but `31_install_deps.sh` did not enforce it for this run.

**Surface for triage.** The `error.json` carries phase + traceback + lineage including the offending `transformers_version=5.8.0`, which is enough to diagnose without re-running. **Validates the BenchmarkError + per-phase context-manager design from Stage 1.5** — the failure produced a clean stub in `runs.jsonl` and a structured `error.json` rather than a stack trace into stderr.

**Mis-classification.** `_classify_error` mapped this to `other`. An `ImportError` on a `Flax*` symbol during `model_load` is unambiguously a framework-version mismatch — worth a dedicated `framework_mismatch` category in a future cleanup so the dashboard can group it with similar drift failures.

**Action items.**
- [ ] Make `31_install_deps.sh` install from `requirements.stage1.lock.txt` strictly, not from `requirements.txt`.
- [ ] Add `framework_mismatch` category to `_classify_error` (matches `ImportError` + `Flax*` substring).
- [ ] Pre-flight check: `python -c "from transformers import FlaxAutoModel"` in `40_verify_jax.sh` so this fails *before* burning compile minutes.

---

## Run 2 — `6f049c5d-…` (SUCCESS)

| Field | Value |
|---|---|
| Status | clean |
| Model / precision | bert_base · bf16 |
| Device | tpu (v5e-1, us-west4-a) |
| transformers | 4.44.2 (pinned) |
| jax | 0.6.2 |
| Latency p50 / p95 / p99 | 0.6407 / 0.6591 / 0.6634 ms |
| Latency CV | **1.31%** (gate: <10%) |
| Throughput @ bs=64 | 5,261 ± 7.8 samples/sec |
| First / subsequent compile | 5.20 s / 0.0008 s |
| Cost per run / per 1k samples | $0.00056 / $0.0000190 |

**Numbers vs. expectations.** 5,261 samp/s on bs=64 implies an effective ~12.16 ms per batch, i.e. **~84% of the ideal `64 × p50`** (40.99 ms). The headroom (~7.7×) is exactly what we expect — bs=1 is decode-style memory-bound; bs=64 lets the v5e MXU stay loaded and amortises sync overhead. **Verdict: numbers are sensible, no anomaly.**

**CV is excellent.** 1.31% on n=300 (3 blocks × 100 passes) is well inside the <10% acceptance gate. The MAD-based outlier removal in `observe/stats.py` did its job; no `high_variance` flag emitted.

**Compile cache works.** First compile 5.20 s; second invocation 0.0008 s — the `compile_controller` clear-and-rerun pattern is verified end-to-end. `compile_cache_hit=False` in the result dict reflects the explicit cache-clear in phase 2 (this is correct — the first compile is *cold by design*, and "warm cache hit" is what the 0.8 ms second-call number represents). **The `compile_cache_hit` field is misnamed for Stage 1 semantics** — what we report is *cold-compile-after-deliberate-clear*, not "did the persistent cache help." Worth renaming or splitting into `cold_compile_explicitly_cleared` + `persistent_cache_hit` once the GCS-backed cache is wired up (Tier 1 R4).

**Lineage gap.** `git_sha=unknown` — the TPU VM checkout was via tarball deploy (`30_deploy_repo.sh`), which strips `.git/`. `lineage.get_git_sha()` returned `unknown` rather than the actual SHA. **R13 (validate lineage against git log) silently passes here because it's a no-op when no `.git/` is present.** The harness should accept a `--git-sha` override (or read `GIT_SHA` env var) and the deploy script should pass it from the host. Without this, the evidence chain breaks at row 1 of every TPU-from-tarball run.

**Action items.**
- [ ] Add `--git-sha` arg to `harness.py`; `30_deploy_repo.sh` exports `GIT_SHA=$(git rev-parse HEAD)` before tarballing.
- [ ] Split `compile_cache_hit` field semantics or rename in next runs.jsonl schema bump.
- [ ] Once probe layer is registered (Stage 1.5 was wired into `runner.py` but no probes were active for this run), rerun smoke and confirm `timing.json` / `memory.json` / `input_fingerprint.json` appear in `run_logs/`.

---

## Synthesis — readiness for Stage 2

**What works:** harness end-to-end; failure capture; statistical gate; cold/warm compile split; cost telemetry; the per-run `lineage.json` artefact.

**What's still soft:**
1. **Lineage is broken on TPU** (`git_sha=unknown`) — every TPU-from-tarball run loses provenance until the override path lands.
2. **Probes were registered but not active** during the smoke run — we have *zero* per-phase timing, fingerprint, or memory data on the TPU side. R19 sign-off should require a re-run of smoke with the default-on probe set (Timing + Memory + InputFingerprint).
3. **Pinning is enforced in-repo but not in-deploy** — Run 1's failure is the proof.
4. **One row is not n=3** — the BF16-free claim (R20) and reproducibility (R23) both need re-runs from a fresh VM, ideally in a different zone.

**Decision.** Stage 1 is *demonstrated* (one clean run, gates honoured, evidence chain intact for the in-repo fields). It is *not yet hardened* — the four soft points above are exactly what Tier 3 R20–R26 are designed to close. **Recommendation: do not start Path 2 until the smoke is rerun with default-on probes and `git_sha` is captured cleanly.**
