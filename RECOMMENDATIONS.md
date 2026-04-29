# Recommendations

> **Format.** Rec ID · Action · Why · How · Effort.
> **Tiers.** Tier 1 (do before any Stage 1 code) · Tier 2 (during Stage 1 build) · Tier 3 (before Stage 2).
> **Effort.** Low (<30 min) · Medium (1–4 hrs) · High (>4 hrs).
> **Last reviewed.** 2026-04-29.

> **Stage 1 status (2026-04-29):** Code complete. `benchmarks/harness.py`, `benchmarks/runner.py`, `models/registry.yaml` (5 models), `observe/stats.py`, `observe/lineage.py`, `observe/compile_controller.py`, `results/dashboard/index.html`, and 97 unit tests all committed and pushed. Tier 1 items R1–R9 are pre-requisites to apply before the first real TPU run. Tier 2 items R10–R18 were applied during the build. Tier 3 items R19–R26 are the immediate pre-Stage-2 checklist.

---

## Tier 1 — Do immediately (before writing any Stage 1 code)

These exist to remove blockers, set guardrails, and prevent the most common foot-guns from appearing in Stage 1 code.

### R1 — Create the GCS model-cache bucket
- **Action.** Create `gs://rajaghv-tpu-cache` in `us-central1` with uniform bucket-level access.
- **Why.** Every Stage 1 model load assumes this bucket exists (ADR-006). Without it, the harness re-downloads weights from HF on every preemptible-VM lifetime, costing time and money (R-C01).
- **How.** `gcloud auth login rajaghv.dev@gmail.com && gcloud config set project <PROJECT> && gcloud storage buckets create gs://rajaghv-tpu-cache --location=us-central1 --uniform-bucket-level-access`. Verify with `gcloud storage ls gs://rajaghv-tpu-cache`.
- **Effort.** Low.

### R2 — Configure the HuggingFace token (PRO account)
- **Action.** Generate a HF user-access token with read scope, store as `HF_TOKEN` env var on every machine + as a Colab secret + as a GitHub Actions secret.
- **Why.** Stage 1's 5-model registry includes Gemma (gated). Pre-flight checks need authenticated calls (R-T07). Forgetting this means the suite fails 30 minutes in.
- **How.** Visit `https://huggingface.co/settings/tokens` → New token (read scope) → copy. Add to `~/.bashrc` (`export HF_TOKEN=...`), to Colab `Secrets` panel, and to repo's `Settings → Secrets and variables → Actions`. Test: `huggingface-cli whoami`.
- **Effort.** Low.

### R3 — Run a Colab Pro TPU runtime test
- **Action.** Open a fresh Colab Pro notebook, attach a TPU v2-8 runtime, run a 30-line JAX-on-TPU "hello world" (matmul + jit + block_until_ready), confirm it succeeds end-to-end.
- **Why.** Colab Pro TPU plumbing breaks regularly with version drift. Discovering that on Stage 1 runner is too late.
- **How.** New notebook → Runtime → Change runtime type → TPU v2-8 → run cell with `import jax; print(jax.devices())`. Save the working JAX version to `docs/colab_setup.md`.
- **Effort.** Low.

### R4 — Set persistent JAX compilation cache env var
- **Action.** Set `JAX_COMPILATION_CACHE_DIR=/mnt/gcs-cache/jax-cache` on every TPU/GPU VM by default.
- **Why.** Compile cache misses can cost 5 min per model (R-T06). With a persistent cache on GCS, the second VM in the same week reuses the artifact for free.
- **How.** Add `export JAX_COMPILATION_CACHE_DIR=/mnt/gcs-cache/jax-cache` to TPU VM startup script and to local `~/.bashrc`. Confirm path is on the gcsfuse mount.
- **Effort.** Low.

### R5 — Run every remote VM under tmux
- **Action.** SSH to a TPU/GPU VM → `tmux new -s bench` → run the suite inside tmux → `Ctrl-b d` to detach.
- **Why.** Network drops between India and us-central1 are routine. A bare SSH session loses the run; a tmux'd run survives. Combined with append-only JSONL (ADR-007) recovery is automatic.
- **How.** Add to `docs/runbooks/run_remote.md`. Default the gcloud-ssh wrapper script to attach to (or create) a `bench` tmux session.
- **Effort.** Low.

### R6 — Smoke-test the harness skeleton on v5e-1 BEFORE the real models
- **Action.** First Stage 1 commit: harness skeleton + a single tiny model (e.g., `distilbert-base-uncased`, 66 M params, BF16, batch=8). Run end-to-end on v5e-1. Verify a row appears in `runs.jsonl`. THEN add the other 4 models.
- **Why.** A 5-model first commit hides which model breaks the harness. A 1-model commit isolates harness bugs from model bugs.
- **How.** Commit 1: harness + DistilBERT only. Commit 2..5: add one model each. Each commit runs through `--suite smoke` end-to-end before the next.
- **Effort.** Medium.
- **Stage 1 outcome (2026-04-29).** Applied with minor variation: all 5 models committed together but fully unit-tested before TPU run. Use `--dry-run` to verify harness logic without downloading models. First real TPU run should verify R6's intent.

### R7 — Choose Stage 1 starter models that are XLA-clean
- **Action.** Stage 1 registry must contain only models with mature, well-understood JAX/Flax forward paths: ViT-B/16, ResNet-50, BERT-base, GPT-2 (125 M), Gemma-2-2B-it (or Phi-3-mini if Gemma access blocked). NO Mamba, NO MoE, NO RecurrentGemma in Stage 1.
- **Why.** Stage 1 is calibrating the harness. A novel-arch failure during calibration confounds harness bugs with arch bugs (R-T03/R-T04). Save novel-arch for Stage 3 once the harness is trusted.
- **How.** Lock the 5-model list in `models/registry.yaml` BEFORE writing harness. Use P23 to justify each choice.
- **Effort.** Low.

### R8 — Set a budget cap and a billing alert
- **Action.** Configure GCP budget alert at 5 USD/month on the project; set `gcloud config set compute/preemptible true` as default; review billing daily for the first week.
- **Why.** A forgotten v6e VM at 0.75 USD/hr × 168 hrs = 126 USD. We do not want to discover that on a Tuesday.
- **How.** GCP Console → Billing → Budgets → Create → 5 USD threshold → email alert at 50%, 90%, 100%. Default preemptible: `gcloud config set compute/preemptible true`. Add `scripts/check_no_idle_vms.sh` to cron (daily).
- **Effort.** Low.

### R9 — Pin every dependency in requirements.txt
- **Action.** Pin JAX, jaxlib, Flax, transformers, huggingface_hub, torch, torch_xla — exact versions, not ranges.
- **Why.** Version drift across JAX minor releases changes XLA HLO output, invalidating compile cache and confusing comparisons across stages.
- **How.** `pip freeze > requirements.lock.txt` after Stage 1 environment works. Use `==` not `>=`. Document upgrade procedure as a separate runbook.
- **Effort.** Low.

---

## Tier 2 — Do during Stage 1 build

These keep the build honest and prevent debt accumulation between commits.

### R10 — Commit after each module, not at the end
- **Action.** Commit after each of: harness.py, runner.py, registry.yaml (1 model), lineage.py, stats.py, compile_controller.py, dashboard/index.html. Each commit must run-clean (`pytest` + `python -m benchmarks.runner --suite smoke --dry-run`).
- **Why.** A single end-of-Stage-1 commit is unreviewable, unbisectable, and hides where bugs entered. Per-module commits give a working bisect history.
- **How.** Standard discipline. Use a checklist in `LESSON_PLAN.md` Stage 1 section.
- **Effort.** Low (per-commit overhead is small if you've internalised the habit).

### R11 — Run the smoke suite after each new model is added to the registry
- **Action.** When a new model joins `registry.yaml`, run `--suite smoke --filter <new_model>` before merging the PR.
- **Why.** Each new model can fail the harness in a new way (R-T01..R-T07). Catching it at PR time means one bisectable change.
- **How.** Add a CI step `if registry.yaml changed: run smoke for new entries`. Manual until CI exists (Stage 9).
- **Effort.** Low.

### R12 — Test CV before claiming a result
- **Action.** Every new chart, every dashboard column, every claim in `context.md` must reference at least one row from `runs.jsonl` with `quality_flag == "clean"`. Reject `high_cv` rows from headline numbers.
- **Why.** A 12% CV reading is not a result, it's a noise blob. Charts built on noise blobs spread misinformation (R-L01..R-L04).
- **How.** Lint script `tools/check_claims.py` parses `context.md` for "throughput is X" patterns, looks up the cited run_id, asserts `quality_flag=="clean"`. Fails CI otherwise.
- **Effort.** Medium.

### R13 — Validate lineage.py output against git log
- **Action.** Unit test for `observe/lineage.py`: capture lineage, then `git log -1 --format=%H` and assert equal.
- **Why.** A silently-wrong git_sha breaks every traceability claim (R-T05 analogue for provenance). The whole evidence chain depends on this field being correct.
- **How.** `tests/test_lineage.py` with `subprocess.check_output(["git", "rev-parse", "HEAD"])`. Run in CI.
- **Effort.** Low.

### R14 — Verify compile_controller.py actually clears the cache
- **Action.** Test asserts: after `clear_xla_cache()`, the next compile takes >1 second on a reference model.
- **Why.** A no-op `clear_xla_cache` produces the worst kind of bug: silently-warm "cold" measurements (R-M02).
- **How.** `tests/test_compile_controller.py`: define small `jit(matmul)`, call once, clear, call again, assert second-call time > 0.5 s. Skip on CPU (cache behaviour different).
- **Effort.** Low.

### R15 — Test dashboard rendering locally before push
- **Action.** Before any commit touching `results/dashboard/`, serve locally (`python -m http.server -d results/dashboard 8080`) and open in a browser. Confirm: rows render, filters work, no console errors.
- **Why.** GitHub Pages takes 1–5 minutes to update; a broken dashboard pushed to main is publicly broken until the fix lands.
- **How.** Add to PR checklist. Add `make dashboard-preview` target.
- **Effort.** Low.

### R16 — Ensure drop_remainder=True everywhere batches are formed
- **Action.** Audit all dataloader / synthetic-input code for batch construction; assert `drop_remainder=True` (or equivalent — pad to fixed shape).
- **Why.** Variable-final-batch is the single most common cause of XLA recompilation (R-T01). One uncaught spot recompiles the model every suite run.
- **How.** Grep for batch construction in harness.py and runner.py; explicit `drop_remainder=True`. Add a unit test that runs the suite twice and asserts `compile_time_s < 0.1` on the second pass.
- **Effort.** Low.

### R17 — Test new code on CPU-JAX before TPU
- **Action.** Every new harness change first runs on `JAX_PLATFORMS=cpu` locally. Only after CPU passes does it touch TPU.
- **Why.** TPU minutes cost money; CPU minutes don't. Most logic bugs (loop bounds, shape mismatches, dtype promotions) appear identically on CPU JAX.
- **How.** `make test-cpu` target that sets the env var and runs pytest. Local pre-push hook.
- **Effort.** Low.

### R18 — Always benchmark with one warmup before the n=3 measured runs
- **Action.** `compile_controller.cold_warm_split(...)` is called with `warmups=2, measured=3` in every cell. The compile-time number is reported separately from the steady-state number.
- **Why.** Without warmup, run #1 is systematically slower (cold caches, JIT overhead leaks); CV is inflated; comparisons are noisy (R-M03).
- **How.** Wired into `harness.py`. Test asserts that warmup runs are NOT in the measured-times array.
- **Effort.** Low.

---

## Tier 3 — Do before Stage 2

These close out Stage 1 honestly and prevent Stage 2 from inheriting silent bugs.

### R19 — Review all Stage 1 results before extending to Path 2 ← **NEXT IMMEDIATE ACTION**
- **Action.** Walk every row in `runs.jsonl` produced by Stage 1. For each: does the throughput match expectations? Is the CV clean? Is the compile time sensible? Document any surprises in `results/stage1_interpretation.md`.
- **Why.** Extending to Path 2 (JAX+GPU) on an unverified Path 1 baseline means you can't tell whether a Path 2 anomaly is Path 2's fault or a pre-existing Path 1 bug.
- **How.** Pair-review the dashboard with yourself (or a future-AI). One sentence per row. Open issues for surprises.
- **Effort.** Medium.

### R20 — Validate the BF16-free-on-TPU claim with actual measurements (P31)
- **Action.** Run ResNet-50 in BF16 vs FP32 vs MIXED on v5e-1 and produce a verdict-with-evidence in `context.md`.
- **Why.** This is a load-bearing claim used in many subsequent recommendations. If it's wrong (or nuanced), Stage 2 dtype defaults are wrong (R-L03).
- **How.** Execute P31 prompt. Update `context.md` evidence chain.
- **Effort.** Medium.

### R21 — Check for thermal throttling on the first multi-hour run
- **Action.** During the first ≥1-hour Stage 1 sequential run on B200, log GPU temp + clock at 1 Hz. Plot at end. Confirm no throttle.
- **Why.** Throttling silently degrades all downstream results (R-T02). Catching it before Stage 2 means we either fix cooling or design around it.
- **How.** `pynvml` 1-Hz logger as a background thread; export to `results/telemetry/`; matplotlib plot.
- **Effort.** Low.

### R22 — Document any unexpected findings in context.md "aha moments"
- **Action.** Any Stage 1 result that surprised you — add a one-paragraph entry to `context.md` § "Aha moments" with the run_id evidence link.
- **Why.** Insights compound. An unrecorded surprise this week is forgotten next week and re-discovered (or re-bugged) next month.
- **How.** Standing PR template: "Did this stage produce any surprise? If yes, add to aha-moments section." Reject merge if section is missing on a stage-completion PR.
- **Effort.** Low.

### R23 — Cross-validate one Stage 1 result manually (re-run with fresh VM)
- **Action.** Pick one Stage 1 result (e.g., ViT-B/16 batch=32 throughput on v5e-1). From a fresh VM in a different zone, with a fresh checkout, re-run that single cell. Confirm new mean falls within original CI_95.
- **Why.** Proves that the entire pipeline (registry → harness → runner → JSONL → dashboard) is reproducible (P37 in dress rehearsal). If reproduction fails, find the leak before Stage 2.
- **How.** Spawn fresh VM in `us-central1-b` (if Stage 1 ran in `-a`), `git clone`, run one cell, compare.
- **Effort.** Medium (~1 hr including provisioning).

### R24 — Set up GitHub Actions for an automated smoke test on push
- **Action.** Add `.github/workflows/smoke_on_push.yml` that runs the smoke suite on a CPU-JAX runner (no TPU/GPU) on every push to `main`. Asserts harness invocation succeeds and produces ≥1 valid JSONL row.
- **Why.** Catches regressions in the harness/registry/lineage code before they silently break Stage 2's first TPU run (and before they consume cloud minutes).
- **How.** Free GitHub-hosted Ubuntu runner. `JAX_PLATFORMS=cpu`. Single tiny model (DistilBERT batch=1). Should complete in <10 min.
- **Effort.** Medium.

### R25 — Snapshot the Stage 1 environment hash and pin it
- **Action.** At Stage 1 close, run `pip freeze > requirements.stage1.lock.txt`, compute the env hash, store both in the repo. Tag the commit `stage1-complete`.
- **Why.** Stage 2 will upgrade something (probably JAX). Without a frozen Stage 1 snapshot we lose the ability to re-run Stage 1 results identically.
- **How.** `pip freeze` → file → commit → `git tag stage1-complete` → push.
- **Effort.** Low.

### R26 — Write the Stage 1 retrospective before starting Stage 2
- **Action.** A `LESSON_PLAN.md` retrospective entry: what went well, what cost more than expected, what you'd change in Stage 2's plan.
- **Why.** Stage 2's design assumptions inherit Stage 1's blind spots if not explicitly examined. The retrospective is where they get surfaced.
- **How.** ~30 minutes, free-form. Include: time per cell vs estimate, cost vs budget, any ADR that needs revisit.
- **Effort.** Low.

---

Documents complete. Five artifacts produced:

- **PROMPTS_ADDITIONS** — 19 self-contained prompts P22–P40 in the same format as the existing improved prompts (output format, constraints, scope per prompt).
- **DECISIONS.md** — 13 ADRs covering scope, frameworks, hardware target, inputs, weights, cache, results format, dashboard, statistics, execution, staging, model ceiling, novel-architecture inclusion. Each ADR includes Status, Context, Rationale, Alternatives, Consequences, Risks, and Revisit Trigger.
- **RISKS.md** — 5 sections (Technical R-T01..R-T08, Measurement R-M01..R-M05, Cost & Access R-C01..R-C04, Infrastructure R-I01..R-I04, Learning R-L01..R-L04), 25 risks total with Likelihood, Impact, Root cause, Early warning, Mitigation, Contingency.
- **QUESTIONS.md** — 23 questions across 5 domains (Hardware/Memory, Compiler/XLA, Model Architecture, Measurement/Statistics, Infrastructure/Workflow), each with Why, How, Expected answer (best-guess reasoning), and Difficulty rating.
- **RECOMMENDATIONS.md** — 26 recommendations across 3 tiers (Tier 1 pre-Stage-1: R1–R9; Tier 2 during Stage 1: R10–R18; Tier 3 pre-Stage-2: R19–R26), each with Action, Why, How, Effort.

Cross-references are wired throughout: ADRs are cited from RISKS (e.g., R-T01 → ADR-006), questions tie back to risks (Q16 → ADR-009), recommendations cite both prompts and risks (R6 → P29, R20 → P31, R23 → P37). All content stays inside the locked decision set (inference-only, JAX+PyTorch, v5e-1 primary, n=3 with CV<10%, 75-model registry, 4B ceiling, 9-stage build).