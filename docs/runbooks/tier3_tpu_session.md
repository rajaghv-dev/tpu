# Tier 3 TPU/B200 Session — Runbook

> **What this is.** Three open Tier 3 items (R20, R21, R23) need a live
> TPU/B200 session to actually run. This runbook is the checklist to execute
> them in order in one session. Costs ~$0.30 + ~1 hour of B200 power.
> **Closes:** R20, R21, R23 from `RECOMMENDATIONS.md`.

## Pre-flight (~5 min)

- [ ] HF token live: `huggingface-cli whoami` returns `rajaghv-dev`
- [ ] gcloud auth: `gcloud auth list` shows `rajaghv@gmail.com` active
- [ ] Project: `gcloud config get-value project` returns `nellaiappar-001`
- [ ] Repo clean: `git status` shows nothing uncommitted
- [ ] Reference values noted (don't re-derive these mid-session):
  - `bert_base bf16` reference: p50=0.6407 ms, tp=5261.2 samp/s, CV=1.31%
  - Reference run_id: `6f049c5d-d1fb-4f1b-aa9a-998c34d2e894`

---

## Phase A — TPU work (R20, R23) on v5e-1

Cost: ~30 min wall, ~$0.20 spot.

### A1. Provision and verify (`./scripts/00..40_*.sh`)

```bash
./scripts/00_validate_local.sh
./scripts/01_validate_gcp.sh
./scripts/02_validate_bucket.sh
./scripts/03_validate_hf.sh
./scripts/10_setup_bucket.sh
./scripts/20_provision_tpu.sh        # picks first zone with quota
./scripts/21_wait_tpu_ready.sh
./scripts/30_deploy_repo.sh
./scripts/31_install_deps.sh         # ⚠ MUST install from requirements.stage1.lock.txt strictly
./scripts/40_verify_jax.sh
```

> ⚠ **Known gap from R19:** `31_install_deps.sh` is the deploy step that
> caused Run 1's failure. Verify it pulls from `requirements.stage1.lock.txt`,
> not `requirements.txt`. Fix before running R20 if not already.

### A2. R20 — BF16 vs FP32 validation

```bash
./scripts/53_run_bf16_validation.sh
```

Acceptance: tp ratio bf16/fp32 within ±5%, p50 ratio within ±5%.
Verdict prints at end: `BF16 is FREE on v5e-1` or `BF16 is NOT FREE on v5e-1`.

After:
- [ ] Pull rows: `./scripts/60_pull_results.sh`
- [ ] Update `context.md §19` with the verdict (add 19.6 entry)
- [ ] If NOT FREE: open an issue, mark ADR-007 (BF16-default) for revisit

### A3. R23 — Cross-zone reproducibility

```bash
./scripts/55_repro_validation.sh
# This script tears down the current VM and re-provisions in a different zone.
```

Acceptance: new tp within ±10% of 5261.2, new p50 within ±10% of 0.6407.

After:
- [ ] Pull rows: `./scripts/60_pull_results.sh`
- [ ] If REPRODUCIBLE: log in `results/stage1_interpretation.md` (add a "Run 3" section)
- [ ] If DRIFT: investigate before Stage 2. Likely culprits:
  - Different zone → different hardware revision
  - Different `requirements.stage1.lock.txt` resolution
  - Persistent JAX cache from a previous run leaking into the "fresh" timing

### A4. Teardown

```bash
./scripts/70_teardown_tpu.sh
./scripts/71_verify_teardown.sh   # asserts \$0/hr
```

---

## Phase B — B200 work (R21) on local DGX

Cost: ~1 hour wall, electricity only (~$0.12 at 1000W × 1hr × $0.12/kWh).

### B1. Pre-flight (on the DGX)

- [ ] `nvidia-smi` works and shows the B200 (Blackwell)
- [ ] `python3 -c "import pandas, matplotlib"` succeeds
- [ ] No other CUDA jobs running (`nvidia-smi` shows GPU util ≈ 0%)
- [ ] Cooling: ambient temp normal, fans audibly running

### B2. R21 — Thermal throttle check

```bash
./scripts/54_thermal_check.sh
```

This runs the quick suite + 10 min padding (~1 hr total) with a 1 Hz
nvidia-smi sampler in the background. Output: CSV + PNG plot in
`results/telemetry/b200_<ts>.{csv,png}`.

Acceptance: SM clock drop <5% from steady-state median AND max temp <87°C.
Verdict prints at end.

After:
- [ ] If NO THROTTLE: log in `context.md §19` as Aha 19.7
- [ ] If THROTTLED: investigate cooling before trusting any B200 number in Stage 2.
  Common causes: case airflow, thermal paste cure, ambient temp, fan curve.
- [ ] Either way, commit the PNG and CSV: `git add results/telemetry/ && git commit -m "R21 thermal data"`

---

## Post-session

- [ ] All three results documented (interpretation doc + aha moments + retrospective)
- [ ] If anything failed reproducibility (R23 drift, R20 non-flat, R21 throttle):
  open an issue and pause Stage 2 until investigated
- [ ] Update `RECOMMENDATIONS.md` Tier 3 status table to reflect what ran
- [ ] Push: `git push origin main`

---

## Aborts and recovery

| Symptom | Likely cause | Fix |
|---|---|---|
| Provision hangs >10 min | Zone quota exhausted | Try alternative zone via `TPU_ZONE=us-west1-a ./scripts/20_provision_tpu.sh` |
| Smoke fails at `model_load` with `ImportError` on `Flax*` | `transformers` not pinned (Run 1's bug) | Re-run `31_install_deps.sh` and ensure it consumes lock file |
| `git_sha=unknown` in new lineage.json | Tarball deploy still strips `.git/` | Apply Stage 2 day-1 fix (export GIT_SHA, accept `--git-sha`) |
| Repro check shows >10% drift | Persistent cache, hw rev, or non-determinism | Compare `lineage.json` env hashes; re-run with cleared GCS cache |
| Thermal sampler shows ~0 W power | nvidia-smi privilege issue or wrong GPU index | Check `nvidia-smi -L`; sampler queries all GPUs |
