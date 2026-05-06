# scripts/ — Staged TPU benchmark orchestration

Numeric-prefixed bash scripts that walk through GCP TPU provisioning, repo
deployment, dependency install, benchmark execution, results collection, and
teardown. Each script is independently runnable and idempotent; `run_all.sh`
ties them together.

Style: `set -euo pipefail`, gcloud + python3 only, ANSI colour on TTY only,
heavy in-script comments + terminal `echo`/log lines so reviewers can audit
behaviour from the source.

## Quickest path — smoke suite, ~8 min, ~$0.05

```bash
./scripts/00_validate_local.sh                      # local preflight (free)
./scripts/01_validate_gcp.sh                        # cloud preflight (free, read-only)
./scripts/91_predict_cost.sh smoke tpu_v5litepod_1  # forecast (~$0.05)
./scripts/run_all.sh --suite smoke --no-gcs         # full pipeline → tears down
```

`--no-gcs` skips the optional gcsfuse mount (Stage 1 5-model registry doesn't
need a model cache; mount becomes important at Stages 5+ with gated/large
models).

## Stages

| #  | Script                         | What it does                                                | State changes? |
|----|--------------------------------|-------------------------------------------------------------|----------------|
| 00 | `00_validate_local.sh`         | bash/gcloud/python3 on PATH, gcloud authed                  | none           |
| 01 | `01_validate_gcp.sh`           | billing, APIs, IAM perms, v5e quota in any zone             | none           |
| 02 | `02_validate_bucket.sh`        | `gs://rajaghv-tpu-cache` exists + R/W probe (standalone use; pipeline skips since 10 calls it) | none (probe deletes self) |
| 03 | `03_validate_hf.sh`            | `HF_TOKEN` valid (only if set or `--required`)              | none           |
| 10 | `10_setup_bucket.sh`           | create the bucket (idempotent)                              | creates bucket |
| 11 | `11_setup_budget.sh`           | $5/mo budget alert (idempotent)                             | creates budget |
| 20 | `20_provision_tpu.sh`          | create v5e-1 spot VM, multi-zone fallback                   | **starts billing** |
| 21 | `21_wait_tpu_ready.sh`         | poll until SSH succeeds                                     | none           |
| 30 | `30_deploy_repo.sh`            | tar + scp repo                                              | writes to VM   |
| 31 | `31_install_deps.sh`           | `pip install -r requirements.txt`                           | writes to VM   |
| 32 | `32_mount_gcs.sh`              | gcsfuse mount + HF_HOME/JAX_COMPILATION_CACHE_DIR vars      | mount on VM    |
| 40 | `40_verify_jax.sh`             | `jax.devices()` shows TPU + matmul probe                    | none           |
| 41 | `41_run_pytests.sh`            | pytest tests/ on VM (97 tests)                              | none           |
| 42 | `42_dry_run.sh`                | harness `--dry-run`                                         | none           |
| 50 | `50_run_smoke.sh`              | smoke suite (1 model, ~8 min) inside tmux                   | writes runs.jsonl |
| 51 | `51_run_quick.sh`              | quick suite (5 models, ~50 min) inside tmux                 | writes runs.jsonl |
| 60 | `60_pull_results.sh`           | scp `results/runs.jsonl` + `run_logs/` back                 | writes locally |
| 70 | `70_teardown_tpu.sh`           | delete TPU                                                  | **stops billing** |
| 71 | `71_verify_teardown.sh`        | confirm zero billable resources remain                      | none           |
| 90 | `90_status.sh`                 | active resources + estimated hourly burn                    | none           |
| 91 | `91_predict_cost.sh`           | forecast cost of `<suite> <device>`                         | none           |
| 92 | `92_idle_check.sh`             | flag VMs/TPUs running >2h ("possibly forgotten")            | none           |

## State file

`20_provision_tpu.sh` writes `.tpu-bench-state/state.env` with the chosen
zone/name. Subsequent stages read it. `70_teardown_tpu.sh` clears it. Override
the path with `TPU_STATE_DIR=...`.

## Library

- `lib/common.sh`  — logging, banner, error trap, exit handlers, state helpers.
  Sourced by every numbered script.
- `lib/config.sh`  — defaults: `TPU_NAME`, `TPU_ZONES_PRIMARY/_FALLBACK`,
  `GCS_BUCKET`, `TMUX_SESSION`, `PRICE_USD_PER_HR`, `SUITE_BASELINE_MINUTES`,
  `DEVICE_SPEEDUP_FACTOR`, `SESSION_BUDGET_USD`. Every value is a
  `: "${VAR:=default}"` so caller env always wins.

## Region / zone — important note

`DECISIONS.md` ADR-003 + ADR-006 specify `us-central1` as canonical (TPU + GCS
bucket co-located for free intra-region reads). As of 2026-05, GCP no longer
offers `v5litepod-1` (v5e single-chip) capacity in `us-central1`. The default
`TPU_ZONES_PRIMARY` is `us-east5-{a,b,c}`; `GCS_BUCKET_REGION` stays at
`us-central1` (matches ADR-006 even though it now incurs cross-region read
egress). To match TPU↔bucket regions, override `GCS_BUCKET` and
`GCS_BUCKET_REGION` to a US-east region and update ADR-006 accordingly.

## Running outside of `run_all.sh`

Each script is independently runnable. Usage:

```bash
# Just check whether anything's billing right now:
./scripts/90_status.sh

# Forecast a quick suite on a v6e-1:
./scripts/91_predict_cost.sh quick tpu_v6e_1

# Tear down a forgotten TPU you find via 92:
./scripts/70_teardown_tpu.sh tpu-bench-v5e us-east5-a

# Resume the pipeline at stage 30 after fixing a 31 install issue:
./scripts/run_all.sh --from 30
```

## Cron suggestions

```cron
# Daily at 09:00 — flag anything running > 2h
0 9 * * * /home/raja/tpu/scripts/92_idle_check.sh > /tmp/tpu-idle.log 2>&1
```

## Existing scripts (preserved)

These ship from earlier sessions and are unchanged:

- `gcloud_setup.sh`, `provision_tpu.sh`, `teardown_tpu.sh` — original TPU
  scripts. Functionally subsumed by `run_all.sh` + `20_provision_tpu.sh` +
  `70_teardown_tpu.sh`; kept for backward compatibility.
- `gcloud_ssh_run.sh`, `gcloud_pod_run.sh`, `gcloud_upload_data.sh` — utility
  helpers; not part of the numbered pipeline.

## Where to look first when a stage fails

- **Stage 01**: missing API → message points to the exact `gcloud services
  enable` line.
- **Stage 20** "RESOURCE_EXHAUSTED": spot capacity oscillates; wait 15 min
  and re-run, or set `TPU_PROVISIONING_MODEL=STANDARD` for on-demand (~3×).
- **Stage 31** pip wheel resolution failure: confirm `python3 --version` ≥ 3.10
  on the VM (the `tpu-ubuntu2204-base` image ships 3.10).
- **Stage 50/51** SSH dropped mid-run: the run is still going inside tmux on
  the VM. Re-attach with `gcloud compute tpus tpu-vm ssh ... --command="tmux
  attach -t bench"`. Or just wait it out and run `60_pull_results.sh`.
