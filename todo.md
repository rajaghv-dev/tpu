# TODO — Provisioning issues uncovered during first `provision_tpu.sh` run

**Context:** Session 5 (2026-05-11). First real provisioning of `tpu-demo` (`v5litepod-1`) preemptible in `us-west4-a` on project `nellaiappar-001`. TPU created successfully. Then `scp --recurse "$(pwd)"` started copying the entire local working tree — including garbage — over a slow gcloud SSH tunnel, while the VM was already billing at $0.36/hr.

## What went wrong

| Issue | Severity | Detail |
|---|---|---|
| `scp --recurse "$(pwd)"` has no excludes | **High** | Copies `.tpu/` venv (~300 MB, useless on TPU), `.git/` (full history, ~10 MB, not needed), `.claude/` (local session state, possibly sensitive) |
| TPU billing during a slow copy | **High** | gcloud SSH tunnel is ~300 KB/s; copying `.tpu/` alone wastes ~15–30 minutes of preemptible time. ~$0.10–0.20 burned for zero benefit. |
| `.tpu/` venv contains a Linux x86_64 Python interpreter | Medium | Might "run" on the TPU VM but has NO jax[tpu]/transformers — the next step (`pip install -r requirements.txt`) installs into the system Python anyway, so the copied venv is dead weight. |
| `.claude/settings.local.json` got copied | Medium | First file in the scp output. Local session settings travelling to a shared cloud machine is a leak vector even if not a credentials leak today. Verify what's in there. |
| SSH key has a passphrase | Low | Every subsequent `gcloud tpu-vm ssh` will prompt. Need `ssh-agent` or passphrase-less key for automation. |
| Script blocks until copy completes | Low | No way to background or cap the copy time. |

## Decisions needed NOW (TPU is billing)

1. **Kill the current run and re-provision with a fixed script** (recommended), OR
2. **Let it finish**, accept the waste, fix the script for next time

### Recommended action (do this now)

```bash
# 1. Ctrl+C the scp in the terminal where provision_tpu.sh is running
# 2. Tear down the half-provisioned VM to stop billing:
./scripts/teardown_tpu.sh tpu-demo us-west4-a
# 3. Confirm zero VMs running:
./scripts/kill_all_tpus.sh --dry-run     # should print "you're at $0/hr"
```

Cost so far if killed now: roughly **$0.05–0.10** (5–10 min × $0.36/hr).

## Fixes for `scripts/provision_tpu.sh` (before next provision)

### Option A — Add excludes to scp (minimal change)
Replace the scp line with a `tar | ssh tar -x` pipeline that filters:

```bash
echo "Copying repo (excluding .tpu/, .git/, .claude/, results/otel/) ..."
tar --exclude='.tpu' \
    --exclude='.git' \
    --exclude='.claude' \
    --exclude='results/otel' \
    --exclude='results/run_logs' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    -czf - . | \
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --command="mkdir -p ~/tpu-examples && tar -xzf - -C ~/tpu-examples"
```

Pros: minimal change, preserves the "fresh tree" semantics.
Cons: still copies untracked files (e.g. anything the user dropped in cwd).

### Option B — `git clone` on the TPU (cleanest)
Replace scp with a git clone from inside the VM:

```bash
echo "Cloning repo on TPU VM (HTTPS, no auth needed for public repo) ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --command="git clone --depth=1 https://github.com/rajaghv-dev/tpu.git ~/tpu-examples"
```

Pros: tiny transfer (only tracked files, latest commit), no local-state leakage, fast.
Cons: requires the latest commit on `origin/main` to be the one to run (no testing of uncommitted local changes). Mitigation: if local has uncommitted changes, fall back to Option A.

### Option C — Hybrid (recommended)
Use git clone for the bulk, then scp only uncommitted changes:

```bash
gcloud ... ssh ... --command="git clone --depth=1 https://github.com/rajaghv-dev/tpu.git ~/tpu-examples"
if ! git diff-index --quiet HEAD --; then
    echo "Local has uncommitted changes — overlaying via tar ..."
    git diff HEAD | gcloud ... ssh ... --command="cd ~/tpu-examples && git apply -"
fi
```

## Other todos surfaced

### SSH key UX
- The script silently let `gcloud` generate `/home/raja/.ssh/google_compute_engine` with a passphrase. Every subsequent `gcloud tpu-vm ssh` prompts. Either:
  - Use a passphrase-less key (acceptable for personal dev, less secure)
  - Set up `ssh-agent` in the shell so the passphrase is cached
  - Document in README.md so future provisions don't surprise

### `.claude/` leak audit
- `.claude/settings.local.json` was copied at the very start. Check:
  ```bash
  cat .claude/settings.local.json | head -20
  ```
- If it contains any credentials/tokens, add `.claude/` to the scp exclusion list AND audit if it should also be in `.gitignore` (it already is — good — but locally it can still travel via scp).

### `gcloud_setup.sh` doesn't request a project check
- The output shows `project nellaiappar-001` — was this set by `gcloud_setup.sh` or pre-existing? Verify `gcp_bootstrap.sh` writes this project somewhere accessible to `provision_tpu.sh` (right now both read `gcloud config get-value project` independently — works but no audit trail).

### Add a "dry-provision" mode
- `./scripts/provision_tpu.sh --dry-run` should print the create + scp + install commands WITHOUT executing — useful for cost estimation before pulling the trigger.

### Add cost telemetry to provision_tpu.sh
- Print elapsed wall-time and estimated cost at each major step (create, copy, install, hello-TPU) so the user sees burn rate in real time.

## Decision matrix

| Action | Time cost | Money cost | Confidence |
|---|---|---|---|
| Kill + re-provision with fixed script | ~10 min | $0.10 wasted, ~$0.05 next run | High |
| Let current finish + fix for next time | ~30 min wait | $0.20–0.30 wasted | Medium (may also OOM the VM disk with junk) |
| Let current finish + manual cleanup on VM | ~30 min + ssh cleanup | $0.30+ | Low ROI |

**Recommendation: kill + fix + re-provision.** The script fix is ~20 lines and saves you $0.20 every provision for the rest of the project.

## Cost & speed optimization opportunities

These compound. The first three give you 80% of the win.

### Tier 1 — Fix what's actively burning money RIGHT NOW

| # | Fix | Saves per provision | Effort |
|---|---|---|---|
| 1 | **`tar --exclude=.tpu --exclude=.git --exclude=.claude` before scp**, OR `git clone --depth=1` on the VM | ~25 min × $0.36/hr ≈ **$0.15** | 20 lines in `provision_tpu.sh` |
| 2 | **Strip `.claude/` from any copy path** | leak risk, not $ | 1 line |
| 3 | **Pin SSH key to ssh-agent or use passphrase-less key** | minutes of typing per provision | one-time |

Cumulative effect: a clean `provision_tpu.sh` run becomes **~3–5 min instead of ~30 min**. At $0.36/hr that's the difference between $0.02 and $0.18 of "just getting onto the VM."

### Tier 2 — Recurring costs across many provisions

| # | Fix | Saves per provision (after 1st) | Effort | Storage cost |
|---|---|---|---|---|
| 4 | **Cache pip wheels in GCS**: `gsutil cp` the resolved wheel set after first install, then `pip install --find-links=gs://$BUCKET/wheels` on subsequent provisions | ~5–10 min (jax[tpu]/torch/transformers/TF) ≈ **$0.04** | ~30 min one-time | ~500 MB × $0.02/GB/mo ≈ $0.01/mo |
| 5 | **Cache HF model weights in GCS** (ADR-006 committed this, not built): persist `~/.cache/huggingface/hub` → `gs://$BUCKET/models/`, re-mount via `gcsfuse` or sync on boot | ~1–2 min per model × 5 models = **~$0.04** per quick suite | ~2 hours | ~5 GB × $0.02 ≈ $0.10/mo |
| 6 | **Persist XLA compile cache to GCS**: `export JAX_COMPILATION_CACHE_DIR=/tmp/xla-cache` then sync to `gs://$BUCKET/xla-cache/` | ~10–60 sec per model after first cold compile | ~30 min | <$0.01/mo |
| 7 | **Build a custom TPU VM image with deps pre-baked** (advanced): `gcloud compute images create` with deps installed | replaces #4 entirely — every provision starts ready | half day | image storage ~$0.02/mo |

After Tier 2 is wired: a re-provision drops from ~10 min (with fixes) to **~2 min** (~$0.01).

### Tier 3 — Architectural cost reduction

| # | Fix | Saves | Effort |
|---|---|---|---|
| 8 | **Run smoke before quick — always.** Catches config errors at $0.05 instead of $0.30 | $0.30 per false-start avoided | habit |
| 9 | **Use Colab Pro TPU for iteration** (already paid, $0 marginal). Reserve v5e-1 for measurement-grade only. | ~80% of dev-cycle cost | Colab notebook wrapper |
| 10 | **Batch experiments per VM lifetime**: provision once, run smoke + quick + custom + teardown in one session. Don't provision per single experiment. | provisioning overhead amortized | already designed |
| 11 | **Auto-teardown safety**: provision script writes a `at` job to kill the VM after N hours; cancelled by successful teardown_tpu.sh | catastrophic-leak insurance ($259/mo if forgotten) | 5 lines |
| 12 | **Try `--spot` instead of `--preemptible`** on v5e-1 | sometimes 10–20% cheaper (~$0.05/hr instead of $0.36/hr — unconfirmed for v5e) | flag swap + test |
| 13 | **Reuse the VM**: don't teardown between runs in the same day. Cost of an idle VM = $0.36/hr; cost of provisioning = ~3–5 min ≈ $0.02. Idle >5 min between runs → cheaper to teardown. | varies | judgment call |

### Tier 4 — Speed (not always cost)

| # | Fix | Speeds up | Effort |
|---|---|---|---|
| 14 | **Parallel scp**: gcloud's scp is serial. Use `tar | ssh tar -x` with `pigz` for parallel compression | copy phase | small |
| 15 | **Use `gcloud compute tpus tpu-vm ssh --strict-host-key-checking=no`** to skip the first-time host-key prompt | UX, not cost | 1 line |
| 16 | **Pre-warm the HF model cache LOCALLY** with `huggingface-cli download` before provisioning, then sync to VM via the GCS path | first-run model download | requires #5 |

### Quick-decision table

If your goal is **"benchmark one model, throw it away"**:
→ Just do Tier 1. ~$0.05 per cycle.

If your goal is **"iterate on the same registry over weeks"**:
→ Tier 1 + Tier 2 (#4, #5, #6). Per-cycle cost drops to ~$0.02 after the first run. Pay ~$0.15/mo for cached storage.

If your goal is **"learn before committing money"**:
→ Tier 3 #9 (Colab) for most work; spend on v5e-1 only when you need the measurement.

If your goal is **"sleep at night not worrying about forgotten VMs"**:
→ Tier 3 #11 (auto-teardown). Most important psychologically; the panic button (`make tpus-kill`) is the manual version.

## Order of operations for next session

1. `./scripts/teardown_tpu.sh tpu-demo us-west4-a` (if not already done)
2. Verify: `./scripts/kill_all_tpus.sh --dry-run` shows zero
3. Audit `.claude/settings.local.json` for leak risk
4. Edit `scripts/provision_tpu.sh` — apply Option C (or A as fallback)
5. Add `--dry-run` mode to `provision_tpu.sh`
6. Commit + push the script fix
7. Re-provision: `./scripts/provision_tpu.sh`
8. Verify the copy is fast and excludes `.tpu/.git/.claude` (use `gcloud ... ssh --command='du -sh ~/tpu-examples/*'` to confirm)
9. Continue with OTel + benchmark run
