# 11 — Cleanup and Cost Safety

> **Learning goal:** stop a Cloud TPU lab session from quietly bleeding money
> after you close the laptop. Understand which resources accrue cost when
> idle, how to prove they are gone, and how to set up billing alarms so a
> mistake costs minutes instead of weeks.

This doc is the safety net for `10_cloud_tpu_setup_playbook.md`. Read both
in the same sitting before you ever run a `create` command.

Cost-bearing commands here are prefixed with `# PAID:` inside code blocks.
Verification / inspection commands are free.

The lab never hardcodes pricing. The canonical reference is:
https://cloud.google.com/tpu/pricing — open it in a tab.

---

## 1. How a "stopped" TPU VM can still cost money

Cloud TPU VMs are not like a generic Compute Engine VM. A few specifics
worth memorising before you trust your intuition:

- **A reserved chip is a billed chip.** The instant `tpu-vm create` returns
  success and the VM moves to `READY`, the chips are allocated to your
  project and you are paying the per-chip-hour rate. Whether or not anyone
  is SSH'd in or running code does not matter.
- **`stop` on a TPU VM is not always "$0".** On some SKU classes, a stopped
  TPU still holds the allocation. Check the pricing page for your specific
  SKU. The reliable zero-cost state is **deleted**.
- **Pod slices bill per chip in the slice.** A `v5p-8` is 8 chips, billed
  at 8 × per-chip-hour. A `v5p-128` is 128 chips. The flag
  `--accelerator-type` is the price.
- **Preemptible / Spot is cheaper but not free.** It is also not the right
  tool for the benchmark methodology in `14_benchmarking_playbook.md` —
  reclaim during a steady-state window invalidates the run.
- **Cost is per chip-hour, but billing granularity is finer than 1 hour.**
  Do not let an idle VM linger for "rounding".

If you only remember one thing: **the only proven zero-state is `delete` +
`list` showing the VM is gone.**

---

## 2. The cleanup checklist

Run this at the end of every session. Better: run it after every benchmark
within the session.

```bash
# Variables — same as in 10_cloud_tpu_setup_playbook.md.
export PROJECT_ID="<your-gcp-project>"
export ZONE="us-central1-a"
export TPU_NAME="cloud-tpu-lab-vm"
```

### 2a. Delete the TPU VM

```bash
# PAID: deleting frees the billing; failing to delete continues billing.
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --quiet
```

### 2b. Verify the delete actually happened

```bash
gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

Acceptable outputs:

- Empty.
- A list that does not contain `$TPU_NAME`.

Not acceptable:

- `STATE: DELETING` — wait, then list again.
- `STATE: READY` after a delete — the delete didn't take. Re-issue the
  delete and check IAM / locks.

Optional belt-and-braces script — fail loud if anything remains:

```bash
remaining=$(gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --filter="name=$TPU_NAME" \
  --format="value(name)")
if [ -n "$remaining" ]; then
  echo "ERROR: TPU VM $TPU_NAME still exists. Cost is still accruing." >&2
  exit 1
fi
echo "OK: TPU VM is gone."
```

### 2c. Check for orphans across the whole project

Even if `$TPU_NAME` is gone, an older session may have left things behind.
Run these in every zone you have used:

```bash
gcloud compute tpus tpu-vm list --project="$PROJECT_ID"
gcloud compute tpus queued-resources list --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute addresses list --project="$PROJECT_ID"
gcloud compute disks list --project="$PROJECT_ID"
gsutil ls -p "$PROJECT_ID"
```

If anything in there is unfamiliar, investigate before deleting — you don't
want to nuke a teammate's resource — but assume it is costing you per hour
until you prove otherwise.

---

## 3. The five common ways to leak money

This list is calibrated to the kind of mistakes that recur even with experienced
operators. Each one has a free fix.

### 3.1 Forgot to delete the VM

Symptom: you closed the terminal at 6pm. At 9am the bill is +$200.

Cause: `create` succeeded, work finished, but no `delete` ran.

Fixes:

- Put `tpu-vm delete` in a shell `trap` so it runs on `EXIT` or `ERR`.
- Add a scheduled job (cron / Cloud Scheduler) that lists running TPUs in
  your project and emails you the inventory daily.
- Use `--max-run-duration` on creating the VM where supported, so the
  resource self-deletes after a fixed window.

Example trap pattern (paste into your provisioning script):

```bash
cleanup_tpu() {
  # PAID: free of charge to call delete; the asset was paid.
  gcloud compute tpus tpu-vm delete "$TPU_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" --quiet || true
}
trap cleanup_tpu EXIT
```

### 3.2 Orphaned static external IP

Symptom: tiny but persistent line item ("static address — unused").

Cause: you reserved a static IP for the VM, deleted the VM, but did not
release the IP. Unused static IPs are billed at a small per-hour rate.

Fixes:

```bash
gcloud compute addresses list --project="$PROJECT_ID"

# PAID: removes the address; the cost was the unused-address rate.
gcloud compute addresses delete <ADDRESS_NAME> \
  --project="$PROJECT_ID" \
  --region="$REGION"
```

### 3.3 Persistent disks left behind

Symptom: small "PD-Balanced" or "PD-SSD" line item every day.

Cause: a TPU runtime image, a Filestore mount, or an unrelated VM you spun
up while debugging leaves a disk behind. The TPU VM itself usually does
*not* leave a separate disk, but the auxiliary debugging VMs do.

Fixes:

```bash
gcloud compute disks list --project="$PROJECT_ID"

# PAID: tiny ongoing cost; one-shot delete.
gcloud compute disks delete <DISK_NAME> \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

### 3.4 GCS bucket with surprise egress

Symptom: storage line item is small but the egress line item is large.

Cause: you set up a bucket for the TPU VM to stage training data or
checkpoints. The bucket itself is cheap. *Egress* — pulling that bucket's
contents to your laptop or to another region — is the expensive part.

Fixes:

- Prefer `scp` of the small JSONL/CSV/MD artefacts (a few MB) to your
  laptop. Do not `gsutil cp` the whole bucket.
- Pull large data from a VM **in the same region** as the bucket.
- When done, either delete the bucket or move it to a Nearline / Coldline
  class — and confirm there are no lifecycle rules creating duplicates.

```bash
gsutil ls -p "$PROJECT_ID"

# Inspect what's in there before deleting.
gsutil du -sh gs://<BUCKET_NAME>

# PAID: deleting halts further storage cost.
gsutil rm -r gs://<BUCKET_NAME>
```

### 3.5 Unused TPU reservation / queued resource

Symptom: you can't create a new VM ("quota / reservation in use") *and* you
see a recurring TPU line item.

Cause: a reservation, queued resource, or auto-scheduled allocation is
still active.

Fixes:

```bash
gcloud compute tpus queued-resources list \
  --project="$PROJECT_ID" \
  --zone="$ZONE"

# PAID: cancellation releases the reservation hold.
gcloud compute tpus queued-resources delete <QUEUED_RESOURCE_NAME> \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --quiet
```

For long-term reservations bought through sales/console, the right path is
the Cloud Console — not gcloud.

---

## 4. Set up billing alerts (free, do this once)

A budget alert is the only thing that protects you from a *script* that
leaks money — your manual checklist won't help if the loop has a `create`
in it.

In the Cloud Console:

1. Billing → Budgets & alerts → Create budget.
2. Scope the budget to your project (not "all billing").
3. Set an absolute amount you would notice if it were spent in a day —
   most learning labs are fine with a small daily figure.
4. Add alert thresholds at 50%, 90%, and 100%. Set the recipients to your
   real email, not a noisy inbox.
5. Optionally connect a Pub/Sub topic so a budget breach can trigger an
   automatic action (e.g. a Cloud Function that runs `tpu-vm delete --quiet`
   on every TPU in your project).

CLI equivalent (the API is `billingbudgets`):

```bash
gcloud billing budgets list \
  --billing-account="<BILLING_ACCOUNT_ID>"
```

Tip: budgets are denominated in your billing currency. Don't hardcode a
USD-only mental model.

---

## 5. End-of-session ritual

The most reliable thing is to make cleanup a ritual that doesn't depend on
remembering. Pin this sequence somewhere visible:

```bash
# 1. Delete the VM.
# PAID:
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" --quiet

# 2. Confirm.
gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" --zone="$ZONE"

# 3. Sweep the project.
gcloud compute tpus tpu-vm list --project="$PROJECT_ID"
gcloud compute addresses list --project="$PROJECT_ID"
gcloud compute disks list --project="$PROJECT_ID"
gsutil ls -p "$PROJECT_ID"

# 4. Log it. Append a single line to docs/progress_log.md.
```

---

## 6. The "is anything running right now?" one-liner

Put this in your shell's `PROMPT_COMMAND`, or run it before opening a new
session:

```bash
gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" \
  --format="table(name, zone, state, acceleratorType)"
```

A blank result is the goal. Treat any output as "billing is happening
*right now*".

---

## 7. Disaster scenarios — what to do if cost has already spiked

If you discover an unexpected bill mid-month, do these things in order:

1. **Stop the bleeding.** Run the cleanup checklist in section 2.
2. **List every TPU in every zone you might have used.** A surprise charge
   often comes from a zone you forgot about.
3. **Check Budgets & alerts.** If the threshold did not fire, the budget
   isn't scoped right. Fix the scope.
4. **Check IAM.** If someone else has `tpu.admin` on the project they may
   have created a VM. The audit log shows who.
5. **File a billing inquiry.** Google may consider one-time credits for
   first-time accidents. Do not rely on this.

---

## 8. Cross-references and code pointers

- `10_cloud_tpu_setup_playbook.md` — the matching create/SSH/run flow.
- `src/common/cost.py` — the lab's cost estimator. Reads
  `--hourly-usd-per-chip` and reports total run USD; never hardcodes a rate.
- `src/profiling/bottleneck_report.py` — has a `cost` severity that flags
  high run cost in the report.

---

## 9. Exercises / TODOs

1. Write a `cleanup_all.sh` that calls the section-2c sweep across the set
   of zones you use, fails loud on any non-empty result, and is safe to run
   multiple times.
2. Configure a budget alert at $1 / day on a throwaway project and confirm
   you receive the email by running a deliberate idle VM for a few minutes.
   Delete the VM and confirm the alert resets.
3. Audit the last 30 days of billing in the Cloud Console. Categorise every
   line item: was it (a) intentional lab work, (b) idle TPU,
   (c) idle non-TPU resource, (d) egress. Numbers in each bucket tell you
   where your next process change should go.
4. Add a `trap` to your provisioning script so a `Ctrl-C` during setup
   does not leave a half-created VM running.
5. Read https://cloud.google.com/tpu/pricing and confirm in writing
   (a note in `progress_log.md`) the per-chip-hour rate for the SKU you use
   today and the date you checked it. Pricing changes; the doc you wrote
   yesterday is still right but the number may not be.
