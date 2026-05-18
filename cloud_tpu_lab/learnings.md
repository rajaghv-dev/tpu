# learnings.md

Things that surprised, broke, or required a fix on the way from "CPU
simulation" to "real Cloud TPU + Grafana on `nellaiappar-001`". Captured
here so the next person doesn't pay the same tax.

---

## 1. Architectural pivots

### 1.1 Simulation never reflects real hardware closely enough to be worth maintaining
The original lab had `src/xla_sim`, `src/pjrt_sim`, `src/sharding`,
`src/memory`, `src/input_pipeline` — first-order analytical models with
fudge factors (`flops_efficiency=0.5`, `hbm_efficiency=0.7`). They
captured the *shape* of bottlenecks but were off by 2–10× in absolute
numbers. There was no calibration loop against real XProf traces.

**Decision:** delete the simulator entirely. Real TPU is the only path.
Anything that exists in the repo must reflect or come from real hardware.

### 1.2 Two-tier metric model
There are two distinct truths about a TPU run, and Grafana needs both:

| Tier               | Source                              | Cadence  | Authoritative for          |
|--------------------|-------------------------------------|----------|----------------------------|
| Workload-level     | JAX → JSONL → Python exporter @ 9100| per-step | Step time, throughput      |
| Infrastructure     | GCP Cloud Monitoring → stackdriver-exporter @ 9255 | 60s | Duty cycle, HBM, network   |

When the two HBM panels disagree, trust the infra tier (it comes from
libtpu firmware, not from JAX bookkeeping).

### 1.3 Keep one canonical workload + make new ones additive
`examples/run_jax_real_tpu.py` ships one workload (matmul) and a
`_BUILDERS` dict. Adding MLP / transformer is a one-line dict-add plus
a builder function. Instrumentation (HLO dumps, XProf, HBM snapshots,
OCT JSONL) is wired once and serves every workload.

---

## 2. GCP / TPU setup gotchas

### 2.1 `us-central2-b` is v4-only — don't default to it for v5e
The `_env.sh` originally defaulted to `us-central2-b`. A
`v5litepod-1` create there returns:

```
PERMISSION_DENIED: Permission denied on 'locations/us-central2-b'
```

The wording is misleading — it's not a permission issue, that location
just doesn't offer v5e. Zone-by-TPU-generation reference:

```
v4   →  us-central2-b (allowlisted projects only)
v5e  →  us-west1-c, us-west4-a, us-east1-c, us-east5-a, europe-west4-b
v5p  →  us-east5-a, europe-west4-b
v6e  →  us-east5-a/b, europe-west4-a, asia-northeast1-b
```

### 2.2 Don't hardcode zone lists — probe them dynamically
Even the lists above go stale. The right pattern:

```bash
for z in $(gcloud compute tpus locations list --project=$PROJECT --format='value(locationId)'); do
    gcloud compute tpus accelerator-types list --zone=$z --project=$PROJECT \
        --filter="type=${ACCEL}" --format="value(type)" 2>/dev/null | \
        grep -qx "$ACCEL" && echo "$z"
done
```

Probe in parallel (`xargs -P 8` or shell `&` with cap) — serial is ~80s,
parallel is ~10s. Implemented in `gcp/find_tpu_zone.sh`.

### 2.3 `gcloud compute tpus accelerator-types list` requires `--zone`
There's no global "list all accelerator types in this project" query.
Must enumerate locations first, then probe each. Same constraint applies
to `gcloud compute tpus tpu-vm list`.

### 2.4 `--filter "type:foo"` is deprecated — use `type=foo`
gcloud emits a `WARNING: --filter : operator evaluation is changing for
consistency across Google APIs` on `:` filters. The new operator is
`=` (exact match). The colon form silently broke our probe later.

### 2.5 The `default` subnet doesn't exist in every region
Newer GCP projects don't auto-create `default` subnets in every region.
A `tpu-vm create --subnetwork=default` in (say) `us-west1` fails with:

```
INVALID_ARGUMENT: The field "Subnetwork" cannot be "default":
requested resource not found
```

**Fix pattern** (implemented in `create_tpu_vm.sh`): derive `REGION`
from `ZONE`, verify subnet existence, otherwise list subnets of
`$NETWORK` in that region and pick the first. If none, print the two
fix commands and exit cleanly without spend.

### 2.6 IAP API is opt-in
`iap.googleapis.com` is not enabled by default. Without it,
`--tunnel-through-iap` fails. Enable explicitly:

```bash
gcloud services enable iap.googleapis.com --project=$PROJECT
```

### 2.7 IAP needs its own firewall rule
Even with IAP enabled, the TPU/GCE firewall must allow Google's IAP
source range (`35.235.240.0/20`) to reach port 22:

```bash
gcloud compute firewall-rules create allow-ssh-from-iap \
    --network=default --direction=INGRESS --action=ALLOW \
    --rules=tcp:22 --source-ranges=35.235.240.0/20 \
    --project=$PROJECT
```

### 2.8 Corporate networks block outbound port 22
The first SSH from `bhar-10476-AIT2` timed out with:

```
ssh: connect to host 34.83.245.217 port 22: Operation timed out
```

`default-allow-ssh` was in place on the GCP side (0.0.0.0/0 → tcp:22),
so the block was outbound from the client network. The IAP fix routes
SSH over 443 (which corporate networks invariably allow).

**Always-on pattern**: every `gcloud compute tpus tpu-vm ssh|scp` call
should include `--tunnel-through-iap`. Patched across all 10 scripts.

### 2.9 Auto-switched zone in `create_tpu_vm.sh` doesn't propagate
When `create_tpu_vm.sh` auto-switches zones, the discovered value lives
only in that script's memory. Subsequent scripts (`install_jax_tpu.sh`,
`run_real_demo.sh`, …) re-source `_env.sh` and still see the old default.

**Workaround** (current): user must `export ZONE=...` between scripts,
or re-discover with `gcloud compute tpus tpu-vm list --zone=$Z` per zone.
**Better fix** (not yet done): each subsequent script should query
`gcloud compute tpus tpu-vm list` to find where the VM actually is.

---

## 3. Docker / Apple Silicon gotchas

### 3.1 `prometheuscommunity/stackdriver-exporter:v0.16.0` is amd64-only
On Apple Silicon (`darwin/arm64`):

```
Error response from daemon: no matching manifest for linux/arm64/v8
```

**Fix**: pin `platform: linux/amd64` on that one service. Docker on M1+
emulates via Rosetta 2. Fine for I/O-bound exporters; the perf hit is
negligible.

The line is a no-op on amd64 hosts (Docker matches the native manifest).
Detection logic in `run_all.sh` step 1 surfaces whether emulation will
kick in.

### 3.2 `version: "3.9"` in compose file emits a deprecation warning
Compose v2 ignores the `version:` key. Just remove it.

### 3.3 Rosetta install on Apple Silicon is non-obvious
If Rosetta 2 isn't installed, amd64-only containers fail silently or
crash on startup. Install command (idempotent):

```bash
softwareupdate --install-rosetta --agree-to-license
```

`run_all.sh` pre-flight now hints at this when it detects darwin/arm64.

---

## 4. Cloud Monitoring exporter gotchas

### 4.1 Metric name mangling
The stackdriver-exporter exposes Prometheus metric names by mangling the
GCP metric type:

```
stackdriver_<monitored_resource_type>_<metric_type_with_non_word_chars_as_underscores>
```

For TPU the monitored resource type is `tpu_worker`. So:

```
tpu.googleapis.com/tpu/duty_cycle
   → stackdriver_tpu_worker_tpu_googleapis_com_tpu_duty_cycle
tpu.googleapis.com/tpu/memory/usage
   → stackdriver_tpu_worker_tpu_googleapis_com_tpu_memory_usage
```

If a Grafana panel returns "No data", first thing to check is whether
the metric name in the panel matches what `curl localhost:9255/metrics`
actually emits. Resource-type assumption (`tpu_worker`) can differ on
older GKE-TPU integrations (sometimes `gce_instance`).

### 4.2 Cloud Monitoring is rate-limited
~6000 read calls/minute per project. `scrape_interval` must be ≥ 60s,
and `--monitoring.metrics-interval=1m --monitoring.metrics-offset=1m`
on the exporter — otherwise Prometheus reads time slices that haven't
been published yet and panels look empty.

### 4.3 Metric names go stale
Google occasionally renames Cloud Monitoring metric paths. The 7
metrics shipped in `cloud_tpu_gcp_metrics.json` were verified against
the docs as of 2026-05; if a panel goes dark, sanity-check:

```bash
curl -s localhost:9255/metrics | grep tpu | awk '{print $1}' | sort -u
```

…and update both `docker-compose.yml` (`--monitoring.metrics-type-prefixes`)
and the panel `expr` fields.

### 4.4 Service account needs only `roles/monitoring.viewer`
No need for `monitoring.editor` or anything write-flavored. The narrower
the role, the safer the JSON key sitting in `~/.config/gcloud/`.

---

## 5. Python / packaging gotchas

### 5.1 `from cloud_tpu_lab.src...` requires the parent on `sys.path`
The tests use absolute imports rooted at the package name:

```python
from cloud_tpu_lab.src.common.cost import CostInputs
```

Running `pytest tests/` from inside `cloud_tpu_lab/` fails with
`ModuleNotFoundError: No module named 'cloud_tpu_lab'`. The right
invocation is:

```bash
cd /Users/AI-Test/raja/tpu && python3 -m pytest cloud_tpu_lab/tests/
```

A `__init__.py` at `cloud_tpu_lab/` would fix this — currently the
scripts manually `sys.path.insert(0, parent.parent)`.

### 5.2 `jax.devices()[0].memory_stats()` is the real HBM truth
Don't trust workload-level HBM bookkeeping (allocated bytes per buffer).
The authoritative number comes from `memory_stats()` on the device,
which reads libtpu's allocator state. Snapshot at three points:
post-init, post-compile, post-final.

### 5.3 `block_until_ready()` is required for honest step timing
JAX is async by default. `time.perf_counter()` around a JIT call
measures dispatch latency, not device time. Always:

```python
result = step_fn(state)
result.block_until_ready()
dt = time.perf_counter() - t0
```

For pytree returns, block on any leaf with `block_until_ready` attribute.

### 5.4 First step is the compile, not the workload
Time step 0 separately. Treat it as `compile_time_s`, take the median
of steps 1..N for steady-state step time and cost math.

---

## 6. Operational gotchas

### 6.1 Idle TPU VMs still bill
A `READY` v5e-1 burns ~$1.30/hr whether or not anything is running on
it. Every paid script in `gcp/` has a `delete_tpu_vm.sh` reminder in
the success message. The `--auto-delete` flag on `run_all.sh` deletes
on success — useful for CI-style invocations.

### 6.2 `gcloud auth login` ≠ `gcloud auth application-default login`
The first authenticates *gcloud*. The second creates the
`application_default_credentials.json` that SDKs read. The
stackdriver-exporter container reads a **separate** service-account
JSON we mount in — neither of the above. Three different credential
files for three different consumers.

### 6.3 Permissions vs. zone errors look identical
`PERMISSION_DENIED: Permission denied on 'locations/X'` can mean:
- The project genuinely lacks permission on that location, OR
- That location doesn't offer the resource at all (e.g., v5e in us-central2).

Always probe `accelerator-types list` before assuming it's an IAM problem.

---

## 7. Things still worth fixing

- **Auto-discover the VM's actual zone** in subsequent scripts (see 2.9)
- **Calibrate cost math** against real billing line items (currently
  a placeholder $/chip/hr)
- **Verify GCP metric names** against live docs (currently best-effort
  per 2026-05; see 4.3)
- **Multi-arch stackdriver-exporter image** (build from source or wait
  for upstream to publish arm64; currently amd64-only via Rosetta)
