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

### 2.8 Corporate networks block outbound port 22 → use Cloud Shell, not IAP
The first SSH from `bhar-10476-AIT2` timed out with:

```
ssh: connect to host 34.83.245.217 port 22: Operation timed out
```

`default-allow-ssh` was in place on the GCP side (0.0.0.0/0 → tcp:22),
so the block was outbound from the client network.

**IAP tunneling does NOT work for TPU VMs** — only for GCE instances:

```
$ gcloud compute tpus tpu-vm ssh ... --tunnel-through-iap
ERROR: unrecognized arguments: --tunnel-through-iap

$ gcloud compute start-iap-tunnel ctl-tpu-vm 22 --zone=us-west1-c ...
ERROR: Could not fetch resource: The resource
'projects/.../zones/.../instances/ctl-tpu-vm' was not found
```

TPU VMs live at `locations/.../nodes/...`, not `zones/.../instances/...`,
and the IAP tunnel command targets GCE specifically. There is no
documented IAP path for TPU VMs as of 2026-05.

**The actual fix**: use **Cloud Shell** (https://console.cloud.google.com →
Activate Cloud Shell). It runs inside Google's network so SSH works
natively, has gcloud preauthed, and is free.

Don't bother adding `--tunnel-through-iap` to TPU VM SSH/SCP — pip
will accept the flag silently in some sub-tools but `tpu-vm ssh` rejects
it.

### 2.9 Auto-switched zone in `create_tpu_vm.sh` doesn't propagate
When `create_tpu_vm.sh` auto-switches zones, the discovered value lives
only in that script's memory. Subsequent scripts (`install_jax_tpu.sh`,
`run_real_demo.sh`, …) re-source `_env.sh` and still see the old default.

**Workaround** (current): user must `export ZONE=...` between scripts,
or re-discover with `gcloud compute tpus tpu-vm list --zone=$Z` per zone.
**Better fix** (not yet done): each subsequent script should query
`gcloud compute tpus tpu-vm list` to find where the VM actually is.

### 2.10 Cloud Shell + local repo can drift, breaking `git pull`
After committing fixes locally and asking Cloud Shell to `git pull`, the
pull was rejected:

```
error: Your local changes to the following files would be overwritten by merge:
        cloud_tpu_lab/gcp/install_jax_tpu.sh
Please commit your changes or stash them before you merge.
```

Cloud Shell had earlier installed the file via an `scp` or web-edit that
changed its mode-bits or content. Fix is one-line and safe:

```bash
git checkout cloud_tpu_lab/gcp/install_jax_tpu.sh    # discard local
git pull
```

(General rule: if you're not editing in Cloud Shell intentionally,
`git checkout <file>` to discard is the right move.)

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

### 5.5 JAX / libtpu install on `tpu-ubuntu2204-base` is a minefield
The base image ships an old `libtpu.so` (e.g. `libtpu_nightly_20241002`)
that pip cannot replace without root. Combined with whatever JAX the
user-site already had, several failure modes hit in this order:

| Symptom                                                       | Cause                                                                                       | Fix                                                                                                  |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `error: unknown attribute code: 22 ... StableHLO_v1.9.6 ... v1.7.5` | Image libtpu (Oct 2024 → StableHLO 1.7.5) vs newer pip-installed JAX 0.6.2 (emits 1.9.6) | Either downgrade JAX to match OR force-reinstall with a wheel that brings a fresh libtpu             |
| `ModuleNotFoundError: No module named 'jax'`                  | Earlier install ran `pip uninstall jax jaxlib libtpu` but the subsequent install failed     | Re-run install; ensure the install command's exit status is checked                                  |
| `jaxlib or libtpu is not installed. Falling back to cpu`      | Plain `jax==0.4.34 jaxlib==0.4.34` (no `[tpu]` extra) doesn't pull the libtpu Python wrapper | Use `jax[tpu]` with `-f https://storage.googleapis.com/jax-releases/libtpu_releases.html`            |
| `backend: cpu` despite the install "succeeding"               | Silent CPU fallback after a broken install                                                  | Add `python -c "import jax; assert jax.default_backend()=='tpu'"` as the FINAL step of any installer |

**Canonical install that worked** (current `install_jax_tpu.sh`):

```bash
pip uninstall -y jax jaxlib libtpu libtpu-nightly
pip install --upgrade --force-reinstall "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c "import jax; assert jax.default_backend()=='tpu'" || exit 1
```

The legacy `libtpu_releases.html` URL is **still authoritative** even in
2026; do not assume pip-only resolution covers libtpu wheels.

### 5.6 Verify `backend == 'tpu'` at install time, not at run time
Without an assertion, a botched install lets the user spin up a paid VM,
SSH in, and run a "TPU" workload on CPU — wasting TPU-hours producing
data that doesn't reflect TPU behaviour at all. The single line costs
nothing and saves a frustrating debug loop.

---

## 5A. Observability pipeline gotchas

### 5A.1 Runner → exporter event-schema mismatch silently produces empty Grafana
The Python metrics exporter recognises very specific event names + field
shapes:

| Event             | Required fields (flat at top of JSON line)                 |
|-------------------|------------------------------------------------------------|
| `xla.compile`     | `compile_time_s`, `cache_hit`                              |
| `runtime.step`    | `step_time_s`, `device_execution_time_s`, `samples_per_second` |
| `hbm.snapshot`    | `hbm_used_bytes`, `hbm_capacity_bytes`, `hbm_utilization_ratio` |

The initial runner emitted `event="train.step"` and put HBM under a
nested `metrics: { used_bytes: ... }`. Result: exporter reads every line,
matches no handler, exposes only `# HELP` / `# TYPE` metadata, no values.
Prometheus scrapes happily but the gauges stay empty. Grafana shows
"No data" with no error anywhere.

**Lesson**: when adding observability, the producer and consumer must
share an explicit schema. Document the schema next to METRIC_NAMES and
fail-fast in tests when an event is emitted that no handler recognises.

### 5A.2 Identity labels need to be injected into EVERY event
The exporter pulls `framework`, `tpu_version`, `workload_name`, and
`run_mode` from each event's top-level fields, defaulting to
`framework="cpu_sim", tpu_version="cpu_sim", workload_name="unknown",
run_mode="local_cpu"` if absent. Dashboards have template variables
defaulting to `framework=jax, tpu_version=v5e` — so even when metric
*values* are correct, the template filter excludes everything.

**Fix**: in the runner, set up an `_identity` dict once and
`fields.setdefault(k, v)` for every emit().

### 5A.3 Promtail's positions cache survives container restart
Promtail stores its file offsets in `/tmp/positions.yaml` inside the
container. After it tails a file to EOF, the position is "stuck" at
file size; restarting the container re-reads the SAME positions file
and seeks back to EOF. The fix:

```bash
docker exec cloud_tpu_lab_promtail rm -f /tmp/positions.yaml
docker restart cloud_tpu_lab_promtail
```

Restart-without-rm does nothing. The `/tmp` inside the container is
its own tmpfs and survives `docker restart` (only `docker rm` clears it).

### 5A.4 Loki's INSTANT query has a narrow default window
`curl 'http://localhost:3100/loki/api/v1/query?query={app=...}'` returns
streams=0 for events more than a few minutes old. Use `query_range`
with explicit `start` / `end`, or in Grafana set the time picker to
"Last 24 hours" before querying.

This trips up CI-style validation: an "is data flowing?" instant query
falsely reports no, even when a 30-minute-old event is sitting in Loki.

### 5A.5 Promtail watches a fixed glob — `from_vm/<RUN_TAG>/` isn't covered
The compose bind-mount maps `../artifacts/logs` → `/var/log/cloud_tpu_lab`,
and Promtail's `__path__` is `/var/log/cloud_tpu_lab/*.jsonl` (not
recursive). Files dropped into `artifacts/from_vm/<RUN_TAG>/` are
invisible. Two options:
1. Copy / symlink the JSONL up to `artifacts/logs/` (current pattern;
   `observability/load_run.sh` does this).
2. Switch the mount to `../artifacts:/var/log/cloud_tpu_lab` and the
   glob to `**/*.jsonl` — durable but loses isolation between runs.

### 5A.6 Three independent data planes must all be alive
Grafana needing real data on the laptop means **three** things must run
that aren't `docker compose up`:

| Source                                         | Lives where                                  | How it starts                              |
|------------------------------------------------|----------------------------------------------|--------------------------------------------|
| **Workload metrics** (JSONL → Prometheus)      | Python exporter on host @ :9100              | `python3 observability/exporters/cloud_tpu_metrics_exporter.py` |
| **Workload logs** (JSONL → Loki)               | Promtail container in compose                | comes up with the stack                    |
| **GCP infra metrics** (Cloud Monitoring → Prom) | stackdriver-exporter container in compose    | comes up with the stack                    |

The host exporter is the most-forgotten one. `observability/load_run.sh`
now handles its lifecycle.

### 5A.7 One-command loader: `observability/load_run.sh`
Wraps the entire "I have a collected run, make Grafana show it" chain:
pick newest run dir → normalise JSONL schema → copy to where Promtail
and the exporter watch → restart the host exporter → clear Promtail's
position cache + restart → verify Prometheus/Loki have the data →
print a deep-link to Grafana with template vars pre-filled.

This script is the answer every time the user asks "why don't I see
anything in Grafana?" after a real run.

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
- **Schema-conformance test for the runner** (see 5A.1) — a unit test
  that emits one of each event type and asserts the exporter handler
  catches it would have prevented the train.step / runtime.step bug
- **Promtail recursive glob** (see 5A.5) so future `from_vm/<RUN_TAG>/`
  artifacts are picked up without a copy step
- **`xprof/` directory often half-written** on short runs — the
  `jax.profiler.stop_trace` flush sometimes only produces `plugins/profile/
  <ts>/` with the trace `.pb` missing. Need either a min-run-length
  guard or explicit `sync` before SCP

## 8. Pipeline in one mental model

```
              ┌─────────────────────────────┐
   TPU VM ───►│ examples/run_jax_real_tpu.py│
              │   ┌───────────────────────┐ │
              │   │ jax.profiler.trace    │ │──► xprof/
              │   │ XLA_FLAGS=--xla_dump  │ │──► hlo/
              │   │ jax.devices()[0]      │ │
              │   │   .memory_stats()     │ │──► run.jsonl  (OCT events)
              │   │ MetricStream          │ │──► run.csv    (Prometheus metrics)
              │   │ render_run_report     │ │──► run.md     (human report)
              │   └───────────────────────┘ │
              └─────────────────────────────┘
                            │
                            │ collect_artifacts.sh / cloudshell download
                            ▼
              ┌─────────────────────────────┐
   LAPTOP ───►│ artifacts/from_vm/<RUN_TAG> │
              └─────────────────────────────┘
                            │
                            │ observability/load_run.sh
                            ▼
              ┌─────────────────────────────────────┐
              │ artifacts/logs/    (Promtail watches)│──► Loki
              │ artifacts/metrics/ (exporter reads) │──► :9100 ──► Prometheus
              └─────────────────────────────────────┘
                                                            │
              ┌─────────────────────────────────────┐       │
              │ GCP Cloud Monitoring API            │──► stackdriver-exporter
              └─────────────────────────────────────┘   (in docker)  │
                                                                     ▼
                                                                  Grafana
                                                                  :3000
```

Three data planes, two halves (workload vs infra). When something's
empty in Grafana, walk the diagram from the source forward — the gap
is always at one specific arrow.
