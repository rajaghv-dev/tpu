# GitHub Issue Backlog

Generated: 2026-05-16. Issues identified during Phase 0-14 audit.

---

## Issue 1: Regenerate requirements.stage1.lock.txt

### Problem
`requirements.stage1.lock.txt` is frozen from 2026-04-29 and is missing opentelemetry packages entirely. Any environment that installs only the lock file will have 17 test failures.

### Evidence
`grep opentelemetry requirements.stage1.lock.txt` → 0 results. Test baseline: 17 failures from `ModuleNotFoundError: No module named 'opentelemetry'`.

### Proposed fix
After a full `pip install -r requirements.txt`, run `pip freeze > requirements.stage1.lock.txt`. Commit the updated lock file.

### Files involved
`requirements.stage1.lock.txt`, `requirements.txt`

### Priority
Critical

### Labels
`ci`, `bug`

### Acceptance criteria
- [ ] `pip install -r requirements.stage1.lock.txt && python3 -m pytest tests/ -q` passes with 0 failures (excluding legitimate OTel-enabled-path tests)

---

## Issue 2: benchmarks/harness.py and train/harness.py code duplication

### Problem
Both files are ~85% identical — same SUITES structure, DEVICE_COSTS dict, load_registry(), build_config(), run_suite(), main() pattern. Any change to one must be manually mirrored in the other.

### Evidence
DEVICE_COSTS appears in both files. load_registry() is identical. run_suite() structure is the same.

### Proposed fix
Extract shared base into `benchmarks/base_harness.py`. Both inference and training harnesses inherit/import from it. DEVICE_COSTS moves to a single location (or to scripts/lib/config.sh).

### Files involved
`benchmarks/harness.py`, `train/harness.py`, new `benchmarks/base_harness.py`

### Priority
High

### Labels
`refactor`

### Acceptance criteria
- [ ] DEVICE_COSTS defined in one place
- [ ] load_registry() defined in one place
- [ ] All existing tests still pass

---

## Issue 3: conftest.py tree_util.tree_map mock is incorrect

### Problem
`tests/conftest.py` sets `jax_mod.tree_util.tree_map = lambda fn, tree: fn(tree)`. This is wrong — tree_map applies fn recursively to all leaves, not to the tree as a whole. Incorrect mock may mask real bugs.

### Evidence
`conftest.py` line ~33. tree_map semantics: `tree_map(lambda x: x+1, [1, 2, 3])` should return `[2, 3, 4]` not `fn([1, 2, 3])`.

### Proposed fix
Replace with correct recursive implementation or use a proper tree_map stub.

### Files involved
`tests/conftest.py`

### Priority
Medium

### Labels
`bug`, `testing`

### Acceptance criteria
- [ ] Mock correctly applies fn to all leaves
- [ ] Existing tests still pass

---

## Issue 4: Stage 2 — Add observe/system_monitor.py (GPU eager counters)

### Problem
Gap C5 (hardware utilization) was documented as addressable via CloudMonitoringProbe but still needs system_monitor.py for local GPU/CPU eager counters (GPU SM%, CPU util, memory BW) without GCP auth requirement.

### Evidence
`observe/system_monitor.py` does not exist. Referenced in README as "Stage 2 will add" and in MEMORY.md gaps table.

### Proposed fix
Implement `observe/system_monitor.py` using `pynvml` for GPU and `psutil` for CPU. Register as `SystemMonitorProbe` following the Probe ABC.

### Files involved
`observe/system_monitor.py` (new), `tests/test_app_probes.py`, `observe/README.md`, `MEMORY.md`

### Priority
High (Stage 2 blocker)

### Labels
`feat`, `agent-task`

### Acceptance criteria
- [ ] SystemMonitorProbe captures GPU SM%, MXU%, power_w, temp_c
- [ ] Works with psutil fallback when pynvml unavailable
- [ ] Tests pass

---

## Issue 5: Add lint CI job

### Problem
`make lint` runs flake8 but there is no CI job for it. Lint regressions can merge undetected.

### Evidence
`.github/workflows/smoke_on_push.yml` has no lint job. `Makefile` has a `lint` target.

### Proposed fix
Add a `lint` job to `.github/workflows/smoke_on_push.yml` that runs `python3 -m flake8 benchmarks/ observe/ tests/ train/ --max-line-length 100`.

### Files involved
`.github/workflows/smoke_on_push.yml`

### Priority
Low

### Labels
`ci`, `good-first-issue`

### Acceptance criteria
- [ ] Lint job runs on every push to main
- [ ] Job fails if flake8 finds errors

---

## Issue 6: Add PR and issue templates

### Problem
No `.github/PULL_REQUEST_TEMPLATE.md` or issue templates exist. Contributors have no structured format.

### Evidence
`ls .github/` → only `workflows/` directory.

### Proposed fix
Create:
- `.github/PULL_REQUEST_TEMPLATE.md` with checklist
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`

### Files involved
`.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/`

### Priority
Low

### Labels
`docs`, `good-first-issue`

---

## Issue 7: Document 4 undocumented probes in main README

### Problem
DeterminismProbe, DeviceInfoProbe, PowerThermalProbe, and XlaCompileProbe exist with full implementations but are only partially documented (observe/README.md updated; main README.md still shows 7 probes).

### Evidence
`grep -rn "^class.*Probe" observe/` shows 14 probe classes. Main README.md probe table has 7 entries.

### Proposed fix
Update the "Probes & Observability" section in README.md to add all 4 new probes and the 3 training probes (TrainingMetricsProbe, StepTimingProbe, CheckpointProbe).

### Files involved
`README.md`, (observe/README.md — already fixed)

### Priority
Medium

### Labels
`docs`, `good-first-issue`

---

## Issue 8: cloud_tpu_lab tests fail without pip install -e .

### Problem
Running `cd cloud_tpu_lab && python3 -m pytest tests/ -q` fails with `ModuleNotFoundError: No module named 'cloud_tpu_lab'` because the package is not installed in the active Python environment.

### Evidence
```
ERROR tests/test_cost_estimator.py - ModuleNotFoundError: No module named 'cloud_tpu_lab'
ERROR tests/test_cpu_simulation_smoke.py - ModuleNotFoundError: No module named 'cloud_tpu_lab'
[...7 total collection errors]
```

### Proposed fix
- Document `pip install -e .` in cloud_tpu_lab/README.md quick start
- Add a CI job that runs `cd cloud_tpu_lab && pip install -e . && python3 -m pytest tests/ -q`

### Files involved
`cloud_tpu_lab/README.md`, `.github/workflows/smoke_on_push.yml`

### Priority
Medium

### Labels
`ci`, `docs`, `good-first-issue`

### Acceptance criteria
- [ ] cloud_tpu_lab/README.md documents `pip install -e .` before test run
- [ ] Either CI validates cloud_tpu_lab or it's explicitly documented as manual-only
