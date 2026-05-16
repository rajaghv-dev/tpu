# Documentation vs Code Consistency Audit

Generated: 2026-05-16 — Phase 1.

## Summary

17 inconsistencies found across docs and code. 3 Critical, 5 High, 6 Medium, 3 Low.

---

## Audit Table

| # | Area | Documentation claim | Repo reality | Evidence | Severity | Required fix |
|---|---|---|---|---|---|---|
| 1 | Test count | README.md line ~200: "224 tests (inference + training + probe tests)" | 272 total collected: 255 passed + 17 failed + 3 skipped (baseline run 2026-05-16) | `python3 -m pytest tests/ -q --tb=no` output | Critical | Update README test count to 272 (or 255 passing + 17 failing with note) |
| 2 | OTel minimal install | README.md: "pip install pytest pyyaml numpy" is sufficient to run tests | 17 tests fail due to `ModuleNotFoundError: No module named 'opentelemetry'` — tests/test_otel.py (10 failures) and tests/test_runner.py (7 failures) | `python3 -m pytest tests/ -q` baseline | Critical | Either add opentelemetry to minimal install, OR fix observe/otel.py to not eagerly import opentelemetry when disabled |
| 3 | Lock file completeness | CI uses `requirements.stage1.lock.txt` which should pass all tests | Lock file (frozen 2026-04-29) is missing opentelemetry packages entirely; tests that import opentelemetry fail in any env using only the lock file | grep opentelemetry requirements.stage1.lock.txt → 0 results | Critical | Regenerate lock file with full `pip freeze` after installing requirements.txt; or confirm CI skips OTel tests |
| 4 | Probe file name | MEMORY.md: `observe/input_fingerprint_probe.py` | Actual file: `observe/input_fingerprint.py` (class InputFingerprintProbe is inside it) | `ls observe/input_fingerprint*` → only input_fingerprint.py | High | Update MEMORY.md reference from `input_fingerprint_probe.py` to `input_fingerprint.py` |
| 5 | 4 undocumented probes | README, observe/README.md, MEMORY.md all document 7 inference probes + 3 training probes | 4 additional complete probe files exist: `observe/determinism_probe.py` (DeterminismProbe), `observe/device_info_probe.py` (DeviceInfoProbe), `observe/power_thermal_probe.py` (PowerThermalProbe), `observe/xla_compile_probe.py` (XlaCompileProbe) | `grep -rn "^class.*Probe" observe/` | High | Add all 4 probes to observe/README.md, MEMORY.md, and main README.md probe table |
| 6 | Makefile python command | Makefile uses `python -m pytest` | `python` binary not found in WSL2 environment — only `python3`. `make test` would fail | `python --version` → command not found; `python3 --version` → OK | High | Change Makefile to use `python3 -m pytest` or add `PYTHON ?= python3` variable |
| 7 | Probe hooks documented | observe/README.md documents 6 lifecycle hooks: before_run, after_run, before_phase, after_phase, on_error, write_log | probe.py actually has 10 hooks including: before_step, after_step, record_metric (training hooks) plus fanout_before_step, fanout_after_step, fanout_record_metric | `grep -n "^def fanout" observe/probe.py` | High | Update observe/README.md to document training-specific hooks (before_step, after_step, record_metric) |
| 8 | SESSION.md last commit | SESSION.md: `Last commit: [update at commit time]` | Stale placeholder — was never updated | cat SESSION.md | High | Update SESSION.md with current last commit (f725bf8) and date 2026-05-10 |
| 9 | Stage 2 planned probes | README "Stage 2 will add: observe/system_monitor.py" | Files observe/system_monitor.py, observe/flops_counter.py, observe/numerics.py, observe/tracer.py, observe/hlo_analyser.py all absent | `ls observe/system_monitor.py` → not found | Medium | These are correctly marked as Stage 2/3 — add explicit "NOT YET IMPLEMENTED" label in docs where referenced |
| 10 | Training probe hooks | Main README probe table: lists 6 hooks total for Probe ABC | Training runner calls before_step/after_step/record_metric not shown in main README probe section | train/runner.py uses fanout_before_step, fanout_after_step, fanout_record_metric | Medium | Update main README "Probes & Observability" section to mention training hooks |
| 11 | LICENSE file | README footer: "License: MIT" | No LICENSE file exists at repo root. cloud_tpu_lab/pyproject.toml says MIT but no LICENSE file anywhere in root | `ls LICENSE` → not found | Medium | Create root LICENSE file with MIT text |
| 12 | CONTRIBUTING.md | No doc currently references a CONTRIBUTING.md | No CONTRIBUTING.md exists | `ls CONTRIBUTING.md` → not found | Medium | Create CONTRIBUTING.md with contribution guide |
| 13 | .env.example | No .env.example exists | Required environment variables (GCP_PROJECT, HF_TOKEN, TPU_BENCH_OTEL, TPU_BENCH_OTEL_ENDPOINT, WHEEL_CACHE_URL, etc.) are undocumented for new users | `ls .env.example` → not found | Medium | Create .env.example documenting all required and optional env vars |
| 14 | AGENTS.md | Not present | No AGENTS.md guide for coding agents | `ls AGENTS.md` → not found | Medium | Create AGENTS.md |
| 15 | GitHub Actions permissions | No explicit permissions block in workflow | smoke_on_push.yml has no top-level permissions declaration | cat .github/workflows/smoke_on_push.yml | Low | Add `permissions: contents: read` to workflow |
| 16 | 30_deploy_repo.sh .claude exclusion | No claim about this but security expectation | .claude/ directory is NOT excluded from the tar command in 30_deploy_repo.sh; .tpu/ also not excluded | cat scripts/30_deploy_repo.sh | Low | Add --exclude='.claude' --exclude='.tpu' to tar command |
| 17 | cloud_tpu_lab Makefile python | cloud_tpu_lab/Makefile uses `$(PYTHON) ?= python3` correctly | But cloud_tpu_lab/README.md says `python3 examples/...` when the Makefile target might use `python` | cloud_tpu_lab/Makefile | Low | Minor inconsistency — no immediate fix needed, both work |

---

## Severity Summary

| Severity | Count | Items |
|---|---|---|
| Critical | 3 | Test count, OTel minimal install, lock file completeness |
| High | 5 | input_fingerprint_probe.py name, 4 undocumented probes, Makefile python vs python3, probe hooks, SESSION.md placeholder |
| Medium | 6 | Stage 2 probe labels, training hooks in README, LICENSE, CONTRIBUTING.md, .env.example, AGENTS.md |
| Low | 3 | CI permissions, .claude exclusion, cloud_tpu_lab Makefile python |

---

## Fix Priority

1. Fix observe/otel.py to not eagerly import opentelemetry when disabled (fixes 17 test failures)
2. Update README test count
3. Update MEMORY.md input_fingerprint file name
4. Add 4 undocumented probes to docs
5. Fix Makefile python → python3
6. Create LICENSE, CONTRIBUTING.md, .env.example, AGENTS.md
7. Add .claude exclusion to 30_deploy_repo.sh
8. Regenerate requirements.stage1.lock.txt
