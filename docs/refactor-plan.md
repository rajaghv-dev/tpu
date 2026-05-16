# Refactor Plan

Generated: 2026-05-16 — based on Phase 1 (doc consistency) + Phase 2 (code quality) audits.

## Goals

1. Fix 17 failing tests (OTel eager import in observe/otel.py)
2. Correct documentation inconsistencies (test count, probe names, undocumented probes)
3. Create missing files (LICENSE, CONTRIBUTING.md, .env.example, AGENTS.md)
4. Fix Makefile python → python3
5. Fix security gap (.claude/ not excluded from deploy tarball)
6. Improve docs to match current code state

## Non-goals

- Large rewrites of runner.py or harness.py
- Extracting shared harness base (deferred — high effort, low urgency)
- Stage 2+ feature additions
- Changing public CLI interfaces
- New framework support (PyTorch, TensorRT)

## Current problems (evidence summary)

| Problem | Evidence | Risk |
|---|---|---|
| 17 failing tests | `pytest tests/ -q` baseline | CI drift; contributor confusion |
| MEMORY.md wrong probe filename | `ls observe/input_fingerprint*` | Documentation rot |
| 4 probes undocumented | `grep -rn "^class.*Probe" observe/` | Discoverability |
| README test count stale (224 vs 272) | pytest baseline | Misleading |
| Makefile `python` not `python3` | `python --version` fails in WSL2 | Local dev breakage |
| No LICENSE | `ls LICENSE` | Legal clarity |
| No CONTRIBUTING.md | `ls CONTRIBUTING.md` | Contributor barrier |
| No .env.example | `ls .env.example` | Setup friction |
| No AGENTS.md | `ls AGENTS.md` | Agent guidance gap |
| .claude/ copied to TPU VM | scripts/30_deploy_repo.sh tar flags | Privacy leak risk |
| CI missing explicit permissions | smoke_on_push.yml | Security best practice |
| SESSION.md last commit placeholder | cat SESSION.md | Stale state |

## Refactor strategy

Work in small, reviewable patches. Fix correctness first (tests), then docs, then missing files, then security.

## Files to change

| File | Change | Risk |
|---|---|---|
| `observe/otel.py` | Guard opentelemetry imports with `if _state["enabled"]` check | Low — adds guard, no behavior change when disabled |
| `MEMORY.md` | Fix `input_fingerprint_probe.py` → `input_fingerprint.py` | Trivial |
| `README.md` | Update test count 224 → 272 | Trivial |
| `Makefile` | `python` → `python3` | Low |
| `scripts/30_deploy_repo.sh` | Add `--exclude='.claude'` to tar | Low |
| `.github/workflows/smoke_on_push.yml` | Add `permissions: contents: read` | Low |
| `observe/README.md` | Add 4 new probes + training hooks | Docs only |
| `SESSION.md` | Update stale placeholder | Docs only |

## Files not to touch

- `benchmarks/runner.py` — no behavioral changes
- `benchmarks/harness.py` — no behavioral changes
- `train/runner.py` — no behavioral changes
- `models/registry.yaml` — no changes
- `results/runs.jsonl` — append-only, never edit

## New files to create

| File | Purpose |
|---|---|
| `LICENSE` | MIT license text |
| `CONTRIBUTING.md` | Contributor guide |
| `.env.example` | Required env vars documentation |
| `AGENTS.md` | Coding agent guide |
| `docs/architecture.md` | This refactor creates it |
| `docs/interfaces.md` | Interface catalog |
| `docs/setup-validation.md` | Setup steps validation |
| `docs/testing.md` | Test strategy |
| `docs/github-readiness.md` | CI/CD readiness |
| `docs/security-audit.md` | Security findings |
| `docs/examples.md` | Examples catalog |
| `docs/observability.md` | Observability guide |
| `docs/agent-handoff.md` | Agent handoff notes |
| `reports/final-validation-report.md` | Final validation |

## Backward compatibility concerns

- `observe/otel.py` fix: no behavior change when OTel is enabled; adds guard only for disabled path
- Makefile fix: `python3` works on all target systems; `python` does not
- All other changes are docs-only or new files

## Test strategy

After fixing observe/otel.py:
- Run `python3 -m pytest tests/test_otel.py -q` → expect 0 failures
- Run `python3 -m pytest tests/test_runner.py -q` → expect 0 failures
- Run `python3 -m pytest tests/ -q` → expect 0 failures (255 → 272 passing)
- Run `make dry-run` → expect no errors

## Rollback plan

Every change is tracked in git. `git revert` any commit to undo. No database migrations, no external state.

## Phase plan

### Phase A — Test fix (highest priority)
Fix observe/otel.py eager import. Target: 17 failures → 0.

### Phase B — Documentation corrections
- Fix MEMORY.md probe filename
- Update README test count
- Add 4 undocumented probes to observe/README.md
- Update SESSION.md placeholder
- Add training hooks to probe docs

### Phase C — Missing files
- Create LICENSE (MIT)
- Create CONTRIBUTING.md
- Create .env.example
- Create AGENTS.md

### Phase D — Tooling fixes
- Fix Makefile python → python3
- Add CI permissions block
- Fix 30_deploy_repo.sh .claude exclusion

### Phase E — Create audit/guide docs
- docs/architecture.md (created in this refactor)
- docs/interfaces.md
- docs/setup-validation.md
- docs/testing.md
- docs/github-readiness.md
- docs/security-audit.md
- docs/examples.md
- docs/observability.md
- AGENTS.md
- docs/agent-handoff.md

### Phase F — Final validation
Run full test suite and verify all docs are consistent.
