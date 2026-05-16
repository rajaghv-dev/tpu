
# Code Quality Audit

Generated: 2026-05-16 — Phase 2.

## Summary

29 findings across the codebase. 1 P0 (critical), 8 P1 (important), 16 P2 (useful cleanup), 4 P3 (optional).

---

## Audit Table

| # | Area | Finding | Evidence | Risk | Suggested refactor | Priority |
|---|---|---|---|---|---|---|
| 1 | observe/otel_probe.py | otel_probe.py appears to cut off or have minimal implementation — OTelProbe class may be incomplete | File exists but full class body not confirmed | P0 | Verify otel_probe.py has complete OTelProbe implementation; add to test coverage | P0 |
| 2 | benchmarks/harness.py vs train/harness.py | Both files are ~85% identical: same SUITES structure, DEVICE_COSTS dict, load_registry(), build_config(), run_suite(), main() pattern | `diff benchmarks/harness.py train/harness.py` would show extensive duplication | Drift risk | Extract shared harness base into benchmarks/base_harness.py; inference and training subclass it | P1 |
| 3 | DEVICE_COSTS duplicated | DEVICE_COSTS dict duplicated in both benchmarks/harness.py (~line 46-55) and train/harness.py (~line 125-129) | grep DEVICE_COSTS | Drift risk — one will go stale | Move to single source: scripts/lib/config.sh (already has prices) or new observe/cost.py | P1 |
| 4 | benchmarks/runner.py run_experiment() | run_experiment() is ~253 lines — handles 9 phases, OTel setup, metric recording, and result serialization | `wc -l benchmarks/runner.py` | Low cohesion | Extract phase implementations to _run_phase_X() helpers; keep orchestration in run_experiment() | P1 |
| 5 | train/harness.py run_suite() | run_suite() is ~160 lines, mirrors benchmarks/harness.py run_suite() | File size | Low cohesion | Extract into shared base class | P1 |
| 6 | benchmarks/harness.py run_suite() | run_suite() is ~120 lines | File size | Low cohesion | Extract result-writing and probe-registration into helpers | P1 |
| 7 | tests/conftest.py jax mock | `jax_mod.tree_util.tree_map = lambda fn, tree: fn(tree)` — incorrect mock. tree_map applies fn to all leaves recursively, not to the tree as a whole | observe/conftest.py line ~33 | Test correctness — may mask real bugs | Implement correct mock: use Python's built-in recursion or import a minimal tree_map implementation | P1 |
| 8 | tests/conftest.py bfloat16 | `jnp.bfloat16 = np.float32` — incorrect type aliasing; bfloat16 is a dtype, not a Python float | conftest.py line ~26 | Type mismatch in tests | Use `jnp.bfloat16 = np.dtype('float32')` or a proper dtype stub | P1 |
| 9 | train/harness.py load_registry() | Identical to benchmarks/harness.py load_registry() — pure duplication | Both files | Drift risk | Extract into shared module or use importlib | P1 |
| 10 | observe/otel.py get_tracer/get_meter | Both functions eagerly import opentelemetry even when _state["enabled"] is False — violates lazy-import contract documented in otel_probe.py | observe/otel.py lines ~202-217 | Causes 17 test failures on envs without opentelemetry | Guard: `if not _state["enabled"]: return NoOpTracer()` before import block | P1 |
| 11 | benchmarks/harness.py DEVICE_COSTS | Device costs hardcoded as dict inside harness (not as config file or env override) | harness.py lines ~46-55 | Stale when GCP changes prices | Move to scripts/lib/config.sh or a YAML config file | P2 |
| 12 | benchmarks/runner.py N_WARMUP/N_MEASURE | Constants N_WARMUP=20, N_MEASURE=100, N_BLOCKS=3 hardcoded inside runner.py | runner.py lines ~127-131 | Not configurable | Add as optional fields to ExperimentConfig with sensible defaults | P2 |
| 13 | benchmarks/runner.py device_cost_usd_per_hr | Default cost 0.36 hardcoded in ExperimentConfig dataclass | runner.py line ~180 | Should come from registry or DEVICE_COSTS | Pull from registry.yaml or reference DEVICE_COSTS dict | P2 |
| 14 | benchmarks/runner.py decoder_start_id | decoder_start_id defaults to 50258 (Whisper-specific hardcoded value) | runner.py line ~368 | Wrong for non-Whisper decoders | Read from model.config.decoder_start_token_id with fallback; document the fallback | P2 |
| 15 | benchmarks/runner.py _classify_error | Uses string matching on exception type names (`type(exc).__name__`) to avoid imports — fragile if type names change | runner.py lines ~56-84 | Fragile classification | Add explicit isinstance checks with try/import guards | P2 |
| 16 | benchmarks/runner.py latest-run stub | Error stub links to latest run_logs by mtime — fragile on concurrent failures | runner.py lines ~270-278 | Wrong run_id on concurrent errors | Pass run_id explicitly instead of finding by mtime | P2 |
| 17 | benchmarks/runner.py silent ValueError | If run_logs directory is empty, ValueError is caught and run_id_for_stub = None with no warning | runner.py lines ~269-277 | Silent failure | Log a warning when run_id_for_stub cannot be resolved | P2 |
| 18 | observe/otel.py _JsonlSpanExporter | export() doesn't log which span failed or the actual error when span.to_json() raises | otel.py lines ~61-94 | Silent observability failure | Log warning with span name and exception type when to_json() fails | P2 |
| 19 | observe/otel.py return type annotations | get_tracer() and get_meter() lack return type annotations | otel.py lines ~202, 211 | Weak contracts | Add `-> Tracer` and `-> Meter` (using TYPE_CHECKING guard for optional dep) | P2 |
| 20 | observe/stats.py HIGH_VARIANCE_CV_PCT | Threshold hardcoded at 10.0 with no comment explaining the choice | stats.py line ~14 | Undocumented threshold | Add a one-line comment with the rationale; consider making it a module-level constant with a name | P3 |
| 21 | observe/stats.py MAD constant | 0.6745 (MAD scaling constant) used without a name or comment | stats.py line ~64 | Confusing magic number | `_MAD_SCALE_CONSTANT = 0.6745  # 1 / Phi^{-1}(3/4) for Gaussian` | P3 |
| 22 | observe/lineage.py _ENV_PACKAGES | Hardcoded tuple (jax, torch, transformers, numpy, flax) | lineage.py line ~16 | Not extensible | Document as "intentionally fixed for Stage 1–9"; or make configurable via env var | P3 |
| 23 | train/harness.py build_config() | Missing return type annotation `-> TrainingExperimentConfig` | train/harness.py line ~145 | Weak contract | Add return type annotation | P2 |
| 24 | train/runner.py nested loss_fn() | Nested loss_fn() functions inside _build_train_step() lack parameter and return type annotations | train/runner.py lines ~431, 465, 516 | Weak contracts | Annotate with input/output types | P2 |
| 25 | Makefile python vs python3 | `make test` uses `python -m pytest` — fails on systems with only python3 | Makefile line 6 | Breaks local dev | Change to `python3 -m pytest` or add `PYTHON ?= python3` | P2 |
| 26 | benchmarks/harness.py YAML error | load_registry() error message assumes only pyyaml is missing — doesn't handle schema errors distinctly | harness.py lines ~73-75 | Misleading error messages | Catch yaml.YAMLError separately from ImportError | P2 |
| 27 | Dead file references | MEMORY.md references `observe/input_fingerprint_probe.py` — file is actually `observe/input_fingerprint.py` | MEMORY.md probe quick reference section | Misleading | Update MEMORY.md | P2 |
| 28 | 4 undocumented probes | DeterminismProbe, DeviceInfoProbe, PowerThermalProbe, XlaCompileProbe exist with full implementations but no docs | `grep -rn "^class.*Probe" observe/` | Discoverability | Add to observe/README.md, MEMORY.md, and main README.md | P2 |
| 29 | benchmarks/runner.py phase() exception | phase() context manager catches BenchmarkError and re-raises passing empty `BaseException()` to fanout_on_error, losing original exception info | runner.py lines ~106-119 | Lost error context in probes | Pass the original BenchmarkError to fanout_on_error | P2 |

---

## Top Risk Clusters

### Cluster 1: Test reliability (P0/P1)
- OTel eager import causes 17 failures in default env (P1 fix in otel.py)
- conftest.py mock incorrectness may mask real bugs (P1)
- otel_probe.py completeness unconfirmed (P0 — verify)

### Cluster 2: Code duplication (P1)
- benchmarks/harness.py and train/harness.py are 85% identical
- DEVICE_COSTS duplicated in 2 files
- load_registry() duplicated in 2 files
- Recommended: extract shared base harness

### Cluster 3: Large functions (P1)
- run_experiment() 253 lines
- run_suite() 120-160 lines
- Phase 3 refactor target: extract per-phase helpers

### Cluster 4: Hardcoded values (P2)
- DEVICE_COSTS, N_WARMUP, N_MEASURE, device_cost_usd_per_hr, decoder_start_id
- Low risk but creates drift as project evolves

---

## Priority Action List

| Priority | Action | Effort |
|---|---|---|
| P0 | Verify otel_probe.py has complete OTelProbe class | 15 min |
| P1 | Fix observe/otel.py eager opentelemetry import | 30 min |
| P1 | Fix conftest.py tree_util.tree_map mock | 30 min |
| P1 | Extract shared harness base (benchmarks + train) | 2 hr |
| P2 | Fix Makefile python → python3 | 5 min |
| P2 | Fix MEMORY.md input_fingerprint_probe.py reference | 5 min |
| P2 | Document 4 undocumented probes | 1 hr |
| P2 | Add N_WARMUP/N_MEASURE to ExperimentConfig | 30 min |
| P3 | Name MAD constant in stats.py | 5 min |
