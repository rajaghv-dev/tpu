## Summary

<!-- 1-3 bullet points describing what this PR does and why -->

## Type of change

- [ ] Bug fix
- [ ] New probe
- [ ] New model (registry.yaml)
- [ ] Documentation update
- [ ] Refactor (no behavior change)
- [ ] CI/tooling change
- [ ] Stage N feature (specify: Stage ___)

## Checklist

- [ ] Tests pass: `python3 -m pytest tests/ -q`
- [ ] No new test failures introduced
- [ ] Lint clean: `python3 -m flake8 benchmarks/ observe/ tests/ train/ --max-line-length 100`
- [ ] Docs updated for any behavior or interface changes
- [ ] New probes added to `observe/README.md` and `MEMORY.md`
- [ ] No secrets committed (`.env`, `.hf-token`, `.claude/`)
- [ ] `results/runs.jsonl` schema unchanged (or migration documented)
- [ ] `AGENTS.md` updated if new safe/restricted files added

## Validation

<!-- Paste relevant test output or dry-run output -->

```bash
# e.g.:
python3 -m pytest tests/ -q
# 265 passed, 7 failed (OTel-enabled tests), 3 skipped
JAX_PLATFORMS=cpu python3 -m benchmarks.harness --suite quick --device cpu --dry-run
# [dry-run] 5 models planned
```

## Breaking changes

<!-- None / describe any public API, CLI, or schema changes -->

## Related issues

<!-- Fixes #N / Relates to #N -->
