# GitHub Readiness

Generated: 2026-05-16.

## Current CI/CD

| File | Jobs | Status | Coverage |
|---|---|---|---|
| `.github/workflows/smoke_on_push.yml` | unit-tests, dry-run-harness, lineage-sanity | ✅ Working | pytest + harness dry-run |

## Missing CI

| Item | Priority | Proposed file |
|---|---|---|
| Lint (flake8) | ✅ Added | lint job in `smoke_on_push.yml` |
| cloud_tpu_lab tests | Low | Add to smoke_on_push.yml or new file |
| Training harness dry-run | Low | Add to dry-run-harness job |
| Dependency update check | Low | `.github/dependabot.yml` |
| Dependabot config | ✅ Added | `.github/dependabot.yml` |
| PR template | ✅ Added | `.github/PULL_REQUEST_TEMPLATE.md` |

## Proposed CI additions

### Lint job (add to smoke_on_push.yml)

```yaml
lint:
  runs-on: ubuntu-latest
  timeout-minutes: 5
  permissions:
    contents: read
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"
        cache: pip
    - run: pip install flake8
    - run: python -m flake8 benchmarks/ observe/ tests/ train/ --max-line-length 100 --extend-ignore=E501
```

### Dependabot

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

## Existing workflow issues

| Issue | Severity | Fix |
|---|---|---|
| No explicit `permissions:` block | Low | Add `permissions: contents: read` at job or workflow level |
| Lock file missing opentelemetry | Critical | Regenerate requirements.stage1.lock.txt |
| Test count comment in workflow is stale | Low | Update comment |

## Recommended permissions block

Add to `.github/workflows/smoke_on_push.yml` after the `on:` block:

```yaml
permissions:
  contents: read
```

## Missing GitHub files

| File | Purpose | Priority |
|---|---|---|
| `.github/PULL_REQUEST_TEMPLATE.md` | PR checklist | Medium |
| `.github/ISSUE_TEMPLATE/bug_report.md` | Bug report template | Low |
| `.github/ISSUE_TEMPLATE/feature_request.md` | Feature request | Low |
| `.github/dependabot.yml` | Dependency updates | Low |
| `.github/CODEOWNERS` | Code ownership | Low |

## Branch protection recommendations

- Require PR reviews before merging to main
- Require status checks: `smoke/unit-tests`, `smoke/dry-run-harness`
- Dismiss stale approvals on new push

## Release readiness

No release workflow exists. Not needed for current stage (personal research repo). Recommend adding when publishing to PyPI or creating versioned releases.

## GitHub Pages

`results/dashboard/index.html` is a static dashboard. Could be served via GitHub Pages. Not configured yet.
