# Security Audit

Generated: 2026-05-16.

## Summary

11 findings: 2 Critical, 2 High, 4 Medium, 3 Low.

No hardcoded secrets found in Python or YAML files.

---

## Findings

| # | File | Lines | Severity | Issue | Fix |
|---|---|---|---|---|---|
| 1 | scripts/30_deploy_repo.sh | tar command | Critical | `.claude/` directory NOT excluded from deployment tarball — local Claude Code session state, memory files, and settings copied to TPU VM | Add `--exclude='.claude'` and `--exclude='.tpu'` to tar flags |
| 2 | scripts/setup_hf.sh | curl -H | Critical | HuggingFace token passed as curl `-H "Authorization: Bearer $tok"` — briefly visible in process list via `ps aux` | Use stdin: `curl ... -H "Authorization: Bearer $(cat -)" <<< "$tok"` or env var approach |
| 3 | scripts/gcp_bootstrap.sh | gpg install | High | `curl \| sudo gpg --dearmor` pipes content without checksum verification — MITM risk | Fetch, verify SHA256 against Google's published value, then dearmor |
| 4 | scripts/lib/common.sh | exit handler | High | Exit handlers stored as strings and executed via `bash -c "$h"` — command injection if handler string contains uncontrolled input | Validate handler input; use argument arrays |
| 5 | .github/workflows/smoke_on_push.yml | — | Medium | No explicit `permissions:` block — relies on GitHub default (currently read-all but can change) | Add `permissions: contents: read` |
| 6 | scripts/30_deploy_repo.sh | remote cmd | Medium | `$HOME` expands on local shell before reaching gcloud SSH — correct by accident but fragile | Escape dollar sign: `\$HOME` for remote expansion |
| 7 | scripts/provision_tpu.sh | gcloud ssh | Medium | TPU_NAME and ZONE inputs used in shell commands without regex validation | Validate with `[[ "$VAR" =~ ^[a-z0-9-]+$ ]]` before use |
| 8 | scripts/setup_hf.sh | API enable | Medium | `gcloud services enable secretmanager.googleapis.com` exit code not checked | Add `\|\| { err "Failed"; return 1; }` |
| 9 | — | — | Low | No `.env.example` documenting required environment variables | Create `.env.example` |
| 10 | scripts/gcp_bootstrap.sh | auth list | Low | Minor unquoted variable in `gcloud auth list` output pipeline | Add quotes for consistency |
| 11 | scripts/30_deploy_repo.sh | mktemp | Low | mktemp race condition (low risk — default permissions 0600 mitigate) | No action required |

---

## Strengths (confirmed safe)

- HF token in `.gitignore` (`.hf-token`, `*.hf-token`)
- `.env` in `.gitignore`
- `.tpu/` venv in `.gitignore`
- `.claude/` in `.gitignore` (but NOT excluded from deploy tarball — see finding #1)
- No hardcoded GCP project IDs or API keys in Python/YAML
- All config via env vars or gcloud defaults
- `set -euo pipefail` in all scripts
- HF token stored in GCP Secret Manager (not plaintext file)
- `shred -u` used for temp token files in setup_hf.sh
- OTel endpoint configurable via env var (not hardcoded)

---

## Fix Priority

1. **Fix 30_deploy_repo.sh** — add `.claude/` exclusion to tar (immediate)
2. **Add .env.example** — document all required env vars (immediate)
3. **Add CI permissions block** — explicit least-privilege (immediate)
4. **Fix HF token in curl** — use stdin approach (medium)
5. **Add GPG checksum verification** — security hardening (low — GCP is trusted source)
6. **Add TPU_NAME/ZONE validation** — input sanitization (low — internal scripts)

---

## Secrets Scan

Files scanned: all .py, .yaml, .yml, .sh, .json, .md, .toml
Result: No plaintext secrets, API keys, or tokens found.
Note: `scripts/lib/config.sh` contains GCP project ID, bucket name, and zone defaults — these are project-specific but not secret.
