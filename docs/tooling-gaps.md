# Tooling Gaps

Generated: 2026-05-16 — Phase 0.

## Overview

Graph/code-intelligence tools were checked for availability. None are installed. Repo audit was completed using direct code reading and grep-based analysis.

## Tool Status

| Tool | Expected use | Available? | Evidence | Alternative used |
|---|---|---|---|---|
| Graphify | Repo-level knowledge graph, doc-code mapping | ❌ Not installed | `which graphify` → not found | Direct file reads + grep |
| CodeGraph | Typed code graph, call relationships, dependency edges | ❌ Not installed | `which codegraph` → not found | `grep -rn` + file reads |
| CodeQL | Semantic security queries, tainted data flows | ❌ Not installed | `which codeql` → not found | Manual security audit via script inspection |
| Pyrefly | Python type checking, API contract drift | ❌ Not installed | `which pyrefly` → not found | Manual type annotation review |
| Memgraph | Graph database backend for persistent code graph | ❌ Not installed | `which mgconsole` → not found | Markdown-based analysis |
| Obsidian | Human-facing knowledge graph, refactor decisions | ❌ Not installed | `which obsidian` → not found | Session notes in MEMORY.md and docs/ |

## Impact on Audit

Without graph tools, the audit relied on:
- Direct file reads (observe/, benchmarks/, train/, tests/)
- `grep -rn` for symbol discovery (probe classes, public functions)
- `ls` for file existence verification
- `python3 -m pytest` for test baseline

This approach is sufficient for the current repo size (~50 Python files). For Stage 2+ (when more modules are added), graph tools would reduce token cost.

## Recommendations

For future agents working on this repo:

1. **Short-term (no tooling):** Use `grep -rn "^class\|^def " observe/ benchmarks/ train/` to find all public symbols. Use `python3 -m pytest tests/ -q` for validation.

2. **Medium-term (if repo grows):** Install pyrefly (`pip install pyrefly`) for type-level change impact analysis. Especially useful before renaming public APIs.

3. **Long-term (Stage 5+):** Consider CodeQL GitHub code scanning (free for public repos) for security queries on the growing codebase.

## Token-saving strategies used (without graph tools)

- Read signatures/headers only for large files (harness.py lines 1-100)
- Used grep output instead of full file reads for probe inventory
- Read existing MEMORY.md first to avoid re-deriving known state
- Used parallel agents for independent file reads and writes
