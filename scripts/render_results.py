#!/usr/bin/env python3
"""
render_results.py — turn results/runs.jsonl + run_logs/ into committable MD.

Generates:
  results/RESULTS.md
      Top-level summary: counts, table of every row, links to per-run details,
      per-probe coverage stats.
  results/run_logs/<run_id>/REPORT.md
      One file per run: full result row, lineage, plus the failure traceback
      from error.json if the row is a failure stub. Every other JSON file in
      the run log dir is rendered as a Probes sub-section.

Designed to be re-runnable after every benchmark session — overwrites the
generated files but never touches the source JSONL/JSON. Safe to commit the
output to git so reviewers can read results without running anything.

Usage:
    python3 scripts/render_results.py
    python3 scripts/render_results.py --jsonl path/to/runs.jsonl --out custom_dir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

# Probes whose presence is structural (not a "probe" in the user-facing sense).
_NON_PROBE_FILES = frozenset({"lineage.json", "error.json"})

# Phase order used by the runner — used to order timing.json rows.
_PHASE_ORDER = (
    "preflight",
    "model_load",
    "compile",
    "warmup",
    "latency",
    "throughput",
    "postflight",
)

# Cap list rendering to keep MD tables readable.
_LIST_TRUNCATE = 5


def _fmt(v: Any) -> str:
    """Render a JSON value for an MD table cell."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return ", ".join(map(str, v)) if v else "—"
    return str(v)


def _row_status(row: Dict[str, Any]) -> str:
    """Classify a row as 'success' or 'failed' (failure stubs carry status=failed)."""
    return "failed" if row.get("status") == "failed" else "success"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError as e:
            print(f"  warning: skipping malformed line: {e}")
    return rows


def _read_run_log(run_logs_dir: Path, run_id: Optional[str]) -> Dict[str, Any]:
    """
    Pull every *.json file from <run_logs_dir>/<run_id>/ into a dict, keyed by
    filename including the .json suffix (e.g. "lineage.json", "timing.json").
    Probe files use the same keying convention so callers can iterate uniformly.
    Malformed JSON is skipped silently — we don't want a corrupt probe file to
    block the rest of the report.
    """
    out: Dict[str, Any] = {}
    if not run_id:
        return out
    log_dir = run_logs_dir / run_id
    if not log_dir.exists():
        return out
    for p in sorted(log_dir.glob("*.json")):
        try:
            out[p.name] = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _probe_files(log_files: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """
    Filter `log_files` down to actual probe outputs (everything that isn't
    lineage.json or error.json). Returns (probe_name, payload) pairs sorted by
    probe name for deterministic output.
    """
    out: List[Tuple[str, Any]] = []
    for fname, payload in log_files.items():
        if fname in _NON_PROBE_FILES:
            continue
        if not fname.endswith(".json"):
            continue
        probe_name = fname[:-len(".json")]
        out.append((probe_name, payload))
    out.sort(key=lambda kv: kv[0])
    return out


# ── Per-run REPORT.md ─────────────────────────────────────────────────────────

_RESULT_FIELD_GROUPS = (
    ("Identity", ["run_id", "timestamp", "device", "framework", "path"]),
    ("Model",    ["model", "domain", "architecture_family", "attention_variant",
                  "positional_encoding", "is_moe", "total_params_M",
                  "active_params_M"]),
    ("Variant",  ["precision", "pruning", "compiled", "compile_mode",
                  "inference_mode", "batch_size", "batch_size_throughput",
                  "seq_len"]),
    ("Compile",  ["first_compile_s", "subsequent_compile_s",
                  "compile_cache_hit"]),
    ("Latency",  ["latency_mean_ms", "latency_std_ms", "latency_cv_pct",
                  "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]),
    ("Throughput", ["throughput_mean_samples_sec",
                    "throughput_std_samples_sec"]),
    ("Quality",  ["flags"]),
    ("Cost",     ["device_cost_usd_per_hr", "experiment_cost_usd",
                  "cost_per_1k_samples_usd"]),
    ("Lineage",  ["git_sha", "jax_version", "torch_version",
                  "transformers_version", "hf_model_revision", "input_seed",
                  "n_independent_runs", "environment_hash"]),
)


def _truncate_list(items: List[Any]) -> str:
    """Render a list as ``a, b, c (…N more)`` if longer than _LIST_TRUNCATE."""
    if not items:
        return "—"
    head = items[:_LIST_TRUNCATE]
    tail = len(items) - len(head)
    rendered = ", ".join(_fmt(x) for x in head)
    if tail > 0:
        rendered += f" (…{tail} more)"
    return rendered


def _render_value_cell(v: Any) -> str:
    """
    Render a single probe value into an MD-table-cell-safe string. Nested dicts
    expand to one level of bullet points. Lists truncate to the first 5
    elements. Pipes are escaped to avoid breaking the surrounding table.
    """
    if isinstance(v, dict):
        if not v:
            return "—"
        # Collapse to a one-level <br> bullet list inside the cell.
        bits = []
        for k, val in v.items():
            if isinstance(val, (dict, list)):
                bits.append(f"- {k}: {_short_repr(val)}")
            else:
                bits.append(f"- {k}: {_fmt(val)}")
        return "<br>".join(bits).replace("|", "\\|")
    if isinstance(v, list):
        return _truncate_list(v).replace("|", "\\|")
    return _fmt(v).replace("|", "\\|")


def _short_repr(v: Any) -> str:
    """Compact one-line repr for deeply-nested values (level 2+)."""
    if isinstance(v, dict):
        if not v:
            return "{}"
        keys = list(v.keys())[:3]
        more = len(v) - len(keys)
        s = "{" + ", ".join(f"{k}: …" for k in keys) + ("…" if more > 0 else "") + "}"
        return s
    if isinstance(v, list):
        return _truncate_list(v)
    return _fmt(v)


def _render_generic_probe(payload: Any) -> List[str]:
    """Default fallback: render any JSON payload as a 2-column key/value table."""
    out: List[str] = []
    if not isinstance(payload, dict):
        # Lists / scalars — just dump as a code block.
        out.append("```json")
        out.append(json.dumps(payload, indent=2, default=str))
        out.append("```")
        out.append("")
        return out
    if not payload:
        out.append("_(empty)_")
        out.append("")
        return out
    out.append("| Key | Value |")
    out.append("|---|---|")
    for k, v in payload.items():
        out.append(f"| {k} | {_render_value_cell(v)} |")
    out.append("")
    return out


# ── Per-probe summarisers ────────────────────────────────────────────────────

def _summarise_timing(payload: Any) -> List[str]:
    """timing.json: render phase_summary as a phase | duration_s table."""
    out: List[str] = []
    if not isinstance(payload, dict):
        return _render_generic_probe(payload)
    phase_summary = payload.get("phase_summary")
    if not isinstance(phase_summary, dict) or not phase_summary:
        return _render_generic_probe(payload)
    # Ordered: known phases first (in runner order), unknown phases trailing.
    known = [p for p in _PHASE_ORDER if p in phase_summary]
    unknown = sorted(p for p in phase_summary if p not in _PHASE_ORDER)
    out.append("| phase | duration_s |")
    out.append("|---|---:|")
    for phase in known + unknown:
        out.append(f"| {phase} | {_fmt(phase_summary[phase])} |")
    out.append("")
    # Surface any other top-level keys beyond phase_summary so we don't hide info.
    extras = {k: v for k, v in payload.items() if k != "phase_summary"}
    if extras:
        out.append("**Other fields**")
        out.append("")
        out.extend(_render_generic_probe(extras))
    return out


def _summarise_cloud_monitoring(payload: Any) -> List[str]:
    """
    cloud_monitoring.json: per_phase_summary as a wide table. If `available`
    is false, just say so — don't pretend we have data we don't.
    """
    out: List[str] = []
    if not isinstance(payload, dict):
        return _render_generic_probe(payload)
    if payload.get("available") is False:
        reason = payload.get("reason") or payload.get("message") or "not available"
        out.append(f"_Cloud Monitoring not available: {reason}._")
        out.append("")
        return out
    per_phase = payload.get("per_phase_summary")
    if not isinstance(per_phase, dict) or not per_phase:
        return _render_generic_probe(payload)
    out.append("| phase | mxu min | mxu mean | mxu max | "
               "hbm util % min | hbm util % mean | hbm util % max |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    known = [p for p in _PHASE_ORDER if p in per_phase]
    unknown = sorted(p for p in per_phase if p not in _PHASE_ORDER)
    for phase in known + unknown:
        bucket = per_phase[phase]
        if not isinstance(bucket, dict):
            out.append(f"| {phase} | {_fmt(bucket)} | | | | | |")
            continue
        mxu = bucket.get("mxu") if isinstance(bucket.get("mxu"), dict) else {}
        hbm = (bucket.get("hbm_util_pct")
               if isinstance(bucket.get("hbm_util_pct"), dict) else {})
        out.append(
            f"| {phase} "
            f"| {_fmt(mxu.get('min'))} "
            f"| {_fmt(mxu.get('mean'))} "
            f"| {_fmt(mxu.get('max'))} "
            f"| {_fmt(hbm.get('min'))} "
            f"| {_fmt(hbm.get('mean'))} "
            f"| {_fmt(hbm.get('max'))} |"
        )
    out.append("")
    return out


def _summarise_hlo_dump(payload: Any) -> List[str]:
    """hlo_dump.json: instruction & fusion counts plus top-5 ops."""
    out: List[str] = []
    if not isinstance(payload, dict):
        return _render_generic_probe(payload)
    out.append("| Field | Value |")
    out.append("|---|---|")
    out.append(f"| instruction_count_estimate | "
               f"{_fmt(payload.get('instruction_count_estimate'))} |")
    out.append(f"| fusion_count_estimate | "
               f"{_fmt(payload.get('fusion_count_estimate'))} |")
    out.append("")
    top_ops = payload.get("top_ops") or payload.get("top_5_ops")
    if isinstance(top_ops, list) and top_ops:
        out.append("**Top ops (first 5)**")
        out.append("")
        out.append("| # | op |")
        out.append("|---:|---|")
        for i, op in enumerate(top_ops[:5], start=1):
            out.append(f"| {i} | {_render_value_cell(op)} |")
        out.append("")
    elif isinstance(top_ops, dict) and top_ops:
        out.append("**Top ops (first 5)**")
        out.append("")
        out.append("| op | count |")
        out.append("|---|---:|")
        for k, v in list(top_ops.items())[:5]:
            out.append(f"| {k} | {_fmt(v)} |")
        out.append("")
    return out


def _summarise_memory(payload: Any) -> List[str]:
    """memory.json: baseline RSS + per-phase delta table."""
    out: List[str] = []
    if not isinstance(payload, dict):
        return _render_generic_probe(payload)
    baseline = payload.get("baseline_rss_mb")
    if baseline is None:
        baseline = payload.get("baseline_rss") or payload.get("baseline")
    if baseline is not None:
        out.append(f"**Baseline RSS:** {_fmt(baseline)} MB")
        out.append("")
    deltas = (payload.get("delta_from_baseline_mb")
              or payload.get("phase_deltas_mb")
              or payload.get("phase_deltas"))
    if isinstance(deltas, dict) and deltas:
        out.append("| phase | delta_from_baseline_mb |")
        out.append("|---|---:|")
        known = [p for p in _PHASE_ORDER if p in deltas]
        unknown = sorted(p for p in deltas if p not in _PHASE_ORDER)
        for phase in known + unknown:
            out.append(f"| {phase} | {_fmt(deltas[phase])} |")
        out.append("")
    else:
        # Fall back to dumping the rest of the payload generically.
        out.extend(_render_generic_probe(
            {k: v for k, v in payload.items()
             if k not in ("baseline_rss_mb", "baseline_rss", "baseline")}
        ))
    return out


_PROBE_SUMMARISERS = {
    "timing": _summarise_timing,
    "cloud_monitoring": _summarise_cloud_monitoring,
    "hlo_dump": _summarise_hlo_dump,
    "memory": _summarise_memory,
}


def _render_probe_section(probe_name: str, payload: Any) -> List[str]:
    """Render a single probe's MD sub-section (heading + body)."""
    out: List[str] = [f"### {probe_name}", ""]
    summariser = _PROBE_SUMMARISERS.get(probe_name)
    if summariser is not None:
        body = summariser(payload)
    else:
        body = _render_generic_probe(payload)
    out.extend(body)
    return out


def _render_run_report(row: Dict[str, Any], log_files: Dict[str, Any]) -> str:
    """Render a single run as a self-contained MD page."""
    status = _row_status(row)
    title_emoji = "✓" if status == "success" else "✗"
    model = row.get("model", "?")
    device = row.get("device", "?")
    precision = row.get("precision", "?")
    timestamp = row.get("timestamp", "?")

    out: List[str] = []
    out.append(f"# Run report — {model} · {precision} · {device} ({status})\n")
    out.append(f"**Status.** {title_emoji} {status}.  "
               f"**Timestamp.** {timestamp}.  "
               f"**Run id.** `{row.get('run_id', '?')}`\n")

    if status == "failed":
        out.append("## Failure summary\n")
        out.append(f"- **Phase.** `{row.get('phase', '?')}`")
        out.append(f"- **Category.** `{row.get('error_category', '?')}`")
        out.append(f"- **Exception type.** `{row.get('exception_type', '?')}`")
        msg = row.get("exception_message", "?")
        out.append(f"- **Message.** {msg}\n")
        err = log_files.get("error.json")
        if err and err.get("traceback"):
            out.append("### Traceback\n")
            out.append("```")
            out.append(err["traceback"].rstrip())
            out.append("```\n")
        if err and err.get("lineage"):
            out.append("### Lineage at failure\n")
            out.append("| Field | Value |")
            out.append("|---|---|")
            for k, v in err["lineage"].items():
                out.append(f"| {k} | {_fmt(v)} |")
            out.append("")
        # Even on failure, surface any probe files so partial observability
        # (e.g. timing.json with whatever phases completed) is still visible.
        probes = _probe_files(log_files)
        if probes:
            out.append("## Probes\n")
            for name, payload in probes:
                out.extend(_render_probe_section(name, payload))
        return "\n".join(out)

    # ── Success path: emit the structured field groups ──────────────────
    for group_name, fields in _RESULT_FIELD_GROUPS:
        present = [(f, row[f]) for f in fields if f in row and row[f] is not None]
        if not present:
            continue
        out.append(f"## {group_name}\n")
        out.append("| Field | Value |")
        out.append("|---|---|")
        for f, v in present:
            out.append(f"| {f} | {_fmt(v)} |")
        out.append("")

    # ── Probes section: every JSON in log_files except lineage/error ────
    probes = _probe_files(log_files)
    if probes:
        out.append("## Probes\n")
        for name, payload in probes:
            out.extend(_render_probe_section(name, payload))

    if log_files.get("lineage.json"):
        out.append("## Auxiliary log files\n")
        out.append(f"- `lineage.json` ({len(json.dumps(log_files['lineage.json']))} bytes)")
        for fname in sorted(log_files):
            if fname in _NON_PROBE_FILES:
                continue
            out.append(f"- `{fname}` ({len(json.dumps(log_files[fname], default=str))} bytes)")
    return "\n".join(out)


# ── Top-level RESULTS.md ──────────────────────────────────────────────────────

def _row_probe_names(run_logs_dir: Path, run_id: Optional[str]) -> List[str]:
    """Cheap probe-name listing for the index table — no JSON parse needed."""
    if not run_id:
        return []
    log_dir = run_logs_dir / run_id
    if not log_dir.exists():
        return []
    names: List[str] = []
    for p in sorted(log_dir.glob("*.json")):
        if p.name in _NON_PROBE_FILES:
            continue
        names.append(p.stem)
    return names


def _render_results_index(
    rows: List[Dict[str, Any]],
    run_logs_dir: Optional[Path] = None,
) -> str:
    succ = [r for r in rows if _row_status(r) == "success"]
    fail = [r for r in rows if _row_status(r) == "failed"]

    out: List[str] = []
    out.append("# Results — TPU × GPU inference benchmark\n")
    out.append("Auto-generated from `results/runs.jsonl` + `results/run_logs/` by")
    out.append("[`scripts/render_results.py`](../scripts/render_results.py). "
               "Re-run after each benchmark session.\n")

    out.append("## Summary\n")
    out.append(f"- **Total rows.** {len(rows)}")
    out.append(f"- **Succeeded.** {len(succ)}")
    out.append(f"- **Failed.**    {len(fail)}\n")

    # Pre-compute probe coverage across all rows. We need this both for the
    # "Probes attached" column and for the coverage section below.
    probes_per_row: Dict[str, List[str]] = {}
    if run_logs_dir is not None:
        for r in rows:
            rid = r.get("run_id")
            if rid:
                probes_per_row[rid] = _row_probe_names(run_logs_dir, rid)

    # ── Successes table ────────────────────────────────────────────────
    if succ:
        out.append("## Successful runs\n")
        out.append("| model | device | precision | first_compile_s | "
                   "p50 ms | p95 ms | p99 ms | CV % | tput sps | "
                   "cost/1k | flags | probes attached | report |")
        out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
        for r in succ:
            run_id = r.get("run_id", "")
            short = run_id[:8] + "…" if run_id else "—"
            link = (f"[{short}](run_logs/{run_id}/REPORT.md)" if run_id else "—")
            probe_names = probes_per_row.get(run_id, [])
            probes_cell = ", ".join(probe_names) if probe_names else "—"
            out.append(
                f"| {r.get('model','?')} "
                f"| {r.get('device','?')} "
                f"| {r.get('precision','?')} "
                f"| {_fmt(r.get('first_compile_s'))} "
                f"| {_fmt(r.get('latency_p50_ms'))} "
                f"| {_fmt(r.get('latency_p95_ms'))} "
                f"| {_fmt(r.get('latency_p99_ms'))} "
                f"| {_fmt(r.get('latency_cv_pct'))} "
                f"| {_fmt(r.get('throughput_mean_samples_sec'))} "
                f"| {_fmt(r.get('cost_per_1k_samples_usd'))} "
                f"| {_fmt(r.get('flags'))} "
                f"| {probes_cell} "
                f"| {link} |"
            )
        out.append("")

    # ── Failures table ─────────────────────────────────────────────────
    if fail:
        out.append("## Failed runs\n")
        out.append("| model | device | precision | phase | category | "
                   "exception | report |")
        out.append("|---|---|---|---|---|---|---|")
        for r in fail:
            run_id = r.get("run_id", "")
            short = run_id[:8] + "…" if run_id else "—"
            link = (f"[{short}](run_logs/{run_id}/REPORT.md)" if run_id else "—")
            msg = (r.get("exception_message") or "")[:80]
            out.append(
                f"| {r.get('model','?')} "
                f"| {r.get('device','?')} "
                f"| {r.get('precision','?')} "
                f"| `{r.get('phase','?')}` "
                f"| `{r.get('error_category','?')}` "
                f"| `{r.get('exception_type','?')}: {msg}` "
                f"| {link} |"
            )
        out.append("")

    # ── Probe coverage ─────────────────────────────────────────────────
    if probes_per_row:
        # Aggregate {probe_name: count_of_rows_with_it}.
        coverage: Dict[str, int] = {}
        for names in probes_per_row.values():
            for n in names:
                coverage[n] = coverage.get(n, 0) + 1
        if coverage:
            out.append("## Probe coverage\n")
            total = len(rows)
            out.append(f"_Across {total} row(s):_\n")
            out.append("| probe | rows with output |")
            out.append("|---|---:|")
            for name in sorted(coverage):
                out.append(f"| {name} | {coverage[name]} / {total} |")
            out.append("")

    out.append("## Reproducing\n")
    out.append("Each row's `lineage` field captures the git SHA, JAX/transformers")
    out.append("versions, HF model revision, and input seed needed to reproduce")
    out.append("identically. See `scripts/run_all.sh --suite smoke` for the")
    out.append("orchestration that produces these rows.\n")
    return "\n".join(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--jsonl", default=str(REPO_ROOT / "results" / "runs.jsonl"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_dir = Path(args.out)
    run_logs_dir = out_dir / "run_logs"

    rows = _read_jsonl(jsonl_path)
    if not rows:
        print(f"No rows in {jsonl_path}; nothing to render.")
        return 0

    # Per-run reports
    n_reports = 0
    for r in rows:
        run_id = r.get("run_id")
        if not run_id:
            continue
        log_files = _read_run_log(run_logs_dir, run_id)
        report_md = _render_run_report(r, log_files)
        target = run_logs_dir / run_id / "REPORT.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report_md)
        n_reports += 1

    # Top-level summary
    index_md = _render_results_index(rows, run_logs_dir=run_logs_dir)
    (out_dir / "RESULTS.md").write_text(index_md)

    print(f"Rendered {n_reports} per-run REPORT.md + 1 summary RESULTS.md")
    print(f"  → {out_dir}/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
