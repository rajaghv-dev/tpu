"""
Markdown report renderer — one self-contained summary per run.

Combines:
  * workload config
  * compile stats (compile_time_s, recompile_count)
  * breakdown by category
  * step-time summary
  * HBM utilization (from `jax.devices()[0].memory_stats()`)
  * cost report
  * bottleneck findings

The file is written to `artifacts/reports/run_<trace_id>.md` and is meant
to be the FIRST thing the user opens after a run.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List, Mapping

from ..common.cost import CostReport
from ..profiling.bottleneck_report import Finding, render_markdown as render_findings
from ..profiling.trace_analyzer import Breakdown, StepSummary


def render_run_report(
    trace_id: str,
    config: Any,
    breakdown: Breakdown,
    summary: StepSummary,
    hbm: Mapping[str, Any],
    compile_stats: Mapping[str, Any],
    cost: CostReport,
    findings: List[Finding],
) -> str:
    """
    Render the Markdown run report.

    `hbm` is a plain dict with keys: used_bytes, capacity_bytes, utilization,
       peak_bytes, oom_events. Build it with
       `profiling.bottleneck_report.hbm_stats_from_jax(...)` on TPU.

    `compile_stats` is a plain dict with keys: compile_time_s, recompile_count.
    """
    cfg_dict = asdict(config) if is_dataclass(config) else dict(vars(config))
    lines: List[str] = []
    lines.append(f"# Cloud TPU Lab — Run Report")
    lines.append("")
    lines.append(f"**trace_id**: `{trace_id}`")
    lines.append("")

    # ── Config ──────────────────────────────────────────────────────────────
    lines.append("## Workload")
    for k, v in cfg_dict.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    # ── Compile ─────────────────────────────────────────────────────────────
    lines.append("## XLA compile")
    lines.append(f"- compile_time_s: {compile_stats.get('compile_time_s', 0.0):.4f}")
    lines.append(f"- recompile_count: {compile_stats.get('recompile_count', 0)}")
    lines.append("")

    # ── Breakdown ───────────────────────────────────────────────────────────
    lines.append("## Time breakdown")
    fr = breakdown.fractions()
    for cat in ("compile", "device", "collective", "input_pipeline",
                "host", "checkpoint"):
        lines.append(
            f"- {cat:20s} {getattr(breakdown, f'{cat}_s'):.4f}s "
            f"({fr[cat]*100:5.1f}%)"
        )
    lines.append("")

    # ── Step summary ────────────────────────────────────────────────────────
    lines.append("## Step time")
    lines.append(f"- n_steps: {summary.n_steps}")
    lines.append(f"- first step: {summary.first_step_s*1000:.2f} ms")
    lines.append(f"- median:     {summary.median_step_s*1000:.2f} ms")
    lines.append(f"- p95:        {summary.p95_step_s*1000:.2f} ms")
    lines.append("")

    # ── HBM ─────────────────────────────────────────────────────────────────
    lines.append("## HBM")
    cap = int(hbm.get("capacity_bytes", 0) or 0)
    used = int(hbm.get("used_bytes", 0) or 0)
    peak = int(hbm.get("peak_bytes", used) or used)
    util = float(hbm.get("utilization", 0.0) or 0.0)
    lines.append(f"- capacity: {cap/1e9:.3f} GB")
    lines.append(f"- used:     {used/1e9:.3f} GB ({util*100:.1f}%)")
    lines.append(f"- peak:     {peak/1e9:.3f} GB")
    lines.append(f"- OOM events: {hbm.get('oom_events', 0)}")
    lines.append("")

    # ── Cost ────────────────────────────────────────────────────────────────
    lines.append("## Cost (placeholder rate; update from cloud.google.com/tpu/pricing)")
    lines.append(f"- total wall time: {cost.total_run_wall_s:.3f} s")
    lines.append(f"- total USD:       ${cost.total_run_usd:.6f}")
    lines.append(f"- per step:        ${cost.cost_per_step_usd:.6f}")
    if cost.cost_per_sample_usd is not None:
        lines.append(f"- per sample:      ${cost.cost_per_sample_usd:.6f}")
    if cost.cost_per_token_usd is not None:
        lines.append(f"- per token:       ${cost.cost_per_token_usd:.6f}")
    lines.append("")

    # ── Findings ────────────────────────────────────────────────────────────
    lines.append(render_findings(findings))
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
