"""
Markdown report renderer — one self-contained summary per run.

Combines:
  * workload config
  * compile cache stats
  * breakdown by category
  * step-time summary
  * HBM utilization
  * cost report
  * bottleneck findings

The file is written to `artifacts/reports/run_<trace_id>.md` and is meant
to be the FIRST thing the user opens after a run.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..common.cost import CostReport
from ..memory.hbm_sim import HbmSimulator
from ..profiling.bottleneck_report import Finding, render_markdown as render_findings
from ..profiling.trace_analyzer import Breakdown, StepSummary
from ..xla_sim.compile_cache import CompileCache


def render_run_report(
    trace_id: str,
    config: Any,
    breakdown: Breakdown,
    summary: StepSummary,
    hbm: HbmSimulator,
    compile_cache: CompileCache,
    cost: CostReport,
    findings: List[Finding],
) -> str:
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

    # ── Compile cache ───────────────────────────────────────────────────────
    lines.append("## XLA compile cache")
    cs = compile_cache.stats()
    lines.append(f"- hits: {cs['hits']}")
    lines.append(f"- misses: {cs['misses']}")
    lines.append(f"- entries: {cs['entries']}")
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
    lines.append(f"- capacity: {hbm.capacity_bytes/1e9:.1f} GB")
    lines.append(f"- used:     {hbm.used_bytes()/1e9:.3f} GB "
                 f"({hbm.utilization()*100:.1f}%)")
    lines.append(f"- OOM events: {hbm.oom_events}")
    for cat, b in hbm.by_category().items():
        lines.append(f"  - {cat:15s} {b/1e9:.3f} GB")
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
