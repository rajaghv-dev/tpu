"""
Bottleneck report — the "what should I do next?" summary.

Given a trace + HBM state + cost report, produce a ranked list of
recommendations. We use rule-of-thumb thresholds documented in
docs/14_benchmarking_playbook.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..memory.hbm_sim import HbmSimulator
from .trace_analyzer import Breakdown


@dataclass
class Finding:
    severity: str   # info | warn | high
    layer: str      # input_pipeline | xla | collective | hbm | host | cost
    message: str
    suggested_fix: str


def diagnose(
    breakdown: Breakdown,
    hbm: HbmSimulator,
    n_chips: int = 1,
    total_run_usd: float = 0.0,
) -> List[Finding]:
    findings: List[Finding] = []
    fr = breakdown.fractions()

    # ── Input pipeline ────────────────────────────────────────────────────
    if fr["input_pipeline"] > 0.10:
        findings.append(Finding(
            severity="high" if fr["input_pipeline"] > 0.25 else "warn",
            layer="input_pipeline",
            message=f"Input pipeline = {fr['input_pipeline']*100:.1f}% of step time",
            suggested_fix=(
                "Increase prefetch depth, use tf.data autotune / "
                "torch DataLoader num_workers, move preprocess to host CPU, "
                "or cache preprocessed shards."
            ),
        ))

    # ── Compile / recompile ───────────────────────────────────────────────
    if fr["compile"] > 0.20:
        findings.append(Finding(
            severity="warn",
            layer="xla",
            message=f"Compile = {fr['compile']*100:.1f}% — likely recompiles",
            suggested_fix=(
                "Stabilise shapes (avoid dynamic batch / seq); set "
                "JAX_COMPILATION_CACHE_DIR to persist compiled executables "
                "across runs; check `jax.config.jax_log_compiles`."
            ),
        ))

    # ── Collective communication ──────────────────────────────────────────
    if n_chips > 1 and fr["collective"] > 0.30:
        findings.append(Finding(
            severity="high",
            layer="collective",
            message=f"Collectives = {fr['collective']*100:.1f}% — communication-bound",
            suggested_fix=(
                "Consider larger batch (amortises all-reduce), shard model "
                "instead of data on small chips, or move to a higher-bandwidth "
                "topology (e.g. v5p instead of v5e)."
            ),
        ))

    # ── HBM pressure ──────────────────────────────────────────────────────
    if hbm.utilization() > 0.85:
        findings.append(Finding(
            severity="high",
            layer="hbm",
            message=f"HBM at {hbm.utilization()*100:.1f}% capacity",
            suggested_fix=(
                "Enable gradient checkpointing, reduce batch size, switch "
                "optimizer state to bf16/fp8, or shard params across more chips."
            ),
        ))
    if hbm.oom_events > 0:
        findings.append(Finding(
            severity="high",
            layer="hbm",
            message=f"OOM events: {hbm.oom_events}",
            suggested_fix="Same fixes as above, plus reduce seq_len or chip count.",
        ))

    # ── Host overhead ─────────────────────────────────────────────────────
    if fr["host"] > 0.15:
        findings.append(Finding(
            severity="warn",
            layer="host",
            message=f"Host overhead = {fr['host']*100:.1f}%",
            suggested_fix=(
                "Move loss / metric reduction into the JIT'd step; avoid "
                "Python-side per-step bookkeeping."
            ),
        ))

    # ── Cost sanity ───────────────────────────────────────────────────────
    if total_run_usd > 10.0:
        findings.append(Finding(
            severity="info",
            layer="cost",
            message=f"Total run cost = ${total_run_usd:.2f}",
            suggested_fix=(
                "Use --dry-run to verify config; enable preemptible/Spot "
                "TPU; reduce n_steps for development; ensure cleanup script "
                "runs even on KeyboardInterrupt."
            ),
        ))

    if not findings:
        findings.append(Finding(
            severity="info",
            layer="overall",
            message="No major bottlenecks detected at the current thresholds.",
            suggested_fix="Nothing to fix — try a larger workload to stress-test.",
        ))
    return findings


def render_markdown(findings: List[Finding]) -> str:
    lines = ["# Bottleneck Report", ""]
    # Group by severity.
    for sev in ("high", "warn", "info"):
        group = [f for f in findings if f.severity == sev]
        if not group:
            continue
        lines.append(f"## {sev.upper()}")
        lines.append("")
        for f in group:
            lines.append(f"- **[{f.layer}]** {f.message}")
            lines.append(f"  - Fix: {f.suggested_fix}")
        lines.append("")
    return "\n".join(lines)
