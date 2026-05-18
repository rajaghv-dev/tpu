"""
Bottleneck report — the "what should I do next?" summary.

Given a trace breakdown + HBM stats + cost, produce a ranked list of
recommendations. Thresholds are rule-of-thumb; tune per workload.

This module is framework-agnostic — it consumes plain Python types
(`Breakdown` from `trace_analyzer`, an `HbmStats` dict, scalars). On a real
TPU the HBM stats come from `jax.devices()[0].memory_stats()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .trace_analyzer import Breakdown


# ── HBM stats: framework-agnostic plain dict ─────────────────────────────────
# Expected keys (all int bytes, except utilization which is a float in [0,1]):
#   used_bytes, capacity_bytes, utilization, oom_events
# `jax.devices()[0].memory_stats()` returns
#   {"bytes_in_use": ..., "bytes_limit": ..., "peak_bytes_in_use": ...}
# Use `hbm_stats_from_jax()` below to normalise.
HbmStats = Mapping[str, Any]


def hbm_stats_from_jax(jax_memory_stats: Mapping[str, Any],
                       oom_events: int = 0) -> Dict[str, Any]:
    """Normalise `jax.devices()[0].memory_stats()` into `HbmStats` shape."""
    used = int(jax_memory_stats.get("bytes_in_use", 0) or 0)
    cap = int(jax_memory_stats.get("bytes_limit", 0) or 0)
    util = (used / cap) if cap > 0 else 0.0
    return {
        "used_bytes": used,
        "capacity_bytes": cap,
        "utilization": util,
        "peak_bytes": int(jax_memory_stats.get("peak_bytes_in_use", used) or used),
        "oom_events": int(oom_events),
    }


def empty_hbm_stats() -> Dict[str, Any]:
    return {"used_bytes": 0, "capacity_bytes": 0, "utilization": 0.0,
            "peak_bytes": 0, "oom_events": 0}


@dataclass
class Finding:
    severity: str   # info | warn | high
    layer: str      # input_pipeline | xla | collective | hbm | host | cost
    message: str
    suggested_fix: str


def diagnose(
    breakdown: Breakdown,
    hbm: HbmStats,
    n_chips: int = 1,
    total_run_usd: float = 0.0,
) -> List[Finding]:
    findings: List[Finding] = []
    fr = breakdown.fractions()
    util = float(hbm.get("utilization", 0.0) or 0.0)
    oom = int(hbm.get("oom_events", 0) or 0)

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
    if util > 0.85:
        findings.append(Finding(
            severity="high",
            layer="hbm",
            message=f"HBM at {util*100:.1f}% capacity",
            suggested_fix=(
                "Enable gradient checkpointing, reduce batch size, switch "
                "optimizer state to bf16/fp8, or shard params across more chips."
            ),
        ))
    if oom > 0:
        findings.append(Finding(
            severity="high",
            layer="hbm",
            message=f"OOM events: {oom}",
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
