"""Report renderer produces well-formed Markdown; diagnose() flags expected layers."""
from cloud_tpu_lab.src.common.config import WorkloadConfig
from cloud_tpu_lab.src.common.cost import CostInputs, estimate_cost
from cloud_tpu_lab.src.observability.report import render_run_report
from cloud_tpu_lab.src.profiling.bottleneck_report import (
    Finding, diagnose, empty_hbm_stats,
)
from cloud_tpu_lab.src.profiling.trace_analyzer import Breakdown, StepSummary


def _empty_breakdown() -> Breakdown:
    return Breakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_renders_minimum_sections() -> None:
    cfg = WorkloadConfig()
    breakdown = _empty_breakdown()
    summary = StepSummary(0, [], 0.0, 0.0, 0.0)
    hbm = empty_hbm_stats()
    hbm["capacity_bytes"] = 1
    cost = estimate_cost(CostInputs(
        chip_count=1, n_steps=1, step_time_s=0.1, hourly_usd_per_chip=1.0,
    ))
    findings = [Finding("info", "overall", "ok", "nothing")]
    md = render_run_report(
        trace_id="TRACE-0001", config=cfg,
        breakdown=breakdown, summary=summary, hbm=hbm,
        compile_stats={"compile_time_s": 0.0, "recompile_count": 0},
        cost=cost, findings=findings,
    )
    assert "# Cloud TPU Lab — Run Report" in md
    assert "## Workload" in md
    assert "## XLA compile" in md
    assert "## Time breakdown" in md
    assert "## HBM" in md
    assert "## Cost" in md
    assert "TRACE-0001" in md


def test_diagnose_flags_input_pipeline_heavy() -> None:
    b = Breakdown(compile_s=0.0, device_s=0.5, collective_s=0.0,
                  input_pipeline_s=0.4, host_s=0.0, checkpoint_s=0.0,
                  other_s=0.0)
    hbm = empty_hbm_stats()
    hbm["capacity_bytes"] = 1_000_000_000
    out = diagnose(b, hbm, n_chips=1, total_run_usd=0.001)
    layers = [f.layer for f in out]
    assert "input_pipeline" in layers


def test_diagnose_returns_at_least_one_finding() -> None:
    b = _empty_breakdown()
    hbm = empty_hbm_stats()
    hbm["capacity_bytes"] = 1_000_000_000
    out = diagnose(b, hbm, n_chips=1, total_run_usd=0.0)
    assert len(out) >= 1
