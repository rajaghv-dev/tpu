"""Report renderer produces well-formed Markdown."""
from cloud_tpu_lab.src.common.config import WorkloadConfig
from cloud_tpu_lab.src.common.cost import CostInputs, estimate_cost
from cloud_tpu_lab.src.memory.hbm_sim import HbmSimulator
from cloud_tpu_lab.src.observability.report import render_run_report
from cloud_tpu_lab.src.profiling.bottleneck_report import Finding, diagnose
from cloud_tpu_lab.src.profiling.trace_analyzer import Breakdown, StepSummary
from cloud_tpu_lab.src.xla_sim.compile_cache import CompileCache


def _empty_breakdown() -> Breakdown:
    return Breakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_renders_minimum_sections() -> None:
    cfg = WorkloadConfig()
    breakdown = _empty_breakdown()
    summary = StepSummary(0, [], 0.0, 0.0, 0.0)
    hbm = HbmSimulator(capacity_bytes=1, bandwidth_bytes_s=1.0)
    cache = CompileCache()
    cost = estimate_cost(CostInputs(
        chip_count=1, n_steps=1, step_time_s=0.1, hourly_usd_per_chip=1.0,
    ))
    findings = [Finding("info", "overall", "ok", "nothing")]
    md = render_run_report(
        trace_id="TRACE-0001", config=cfg,
        breakdown=breakdown, summary=summary, hbm=hbm,
        compile_cache=cache, cost=cost, findings=findings,
    )
    assert "# Cloud TPU Lab — Run Report" in md
    assert "## Workload" in md
    assert "## XLA compile cache" in md
    assert "## Time breakdown" in md
    assert "## HBM" in md
    assert "## Cost" in md
    assert "TRACE-0001" in md


def test_diagnose_flags_input_pipeline_heavy() -> None:
    b = Breakdown(compile_s=0.0, device_s=0.5, collective_s=0.0,
                  input_pipeline_s=0.4, host_s=0.0, checkpoint_s=0.0,
                  other_s=0.0)
    hbm = HbmSimulator(capacity_bytes=1_000_000_000, bandwidth_bytes_s=1.0)
    out = diagnose(b, hbm, n_chips=1, total_run_usd=0.001)
    layers = [f.layer for f in out]
    assert "input_pipeline" in layers


def test_diagnose_returns_at_least_one_finding() -> None:
    b = _empty_breakdown()
    hbm = HbmSimulator(capacity_bytes=1_000_000_000, bandwidth_bytes_s=1.0)
    out = diagnose(b, hbm, n_chips=1, total_run_usd=0.0)
    assert len(out) >= 1
