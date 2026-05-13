"""End-to-end smoke — the demo runs and produces all four artefact files."""
from pathlib import Path

from cloud_tpu_lab.src.common.config import SimulationConfig, WorkloadConfig
from cloud_tpu_lab.examples.run_cpu_simulation_demo import run_demo


def test_demo_runs_and_writes_artefacts(tmp_path: Path) -> None:
    sim = SimulationConfig(
        log_dir=str(tmp_path / "logs"),
        metrics_dir=str(tmp_path / "metrics"),
        trace_dir=str(tmp_path / "traces"),
        report_dir=str(tmp_path / "reports"),
        plot_dir=str(tmp_path / "plots"),
        make_plot=False,
    )
    workload = WorkloadConfig(
        name="smoke", n_steps=3, batch_size=4, hidden_size=32, num_layers=2,
    )
    summary = run_demo(workload, sim, quiet=True)

    assert summary["n_log_events"] > 0
    assert summary["n_metric_rows"] == 3
    assert summary["n_trace_events"] > 0
    for key in ("log", "metrics", "trace", "report"):
        p = Path(summary["artifacts"][key])
        assert p.exists(), f"missing artefact: {p}"
        assert p.stat().st_size > 0, f"empty artefact: {p}"


def test_demo_two_runs_produce_different_trace_ids(tmp_path: Path) -> None:
    sim = SimulationConfig(
        log_dir=str(tmp_path / "logs"),
        metrics_dir=str(tmp_path / "metrics"),
        trace_dir=str(tmp_path / "traces"),
        report_dir=str(tmp_path / "reports"),
        plot_dir=str(tmp_path / "plots"),
        make_plot=False,
    )
    workload = WorkloadConfig(name="smoke", n_steps=1)
    s1 = run_demo(workload, sim, quiet=True)
    s2 = run_demo(workload, sim, quiet=True)
    assert s1["trace_id"] != s2["trace_id"]


def test_demo_summary_includes_cost() -> None:
    sim = SimulationConfig(make_plot=False)
    workload = WorkloadConfig(name="cost_smoke", n_steps=2)
    summary = run_demo(workload, sim, quiet=True)
    assert summary["total_run_usd"] >= 0.0
    assert summary["hbm_utilization"] >= 0.0
