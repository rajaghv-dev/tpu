"""Cost math is correct and divides-by-zero-safe."""
from cloud_tpu_lab.src.common.cost import CostInputs, estimate_cost


def test_basic_total_cost() -> None:
    # 100 steps × 0.36s = 36s = 0.01 hours. × 4 chips × $1.20/hr = $0.048.
    r = estimate_cost(CostInputs(
        chip_count=4, n_steps=100, step_time_s=0.36,
        hourly_usd_per_chip=1.20, samples_per_step=8,
    ))
    assert abs(r.total_run_wall_s - 36.0) < 1e-6
    assert abs(r.total_run_usd - 0.048) < 1e-6
    assert abs(r.cost_per_step_usd - 0.00048) < 1e-6
    assert abs(r.cost_per_sample_usd - 0.00006) < 1e-6


def test_zero_samples_yields_none() -> None:
    r = estimate_cost(CostInputs(
        chip_count=1, n_steps=10, step_time_s=0.1,
        hourly_usd_per_chip=1.0,
    ))
    assert r.cost_per_sample_usd is None
    assert r.cost_per_token_usd is None


def test_zero_steps_does_not_divide_by_zero() -> None:
    r = estimate_cost(CostInputs(
        chip_count=1, n_steps=0, step_time_s=0.1,
        hourly_usd_per_chip=1.0,
    ))
    # Total wall = 0, total usd = 0, cost per step = 0 / max(0,1) = 0.
    assert r.total_run_wall_s == 0
    assert r.total_run_usd == 0
    assert r.cost_per_step_usd == 0.0


def test_utilization_inflates_adjusted_cost() -> None:
    r = estimate_cost(CostInputs(
        chip_count=1, n_steps=10, step_time_s=0.1,
        hourly_usd_per_chip=1.0, utilization=0.5,
    ))
    assert r.utilization_adjusted_usd > r.total_run_usd
