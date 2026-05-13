"""
Cost / performance estimator.

Cloud TPU pricing changes — never hardcode a current rate as a default
that's hidden from the user. The estimator takes `hourly_usd_per_chip`
explicitly and derives cost-per-step / cost-per-sample / cost-per-token
from `chip_count`, `n_steps`, `step_time_s`, `samples_per_step`,
`tokens_per_step`.

Sources for the hourly price:
  https://cloud.google.com/tpu/pricing       (public list price)
  https://cloud.google.com/tpu/docs/regions-zones   (per-region availability)

These prices vary by region, commitment, and Spot vs. on-demand. Update
your `WorkloadConfig.hourly_usd_per_chip` from the official page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CostInputs:
    chip_count: int
    n_steps: int
    step_time_s: float
    hourly_usd_per_chip: float
    samples_per_step: int = 0
    tokens_per_step: int = 0
    utilization: float = 1.0  # fraction of wall-clock spent on real work


@dataclass
class CostReport:
    total_run_wall_s: float
    total_run_usd: float
    cost_per_step_usd: float
    cost_per_sample_usd: Optional[float]
    cost_per_token_usd: Optional[float]
    cost_per_epoch_usd: Optional[float]
    utilization_adjusted_usd: float

    def to_dict(self) -> dict:
        return {
            "total_run_wall_s": self.total_run_wall_s,
            "total_run_usd": self.total_run_usd,
            "cost_per_step_usd": self.cost_per_step_usd,
            "cost_per_sample_usd": self.cost_per_sample_usd,
            "cost_per_token_usd": self.cost_per_token_usd,
            "cost_per_epoch_usd": self.cost_per_epoch_usd,
            "utilization_adjusted_usd": self.utilization_adjusted_usd,
        }


def estimate_cost(inputs: CostInputs, samples_per_epoch: int = 0) -> CostReport:
    """Wall-clock × chip_count × hourly_rate, with derived per-unit columns."""
    total_wall_s = inputs.n_steps * inputs.step_time_s
    chip_hours = (total_wall_s / 3600.0) * inputs.chip_count
    total_usd = chip_hours * inputs.hourly_usd_per_chip

    cost_per_step = total_usd / max(inputs.n_steps, 1)
    cost_per_sample = (
        cost_per_step / inputs.samples_per_step
        if inputs.samples_per_step > 0 else None
    )
    cost_per_token = (
        cost_per_step / inputs.tokens_per_step
        if inputs.tokens_per_step > 0 else None
    )
    cost_per_epoch = (
        cost_per_sample * samples_per_epoch
        if cost_per_sample is not None and samples_per_epoch > 0 else None
    )

    # Utilization-adjusted: what would the same workload cost if we used the
    # accelerator 100 % of wall time? Lower number = "you're being charged
    # for X but only getting Y of useful work."
    util = max(min(inputs.utilization, 1.0), 1e-6)
    util_adjusted = total_usd / util

    return CostReport(
        total_run_wall_s=total_wall_s,
        total_run_usd=total_usd,
        cost_per_step_usd=cost_per_step,
        cost_per_sample_usd=cost_per_sample,
        cost_per_token_usd=cost_per_token,
        cost_per_epoch_usd=cost_per_epoch,
        utilization_adjusted_usd=util_adjusted,
    )
