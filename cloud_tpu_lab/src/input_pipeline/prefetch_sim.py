"""
Plot helper: prefetch-depth vs. effective input wait.

Used by `examples/run_cpu_simulation_demo.py` and notebook 10 to show
that "increasing prefetch depth" eventually stops helping — past a certain
point the host can stage faster than the TPU consumes.
"""
from __future__ import annotations

from typing import List, Tuple

from .dataloader_sim import simulate_input_pipeline


def sweep_prefetch_depth(
    depths: List[int],
    *,
    batch_size: int,
    bytes_per_sample: int,
    device_step_time_s: float,
) -> List[Tuple[int, float]]:
    """Return [(depth, mean_wait_s), ...] for plotting."""
    out: List[Tuple[int, float]] = []
    for d in depths:
        costs = simulate_input_pipeline(
            n_steps=1, batch_size=batch_size,
            bytes_per_sample=bytes_per_sample,
            prefetch_depth=d,
            device_step_time_s=device_step_time_s,
        )
        out.append((d, costs[0].effective_wait_s))
    return out
