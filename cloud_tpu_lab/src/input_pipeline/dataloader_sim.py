"""
Simulated input pipeline — the canonical "TPU is idle but you don't know why".

Real TPU pipelines stage:

    disk → host RAM → preprocess → host-to-device buffer → TPU HBM

Any one of those can starve the TPU. We model it as fixed per-batch
load + preprocess times, and a prefetch depth that lets some of the host
work overlap with TPU compute.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class InputPipelineCost:
    load_time_s: float
    preprocess_time_s: float
    host_to_device_time_s: float
    effective_wait_s: float  # what the TPU actually saw — accounts for prefetch


def simulate_input_pipeline(
    n_steps: int,
    batch_size: int,
    bytes_per_sample: int,
    disk_bandwidth_bytes_s: float = 200e6,   # ~200 MB/s SSD-class
    preprocess_s_per_sample: float = 0.001,
    h2d_bandwidth_bytes_s: float = 12e9,     # PCIe Gen4 ~12 GB/s
    prefetch_depth: int = 2,
    device_step_time_s: float = 0.01,
) -> List[InputPipelineCost]:
    """
    Returns one InputPipelineCost per step.

    Prefetch depth N lets up to N batches be staged ahead. As long as the
    device step (compute time on TPU) is longer than (load + preprocess +
    H2D) the TPU sees zero wait. When prefetch is too shallow the TPU
    waits the difference each step.
    """
    bytes_per_batch = batch_size * bytes_per_sample
    load_s = bytes_per_batch / max(disk_bandwidth_bytes_s, 1.0)
    pp_s = batch_size * preprocess_s_per_sample
    h2d_s = bytes_per_batch / max(h2d_bandwidth_bytes_s, 1.0)
    pipeline_s = load_s + pp_s + h2d_s

    # Effective wait is positive only when the host can't keep up.
    overlap_s = min(device_step_time_s * prefetch_depth, pipeline_s)
    wait_s = max(pipeline_s - overlap_s, 0.0)

    return [
        InputPipelineCost(
            load_time_s=load_s,
            preprocess_time_s=pp_s,
            host_to_device_time_s=h2d_s,
            effective_wait_s=wait_s,
        )
        for _ in range(n_steps)
    ]
