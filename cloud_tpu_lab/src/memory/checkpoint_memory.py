"""
Checkpoint memory + I/O time estimator.

Used by the cost report to remind the user that "checkpoint frequency" is
itself a cost knob — frequent checkpoints add wall-clock and (on a real
TPU pod) bandwidth pressure.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckpointEstimate:
    param_bytes: int
    optimizer_state_bytes: int
    total_bytes: int
    write_time_s: float


def estimate_checkpoint(
    param_bytes: int,
    optimizer_state_multiplier: float = 2.0,   # Adam keeps m + v
    disk_bandwidth_bytes_s: float = 500e6,     # ~500 MB/s GCS sustained
) -> CheckpointEstimate:
    opt = int(param_bytes * optimizer_state_multiplier)
    total = param_bytes + opt
    return CheckpointEstimate(
        param_bytes=param_bytes,
        optimizer_state_bytes=opt,
        total_bytes=total,
        write_time_s=total / max(disk_bandwidth_bytes_s, 1.0),
    )
