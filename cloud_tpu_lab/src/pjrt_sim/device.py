"""
Fake TPU device for the simulator.

A `TpuDevice` knows its peak FLOPS and HBM bandwidth (looked up from the
catalog), so the runtime can estimate the time to execute an HLO op by
the well-known roofline model:

    t_compute = flops / peak_flops_per_sec
    t_memory  = bytes / hbm_bandwidth_per_sec
    t_op      = max(t_compute, t_memory)

The roofline model is a *simulation* — it ignores compiler optimisations,
fusion, async transfers, MXU efficiency at small sizes, and on-chip SRAM
reuse. Treat the numbers as illustrative, not benchmark-grade.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..tpu_versions.cloud_tpu_catalog import TpuSpec


@dataclass
class TpuDevice:
    device_id: int
    spec: TpuSpec
    # Effective fractions — multiplied into peak FLOPS / bandwidth. Used to
    # simulate "you'll never hit peak in practice".
    flops_efficiency: float = 0.5
    hbm_efficiency: float = 0.7

    @property
    def effective_flops_per_s(self) -> float:
        return self.spec.peak_bf16_tflops * 1e12 * self.flops_efficiency

    @property
    def effective_hbm_bw_per_s(self) -> float:
        return self.spec.hbm_bandwidth_gbps * 1e9 * self.hbm_efficiency

    def roofline_op_time_s(self, flops: int, bytes_moved: int) -> float:
        t_c = flops / max(self.effective_flops_per_s, 1.0)
        t_m = bytes_moved / max(self.effective_hbm_bw_per_s, 1.0)
        return max(t_c, t_m)
