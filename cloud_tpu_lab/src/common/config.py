"""
Workload + simulation configuration objects.

Everything the simulation needs is captured in a single `WorkloadConfig`
dataclass plus a `SimulationConfig`. Knobs the user changes most often
(batch size, mesh shape, n_steps, tpu_version) are at the top of
`WorkloadConfig`; less-frequently-changed details (HBM bandwidth, MXU
flops) live in derived helpers in src/tpu_versions/ and are looked up from
`tpu_version`.

This module deliberately depends on nothing else in the project so it can
be imported by tests without pulling in jax / matplotlib / numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Workload knobs ────────────────────────────────────────────────────────────


@dataclass
class WorkloadConfig:
    """One end-to-end run of the simulator."""
    name: str = "tiny_mlp_demo"
    framework: str = "jax"  # jax | pytorch_xla | tensorflow
    model_kind: str = "mlp"  # mlp | transformer
    # Shape knobs — all small by default so the demo is fast on CPU.
    batch_size: int = 8
    seq_len: int = 32         # transformer only
    hidden_size: int = 128
    num_layers: int = 2
    vocab_size: int = 1024    # transformer only
    input_dim: int = 64       # mlp only
    output_dim: int = 10      # mlp only
    # Training knobs
    n_steps: int = 5
    precision: str = "bf16"   # bf16 | fp32
    # Hardware target (used for HBM capacity, MXU flops, pricing) — looked up
    # in src/tpu_versions/cloud_tpu_catalog.py at runtime.
    tpu_version: str = "v5e"
    chip_count: int = 1
    # Mesh layout (list of ints). [chip_count] = pure data-parallel.
    mesh_shape: Optional[List[int]] = None
    # Pricing — placeholder, user overrides for real estimates.
    hourly_usd_per_chip: float = 1.20

    def __post_init__(self) -> None:
        if self.mesh_shape is None:
            self.mesh_shape = [self.chip_count]


@dataclass
class SimulationConfig:
    """Simulator-only knobs (no real TPU equivalent)."""
    # If True, the simulated step time is derived from FLOPS + bandwidth +
    # collective cost. If False, the step time is the wall-clock of the
    # numeric forward pass (more accurate but only meaningful for CPU runs).
    use_analytical_step_time: bool = True
    # Output dirs (relative to repo root unless absolute).
    log_dir: str = "cloud_tpu_lab/artifacts/logs"
    metrics_dir: str = "cloud_tpu_lab/artifacts/metrics"
    trace_dir: str = "cloud_tpu_lab/artifacts/traces"
    report_dir: str = "cloud_tpu_lab/artifacts/reports"
    plot_dir: str = "cloud_tpu_lab/artifacts/plots"
    # Whether to emit a Matplotlib timeline plot at the end. Set False on
    # headless CI hosts without a backend.
    make_plot: bool = True
