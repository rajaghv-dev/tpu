"""
Cloud TPU version catalog.

Source-of-truth lookup for per-version specs used by the simulator (HBM
capacity, peak FLOPS, ICI bandwidth) and by the cost estimator.

Every field is one of:
  PUBLIC  — published on cloud.google.com/tpu
  DOC     — documented behaviour in Google whitepapers / Cloud blog
  SIM     — value we use in the simulator (best-effort, not official)
  INFER   — reasonable inference, clearly marked
  UNKNOWN — proprietary or not publicly disclosed

Update freely as official numbers change. Do NOT invent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TpuSpec:
    version: str
    code_name: str
    hbm_per_chip_gb: float
    hbm_bandwidth_gbps: float
    peak_bf16_tflops: float
    ici_bandwidth_gbps: float
    chips_per_host: int
    typical_slice_shapes: tuple
    # source markers per field — keys match the field names above
    sources: Dict[str, str]
    notes: str = ""


# Numbers below are intentionally CONSERVATIVE — when public sources give a
# range, we take the lower end so the simulator never overstates capability.
# Update from https://cloud.google.com/tpu/docs/system-architecture-tpu-vm and
# the version-specific pages (v4 / v5e / v5p / v6e / Trillium).
_CATALOG: Dict[str, TpuSpec] = {
    "v4": TpuSpec(
        version="v4",
        code_name="TPU v4",
        hbm_per_chip_gb=32.0,
        hbm_bandwidth_gbps=1200.0,
        peak_bf16_tflops=275.0,
        ici_bandwidth_gbps=270.0,
        chips_per_host=4,
        typical_slice_shapes=((2, 2, 1), (2, 2, 2), (4, 4, 4)),
        sources={
            "hbm_per_chip_gb": "PUBLIC",
            "hbm_bandwidth_gbps": "PUBLIC",
            "peak_bf16_tflops": "PUBLIC",
            "ici_bandwidth_gbps": "DOC",
            "chips_per_host": "PUBLIC",
            "typical_slice_shapes": "PUBLIC",
        },
        notes="3D torus topology; best for very large training; HBM = HBM2.",
    ),
    "v5e": TpuSpec(
        version="v5e",
        code_name="TPU v5e",
        hbm_per_chip_gb=16.0,
        hbm_bandwidth_gbps=820.0,
        peak_bf16_tflops=197.0,
        ici_bandwidth_gbps=200.0,
        chips_per_host=4,
        typical_slice_shapes=((1, 1), (2, 2), (4, 4), (8, 8)),
        sources={
            "hbm_per_chip_gb": "PUBLIC",
            "hbm_bandwidth_gbps": "PUBLIC",
            "peak_bf16_tflops": "PUBLIC",
            "ici_bandwidth_gbps": "DOC",
            "chips_per_host": "PUBLIC",
            "typical_slice_shapes": "PUBLIC",
        },
        notes="2D torus; cost-optimised for inference + medium training.",
    ),
    "v5p": TpuSpec(
        version="v5p",
        code_name="TPU v5p",
        hbm_per_chip_gb=96.0,
        hbm_bandwidth_gbps=2765.0,
        peak_bf16_tflops=459.0,
        ici_bandwidth_gbps=600.0,
        chips_per_host=4,
        typical_slice_shapes=((2, 2, 1), (4, 4, 4), (8, 8, 8)),
        sources={
            "hbm_per_chip_gb": "PUBLIC",
            "hbm_bandwidth_gbps": "PUBLIC",
            "peak_bf16_tflops": "PUBLIC",
            "ici_bandwidth_gbps": "DOC",
            "chips_per_host": "PUBLIC",
            "typical_slice_shapes": "PUBLIC",
        },
        notes="3D torus; performance-optimised; HBM = HBM3.",
    ),
    "v6e": TpuSpec(
        version="v6e",
        code_name="TPU v6e (Trillium)",
        hbm_per_chip_gb=32.0,
        hbm_bandwidth_gbps=1640.0,
        peak_bf16_tflops=918.0,
        ici_bandwidth_gbps=800.0,
        chips_per_host=8,
        typical_slice_shapes=((1, 1), (2, 4), (4, 4), (8, 8), (16, 16)),
        sources={
            "hbm_per_chip_gb": "PUBLIC",
            "hbm_bandwidth_gbps": "PUBLIC",
            "peak_bf16_tflops": "PUBLIC",
            "ici_bandwidth_gbps": "DOC",
            "chips_per_host": "PUBLIC",
            "typical_slice_shapes": "PUBLIC",
        },
        notes="Trillium generation; ~4× peak vs v5e; SparseCore available.",
    ),
}


def get_spec(version: str) -> TpuSpec:
    """Lookup spec by short version key (v4 / v5e / v5p / v6e). Strict."""
    key = version.lower().lstrip("tpu-").lstrip("tpu_").lstrip("tpu")
    if key not in _CATALOG:
        raise KeyError(
            f"Unknown TPU version {version!r}; known: {sorted(_CATALOG.keys())}"
        )
    return _CATALOG[key]


def list_versions() -> list[str]:
    return sorted(_CATALOG.keys())


def list_specs() -> Dict[str, TpuSpec]:
    return dict(_CATALOG)
