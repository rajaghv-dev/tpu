"""
Pretty-print and side-by-side compare known Cloud TPU versions.

Used by `examples/run_cpu_simulation_demo.py --show-versions` and the
notebook `01_cloud_tpu_big_picture.ipynb`. No external deps.
"""
from __future__ import annotations

from .cloud_tpu_catalog import TpuSpec, get_spec, list_specs


_FIELDS = (
    ("HBM / chip (GB)",       "hbm_per_chip_gb"),
    ("HBM bandwidth (GB/s)",  "hbm_bandwidth_gbps"),
    ("Peak BF16 (TFLOPS)",    "peak_bf16_tflops"),
    ("ICI bandwidth (GB/s)",  "ici_bandwidth_gbps"),
    ("Chips per host",        "chips_per_host"),
)


def render_table(versions: list[str] | None = None) -> str:
    """ASCII table of per-version specs."""
    specs = list_specs() if not versions else {v: get_spec(v) for v in versions}
    keys = list(specs.keys())
    col_w = 24
    head = "Spec".ljust(col_w) + "".join(k.ljust(col_w) for k in keys)
    sep = "-" * len(head)
    rows = [head, sep]
    for label, attr in _FIELDS:
        row = label.ljust(col_w) + "".join(
            f"{getattr(specs[k], attr):.1f}".ljust(col_w) for k in keys
        )
        rows.append(row)
    rows.append("")
    rows.append("Notes:")
    for k in keys:
        rows.append(f"  {k}: {specs[k].notes}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(render_table())
