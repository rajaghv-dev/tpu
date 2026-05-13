#!/usr/bin/env python3
"""
Sharding simulation demo — sweep chip count and plot scaling behaviour.

CPU-only. Produces:
  artifacts/plots/sharding_<trace_id>.png       (scaling-efficiency plot)
  artifacts/reports/sharding_<trace_id>.md      (per-chip-count table)

Usage:
  python -m cloud_tpu_lab.examples.run_sharding_simulation \
      --max-chips 32 --payload-mb 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cloud_tpu_lab.src.common.trace import new_trace_id
from cloud_tpu_lab.src.sharding.all_reduce import all_reduce_time
from cloud_tpu_lab.src.sharding.mesh import PartitionSpec, make_1d_mesh
from cloud_tpu_lab.src.sharding.partitioner import partition_tensor
from cloud_tpu_lab.src.tpu_versions.cloud_tpu_catalog import get_spec


def sweep_chip_counts(
    chip_counts: List[int],
    payload_bytes: int,
    tpu_version: str,
    compute_per_chip_s: float,
) -> List[Tuple[int, float, float, float]]:
    """Return [(n_chips, step_s, comm_s, scaling_eff), ...]."""
    spec = get_spec(tpu_version)
    ici_bw = spec.ici_bandwidth_gbps * 1e9
    rows = []
    baseline_step = None
    for n in chip_counts:
        coll = all_reduce_time(payload_bytes, n, ici_bw)
        # Compute scales as 1/n (ideal): compute_per_chip_s / n.
        compute_s = compute_per_chip_s / max(n, 1)
        step_s = compute_s + coll.sim_duration_s
        if baseline_step is None:
            baseline_step = step_s * n  # ideal "1-chip" time = n × compute_per_chip_s/n × n
        # Scaling efficiency = ideal_speedup / actual_speedup vs single chip.
        # Use the n=1 measurement as the baseline.
        rows.append((n, step_s, coll.sim_duration_s, 0.0))

    # Compute scaling efficiency relative to n=1.
    base = rows[0][1] if rows else 1.0
    out = []
    for n, step, comm, _ in rows:
        speedup = base / max(step, 1e-12)
        eff = speedup / n  # 1.0 = perfect
        out.append((n, step, comm, eff))
    return out


def render_markdown(rows, payload_bytes, tpu_version, trace_id) -> str:
    lines = [
        f"# Sharding simulation — trace {trace_id}",
        f"- payload: {payload_bytes/1e6:.1f} MB",
        f"- tpu_version: {tpu_version}",
        "",
        "| chips | step_s | comm_s | comm_fraction | scaling_efficiency |",
        "|------:|-------:|-------:|--------------:|-------------------:|",
    ]
    for n, step, comm, eff in rows:
        frac = comm / max(step, 1e-12)
        lines.append(f"| {n} | {step:.6f} | {comm:.6f} | {frac*100:5.1f}% | {eff*100:5.1f}% |")
    return "\n".join(lines)


def maybe_plot(rows, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ns = [r[0] for r in rows]
    steps = [r[1] for r in rows]
    effs = [r[3] * 100 for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(ns, steps, "o-")
    axs[0].set_xlabel("chips")
    axs[0].set_ylabel("step time (s)")
    axs[0].set_title("Step time vs chip count")
    axs[1].plot(ns, effs, "o-")
    axs[1].set_xlabel("chips")
    axs[1].set_ylabel("scaling efficiency (%)")
    axs[1].set_title("Scaling efficiency vs chip count")
    axs[1].axhline(100, color="green", linestyle="--", alpha=0.4, label="ideal")
    axs[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-chips", type=int, default=16)
    p.add_argument("--payload-mb", type=float, default=8.0)
    p.add_argument("--tpu-version", default="v5e",
                   choices=["v4", "v5e", "v5p", "v6e"])
    p.add_argument("--compute-per-chip-ms", type=float, default=10.0,
                   help="Ideal per-chip compute time at n=1 (ms)")
    args = p.parse_args(argv)

    trace_id = new_trace_id()
    chips = [1, 2, 4, 8, 16, 32]
    chips = [c for c in chips if c <= args.max_chips]
    rows = sweep_chip_counts(
        chip_counts=chips,
        payload_bytes=int(args.payload_mb * 1e6),
        tpu_version=args.tpu_version,
        compute_per_chip_s=args.compute_per_chip_ms / 1000.0,
    )

    report_dir = Path("cloud_tpu_lab/artifacts/reports")
    plot_dir = Path("cloud_tpu_lab/artifacts/plots")
    report_path = report_dir / f"sharding_{trace_id}.md"
    plot_path = plot_dir / f"sharding_{trace_id}.png"
    report_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(rows, int(args.payload_mb * 1e6),
                         args.tpu_version, trace_id)
    report_path.write_text(md)
    plotted = maybe_plot(rows, plot_path)

    print(md)
    print()
    print(f"Report: {report_path}")
    if plotted:
        print(f"Plot:   {plot_path}")
    else:
        print("Plot:   (matplotlib not installed — skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
