#!/usr/bin/env python3
"""
Cost analysis demo — sweep across (tpu_version × chip_count × batch_size)
and print a Markdown table of cost-per-step / per-sample.

The hourly rate is a placeholder — pass --hourly-usd-per-chip to use a real
rate from https://cloud.google.com/tpu/pricing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cloud_tpu_lab.src.common.cost import CostInputs, estimate_cost
from cloud_tpu_lab.src.tpu_versions.cloud_tpu_catalog import get_spec, list_versions


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--versions", nargs="+", default=list_versions())
    p.add_argument("--chip-counts", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32, 128])
    p.add_argument("--step-time-ms", type=float, default=50.0,
                   help="Assumed steady-state step time (ms)")
    p.add_argument("--hourly-usd-per-chip", type=float, default=1.20,
                   help="Placeholder; check cloud.google.com/tpu/pricing")
    p.add_argument("--n-steps", type=int, default=1000)
    args = p.parse_args(argv)

    rows: List[Tuple[str, int, int, float, float, float]] = []
    for v in args.versions:
        spec = get_spec(v)
        for chips in args.chip_counts:
            for bs in args.batch_sizes:
                r = estimate_cost(CostInputs(
                    chip_count=chips, n_steps=args.n_steps,
                    step_time_s=args.step_time_ms / 1000.0,
                    hourly_usd_per_chip=args.hourly_usd_per_chip,
                    samples_per_step=bs * chips,  # data-parallel
                ))
                rows.append((
                    v, chips, bs, r.total_run_usd,
                    r.cost_per_step_usd, r.cost_per_sample_usd or 0.0,
                ))

    print("| tpu | chips | batch | total $ | $/step | $/sample |")
    print("|-----|------:|------:|--------:|-------:|---------:|")
    for r in rows:
        v, chips, bs, total, per_step, per_sample = r
        print(f"| {v} | {chips} | {bs} | {total:.6f} | {per_step:.6f} | {per_sample:.8f} |")
    print()
    print("⚠ Pricing placeholder — confirm at https://cloud.google.com/tpu/pricing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
