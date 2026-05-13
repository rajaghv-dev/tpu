#!/usr/bin/env python3
"""
Profiler trace demo — generate a Chrome-compatible trace JSON for a fake
training step. CPU-only.

Usage:
  python -m cloud_tpu_lab.examples.run_profiler_trace_demo
Then open the produced .json in chrome://tracing or https://ui.perfetto.dev/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cloud_tpu_lab.src.common.trace import new_step_id, new_trace_id
from cloud_tpu_lab.src.profiling.profiler_trace import ProfilerTrace


def main() -> int:
    trace_id = new_trace_id()
    pt = ProfilerTrace(trace_id=trace_id)
    pt.start()

    # Compile (one-shot, before training starts).
    pt.add_event("xla.compile", "compile", dur_s=0.42, tid=0,
                 args={"trace_id": trace_id})

    # 5 training steps with input pipeline + device + collective.
    for i in range(5):
        step_id = new_step_id()
        pt.add_event(f"input.load_batch", "input_pipeline",
                     dur_s=0.003 if i > 0 else 0.012, tid=1,
                     args={"trace_id": trace_id, "step_id": step_id})
        pt.add_event(f"step.{i}", "device",
                     dur_s=0.015 + (0.020 if i == 0 else 0.0), tid=2,
                     args={"trace_id": trace_id, "step_id": step_id})
        pt.add_event("collective.all_reduce", "collective",
                     dur_s=0.002, tid=3,
                     args={"trace_id": trace_id, "step_id": step_id})

    pt.add_event("checkpoint.save", "checkpoint", dur_s=0.080, tid=4,
                 args={"trace_id": trace_id})

    out = Path("cloud_tpu_lab/artifacts/traces") / f"demo_{trace_id}.json"
    pt.write_chrome_json(out)
    print(f"Wrote {out}")
    print(f"Breakdown by category: {pt.breakdown_by_cat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
