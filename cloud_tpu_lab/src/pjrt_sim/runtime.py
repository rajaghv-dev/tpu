"""
Fake PJRT-style runtime.

Real PJRT is the runtime layer that owns devices, manages buffers, and
calls into the platform-specific executable (TPU / GPU / CPU). Our fake:

  * holds a list of `TpuDevice`s
  * executes a `CompiledExecutable` by walking its ops in order
  * times each op via the device's roofline model
  * emits one `device_event_id` per op execution to the profiler
  * tracks total flops + bytes moved per step

The runtime is single-threaded and synchronous — perfectly fine for a
teaching simulator. Async / streams / overlap can be a later module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..common.trace import TraceContext, new_device_event_id
from ..xla_sim.fake_hlo import HloOp
from .device import TpuDevice
from .executable import CompiledExecutable


@dataclass
class OpExecutionRecord:
    """One op execution — emitted to the profiler trace."""
    device_event_id: str
    device_id: int
    op_id: str
    kind: str
    layer_name: Optional[str]
    model_layer_id: Optional[str]
    sim_duration_s: float
    flops: int
    bytes_moved: int


@dataclass
class PjrtRuntime:
    devices: List[TpuDevice]
    # Sink for op-execution records (the profiler reads this).
    op_sink: List[OpExecutionRecord] = field(default_factory=list)

    def execute(
        self,
        executable: CompiledExecutable,
        trace: TraceContext,
        log_event: Optional[Callable[[str, dict], None]] = None,
    ) -> float:
        """
        Run the executable once. Returns simulated step time in seconds.

        `log_event(event, fields)` is the optional structured-log callback.
        """
        primary = self.devices[0]
        step_t = 0.0
        for op in executable.ops():
            ev_id = new_device_event_id()
            op_dur = primary.roofline_op_time_s(op.flops, op.bytes_in + op.bytes_out)
            step_t += op_dur
            self.op_sink.append(OpExecutionRecord(
                device_event_id=ev_id,
                device_id=primary.device_id,
                op_id=op.op_id,
                kind=op.kind.value,
                layer_name=op.layer_name,
                model_layer_id=op.model_layer_id,
                sim_duration_s=op_dur,
                flops=op.flops,
                bytes_moved=op.bytes_in + op.bytes_out,
            ))
            if log_event is not None:
                log_event("tpu.execute_op", {
                    **trace.as_log_fields(),
                    "device_event_id": ev_id,
                    "hlo_op_id": op.op_id,
                    "model_layer_id": op.model_layer_id,
                    "kind": op.kind.value,
                    "sim_duration_s": op_dur,
                    "flops": op.flops,
                    "bytes_moved": op.bytes_in + op.bytes_out,
                })
        return step_t
