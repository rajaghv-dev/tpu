"""
Fake HLO — a minimal data model that mimics XLA's HLO IR for teaching.

Real HLO is much richer (sharding annotations, dataflow, async, custom
calls). We only model what's needed to demonstrate:

  * Lowering: a high-level layer like `Linear(in=64, out=128)` becomes a
    sequence of `DotGeneral` + `Add` HLO ops.
  * Traceability: each HLO op carries `model_layer_id` so a profiler can
    correlate device-time back to the layer that produced it.
  * Cost: each op carries `flops` and `bytes_in/out` so the runtime can
    estimate step time without a real device.

HloOpKind enum mirrors a tiny subset of XLA op names so the strings look
familiar when you see them in real HLO dumps later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class HloOpKind(str, Enum):
    DOT_GENERAL = "DotGeneral"
    ADD = "Add"
    MUL = "Multiply"
    BROADCAST = "Broadcast"
    REDUCE = "Reduce"
    REDUCE_MEAN = "ReduceMean"
    VARIANCE = "Variance"
    NORMALIZE = "Normalize"
    SOFTMAX = "Softmax"
    CONVOLUTION = "Convolution"
    TRANSPOSE = "Transpose"
    RESHAPE = "Reshape"
    PARAMETER = "Parameter"
    CONSTANT = "Constant"
    ALL_REDUCE = "AllReduce"
    ALL_GATHER = "AllGather"
    REDUCE_SCATTER = "ReduceScatter"


@dataclass
class HloOp:
    """One fake HLO op."""
    op_id: str
    kind: HloOpKind
    shape: Tuple[int, ...]            # output shape
    dtype: str                        # bf16 | fp32 | int32 ...
    inputs: List[str] = field(default_factory=list)  # op_ids of producers
    # Traceability — back to the model.
    model_layer_id: Optional[str] = None
    layer_name: Optional[str] = None
    # Cost estimates (used by the runtime simulator).
    flops: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    # Free-form metadata for the dashboard.
    metadata: dict = field(default_factory=dict)


@dataclass
class HloModule:
    """A whole compiled model — entry_op is the output of the graph."""
    name: str
    ops: List[HloOp]
    entry_op_id: str
    # Lineage back to the source model.
    model_name: str = "unknown"

    def op_count_by_kind(self) -> dict:
        out: dict = {}
        for op in self.ops:
            out[op.kind.value] = out.get(op.kind.value, 0) + 1
        return out

    def total_flops(self) -> int:
        return sum(op.flops for op in self.ops)

    def total_bytes_moved(self) -> int:
        return sum(op.bytes_in + op.bytes_out for op in self.ops)
