"""
Fake PJRT-style compiled executable.

In real JAX / PyTorch-XLA, `pjrt.Executable` is the artifact you call
`execute()` on. We model it here as a tiny bundle: the HLO module it came
from, its executable_id, and a per-op summary the runtime uses to time
execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..xla_sim.fake_hlo import HloModule, HloOp


@dataclass
class CompiledExecutable:
    executable_id: str
    module: HloModule
    # Sharding strategy this executable was compiled against — bookkeeping
    # only, the actual partitioning happens elsewhere in src/sharding/.
    mesh_shape: tuple = (1,)
    dtype: str = "bf16"

    def ops(self) -> List[HloOp]:
        return self.module.ops

    def total_flops(self) -> int:
        return self.module.total_flops()

    def total_bytes_moved(self) -> int:
        return self.module.total_bytes_moved()
