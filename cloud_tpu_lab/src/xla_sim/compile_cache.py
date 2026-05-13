"""
Fake XLA compile cache — keyed on (HLO ops, shapes, dtype, mesh).

Used by the demo to show:
  * first-step compile cost  (cache miss → "compile time" in trace)
  * subsequent-step cache hit (≈ zero compile time)
  * shape change → cache miss (the "silent recompile" footgun)

The numbers (`base_compile_s`, `flops_factor`) are deliberately a simulation
model — not measured TPU compile times. They're tuned so the simulator
produces dashboard plots that look qualitatively right.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .fake_hlo import HloModule


@dataclass
class CompileResult:
    cache_key: str
    compile_time_s: float
    cache_hit: bool
    executable_id: str
    op_count: int
    total_flops: int


@dataclass
class CompileCache:
    """In-memory cache. Maps cache_key → executable_id."""
    entries: Dict[str, str] = field(default_factory=dict)
    # Counters surfaced to the dashboard.
    hits: int = 0
    misses: int = 0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "entries": len(self.entries)}


def _compute_key(module: HloModule, mesh_shape: tuple, dtype: str) -> str:
    h = hashlib.sha256()
    h.update(module.model_name.encode())
    h.update(dtype.encode())
    h.update(str(mesh_shape).encode())
    for op in module.ops:
        h.update(op.kind.value.encode())
        h.update(str(op.shape).encode())
        h.update(op.dtype.encode())
    return h.hexdigest()[:16]


def compile_hlo(
    module: HloModule,
    cache: CompileCache,
    mesh_shape: tuple = (1,),
    dtype: str = "bf16",
    base_compile_s: float = 0.30,
    flops_factor: float = 1e-12,
) -> CompileResult:
    """
    Simulate XLA compilation.

    Compile time model (cold compile):
        t = base_compile_s + flops_factor * total_flops

    Cache hit returns ≈0 immediately.

    We don't actually `time.sleep()` — the demo would be too slow. Instead
    we *record* the simulated compile cost so it shows up in the profiler
    trace, but real wall-clock is just a microsecond.
    """
    from ..common.trace import new_executable_id  # local import to avoid cycle

    key = _compute_key(module, mesh_shape, dtype)
    if key in cache.entries:
        cache.hits += 1
        return CompileResult(
            cache_key=key,
            compile_time_s=0.0,
            cache_hit=True,
            executable_id=cache.entries[key],
            op_count=len(module.ops),
            total_flops=module.total_flops(),
        )

    # Cache miss → simulated compile cost. (We don't actually sleep; the
    # profiler trace records the simulated cost so it shows up in plots.)
    sim_compile_s = base_compile_s + flops_factor * module.total_flops()
    exe_id = new_executable_id()
    cache.entries[key] = exe_id
    cache.misses += 1

    return CompileResult(
        cache_key=key,
        compile_time_s=sim_compile_s,
        cache_hit=False,
        executable_id=exe_id,
        op_count=len(module.ops),
        total_flops=module.total_flops(),
    )
