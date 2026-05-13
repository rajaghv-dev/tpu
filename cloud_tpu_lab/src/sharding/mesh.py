"""
Device mesh — the SPMD substrate.

A mesh is an N-D arrangement of devices with named axes. Common shapes:

    1D: [4]               name=("data",)            — pure data-parallel
    2D: [2, 2]            name=("data", "model")    — DP × TP
    3D: [2, 2, 2]         name=("data", "model", "pipeline")

Sharding annotations on tensors are expressed as `PartitionSpec` —
tuple of (axis_name | None) per tensor dim.

Examples:
    PartitionSpec(None,)          replicated
    PartitionSpec("data",)        sharded along the "data" axis
    PartitionSpec("data","model") sharded along two axes

This module is the substrate; `partitioner.py` does the actual placement
and `all_reduce.py` simulates collective cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Mesh:
    shape: Tuple[int, ...]
    axis_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.axis_names):
            raise ValueError(
                f"shape and axis_names must have same length, "
                f"got {self.shape} and {self.axis_names}"
            )

    @property
    def n_devices(self) -> int:
        return reduce(mul, self.shape, 1)

    def axis_size(self, name: str) -> int:
        idx = self.axis_names.index(name)
        return self.shape[idx]


@dataclass(frozen=True)
class PartitionSpec:
    """
    One entry per tensor dim. None = replicated on that dim; "axis_name"
    = sharded along that mesh axis.
    """
    spec: Tuple[Optional[str], ...]

    def is_replicated(self) -> bool:
        return all(s is None for s in self.spec)


def make_1d_mesh(n_devices: int, name: str = "data") -> Mesh:
    return Mesh(shape=(n_devices,), axis_names=(name,))


def make_2d_mesh(rows: int, cols: int,
                 names: Tuple[str, str] = ("data", "model")) -> Mesh:
    return Mesh(shape=(rows, cols), axis_names=names)
