"""
Tensor partitioner — splits a logical tensor into shards according to a
`PartitionSpec` over a `Mesh`.

This is a simulator: we don't move real bytes. We just compute the shard
shape and emit one `shard_id` per shard, plus the device(s) that would
own each shard. Useful for showing in dashboards:

    "embedding_table (2048×768) — sharded(data) on 4-chip mesh →
     4 shards of (512×768), one per chip"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..common.trace import new_shard_id, new_tensor_id
from .mesh import Mesh, PartitionSpec


@dataclass
class Shard:
    tensor_id: str
    shard_id: str
    device_id: int
    logical_shape: Tuple[int, ...]
    shard_shape: Tuple[int, ...]
    sharded_axes: Tuple[str, ...]


def partition_tensor(
    logical_shape: Tuple[int, ...],
    spec: PartitionSpec,
    mesh: Mesh,
    tensor_id: str | None = None,
) -> List[Shard]:
    """Return one Shard per device in the mesh for this tensor."""
    if len(spec.spec) != len(logical_shape):
        raise ValueError(
            f"PartitionSpec rank {len(spec.spec)} != tensor rank {len(logical_shape)}"
        )

    tensor_id = tensor_id or new_tensor_id()

    # Per-dim shard count from the spec.
    shard_factors: List[int] = []
    sharded_axes: List[str] = []
    for dim_spec in spec.spec:
        if dim_spec is None:
            shard_factors.append(1)
        else:
            shard_factors.append(mesh.axis_size(dim_spec))
            sharded_axes.append(dim_spec)

    shard_shape = tuple(
        max(d // f, 1) for d, f in zip(logical_shape, shard_factors)
    )

    shards: List[Shard] = []
    for dev_id in range(mesh.n_devices):
        shards.append(Shard(
            tensor_id=tensor_id,
            shard_id=new_shard_id(),
            device_id=dev_id,
            logical_shape=logical_shape,
            shard_shape=shard_shape,
            sharded_axes=tuple(sharded_axes),
        ))
    return shards
