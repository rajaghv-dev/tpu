"""
Collective communication cost model.

We use a simplified `α + βn` model where each collective has a fixed
latency `α` (link-level latency × ring distance) and a per-byte cost `β`
(inverse of effective ICI bandwidth).

For an all-reduce on a ring of N chips with a payload of B bytes:

    t_allreduce ≈ 2 * (N - 1) / N * B / ICI_BW

That's the standard ring-allreduce bandwidth-optimal formula. Reality is
messier (mesh vs ring, hierarchical reductions, async with compute), but
this is the right ballpark for teaching.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..common.trace import new_collective_id


@dataclass
class CollectiveCost:
    collective_id: str
    kind: str          # all_reduce | all_gather | reduce_scatter
    payload_bytes: int
    n_participants: int
    sim_duration_s: float


def all_reduce_time(
    payload_bytes: int,
    n_chips: int,
    ici_bandwidth_bytes_s: float,
    link_latency_s: float = 2e-6,
) -> CollectiveCost:
    if n_chips <= 1:
        return CollectiveCost(
            collective_id=new_collective_id(),
            kind="all_reduce",
            payload_bytes=payload_bytes,
            n_participants=n_chips,
            sim_duration_s=0.0,
        )
    # Ring-AllReduce bandwidth-optimal: each chip sends and receives
    # 2(N-1)/N * B bytes total.
    bw_s = (2.0 * (n_chips - 1) / n_chips) * payload_bytes / max(ici_bandwidth_bytes_s, 1.0)
    lat_s = link_latency_s * (n_chips - 1)
    return CollectiveCost(
        collective_id=new_collective_id(),
        kind="all_reduce",
        payload_bytes=payload_bytes,
        n_participants=n_chips,
        sim_duration_s=lat_s + bw_s,
    )


def all_gather_time(
    payload_bytes: int, n_chips: int, ici_bandwidth_bytes_s: float,
    link_latency_s: float = 2e-6,
) -> CollectiveCost:
    if n_chips <= 1:
        return CollectiveCost(
            collective_id=new_collective_id(), kind="all_gather",
            payload_bytes=payload_bytes, n_participants=n_chips,
            sim_duration_s=0.0,
        )
    bw_s = ((n_chips - 1) / n_chips) * (payload_bytes * n_chips) / max(ici_bandwidth_bytes_s, 1.0)
    lat_s = link_latency_s * (n_chips - 1)
    return CollectiveCost(
        collective_id=new_collective_id(), kind="all_gather",
        payload_bytes=payload_bytes, n_participants=n_chips,
        sim_duration_s=lat_s + bw_s,
    )


def reduce_scatter_time(
    payload_bytes: int, n_chips: int, ici_bandwidth_bytes_s: float,
    link_latency_s: float = 2e-6,
) -> CollectiveCost:
    if n_chips <= 1:
        return CollectiveCost(
            collective_id=new_collective_id(), kind="reduce_scatter",
            payload_bytes=payload_bytes, n_participants=n_chips,
            sim_duration_s=0.0,
        )
    bw_s = ((n_chips - 1) / n_chips) * payload_bytes / max(ici_bandwidth_bytes_s, 1.0)
    lat_s = link_latency_s * (n_chips - 1)
    return CollectiveCost(
        collective_id=new_collective_id(), kind="reduce_scatter",
        payload_bytes=payload_bytes, n_participants=n_chips,
        sim_duration_s=lat_s + bw_s,
    )
