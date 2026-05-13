"""
HBM (High-Bandwidth Memory) simulator.

The HBM model tracks allocations by name and category so we can answer:

  * Are we over capacity?
  * How much of HBM is parameters vs. optimizer state vs. activations?
  * How much bandwidth do we need at this batch size?

For training the rough budget is:

    parameters         × 1   (one copy)
    gradients          × 1   (one copy, same shape as params)
    optimizer state    × 2   (Adam: m, v — two copies of param shape)
    activations        × O(batch × seq × hidden × n_layers)
    workspace          × small

Activation memory is the one that scales with batch size & seq_len, so it
dominates for any non-trivial transformer. This is why mixed-precision +
gradient checkpointing matter so much for big models on small chips.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Allocation:
    name: str
    category: str          # parameters | gradients | optimizer | activations | workspace
    bytes: int


@dataclass
class HbmSimulator:
    capacity_bytes: int    # total HBM available on the chip (catalog × 1e9)
    bandwidth_bytes_s: float
    allocations: List[Allocation] = field(default_factory=list)
    oom_events: int = 0

    # ── Accounting ──────────────────────────────────────────────────────────

    def used_bytes(self) -> int:
        return sum(a.bytes for a in self.allocations)

    def free_bytes(self) -> int:
        return max(self.capacity_bytes - self.used_bytes(), 0)

    def utilization(self) -> float:
        return self.used_bytes() / max(self.capacity_bytes, 1)

    def by_category(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in self.allocations:
            out[a.category] = out.get(a.category, 0) + a.bytes
        return out

    # ── Operations ──────────────────────────────────────────────────────────

    def allocate(self, name: str, category: str, n_bytes: int) -> bool:
        """Returns True on success, False on OOM (and records the event)."""
        if self.used_bytes() + n_bytes > self.capacity_bytes:
            self.oom_events += 1
            return False
        self.allocations.append(Allocation(name=name, category=category, bytes=n_bytes))
        return True

    def free(self, name: str) -> None:
        self.allocations = [a for a in self.allocations if a.name != name]

    def reset(self) -> None:
        self.allocations.clear()
        self.oom_events = 0


# ── Convenience builders ─────────────────────────────────────────────────────


def make_hbm_for_spec(spec, hbm_efficiency: float = 0.7) -> HbmSimulator:
    """Build an HbmSimulator from a TpuSpec."""
    return HbmSimulator(
        capacity_bytes=int(spec.hbm_per_chip_gb * 1e9),
        bandwidth_bytes_s=spec.hbm_bandwidth_gbps * 1e9 * hbm_efficiency,
    )
