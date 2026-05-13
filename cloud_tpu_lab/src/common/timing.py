"""
Tiny timing helper — `perf_counter`-based stopwatches that don't require any
external dep. Used everywhere we need a phase duration in the simulation.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator


@dataclass
class Stopwatch:
    """Cumulative timer with named segments."""
    segments_s: Dict[str, float] = field(default_factory=dict)
    _starts: Dict[str, float] = field(default_factory=dict)

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if name not in self._starts:
            raise KeyError(f"Stopwatch.stop({name!r}) called without start")
        dur = time.perf_counter() - self._starts.pop(name)
        self.segments_s[name] = self.segments_s.get(name, 0.0) + dur
        return dur

    @contextmanager
    def segment(self, name: str) -> Iterator[None]:
        """`with sw.segment("xla.compile"):` — accumulates into `segments_s`."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def total_s(self) -> float:
        return sum(self.segments_s.values())
