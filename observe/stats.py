"""
Statistical analysis for benchmark timing measurements.

Stage 1 gap fixed: C2 (multi-run statistics with CV check).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# CV threshold above which we flag the measurement as unreliable.
HIGH_VARIANCE_CV_PCT = 10.0


@dataclass
class TimingStats:
    mean_ms: float
    std_ms: float
    cv_pct: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_valid: int
    n_outliers: int
    high_variance: bool


@dataclass
class ThroughputStats:
    mean_samples_sec: float
    std_samples_sec: float
    cv_pct: float


def _iterative_sigma_mask(values: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Iterative MAD-based outlier detection (modified Z-score).

    Uses median and Median Absolute Deviation instead of mean and std so that
    a single extreme outlier does not inflate the threshold and hide itself.

    Returns a boolean mask where True = outlier. Removes at most one outlier
    per iteration (the most extreme) until no value exceeds the threshold.

    Modified Z-score: 0.6745 * |xi - median| / MAD.
    Falls back to 3-sigma when MAD is zero (all remaining values identical).
    """
    mask = np.zeros(len(values), dtype=bool)
    remaining_idx = np.arange(len(values))

    while len(remaining_idx) >= 4:
        vals = values[remaining_idx]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad == 0.0:
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            if std == 0.0:
                break
            scores = np.abs(vals - mean) / std
        else:
            scores = 0.6745 * np.abs(vals - med) / mad

        worst = int(np.argmax(scores))
        if scores[worst] > sigma:
            mask[remaining_idx[worst]] = True
            remaining_idx = np.delete(remaining_idx, worst)
        else:
            break

    return mask


def compute_timing_stats(timings_ms: Sequence[float]) -> TimingStats:
    """
    Compute benchmark statistics from raw timing measurements.

    Applies iterative MAD-based outlier removal (modified Z-score), then
    computes percentile latencies and coefficient of variation.
    Flags high_variance when CV ≥ 10%.

    Args:
        timings_ms: Wall-clock pass durations in milliseconds.

    Returns:
        TimingStats dataclass with all metric fields populated.
    """
    arr = np.asarray(timings_ms, dtype=float)
    if len(arr) == 0:
        raise ValueError("timings_ms must not be empty")

    outlier_mask = _iterative_sigma_mask(arr)
    valid = arr[~outlier_mask]

    # Safety: never discard everything
    if len(valid) == 0:
        valid = arr
        outlier_mask = np.zeros(len(arr), dtype=bool)

    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    cv_pct = (std / mean * 100.0) if mean > 0 else 0.0

    return TimingStats(
        mean_ms=round(mean, 4),
        std_ms=round(std, 4),
        cv_pct=round(cv_pct, 2),
        p50_ms=round(float(np.percentile(valid, 50)), 4),
        p95_ms=round(float(np.percentile(valid, 95)), 4),
        p99_ms=round(float(np.percentile(valid, 99)), 4),
        n_valid=int(len(valid)),
        n_outliers=int(np.sum(outlier_mask)),
        high_variance=cv_pct >= HIGH_VARIANCE_CV_PCT,
    )


def throughput_stats(timings_ms: Sequence[float], batch_size: int) -> ThroughputStats:
    """
    Convert per-batch timing measurements to throughput statistics.

    Args:
        timings_ms: Per-batch wall-clock times in milliseconds.
        batch_size: Number of samples per batch.

    Returns:
        ThroughputStats dataclass with mean, std, and CV.
    """
    stats = compute_timing_stats(timings_ms)
    mean_s = stats.mean_ms / 1000.0
    std_s = stats.std_ms / 1000.0
    if mean_s <= 0:
        return ThroughputStats(
            mean_samples_sec=0.0,
            std_samples_sec=0.0,
            cv_pct=0.0,
        )
    # Error propagation: σ_tp ≈ (batch_size / mean_s²) * σ_s
    mean_tp = batch_size / mean_s
    std_tp = (batch_size / (mean_s ** 2)) * std_s
    return ThroughputStats(
        mean_samples_sec=round(mean_tp, 1),
        std_samples_sec=round(std_tp, 1),
        cv_pct=stats.cv_pct,
    )
