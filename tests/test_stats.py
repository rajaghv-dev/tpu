"""Tests for observe/stats.py — pure numpy, no JAX required."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from observe.stats import (
    HIGH_VARIANCE_CV_PCT,
    TimingStats,
    _iterative_sigma_mask,
    compute_timing_stats,
    throughput_stats,
)


class TestIterativeSigmaMask:
    def test_no_outliers(self):
        arr = np.array([10.0, 10.1, 9.9, 10.05, 10.02])
        mask = _iterative_sigma_mask(arr)
        assert not mask.any(), "No values should be flagged as outliers"

    def test_single_extreme_outlier(self):
        arr = np.array([10.0, 10.1, 9.9, 10.05, 1000.0])
        mask = _iterative_sigma_mask(arr)
        # The 1000.0 value must be flagged
        assert mask[-1], "Extreme outlier at index 4 should be flagged"
        # Others should not
        assert not mask[:-1].any(), "Non-outlier values should not be flagged"

    def test_uniform_array_no_outliers(self):
        arr = np.ones(50) * 5.0
        mask = _iterative_sigma_mask(arr)
        assert not mask.any()

    def test_very_small_array_untouched(self):
        # With fewer than 4 elements, the loop body never executes
        arr = np.array([1.0, 2.0, 3.0])
        mask = _iterative_sigma_mask(arr)
        assert not mask.any()

    def test_custom_sigma(self):
        arr = np.array([10.0, 10.0, 10.0, 10.0, 14.0])
        mask_tight = _iterative_sigma_mask(arr, sigma=1.0)
        mask_loose = _iterative_sigma_mask(arr, sigma=5.0)
        # With sigma=1.0 the 14.0 should be flagged; with sigma=5.0 it may not be
        assert mask_tight[-1]
        assert not mask_loose[-1]


class TestComputeTimingStats:
    def _stable_timings(self, n: int = 300, mean: float = 10.0, noise: float = 0.1):
        rng = np.random.default_rng(42)
        return list(rng.normal(mean, noise, n))

    def test_basic_fields_populated(self):
        timings = self._stable_timings()
        stats = compute_timing_stats(timings)
        assert isinstance(stats, TimingStats)
        assert stats.mean_ms > 0
        assert stats.std_ms >= 0
        assert stats.cv_pct >= 0
        assert stats.p50_ms > 0
        assert stats.p95_ms >= stats.p50_ms
        assert stats.p99_ms >= stats.p95_ms
        assert stats.n_valid > 0
        assert stats.n_outliers >= 0

    def test_low_variance_not_flagged(self):
        timings = self._stable_timings(n=300, mean=10.0, noise=0.05)
        stats = compute_timing_stats(timings)
        assert not stats.high_variance, f"CV={stats.cv_pct}% should be < 10%"

    def test_high_variance_flagged(self):
        rng = np.random.default_rng(7)
        # CV = std/mean * 100 = 5/10 * 100 = 50% → high variance
        timings = list(rng.normal(10.0, 5.0, 300))
        stats = compute_timing_stats(timings)
        assert stats.high_variance, f"CV={stats.cv_pct}% should be >= {HIGH_VARIANCE_CV_PCT}%"

    def test_mean_close_to_true_mean(self):
        rng = np.random.default_rng(0)
        true_mean = 8.0
        timings = list(rng.normal(true_mean, 0.05, 300))
        stats = compute_timing_stats(timings)
        assert abs(stats.mean_ms - true_mean) < 0.1

    def test_outlier_removal_improves_accuracy(self):
        rng = np.random.default_rng(42)
        true_mean = 10.0
        timings = list(rng.normal(true_mean, 0.1, 295)) + [500.0, 600.0, 700.0, 800.0, 900.0]
        stats = compute_timing_stats(timings)
        assert stats.n_outliers > 0, "Extreme outliers should be removed"
        assert abs(stats.mean_ms - true_mean) < 1.0, "Mean should be close to true_mean after removal"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_timing_stats([])

    def test_single_value(self):
        stats = compute_timing_stats([42.0])
        assert stats.mean_ms == pytest.approx(42.0)
        assert stats.std_ms == 0.0
        assert stats.n_valid == 1

    def test_all_same_values(self):
        timings = [5.0] * 100
        stats = compute_timing_stats(timings)
        assert stats.mean_ms == pytest.approx(5.0)
        assert stats.cv_pct == pytest.approx(0.0)
        assert not stats.high_variance

    def test_p50_p95_p99_ordering(self):
        rng = np.random.default_rng(1)
        timings = list(rng.exponential(10.0, 300))
        stats = compute_timing_stats(timings)
        assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms


class TestThroughputStats:
    def test_basic_throughput(self):
        # 32 samples / 10ms per batch → 3200 samples/sec
        timings = [10.0] * 100  # constant 10ms
        result = throughput_stats(timings, batch_size=32)
        assert result["throughput_mean_samples_sec"] == pytest.approx(3200.0, rel=0.01)

    def test_std_propagation(self):
        rng = np.random.default_rng(42)
        timings = list(rng.normal(10.0, 0.1, 300))
        result = throughput_stats(timings, batch_size=16)
        assert result["throughput_std_samples_sec"] >= 0
        assert result["throughput_mean_samples_sec"] > 0

    def test_returns_required_keys(self):
        timings = [5.0] * 100
        result = throughput_stats(timings, batch_size=8)
        assert "throughput_mean_samples_sec" in result
        assert "throughput_std_samples_sec" in result
        assert "throughput_cv_pct" in result
