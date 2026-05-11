"""
Tests for observe/cloud_monitoring_probe.py.

We only exercise the no-op path (GCP libs/creds absent) plus the pure
aggregation function. The live polling path requires real GCP creds + a
running TPU, which CI does not have.
"""
from __future__ import annotations

import builtins
import sys
from pathlib import Path
from unittest import mock

import pytest

from observe.cloud_monitoring_probe import (
    CloudMonitoringProbe,
    aggregate_per_phase_summary,
)


# ── No-op fallback construction ──────────────────────────────────────────────

def _force_noop_env(monkeypatch):
    """Strip env vars and gcloud so the probe has no project/tpu/zone source."""
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.delenv("TPU_ZONE", raising=False)
    # Block the gcloud subprocess fallback by making it raise.
    monkeypatch.setattr(
        "observe.cloud_monitoring_probe._gcloud_default_project",
        lambda: None,
    )
    # Block the state.env fallback by pointing at a guaranteed-missing path.
    monkeypatch.setattr(
        "observe.cloud_monitoring_probe._STATE_ENV_PATH",
        Path("/tmp/__definitely_does_not_exist_state__.env"),
    )


class TestConstructionNoCreds:
    def test_constructs_when_no_env_no_state(self, monkeypatch):
        _force_noop_env(monkeypatch)
        probe = CloudMonitoringProbe()
        assert probe._available is False
        assert probe._project is None
        assert probe._tpu_name is None
        assert probe._zone is None

    def test_constructs_when_monitoring_v3_missing(self, monkeypatch):
        # Provide identifiers so we get past the early bail-out, then make
        # the import fail.
        monkeypatch.setenv("GCP_PROJECT", "my-proj")
        monkeypatch.setenv("TPU_NAME", "my-tpu")
        monkeypatch.setenv("TPU_ZONE", "us-central1-a")

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("google.cloud") or name == "google.cloud.monitoring_v3":
                raise ImportError(f"simulated missing: {name}")
            return real_import(name, *args, **kwargs)

        # Make sure no cached module satisfies the import.
        monkeypatch.delitem(sys.modules, "google.cloud.monitoring_v3", raising=False)
        with mock.patch.object(builtins, "__import__", side_effect=fake_import):
            probe = CloudMonitoringProbe()
        assert probe._available is False
        # Identifiers still captured so write_log can report them.
        assert probe._project == "my-proj"
        assert probe._tpu_name == "my-tpu"
        assert probe._zone == "us-central1-a"


class TestNoopHooks:
    """All lifecycle hooks must be safely callable on a no-op probe."""

    def test_all_hooks_are_noops(self, monkeypatch, tmp_path):
        _force_noop_env(monkeypatch)
        probe = CloudMonitoringProbe()
        assert probe._available is False

        # before_run: must NOT spawn a thread on a no-op probe.
        probe.before_run("run-id-x", config=object(), log_dir=tmp_path)
        assert probe._thread is None

        # phase hooks: just toggle _current_phase, never raise.
        probe.before_phase("compile")
        assert probe._current_phase == "compile"
        probe.after_phase("compile", duration_s=1.23)
        assert probe._current_phase == "compile"
        probe.on_error("latency", RuntimeError("boom"))
        assert probe._current_phase == "latency"

        # after_run: must be safe with no thread to join.
        probe.after_run("run-id-x", result=None)


class TestWriteLog:
    def test_returns_dict_with_available_false(self, monkeypatch):
        _force_noop_env(monkeypatch)
        probe = CloudMonitoringProbe()
        log = probe.write_log()
        assert isinstance(log, dict)
        assert log["available"] is False
        assert log["polling_interval_s"] == 1.0
        assert log["n_samples"] == 0
        assert log["samples"] == []
        assert log["per_phase_summary"] == {}
        # Identity fields present (None on no-op).
        assert "tpu_name" in log
        assert "zone" in log
        assert "project" in log


# ── Aggregation function ─────────────────────────────────────────────────────

class TestAggregation:
    def test_empty_input(self):
        assert aggregate_per_phase_summary([]) == {}

    def test_single_phase_single_metric(self):
        rows = [
            {"ts": 1.0, "phase": "compile", "metric": "mxu_utilization", "value": 10.0},
            {"ts": 2.0, "phase": "compile", "metric": "mxu_utilization", "value": 20.0},
            {"ts": 3.0, "phase": "compile", "metric": "mxu_utilization", "value": 30.0},
        ]
        out = aggregate_per_phase_summary(rows)
        assert out == {
            "compile": {
                "mxu_utilization": {"min": 10.0, "mean": 20.0, "max": 30.0},
            },
        }

    def test_multiple_phases_multiple_metrics(self):
        rows = [
            {"ts": 1.0, "phase": "compile",  "metric": "mxu_utilization",   "value": 5.0},
            {"ts": 2.0, "phase": "compile",  "metric": "mxu_utilization",   "value": 15.0},
            {"ts": 3.0, "phase": "compile",  "metric": "memory_utilization","value": 50.0},
            {"ts": 4.0, "phase": "latency",  "metric": "mxu_utilization",   "value": 80.0},
            {"ts": 5.0, "phase": "latency",  "metric": "mxu_utilization",   "value": 90.0},
        ]
        out = aggregate_per_phase_summary(rows)

        assert out["compile"]["mxu_utilization"] == {"min": 5.0, "mean": 10.0, "max": 15.0}
        assert out["compile"]["memory_utilization"] == {
            "min": 50.0, "mean": 50.0, "max": 50.0,
        }
        assert out["latency"]["mxu_utilization"] == {
            "min": 80.0, "mean": 85.0, "max": 90.0,
        }

    def test_skips_nan_and_inf(self):
        rows = [
            {"ts": 1.0, "phase": "x", "metric": "m", "value": 1.0},
            {"ts": 2.0, "phase": "x", "metric": "m", "value": float("nan")},
            {"ts": 3.0, "phase": "x", "metric": "m", "value": float("inf")},
            {"ts": 4.0, "phase": "x", "metric": "m", "value": 3.0},
        ]
        out = aggregate_per_phase_summary(rows)
        assert out["x"]["m"] == {"min": 1.0, "mean": 2.0, "max": 3.0}

    def test_skips_malformed_rows(self):
        rows = [
            {"ts": 1.0, "phase": "x", "metric": "m", "value": 7.0},
            {"phase": "x", "metric": "m"},               # missing value
            {"ts": 2.0, "phase": "x", "value": 9.0},     # missing metric
            {"ts": 3.0, "phase": "x", "metric": "m", "value": "not-a-number"},
        ]
        out = aggregate_per_phase_summary(rows)
        # Only the first row contributes.
        assert out == {"x": {"m": {"min": 7.0, "mean": 7.0, "max": 7.0}}}

    def test_float_coercion_from_int_values(self):
        rows = [
            {"ts": 1, "phase": "p", "metric": "m", "value": 1},
            {"ts": 2, "phase": "p", "metric": "m", "value": 3},
        ]
        out = aggregate_per_phase_summary(rows)
        s = out["p"]["m"]
        assert s["min"] == 1.0 and s["max"] == 3.0
        assert s["mean"] == pytest.approx(2.0)
