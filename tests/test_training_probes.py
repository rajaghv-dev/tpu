"""Tests for training-specific probes (TrainingMetricsProbe, StepTimingProbe,
CheckpointProbe) and the new step-level Probe hooks added in Stage 1.6."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from observe.checkpoint_probe import CheckpointProbe
from observe.probe import (
    Probe,
    clear_probes,
    fanout_after_step,
    fanout_before_step,
    fanout_record_metric,
    register_probe,
)
from observe.step_timing_probe import StepTimingProbe
from observe.training_metrics_probe import TrainingMetricsProbe


# ── Probe ABC step hooks default to no-op ────────────────────────────────────


class TestProbeStepHooksAreNoops:
    """A vanilla Probe subclass without step hooks should not crash on fanout."""

    def test_default_step_hooks_are_safe(self):
        clear_probes()

        class P(Probe):
            name = "vanilla"

        register_probe(P())
        # These would raise AttributeError if the base class didn't define them.
        fanout_before_step(0)
        fanout_after_step(0, {"loss": 1.0})
        fanout_record_metric("eval_loss", 0.5, step=10)
        clear_probes()


# ── TrainingMetricsProbe ─────────────────────────────────────────────────────


class TestTrainingMetricsProbe:
    def test_name(self):
        assert TrainingMetricsProbe.name == "training_metrics"

    def test_after_step_records_scalars(self):
        p = TrainingMetricsProbe()
        p.after_step(0, {"loss": 1.5, "lr": 1.0e-5, "grad_norm": 0.3})
        p.after_step(1, {"loss": 1.4, "lr": 1.0e-5, "grad_norm": 0.28})
        log = p.write_log()
        assert log["n_steps"] == 2
        assert set(log["metric_keys"]) == {"loss", "lr", "grad_norm"}
        assert log["history"][0]["step"] == 0
        assert log["history"][1]["loss"] == 1.4

    def test_non_scalar_values_silently_dropped(self):
        p = TrainingMetricsProbe()
        # A list cannot be cast to float — should be skipped, not raise.
        p.after_step(0, {"loss": 1.0, "logits": [1, 2, 3]})
        log = p.write_log()
        assert log["history"][0].get("loss") == 1.0
        assert "logits" not in log["history"][0]

    def test_record_metric_appends_ad_hoc(self):
        p = TrainingMetricsProbe()
        p.record_metric("eval_loss", 0.42, step=100)
        p.record_metric("eval_accuracy", 0.89, step=100)
        log = p.write_log()
        assert log["ad_hoc"] == [
            {"name": "eval_loss", "value": 0.42, "step": 100},
            {"name": "eval_accuracy", "value": 0.89, "step": 100},
        ]

    def test_string_metric_passes_through(self):
        p = TrainingMetricsProbe()
        p.record_metric("checkpoint_path", "/tmp/ckpt")
        log = p.write_log()
        assert log["ad_hoc"][0]["value"] == "/tmp/ckpt"


# ── StepTimingProbe ──────────────────────────────────────────────────────────


class TestStepTimingProbe:
    def test_name(self):
        assert StepTimingProbe.name == "step_timing"

    def test_records_durations(self):
        p = StepTimingProbe(rolling_window=3)
        for i in range(4):
            p.before_step(i)
            time.sleep(0.005)
            p.after_step(i, {"loss": 1.0, "samples_in_batch": 8, "tokens_in_batch": 1024})
        log = p.write_log()
        assert log["n_steps"] == 4
        assert log["first_step_s"] >= 0.004
        assert log["median_throughput_samples_sec"] is not None
        assert log["median_throughput_samples_sec"] > 0
        assert log["median_throughput_tokens_sec"] is not None

    def test_rolling_throughput(self):
        p = StepTimingProbe(rolling_window=2)
        for i in range(3):
            p.before_step(i)
            time.sleep(0.002)
            p.after_step(i, {"samples_in_batch": 4})
        rolling = p.rolling_throughput()
        assert rolling["samples_sec"] is not None
        assert rolling["tokens_sec"] is None  # we never supplied tokens

    def test_handles_no_steps_gracefully(self):
        p = StepTimingProbe()
        log = p.write_log()
        assert log["n_steps"] == 0
        assert log["median_step_s"] is None

    def test_after_step_without_before_is_safe(self):
        # Should not crash if a runner skips before_step (degraded mode).
        p = StepTimingProbe()
        p.after_step(0, {"loss": 1.0})
        log = p.write_log()
        # No duration was recorded — n_steps stays 0.
        assert log["n_steps"] == 0


# ── CheckpointProbe ──────────────────────────────────────────────────────────


class TestCheckpointProbe:
    def test_name(self):
        assert CheckpointProbe.name == "checkpoint"

    def test_pairs_metric_records_by_step(self, tmp_path: Path):
        p = CheckpointProbe()
        p.before_run("rid", None, tmp_path)
        p.record_metric("checkpoint_write", 0.5, step=100)
        p.record_metric("checkpoint_size_bytes", 1024, step=100)
        p.record_metric("checkpoint_path", str(tmp_path / "ckpt-100"), step=100)
        log = p.write_log()
        assert log["n_checkpoints"] == 1
        ev = log["events"][0]
        assert ev["step"] == 100
        assert ev["duration_s"] == 0.5
        assert ev["size_bytes"] == 1024

    def test_unrelated_metrics_ignored(self, tmp_path: Path):
        p = CheckpointProbe()
        p.before_run("rid", None, tmp_path)
        p.record_metric("loss", 0.42, step=10)
        p.record_metric("eval_accuracy", 0.9, step=10)
        log = p.write_log()
        assert log["n_checkpoints"] == 0

    def test_discovered_files_when_runner_silent(self, tmp_path: Path):
        # Runner forgot to call record_metric, but a checkpoint exists on disk.
        p = CheckpointProbe()
        p.before_run("rid", None, tmp_path)
        ckpt_dir = tmp_path / "checkpoints" / "step-50"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "params.npz").write_bytes(b"x" * 4096)

        log = p.write_log()
        assert log["n_checkpoints"] == 0
        assert any(d["path"].endswith("params.npz") for d in log["discovered_files"])
        assert log["total_size_bytes"] == 4096

    def test_partial_event_still_recorded(self, tmp_path: Path):
        # Only duration is supplied, no size, no path. Should still flush.
        p = CheckpointProbe()
        p.before_run("rid", None, tmp_path)
        p.record_metric("checkpoint_write", 1.2, step=50)
        log = p.write_log()
        # Without size or path the event is held pending; flush at write_log.
        assert log["n_checkpoints"] == 1
        assert log["events"][0]["duration_s"] == 1.2


# ── Probe registry fan-out integration ───────────────────────────────────────


class TestStepFanout:
    def test_metrics_probe_receives_after_step(self):
        clear_probes()
        m = TrainingMetricsProbe()
        s = StepTimingProbe()
        register_probe(m)
        register_probe(s)
        for i in range(3):
            fanout_before_step(i)
            time.sleep(0.001)
            fanout_after_step(i, {"loss": 0.1 * (3 - i), "samples_in_batch": 16})

        m_log = m.write_log()
        s_log = s.write_log()
        assert m_log["n_steps"] == 3
        assert s_log["n_steps"] == 3
        clear_probes()

    def test_record_metric_fans_out_to_all_probes(self):
        clear_probes()
        m = TrainingMetricsProbe()
        c = CheckpointProbe()
        register_probe(m)
        register_probe(c)
        # CheckpointProbe needs a log_dir for discovered_files; provide one.
        # before_run isn't fanned out by record_metric so call directly.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            c.before_run("rid", None, Path(td))
            fanout_record_metric("checkpoint_write", 0.3, step=10)
            fanout_record_metric("checkpoint_size_bytes", 100, step=10)
            fanout_record_metric("eval_loss", 0.42, step=10)

            m_log = m.write_log()
            c_log = c.write_log()

            assert any(e["name"] == "eval_loss" for e in m_log["ad_hoc"])
            # CheckpointProbe should NOT have grabbed eval_loss.
            assert c_log["n_checkpoints"] == 1
        clear_probes()

    def test_buggy_probe_does_not_break_fanout(self):
        """Probe exceptions in step hooks must be swallowed (matches phase
        hooks; observability code never fails the run)."""
        clear_probes()

        class BadProbe(Probe):
            name = "bad"
            def before_step(self, step):
                raise RuntimeError("intentional")
            def after_step(self, step, metrics):
                raise RuntimeError("intentional")
            def record_metric(self, name, value, step=None):
                raise RuntimeError("intentional")

        good = TrainingMetricsProbe()
        register_probe(BadProbe())
        register_probe(good)
        # All three should complete without raising:
        fanout_before_step(0)
        fanout_after_step(0, {"loss": 1.0})
        fanout_record_metric("eval_loss", 0.1)
        # Good probe still got the data.
        assert good.write_log()["n_steps"] == 1
        clear_probes()
