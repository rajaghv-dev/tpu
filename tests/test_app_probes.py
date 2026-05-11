"""Tests for the application-layer probes (timing, memory, input_fingerprint)."""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


# ── Fake JAX installation (matches test_runner.py) ────────────────────────────

def _install_fake_jax():
    """Register a minimal fake jax module before importing runner."""
    if "jax" in sys.modules:
        return sys.modules["jax"]
    jax_mod = types.ModuleType("jax")
    jnp_mod = types.ModuleType("jax.numpy")

    jnp_mod.bfloat16 = np.float32
    jnp_mod.float32 = np.float32

    jax_mod.numpy = jnp_mod
    jax_mod.local_devices = lambda: [object()]
    jax_mod.block_until_ready = lambda x: x
    jax_mod.jit = lambda fn: fn
    jax_mod.tree_util = types.ModuleType("jax.tree_util")
    jax_mod.tree_util.tree_map = lambda fn, tree: fn(tree)

    sys.modules.setdefault("jax", jax_mod)
    sys.modules.setdefault("jax.numpy", jnp_mod)

    return jax_mod


_install_fake_jax()

from benchmarks.runner import ExperimentConfig  # noqa: E402
from observe.input_fingerprint import InputFingerprintProbe  # noqa: E402
from observe.memory_probe import MemoryProbe  # noqa: E402
from observe.timing_probe import TimingProbe  # noqa: E402


# ── Config factory ────────────────────────────────────────────────────────────

def _base_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        model_id="bert_base",
        hf_id="bert-base-uncased",
        task="sequence-classification",
        domain="nlp_encoder",
        architecture_family="transformer_encoder",
        attention_variant="mha",
        positional_encoding="absolute",
        is_moe=False,
        total_params_M=110,
        active_params_M=110,
        input_type="text",
        precision="bf16",
        framework="jax",
        device="tpu",
        seq_len=128,
        batch_size_latency=1,
        batch_size_throughput=8,
        input_seed=42,
        vocab_size=30522,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ── TimingProbe ───────────────────────────────────────────────────────────────

class TestTimingProbe:
    def test_name(self):
        assert TimingProbe.name == "timing"

    def test_basic_lifecycle_records_total_run(self, tmp_path):
        probe = TimingProbe()
        cfg = _base_config()
        probe.before_run("rid", cfg, tmp_path)
        # Sleep just enough to make `total_run_s > 0` reliably.
        time.sleep(0.01)
        probe.after_run("rid", {})
        log = probe.write_log()

        assert isinstance(log, dict)
        assert log["total_run_s"] > 0
        assert "timeline" in log
        assert "phase_summary" in log

    def test_phase_calls_populate_timeline(self, tmp_path):
        probe = TimingProbe()
        cfg = _base_config()
        probe.before_run("rid", cfg, tmp_path)
        for ph in ("preflight", "compile", "latency"):
            probe.before_phase(ph)
            time.sleep(0.001)
            probe.after_phase(ph, 0.001)
        probe.after_run("rid", {})
        log = probe.write_log()

        phases = [e["phase"] for e in log["timeline"]]
        assert phases == ["preflight", "compile", "latency"]
        for entry in log["timeline"]:
            assert entry["duration_s"] is not None
            assert "ts" in entry

    def test_phase_summary_keyed_by_name(self, tmp_path):
        probe = TimingProbe()
        cfg = _base_config()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("compile")
        probe.after_phase("compile", 0.5)
        probe.after_run("rid", {})
        log = probe.write_log()

        assert "compile" in log["phase_summary"]
        assert log["phase_summary"]["compile"]["duration_s"] == 0.5

    def test_on_error_recorded(self, tmp_path):
        probe = TimingProbe()
        cfg = _base_config()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("model_load")
        probe.on_error("model_load", RuntimeError("boom"))
        probe.after_run("rid", None)
        log = probe.write_log()

        err_entries = [e for e in log["timeline"] if e.get("error")]
        assert len(err_entries) == 1
        assert err_entries[0]["phase"] == "model_load"

    def test_write_log_before_run_returns_none(self):
        # write_log called without before_run must not crash.
        probe = TimingProbe()
        assert probe.write_log() is None


# ── MemoryProbe ───────────────────────────────────────────────────────────────

class TestMemoryProbe:
    def test_name(self):
        assert MemoryProbe.name == "memory"

    def test_lifecycle_no_crash(self, tmp_path):
        probe = MemoryProbe()
        cfg = _base_config()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("preflight")
        probe.after_phase("preflight", 0.01)
        probe.after_run("rid", {})
        log = probe.write_log()

        assert isinstance(log, dict)
        assert "available" in log
        assert "snapshots" in log

    def test_snapshot_list_grows_when_available(self, tmp_path):
        probe = MemoryProbe()
        if not probe._available:
            pytest.skip("psutil not installed")

        cfg = _base_config()
        probe.before_run("rid", cfg, tmp_path)
        n0 = len(probe._snapshots)

        for ph in ("preflight", "compile"):
            probe.before_phase(ph)
            probe.after_phase(ph, 0.01)

        n1 = len(probe._snapshots)
        assert n1 > n0
        # 2 phases × 2 snapshots each = 4
        assert n1 - n0 == 4

    def test_baseline_recorded_when_available(self, tmp_path):
        probe = MemoryProbe()
        if not probe._available:
            pytest.skip("psutil not installed")

        cfg = _base_config()
        probe.before_run("rid", cfg, tmp_path)
        log = probe.write_log()
        assert log["available"] is True
        assert log["baseline_rss_mb"] is not None
        assert log["baseline_rss_mb"] > 0

    def test_on_error_appends_snapshot(self, tmp_path):
        probe = MemoryProbe()
        if not probe._available:
            pytest.skip("psutil not installed")

        cfg = _base_config()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("model_load")
        n_before = len(probe._snapshots)
        probe.on_error("model_load", RuntimeError("simulated"))
        n_after = len(probe._snapshots)
        assert n_after == n_before + 1
        assert probe._snapshots[-1]["when"] == "on_error"

    def test_unavailable_psutil_degrades_to_noop(self, tmp_path, monkeypatch):
        # Force the lazy import inside __init__ to fail.
        original_psutil = sys.modules.pop("psutil", None)
        monkeypatch.setitem(sys.modules, "psutil", None)
        try:
            probe = MemoryProbe()
            cfg = _base_config()
            probe.before_run("rid", cfg, tmp_path)
            probe.before_phase("preflight")
            probe.after_phase("preflight", 0.0)
            probe.after_run("rid", {})
            log = probe.write_log()
            assert log["available"] is False
            assert log["snapshots"] == []
        finally:
            if original_psutil is not None:
                sys.modules["psutil"] = original_psutil


# ── InputFingerprintProbe ─────────────────────────────────────────────────────

class TestInputFingerprintProbe:
    def test_name(self):
        assert InputFingerprintProbe.name == "input_fingerprint"

    def test_fingerprint_computed_on_latency_phase(self, tmp_path):
        cfg = _base_config(input_seed=42, input_type="text")
        probe = InputFingerprintProbe()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("latency")
        log = probe.write_log()

        assert log["input_seed"] == 42
        assert log["fingerprint_sha256_16"] is not None
        assert len(log["fingerprint_sha256_16"]) == 16
        # Hex-only.
        int(log["fingerprint_sha256_16"], 16)

    def test_fingerprint_deterministic_across_two_probes(self, tmp_path):
        cfg = _base_config(input_seed=42, input_type="text")
        probe_a = InputFingerprintProbe()
        probe_a.before_run("rid_a", cfg, tmp_path)
        probe_a.before_phase("latency")

        probe_b = InputFingerprintProbe()
        probe_b.before_run("rid_b", cfg, tmp_path)
        probe_b.before_phase("latency")

        log_a = probe_a.write_log()
        log_b = probe_b.write_log()
        assert log_a["fingerprint_sha256_16"] == log_b["fingerprint_sha256_16"]

    def test_different_seeds_produce_different_fingerprints(self, tmp_path):
        cfg_a = _base_config(input_seed=42, input_type="text")
        cfg_b = _base_config(input_seed=99, input_type="text")

        probe_a = InputFingerprintProbe()
        probe_a.before_run("rid_a", cfg_a, tmp_path)
        probe_a.before_phase("latency")

        probe_b = InputFingerprintProbe()
        probe_b.before_run("rid_b", cfg_b, tmp_path)
        probe_b.before_phase("latency")

        assert (
            probe_a.write_log()["fingerprint_sha256_16"]
            != probe_b.write_log()["fingerprint_sha256_16"]
        )

    def test_non_latency_phase_does_not_fingerprint(self, tmp_path):
        cfg = _base_config(input_seed=42, input_type="text")
        probe = InputFingerprintProbe()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("compile")
        log = probe.write_log()
        assert log["fingerprint_sha256_16"] is None

    def test_input_shapes_and_dtypes_recorded(self, tmp_path):
        cfg = _base_config(input_seed=42, input_type="text", seq_len=128)
        probe = InputFingerprintProbe()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("latency")
        log = probe.write_log()

        # bs=1 inputs
        assert "input_ids" in log["input_shapes"]
        assert log["input_shapes"]["input_ids"] == [1, 128]
        assert log["input_dtypes"]["input_ids"] == "int32"

    def test_image_input_type_fingerprints(self, tmp_path):
        cfg = _base_config(
            input_type="image",
            task="image-classification",
            image_size=[3, 224, 224],
        )
        probe = InputFingerprintProbe()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("latency")
        log = probe.write_log()

        assert log["fingerprint_sha256_16"] is not None
        assert log["input_shapes"]["pixel_values"] == [1, 3, 224, 224]

    def test_idempotent_on_repeat_latency_call(self, tmp_path):
        cfg = _base_config(input_seed=42, input_type="text")
        probe = InputFingerprintProbe()
        probe.before_run("rid", cfg, tmp_path)
        probe.before_phase("latency")
        first = probe.write_log()["fingerprint_sha256_16"]
        # If the runner ever called latency twice, we shouldn't recompute.
        probe.before_phase("latency")
        second = probe.write_log()["fingerprint_sha256_16"]
        assert first == second
