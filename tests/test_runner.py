"""Tests for benchmarks/runner.py — mocks JAX and HF transformers."""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest import mock

import pytest

np = pytest.importorskip("numpy")


# ── Fake JAX installation ─────────────────────────────────────────────────────

def _install_fake_jax():
    """Register a minimal fake jax module before importing runner."""
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

from benchmarks.runner import (  # noqa: E402 — after fake jax installed
    ExperimentConfig,
    _TASK_ARGS,
    _inputs_to_args,
    make_synthetic_inputs,
)


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


# ── _TASK_ARGS consistency ────────────────────────────────────────────────────

class TestTaskArgs:
    def test_all_standard_tasks_covered(self):
        expected = {
            "sequence-classification",
            "image-classification",
            "causal-lm",
            "zero-shot-image-classification",
        }
        assert set(_TASK_ARGS.keys()) == expected

    def test_each_task_has_non_empty_key_list(self):
        for task, keys in _TASK_ARGS.items():
            assert len(keys) >= 1, f"Task {task} has empty key list"

    def test_asr_not_in_task_args(self):
        # ASR is handled separately (decoder_ids injection)
        assert "automatic-speech-recognition" not in _TASK_ARGS


# ── make_synthetic_inputs ─────────────────────────────────────────────────────

class TestMakeSyntheticInputs:
    def test_text_shape(self):
        cfg = _base_config(input_type="text", seq_len=128, vocab_size=30522)
        inputs = make_synthetic_inputs(cfg, batch_size=4)
        assert inputs["input_ids"].shape == (4, 128)
        assert inputs["attention_mask"].shape == (4, 128)
        assert inputs["token_type_ids"].shape == (4, 128)

    def test_text_token_range(self):
        cfg = _base_config(input_type="text", vocab_size=30522)
        inputs = make_synthetic_inputs(cfg, batch_size=2)
        assert inputs["input_ids"].min() >= 1
        assert inputs["input_ids"].max() < 30522

    def test_attention_mask_all_ones(self):
        cfg = _base_config(input_type="text")
        inputs = make_synthetic_inputs(cfg, batch_size=2)
        assert (inputs["attention_mask"] == 1).all()

    def test_token_type_ids_all_zeros(self):
        cfg = _base_config(input_type="text")
        inputs = make_synthetic_inputs(cfg, batch_size=2)
        assert (inputs["token_type_ids"] == 0).all()

    def test_image_shape(self):
        cfg = _base_config(input_type="image", image_size=[3, 224, 224])
        inputs = make_synthetic_inputs(cfg, batch_size=2)
        assert inputs["pixel_values"].shape == (2, 3, 224, 224)
        assert inputs["pixel_values"].dtype == np.float32

    def test_audio_shape(self):
        cfg = _base_config(input_type="audio", n_mels=80, n_frames=3000)
        inputs = make_synthetic_inputs(cfg, batch_size=4)
        assert inputs["input_features"].shape == (4, 80, 3000)

    def test_image_text_keys(self):
        cfg = _base_config(
            input_type="image_text",
            image_size=[3, 224, 224],
            seq_len=77,
            vocab_size=49408,
        )
        inputs = make_synthetic_inputs(cfg, batch_size=2)
        assert "pixel_values" in inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

    def test_reproducible_with_same_seed(self):
        cfg = _base_config(input_type="text", input_seed=42)
        inputs_a = make_synthetic_inputs(cfg, batch_size=2)
        inputs_b = make_synthetic_inputs(cfg, batch_size=2)
        assert (inputs_a["input_ids"] == inputs_b["input_ids"]).all()

    def test_different_seeds_differ(self):
        cfg_a = _base_config(input_type="text", input_seed=42)
        cfg_b = _base_config(input_type="text", input_seed=99)
        inputs_a = make_synthetic_inputs(cfg_a, batch_size=2)
        inputs_b = make_synthetic_inputs(cfg_b, batch_size=2)
        assert not (inputs_a["input_ids"] == inputs_b["input_ids"]).all()

    def test_unknown_input_type_raises(self):
        cfg = _base_config(input_type="video")
        with pytest.raises(ValueError, match="Unknown input_type"):
            make_synthetic_inputs(cfg, batch_size=1)


# ── _inputs_to_args ───────────────────────────────────────────────────────────

class TestInputsToArgs:
    def test_text_returns_three_arrays(self):
        cfg = _base_config(task="sequence-classification")
        inputs = make_synthetic_inputs(cfg, batch_size=1)
        args = _inputs_to_args(inputs, "sequence-classification")
        assert len(args) == 3

    def test_image_returns_one_array(self):
        cfg = _base_config(input_type="image", task="image-classification", image_size=[3, 224, 224])
        inputs = make_synthetic_inputs(cfg, batch_size=1)
        args = _inputs_to_args(inputs, "image-classification")
        assert len(args) == 1

    def test_causal_lm_returns_two_arrays(self):
        cfg = _base_config(input_type="text", task="causal-lm")
        inputs = make_synthetic_inputs(cfg, batch_size=1)
        args = _inputs_to_args(inputs, "causal-lm")
        assert len(args) == 2

    def test_whisper_returns_two_arrays_with_decoder_ids(self):
        cfg = _base_config(input_type="audio", task="automatic-speech-recognition",
                           n_mels=80, n_frames=3000)
        inputs = make_synthetic_inputs(cfg, batch_size=2)
        args = _inputs_to_args(inputs, "automatic-speech-recognition", decoder_start_id=50258)
        assert len(args) == 2
        assert args[1].shape == (2, 1)

    def test_clip_returns_three_arrays(self):
        cfg = _base_config(
            input_type="image_text",
            task="zero-shot-image-classification",
            image_size=[3, 224, 224],
            seq_len=77,
        )
        inputs = make_synthetic_inputs(cfg, batch_size=1)
        args = _inputs_to_args(inputs, "zero-shot-image-classification")
        assert len(args) == 3

    def test_unknown_task_raises(self):
        inputs = {"input_ids": np.ones((1, 10), dtype=np.int32)}
        with pytest.raises(ValueError, match="Unknown task"):
            _inputs_to_args(inputs, "unsupported-task")

    def test_arg_order_matches_task_args_dict(self):
        """_inputs_to_args must return args in the same order as _TASK_ARGS."""
        cfg = _base_config(task="causal-lm", input_type="text")
        inputs = make_synthetic_inputs(cfg, batch_size=1)
        args = _inputs_to_args(inputs, "causal-lm")
        expected_keys = _TASK_ARGS["causal-lm"]
        assert len(args) == len(expected_keys)
        for arg, key in zip(args, expected_keys):
            assert (arg == inputs[key]).all()


# ── ExperimentConfig ──────────────────────────────────────────────────────────

class TestExperimentConfig:
    def test_default_precision_is_bf16(self):
        cfg = _base_config()
        assert cfg.precision == "bf16"

    def test_default_seed_is_42(self):
        cfg = _base_config()
        assert cfg.input_seed == 42

    def test_fields_accessible(self):
        cfg = _base_config(model_id="test_model", total_params_M=42)
        assert cfg.model_id == "test_model"
        assert cfg.total_params_M == 42


# ── run_experiment integration ────────────────────────────────────────────────

class TestRunExperiment:
    """
    Integration test for run_experiment using a mock model loader.

    Patches N_WARMUP/N_MEASURE/N_BLOCKS to tiny values so the test
    finishes in milliseconds, then verifies the result dict structure.
    """

    REQUIRED_RESULT_FIELDS = {
        "run_id", "timestamp",
        "git_sha", "jax_version", "hf_model_revision", "input_seed",
        "device", "framework", "path",
        "model", "domain", "precision",
        "first_compile_s", "subsequent_compile_s",
        "latency_mean_ms", "latency_std_ms", "latency_cv_pct",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "throughput_mean_samples_sec", "throughput_std_samples_sec",
        "flags", "device_cost_usd_per_hr", "experiment_cost_usd",
    }

    def _make_fake_loader(self):
        """Return a loader that produces a callable fake model."""
        class _FakeModel:
            class config:
                decoder_start_token_id = 50258
                _commit_hash = "deadbeef"

            def __init__(self):
                self.params = {"w": np.ones((4, 4), dtype=np.float32)}

            def __call__(self, **kwargs: Any):
                return np.ones((1, 10), dtype=np.float32)

        model = _FakeModel()
        return lambda cfg: (model, model.params, "deadbeef")

    def test_returns_dict_with_required_fields(self, tmp_path):
        import benchmarks.runner as rm

        loader = self._make_fake_loader()
        cfg = _base_config(batch_size_throughput=2)

        with (
            mock.patch.object(rm, "N_WARMUP", 2),
            mock.patch.object(rm, "N_MEASURE", 3),
            mock.patch.object(rm, "N_BLOCKS", 2),
        ):
            result = rm.run_experiment(cfg, results_dir=str(tmp_path), _loader=loader)

        assert isinstance(result, dict)
        for field in self.REQUIRED_RESULT_FIELDS:
            assert field in result, f"Missing field in result: {field}"

    def test_latency_stats_are_numeric(self, tmp_path):
        import benchmarks.runner as rm

        loader = self._make_fake_loader()
        cfg = _base_config(batch_size_throughput=2)

        with (
            mock.patch.object(rm, "N_WARMUP", 2),
            mock.patch.object(rm, "N_MEASURE", 3),
            mock.patch.object(rm, "N_BLOCKS", 2),
        ):
            result = rm.run_experiment(cfg, results_dir=str(tmp_path), _loader=loader)

        assert isinstance(result["latency_mean_ms"], float)
        assert isinstance(result["latency_p50_ms"], float)
        assert result["latency_p50_ms"] <= result["latency_p99_ms"]
        assert result["latency_cv_pct"] >= 0

    def test_lineage_json_written(self, tmp_path):
        import benchmarks.runner as rm

        loader = self._make_fake_loader()
        cfg = _base_config(batch_size_throughput=2)

        with (
            mock.patch.object(rm, "N_WARMUP", 2),
            mock.patch.object(rm, "N_MEASURE", 3),
            mock.patch.object(rm, "N_BLOCKS", 2),
        ):
            result = rm.run_experiment(cfg, results_dir=str(tmp_path), _loader=loader)

        run_id = result["run_id"]
        lineage_path = tmp_path / "run_logs" / run_id / "lineage.json"
        assert lineage_path.exists(), "lineage.json should be written to results_dir"

    def test_flags_list_present(self, tmp_path):
        import benchmarks.runner as rm

        loader = self._make_fake_loader()
        cfg = _base_config(batch_size_throughput=2)

        with (
            mock.patch.object(rm, "N_WARMUP", 2),
            mock.patch.object(rm, "N_MEASURE", 3),
            mock.patch.object(rm, "N_BLOCKS", 2),
        ):
            result = rm.run_experiment(cfg, results_dir=str(tmp_path), _loader=loader)

        assert isinstance(result["flags"], list)

    def test_throughput_positive(self, tmp_path):
        import benchmarks.runner as rm

        loader = self._make_fake_loader()
        cfg = _base_config(batch_size_throughput=4)

        with (
            mock.patch.object(rm, "N_WARMUP", 2),
            mock.patch.object(rm, "N_MEASURE", 3),
            mock.patch.object(rm, "N_BLOCKS", 2),
        ):
            result = rm.run_experiment(cfg, results_dir=str(tmp_path), _loader=loader)

        assert result["throughput_mean_samples_sec"] > 0


# ── Exception capture ─────────────────────────────────────────────────────────

class TestBenchmarkErrorCapture:
    """
    Verifies the structured-failure path: when any phase raises, the runner
    wraps the original exception in BenchmarkError with the phase name +
    error category, AND writes a results/run_logs/<run_id>/error.json.
    """

    def test_model_load_failure_writes_error_json(self, tmp_path):
        import benchmarks.runner as rm

        def failing_loader(cfg):
            raise OSError("simulated HF download failure")

        cfg = _base_config(batch_size_throughput=2)
        with pytest.raises(rm.BenchmarkError) as exc_info:
            with (
                mock.patch.object(rm, "N_WARMUP", 1),
                mock.patch.object(rm, "N_MEASURE", 1),
                mock.patch.object(rm, "N_BLOCKS", 1),
            ):
                rm.run_experiment(cfg, results_dir=str(tmp_path), _loader=failing_loader)

        err = exc_info.value
        assert err.phase == "model_load"
        # OSError in model_load → 'network' category (per _classify_error)
        assert err.error_category == "network"
        assert "simulated HF download failure" in err.original_message

        # error.json should exist for at least one run_id under tmp_path/run_logs/
        run_logs = tmp_path / "run_logs"
        assert run_logs.exists()
        error_files = list(run_logs.glob("*/error.json"))
        assert len(error_files) == 1, f"expected 1 error.json, found {error_files}"
        import json
        payload = json.loads(error_files[0].read_text())
        assert payload["phase"] == "model_load"
        assert payload["error_category"] == "network"
        assert payload["lineage"]["input_seed"] == cfg.input_seed

    def test_keyboard_interrupt_categorised(self, tmp_path):
        import benchmarks.runner as rm

        def interrupting_loader(cfg):
            raise KeyboardInterrupt()

        cfg = _base_config(batch_size_throughput=2)
        with pytest.raises(rm.BenchmarkError) as exc_info:
            with (
                mock.patch.object(rm, "N_WARMUP", 1),
                mock.patch.object(rm, "N_MEASURE", 1),
                mock.patch.object(rm, "N_BLOCKS", 1),
            ):
                rm.run_experiment(cfg, results_dir=str(tmp_path), _loader=interrupting_loader)

        assert exc_info.value.error_category == "interrupted"
