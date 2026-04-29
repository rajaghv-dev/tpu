"""Tests for benchmarks/runner.py — mocks JAX and HF transformers."""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest import mock

import pytest

np = pytest.importorskip("numpy")

# ── Helpers to install fake JAX ───────────────────────────────────────────────

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
    _inputs_to_args,
    make_synthetic_inputs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

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
        # decoder_input_ids should have shape (batch, 1)
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
