"""Tests for train/runner.py — config dataclass + synthetic-data helper.

The full run_training() call requires a working Flax + optax + jax stack and
a real model checkpoint, so it isn't exercised here. Tests focus on the
deterministic, mock-friendly surface: the config defaults, the synthetic
batch generator, and the harness wiring (probe set registration).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")


# ── Fake jax (matches tests/test_runner.py) ──────────────────────────────────

def _install_fake_jax():
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

from train.runner import (  # noqa: E402
    TrainingExperimentConfig,
    make_synthetic_train_batch,
)


def _base_cfg(**overrides) -> TrainingExperimentConfig:
    defaults = dict(
        task_id="bert_finetune",
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
        batch_size=8,
        vocab_size=30522,
        num_labels=2,
        n_steps=10,
        n_eval_steps=2,
        n_warmup_steps_train=2,
        lr=2.0e-5,
        lr_warmup_steps=2,
        weight_decay=0.01,
        optimizer="adamw",
        input_seed=42,
        init_seed=0,
        save_checkpoint=False,
        device_cost_usd_per_hr=0.36,
    )
    defaults.update(overrides)
    return TrainingExperimentConfig(**defaults)


# ── TrainingExperimentConfig ─────────────────────────────────────────────────


class TestConfig:
    def test_defaults_are_sane(self):
        cfg = _base_cfg()
        assert cfg.precision == "bf16"
        assert cfg.batch_size == 8
        assert cfg.optimizer == "adamw"

    def test_construction_with_overrides(self):
        cfg = _base_cfg(precision="fp32", n_steps=50)
        assert cfg.precision == "fp32"
        assert cfg.n_steps == 50


# ── make_synthetic_train_batch ───────────────────────────────────────────────


class TestSyntheticBatch:
    def test_text_batch_shapes(self):
        cfg = _base_cfg(batch_size=4, seq_len=16)
        rng = np.random.default_rng(42)
        batch = make_synthetic_train_batch(cfg, rng)
        assert batch["input_ids"].shape == (4, 16)
        assert batch["attention_mask"].shape == (4, 16)
        assert batch["token_type_ids"].shape == (4, 16)
        assert batch["labels"].shape == (4,)

    def test_labels_in_range(self):
        cfg = _base_cfg(num_labels=3)
        rng = np.random.default_rng(42)
        batch = make_synthetic_train_batch(cfg, rng)
        assert batch["labels"].min() >= 0
        assert batch["labels"].max() < 3

    def test_input_ids_in_vocab_range(self):
        cfg = _base_cfg(vocab_size=100)
        rng = np.random.default_rng(42)
        batch = make_synthetic_train_batch(cfg, rng)
        assert batch["input_ids"].min() >= 1
        assert batch["input_ids"].max() < 100

    def test_attention_mask_all_ones(self):
        cfg = _base_cfg()
        rng = np.random.default_rng(42)
        batch = make_synthetic_train_batch(cfg, rng)
        assert (batch["attention_mask"] == 1).all()

    def test_token_type_ids_all_zeros(self):
        cfg = _base_cfg()
        rng = np.random.default_rng(42)
        batch = make_synthetic_train_batch(cfg, rng)
        assert (batch["token_type_ids"] == 0).all()

    def test_seed_determinism(self):
        cfg = _base_cfg()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        b1 = make_synthetic_train_batch(cfg, rng1)
        b2 = make_synthetic_train_batch(cfg, rng2)
        assert (b1["input_ids"] == b2["input_ids"]).all()
        assert (b1["labels"] == b2["labels"]).all()

    def test_unsupported_input_type_raises(self):
        cfg = _base_cfg(input_type="image")
        with pytest.raises(ValueError, match="text"):
            make_synthetic_train_batch(cfg, np.random.default_rng())


# ── Registry yaml is well-formed ─────────────────────────────────────────────


class TestRegistry:
    def test_registry_loads_and_has_required_keys(self):
        path = Path(__file__).parent.parent / "train" / "registry.yaml"
        data = yaml.safe_load(path.read_text())
        assert "tasks" in data
        assert len(data["tasks"]) >= 1
        required = {
            "id", "hf_id", "task", "domain", "input_type",
            "total_params_M", "active_params_M",
            "default_seq_len", "default_batch_size",
            "default_steps", "default_lr",
        }
        for entry in data["tasks"]:
            missing = required - set(entry)
            assert not missing, f"task {entry.get('id')!r} missing keys: {missing}"


# ── Harness CLI surface ──────────────────────────────────────────────────────


class TestHarnessCLI:
    def test_dry_run_smoke_does_not_import_jax_heavyweights(self):
        # `--dry-run` only parses + prints; it must not crash even on a host
        # without optax / transformers / real jax. The fake jax module covers
        # for the import in build_config (which goes through TrainingExperimentConfig).
        from train.harness import main
        rc = main(["--suite", "smoke", "--device", "cpu", "--dry-run"])
        assert rc == 0

    def test_dry_run_unknown_task_returns_error(self):
        from train.harness import main
        rc = main(["--task", "does_not_exist", "--device", "cpu", "--dry-run"])
        assert rc == 1

    def test_dry_run_with_explicit_task(self):
        from train.harness import main
        rc = main(["--task", "bert_finetune", "--device", "cpu", "--dry-run"])
        assert rc == 0
