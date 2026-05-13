"""Tests for train/runner.py — config dataclass + synthetic-data helper.

The full run_training() call requires a working Flax + optax + jax stack and
a real model checkpoint, so it isn't exercised here. Tests focus on the
deterministic, mock-friendly surface: the config defaults, the synthetic
batch generator (across all three task families), the optimizer / schedule
factories, and the harness wiring (probe set registration + CLI surface).
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
        lr_schedule="linear",
        weight_decay=0.01,
        optimizer="adamw",
        max_grad_norm=1.0,
        grad_accum_steps=1,
        input_seed=42,
        init_seed=0,
        eval_seed=1337,
        deterministic=False,
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

    def test_new_control_fields_default(self):
        cfg = _base_cfg()
        assert cfg.lr_schedule == "linear"
        assert cfg.max_grad_norm == 1.0
        assert cfg.grad_accum_steps == 1
        assert cfg.eval_seed == 1337
        assert cfg.deterministic is False

    def test_optimizer_choice_accepted(self):
        for opt in ("adamw", "sgd", "lion", "adafactor"):
            cfg = _base_cfg(optimizer=opt)
            assert cfg.optimizer == opt

    def test_lr_schedule_choice_accepted(self):
        for sched in ("linear", "cosine", "constant"):
            cfg = _base_cfg(lr_schedule=sched)
            assert cfg.lr_schedule == sched


# ── make_synthetic_train_batch — sequence-classification ─────────────────────


class TestSyntheticBatchSeqCls:
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

    def test_text_input_type_mismatch_raises(self):
        # image input + seq-cls task → ValueError mentioning 'text'.
        cfg = _base_cfg(input_type="image")
        with pytest.raises(ValueError, match="text"):
            make_synthetic_train_batch(cfg, np.random.default_rng())


# ── make_synthetic_train_batch — causal-lm ───────────────────────────────────


class TestSyntheticBatchCausalLM:
    def _cfg(self, **overrides):
        base = dict(
            task_id="gpt2_lm", hf_id="gpt2", task="causal-lm",
            domain="nlp_decoder",
            architecture_family="transformer_decoder",
            vocab_size=50257, batch_size=4, seq_len=16,
        )
        base.update(overrides)
        return _base_cfg(**base)

    def test_shapes(self):
        rng = np.random.default_rng(42)
        b = make_synthetic_train_batch(self._cfg(), rng)
        assert b["input_ids"].shape == (4, 16)
        assert b["attention_mask"].shape == (4, 16)
        # causal-lm batches do NOT include a separate `labels` field —
        # labels are derived in-step by shifting input_ids.
        assert "labels" not in b
        assert "token_type_ids" not in b

    def test_input_ids_in_vocab_range(self):
        rng = np.random.default_rng(42)
        b = make_synthetic_train_batch(self._cfg(vocab_size=512), rng)
        assert b["input_ids"].min() >= 1
        assert b["input_ids"].max() < 512

    def test_attention_mask_all_ones(self):
        rng = np.random.default_rng(42)
        b = make_synthetic_train_batch(self._cfg(), rng)
        assert (b["attention_mask"] == 1).all()

    def test_seed_determinism(self):
        b1 = make_synthetic_train_batch(self._cfg(), np.random.default_rng(7))
        b2 = make_synthetic_train_batch(self._cfg(), np.random.default_rng(7))
        assert (b1["input_ids"] == b2["input_ids"]).all()

    def test_image_input_type_raises(self):
        cfg = self._cfg(input_type="image")
        with pytest.raises(ValueError, match="text"):
            make_synthetic_train_batch(cfg, np.random.default_rng())


# ── make_synthetic_train_batch — image-classification ────────────────────────


class TestSyntheticBatchImageCls:
    def _cfg(self, **overrides):
        base = dict(
            task_id="vit_b16_finetune",
            hf_id="google/vit-base-patch16-224",
            task="image-classification",
            domain="vision_cls",
            input_type="image",
            image_size=[3, 32, 32],
            num_labels=10,
            batch_size=4,
        )
        base.update(overrides)
        return _base_cfg(**base)

    def test_shapes(self):
        rng = np.random.default_rng(42)
        b = make_synthetic_train_batch(self._cfg(), rng)
        assert b["pixel_values"].shape == (4, 3, 32, 32)
        assert b["labels"].shape == (4,)
        assert b["pixel_values"].dtype == np.float32
        assert b["labels"].dtype == np.int32

    def test_labels_in_range(self):
        rng = np.random.default_rng(42)
        b = make_synthetic_train_batch(self._cfg(num_labels=5), rng)
        assert b["labels"].min() >= 0
        assert b["labels"].max() < 5

    def test_image_input_distribution(self):
        # standard_normal → roughly mean 0, std 1 over a 4×3×32×32 sample.
        rng = np.random.default_rng(42)
        b = make_synthetic_train_batch(self._cfg(batch_size=64), rng)
        assert abs(float(b["pixel_values"].mean())) < 0.2
        assert 0.8 < float(b["pixel_values"].std()) < 1.2

    def test_text_input_type_raises(self):
        cfg = self._cfg(input_type="text")
        with pytest.raises(ValueError, match="image"):
            make_synthetic_train_batch(cfg, np.random.default_rng())

    def test_missing_image_size_raises(self):
        cfg = self._cfg(image_size=None)
        with pytest.raises(ValueError, match="image_size"):
            make_synthetic_train_batch(cfg, np.random.default_rng())


class TestSyntheticBatchTaskDispatch:
    def test_unknown_task_raises(self):
        cfg = _base_cfg(task="zero-shot-image-classification")
        with pytest.raises(ValueError, match="Unsupported task"):
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
            "default_batch_size",
            "default_steps", "default_lr",
        }
        for entry in data["tasks"]:
            missing = required - set(entry)
            assert not missing, f"task {entry.get('id')!r} missing keys: {missing}"

    def test_registry_has_multiple_task_families(self):
        path = Path(__file__).parent.parent / "train" / "registry.yaml"
        data = yaml.safe_load(path.read_text())
        tasks = {t["task"] for t in data["tasks"]}
        # The expanded registry must cover all three task families.
        assert {"sequence-classification", "causal-lm", "image-classification"} <= tasks

    def test_text_tasks_carry_seq_len_and_vocab(self):
        path = Path(__file__).parent.parent / "train" / "registry.yaml"
        data = yaml.safe_load(path.read_text())
        for t in data["tasks"]:
            if t["input_type"] == "text":
                assert "default_seq_len" in t, t["id"]
                assert "vocab_size" in t, t["id"]

    def test_image_tasks_carry_image_size(self):
        path = Path(__file__).parent.parent / "train" / "registry.yaml"
        data = yaml.safe_load(path.read_text())
        for t in data["tasks"]:
            if t["input_type"] == "image":
                assert "image_size" in t, t["id"]
                assert len(t["image_size"]) == 3, t["id"]

    def test_registry_size_grew(self):
        # Sanity check that the expansion stuck. Stage 1.6 added 13 tasks
        # on top of the original bert_finetune; expect at least 10.
        path = Path(__file__).parent.parent / "train" / "registry.yaml"
        data = yaml.safe_load(path.read_text())
        assert len(data["tasks"]) >= 10


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

    def test_dry_run_with_causal_lm_task(self):
        from train.harness import main
        rc = main(["--task", "gpt2_lm", "--device", "cpu", "--dry-run"])
        assert rc == 0

    def test_dry_run_with_image_task(self):
        from train.harness import main
        rc = main(["--task", "vit_b16_finetune", "--device", "cpu", "--dry-run"])
        assert rc == 0

    def test_dry_run_causal_smoke_suite(self):
        from train.harness import main
        rc = main(["--suite", "causal_smoke", "--device", "cpu", "--dry-run"])
        assert rc == 0

    def test_dry_run_vit_smoke_suite(self):
        from train.harness import main
        rc = main(["--suite", "vit_smoke", "--device", "cpu", "--dry-run"])
        assert rc == 0

    def test_dry_run_diverse_suite(self):
        from train.harness import main
        rc = main(["--suite", "diverse", "--device", "cpu", "--dry-run"])
        assert rc == 0

    def test_dry_run_scaling_suite(self):
        from train.harness import main
        rc = main(["--suite", "scaling", "--device", "cpu", "--dry-run"])
        assert rc == 0

    def test_dry_run_with_optimizer_override(self, capsys):
        from train.harness import main
        rc = main([
            "--task", "bert_finetune", "--device", "cpu", "--dry-run",
            "--optimizer", "lion",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "opt=lion" in out

    def test_dry_run_with_grad_accum_override(self, capsys):
        from train.harness import main
        rc = main([
            "--task", "bert_finetune", "--device", "cpu", "--dry-run",
            "--grad-accum", "4",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "accum=4" in out

    def test_dry_run_with_deterministic_flag(self, capsys):
        from train.harness import main
        rc = main([
            "--task", "bert_finetune", "--device", "cpu", "--dry-run",
            "--deterministic",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[deterministic]" in out


# ── build_config control-knob propagation ────────────────────────────────────


class TestBuildConfigOverrides:
    def _load_bert_entry(self):
        path = Path(__file__).parent.parent / "train" / "registry.yaml"
        data = yaml.safe_load(path.read_text())
        return next(t for t in data["tasks"] if t["id"] == "bert_finetune")

    def test_cli_override_beats_registry(self):
        from train.harness import build_config
        entry = self._load_bert_entry()
        cfg = build_config(
            entry, precision="bf16", device="cpu",
            overrides={"optimizer": "lion", "max_grad_norm": 0.5,
                       "grad_accum_steps": 8, "lr_schedule": "cosine",
                       "eval_seed": 99, "deterministic": True},
        )
        assert cfg.optimizer == "lion"
        assert cfg.max_grad_norm == 0.5
        assert cfg.grad_accum_steps == 8
        assert cfg.lr_schedule == "cosine"
        assert cfg.eval_seed == 99
        assert cfg.deterministic is True

    def test_registry_default_used_when_override_none(self):
        from train.harness import build_config
        entry = self._load_bert_entry()
        cfg = build_config(entry, precision="bf16", device="cpu", overrides={})
        # The registry sets these defaults — make sure they propagate.
        assert cfg.optimizer == entry.get("default_optimizer", "adamw")
        assert cfg.max_grad_norm == entry.get("default_max_grad_norm", 1.0)
        assert cfg.grad_accum_steps == entry.get("default_grad_accum_steps", 1)

    def test_image_size_propagates_for_image_tasks(self):
        path = Path(__file__).parent.parent / "train" / "registry.yaml"
        data = yaml.safe_load(path.read_text())
        entry = next(t for t in data["tasks"] if t["id"] == "vit_b16_finetune")
        from train.harness import build_config
        cfg = build_config(entry, precision="bf16", device="cpu")
        assert cfg.image_size == [3, 224, 224]


# ── Probe-set registration ───────────────────────────────────────────────────


class TestProbeSetRegistration:
    def setup_method(self):
        # Each test starts with a clean registry — _register_probe_set
        # already clears, but we don't want leaks across the test session.
        from observe.probe import clear_probes
        clear_probes()

    def teardown_method(self):
        from observe.probe import clear_probes
        clear_probes()

    def test_none_registers_nothing(self):
        from train.harness import _register_probe_set
        names, skipped = _register_probe_set("none")
        assert names == []
        assert skipped == []

    def test_minimal_registers_four_baseline_probes(self):
        from train.harness import _register_probe_set
        names, _ = _register_probe_set("minimal")
        # Order is the registration order — keep it stable for the dashboard.
        assert names == ["timing", "memory", "training_metrics", "step_timing"]

    def test_default_includes_minimal_plus_new_observability_probes(self):
        from train.harness import _register_probe_set
        names, _ = _register_probe_set("default")
        # Minimal subset always present.
        for n in ("timing", "memory", "training_metrics", "step_timing"):
            assert n in names
        # New Stage-1.6 probes (any subset that imports successfully).
        # InputFingerprint + Checkpoint are mandatory in default; the
        # device_info / determinism / xla_compile probes are optional —
        # they're skipped (not raised) if their import fails.
        assert "input_fingerprint" in names
        assert "checkpoint" in names

    def test_full_is_strict_superset_of_default(self):
        from train.harness import _register_probe_set
        names_default, _ = _register_probe_set("default")
        names_full, _ = _register_probe_set("full")
        assert set(names_default) <= set(names_full)
