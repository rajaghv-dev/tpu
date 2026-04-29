"""Tests for models/registry.yaml — verifies structure and content."""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REGISTRY_PATH = Path(__file__).parent.parent / "models" / "registry.yaml"

REQUIRED_FIELDS = {
    "id",
    "hf_id",
    "task",
    "domain",
    "architecture_family",
    "attention_variant",
    "positional_encoding",
    "is_moe",
    "total_params_M",
    "active_params_M",
    "input_type",
}

VALID_INPUT_TYPES = {"text", "image", "audio", "image_text"}
VALID_TASKS = {
    "sequence-classification",
    "image-classification",
    "causal-lm",
    "automatic-speech-recognition",
    "zero-shot-image-classification",
}
STAGE_1_MODEL_IDS = {"bert_base", "vit_b16", "gpt2", "whisper_base", "clip_vit_b32"}


@pytest.fixture(scope="module")
def registry():
    with REGISTRY_PATH.open() as fh:
        data = yaml.safe_load(fh)
    return data["models"]


class TestRegistryFile:
    def test_file_exists(self):
        assert REGISTRY_PATH.exists(), f"Registry not found at {REGISTRY_PATH}"

    def test_top_level_structure(self):
        with REGISTRY_PATH.open() as fh:
            data = yaml.safe_load(fh)
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0


class TestRegistryContent:
    def test_stage1_has_five_models(self, registry):
        assert len(registry) == 5, f"Stage 1 requires exactly 5 models, got {len(registry)}"

    def test_all_stage1_model_ids_present(self, registry):
        ids = {m["id"] for m in registry}
        missing = STAGE_1_MODEL_IDS - ids
        assert not missing, f"Missing models: {missing}"

    def test_required_fields_on_every_model(self, registry):
        for model in registry:
            missing = REQUIRED_FIELDS - set(model.keys())
            assert not missing, f"Model {model.get('id')} missing fields: {missing}"

    def test_ids_are_unique(self, registry):
        ids = [m["id"] for m in registry]
        assert len(ids) == len(set(ids)), "Duplicate model ids in registry"

    def test_valid_input_types(self, registry):
        for model in registry:
            assert model["input_type"] in VALID_INPUT_TYPES, (
                f"Model {model['id']} has unknown input_type: {model['input_type']}"
            )

    def test_valid_tasks(self, registry):
        for model in registry:
            assert model["task"] in VALID_TASKS, (
                f"Model {model['id']} has unknown task: {model['task']}"
            )

    def test_params_positive(self, registry):
        for model in registry:
            assert model["total_params_M"] > 0
            assert model["active_params_M"] > 0
            assert model["active_params_M"] <= model["total_params_M"]

    def test_is_moe_is_bool(self, registry):
        for model in registry:
            assert isinstance(model["is_moe"], bool), (
                f"Model {model['id']}: is_moe must be bool"
            )

    def test_bert_base_fields(self, registry):
        bert = next(m for m in registry if m["id"] == "bert_base")
        assert bert["hf_id"] == "bert-base-uncased"
        assert bert["input_type"] == "text"
        assert bert["total_params_M"] == 110
        assert bert["vocab_size"] == 30522

    def test_whisper_base_has_audio_fields(self, registry):
        whisper = next(m for m in registry if m["id"] == "whisper_base")
        assert whisper["input_type"] == "audio"
        assert "n_mels" in whisper
        assert "n_frames" in whisper
        assert whisper["n_mels"] > 0
        assert whisper["n_frames"] > 0

    def test_image_models_have_image_size(self, registry):
        for model in registry:
            if model["input_type"] in ("image", "image_text"):
                assert "image_size" in model, (
                    f"Model {model['id']} with input_type={model['input_type']} "
                    "must have image_size"
                )
                assert len(model["image_size"]) == 3
