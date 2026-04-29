"""Tests for observe/lineage.py — mocks subprocess and importlib."""
from __future__ import annotations

from unittest import mock

import pytest

from observe.lineage import (
    build_environment_hash,
    build_lineage,
    get_git_sha,
    get_package_version,
)


class TestGetGitSha:
    def test_returns_sha_on_success(self):
        fake_sha = "a1b2c3d4e5f6" * 3
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0, stdout=fake_sha + "\n"
            )
            assert get_git_sha() == fake_sha

    def test_returns_unknown_on_failure(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="")
            assert get_git_sha() == "unknown"

    def test_returns_unknown_on_exception(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert get_git_sha() == "unknown"

    def test_returns_unknown_on_timeout(self):
        import subprocess
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)):
            assert get_git_sha() == "unknown"


class TestGetPackageVersion:
    def test_installed_package(self):
        # pip is always installed
        version = get_package_version("pip")
        assert version != "not_installed"
        assert "." in version  # version strings have dots

    def test_missing_package(self):
        version = get_package_version("__nonexistent_package_xyz__")
        assert version == "not_installed"


class TestBuildEnvironmentHash:
    def test_returns_16_hex_chars(self):
        h = build_environment_hash(packages=("pip",))
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        h1 = build_environment_hash(packages=("pip",))
        h2 = build_environment_hash(packages=("pip",))
        assert h1 == h2

    def test_different_packages_different_hash(self):
        h1 = build_environment_hash(packages=("pip",))
        h2 = build_environment_hash(packages=("__nonexistent_xyz__",))
        assert h1 != h2

    def test_version_change_changes_hash(self):
        with mock.patch("observe.lineage.get_package_version", return_value="1.0.0"):
            h1 = build_environment_hash(packages=("pip",))
        with mock.patch("observe.lineage.get_package_version", return_value="2.0.0"):
            h2 = build_environment_hash(packages=("pip",))
        assert h1 != h2


class TestBuildLineage:
    def test_all_required_keys_present(self):
        with mock.patch("observe.lineage.get_git_sha", return_value="deadbeef"):
            lineage = build_lineage("bert_base", hf_revision="abc", input_seed=42)
        required_keys = {
            "git_sha",
            "jax_version",
            "torch_version",
            "transformers_version",
            "hf_model_revision",
            "input_seed",
            "n_independent_runs",
            "environment_hash",
        }
        assert required_keys.issubset(lineage.keys())

    def test_git_sha_used(self):
        with mock.patch("observe.lineage.get_git_sha", return_value="cafebabe"):
            lineage = build_lineage("bert_base")
        assert lineage["git_sha"] == "cafebabe"

    def test_hf_revision_stored(self):
        lineage = build_lineage("bert_base", hf_revision="rev123")
        assert lineage["hf_model_revision"] == "rev123"

    def test_missing_hf_revision_defaults_to_unknown(self):
        lineage = build_lineage("bert_base", hf_revision=None)
        assert lineage["hf_model_revision"] == "unknown"

    def test_input_seed_stored(self):
        lineage = build_lineage("bert_base", input_seed=99)
        assert lineage["input_seed"] == 99

    def test_n_independent_runs_is_three(self):
        lineage = build_lineage("bert_base")
        assert lineage["n_independent_runs"] == 3
