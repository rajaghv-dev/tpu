"""Shared pytest fixtures and test helpers."""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest import mock

import pytest


# ── Fake JAX module ───────────────────────────────────────────────────────────

class _FakeDevice:
    def __repr__(self) -> str:
        return "FakeTpuDevice(id=0)"


def _make_fake_jax() -> types.ModuleType:
    """Build a minimal fake jax module that satisfies runner.py imports."""
    jax_mod = types.ModuleType("jax")
    jnp_mod = types.ModuleType("jax.numpy")

    # numpy-backed jnp stubs
    import numpy as np
    jnp_mod.bfloat16 = np.float32  # bfloat16 not in numpy; use float32 as stand-in

    jax_mod.numpy = jnp_mod
    jax_mod.local_devices = lambda: [_FakeDevice()]
    jax_mod.block_until_ready = lambda x: x
    jax_mod.jit = lambda fn: fn  # identity decorator — no actual JIT
    jax_mod.tree_util = types.ModuleType("jax.tree_util")
    jax_mod.tree_util.tree_map = lambda fn, tree: fn(tree)

    return jax_mod


@pytest.fixture
def fake_jax():
    """Inject a minimal fake jax into sys.modules for the duration of a test."""
    jax_mod = _make_fake_jax()
    with mock.patch.dict(sys.modules, {"jax": jax_mod, "jax.numpy": jax_mod.numpy}):
        yield jax_mod


# ── Fake model ────────────────────────────────────────────────────────────────

class FakeModel:
    """Minimal fake HF Flax model for runner tests."""

    class config:
        decoder_start_token_id = 50258
        _commit_hash = "abc123"
        vocab_size = 30522

    def __init__(self, output_shape: tuple = (1, 128, 30522)):
        import numpy as np
        self._output = np.ones(output_shape, dtype=np.float32)
        self.params = {"weight": np.ones((10, 10), dtype=np.float32)}

    def __call__(self, **kwargs: Any) -> Any:
        return self._output


@pytest.fixture
def fake_loader():
    """
    Returns a _loader function compatible with run_experiment's _loader param.
    """
    def _loader(config):
        model = FakeModel()
        params = model.params
        hf_revision = "deadbeef"
        return model, params, hf_revision

    return _loader
