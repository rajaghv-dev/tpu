"""Tests for observe/compile_controller.py — mocks filesystem and time."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from observe.compile_controller import (
    clear_xla_cache,
    measure_compile,
    timed_call,
)


class TestClearXlaCache:
    def test_clears_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "xla_cache"
            cache.mkdir()
            (cache / "some_file.pb").write_text("dummy")
            result = clear_xla_cache(str(cache))
            assert result is True
            assert cache.exists()
            assert not list(cache.iterdir()), "Cache directory should be empty after clear"

    def test_returns_false_when_dir_missing(self):
        result = clear_xla_cache("/tmp/__nonexistent_xla_cache_xyz__")
        assert result is False

    def test_returns_false_when_no_dir_given(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            result = clear_xla_cache(cache_dir=None)
        assert result is False

    def test_reads_env_var(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "xla_cache"
            cache.mkdir()
            with mock.patch.dict("os.environ", {"JAX_COMPILATION_CACHE_DIR": str(cache)}):
                result = clear_xla_cache(cache_dir=None)
            assert result is True


class TestTimedCall:
    def test_returns_elapsed_and_result(self):
        def fast_fn(x):
            return x * 2

        elapsed, result = timed_call(fast_fn, (21,))
        assert result == 42
        assert elapsed >= 0.0
        assert elapsed < 1.0  # should be near-instant

    def test_sync_fn_called(self):
        called_with = []

        def fn(x):
            return x + 1

        def sync(out):
            called_with.append(out)

        _, result = timed_call(fn, (5,), sync_fn=sync)
        assert result == 6
        assert called_with == [6]

    def test_elapsed_reflects_slow_function(self):
        import time

        def slow_fn():
            time.sleep(0.05)
            return "done"

        elapsed, result = timed_call(slow_fn, ())
        assert elapsed >= 0.04
        assert result == "done"


class TestMeasureCompile:
    def test_returns_required_keys(self):
        call_count = {"n": 0}

        def fn(x):
            call_count["n"] += 1
            return x * 2

        result = measure_compile(fn, (5,), clear_cache=False)
        assert "first_compile_s" in result
        assert "subsequent_compile_s" in result
        assert "compile_cache_hit" in result

    def test_fn_called_twice(self):
        call_count = {"n": 0}

        def fn(x):
            call_count["n"] += 1
            return x

        measure_compile(fn, (1,), clear_cache=False)
        assert call_count["n"] == 2

    def test_compile_cache_hit_is_false(self):
        result = measure_compile(lambda x: x, (1,), clear_cache=False)
        assert result["compile_cache_hit"] is False

    def test_times_are_non_negative(self):
        result = measure_compile(lambda x: x * 2, (3,), clear_cache=False)
        assert result["first_compile_s"] >= 0
        assert result["subsequent_compile_s"] >= 0

    def test_cache_cleared_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "xla"
            cache.mkdir()
            (cache / "stale.bin").write_text("old")
            result = measure_compile(
                lambda x: x,
                (1,),
                clear_cache=True,
                cache_dir=str(cache),
            )
        assert result["first_compile_s"] >= 0
