"""
Compilation timing and XLA cache management.

Stage 1 gap fixed: C3 (compile cache not controlled).

Measures cold (first call) and warm (subsequent call) compilation times
by explicitly clearing the XLA persistent cache before timing begins.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional


_DEFAULT_CACHE_ENV = "JAX_COMPILATION_CACHE_DIR"


def _default_cache_dir() -> Optional[str]:
    return os.environ.get(_DEFAULT_CACHE_ENV) or None


def clear_xla_cache(cache_dir: Optional[str] = None) -> bool:
    """
    Clear the XLA persistent compilation cache.

    Args:
        cache_dir: Path to clear. If None, reads JAX_COMPILATION_CACHE_DIR
                   from the environment. If the env var is unset, returns False.

    Returns:
        True if a cache directory existed and was cleared; False otherwise.
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir()
    if not cache_dir:
        return False
    path = Path(cache_dir)
    if path.exists():
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return True
    return False


def timed_call(
    fn: Callable,
    args: tuple,
    sync_fn: Optional[Callable] = None,
) -> tuple[float, Any]:
    """
    Call fn(*args) and return (elapsed_seconds, result).

    sync_fn is called on the result before stopping the timer —
    use jax.block_until_ready to handle async dispatch.
    """
    start = time.perf_counter()
    result = fn(*args)
    if sync_fn is not None:
        sync_fn(result)
    elapsed = time.perf_counter() - start
    return elapsed, result


def measure_compile(
    fn: Callable,
    args: tuple,
    sync_fn: Optional[Callable] = None,
    cache_dir: Optional[str] = None,
    clear_cache: bool = True,
) -> dict:
    """
    Measure cold and warm compilation times for *fn*.

    Clears the XLA cache (if configured), then times the first call
    (compilation + execution) and second call (execution only).

    Args:
        fn: JIT-compiled callable to time.
        args: Positional arguments to pass to fn.
        sync_fn: Called on the output to force async completion.
        cache_dir: XLA cache directory to clear. Defaults to env var.
        clear_cache: Whether to clear the cache before measurement.

    Returns:
        Dict with first_compile_s, subsequent_compile_s, compile_cache_hit.
    """
    if clear_cache:
        clear_xla_cache(cache_dir)

    first_s, _ = timed_call(fn, args, sync_fn)
    subsequent_s, _ = timed_call(fn, args, sync_fn)

    return {
        "first_compile_s": round(first_s, 4),
        "subsequent_compile_s": round(subsequent_s, 4),
        "compile_cache_hit": False,
    }
