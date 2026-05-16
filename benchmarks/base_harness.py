"""Shared constants and utilities for inference and training harnesses.

Both benchmarks/harness.py (inference) and train/harness.py (training) share
the same device cost table and the same low-level YAML-loading logic.  Rather
than keeping two copies in sync, this module owns the single source of truth
and re-exports what each harness needs.

NOTE on load_registry
---------------------
The two harnesses point at *different* registry files and return *different*
top-level keys (``data["models"]`` vs ``data["tasks"]``), so a single
zero-argument ``load_registry()`` cannot serve both.  The shared primitive
here is ``_load_yaml_registry``, which handles the common mechanics (YAML
import guard, path resolution, file open).  Each harness wraps it in its own
``load_registry()`` that supplies the correct default path and key name.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Cost per preemptible hour in USD for each device type.
# Shared by both the inference harness and the training harness.
# ---------------------------------------------------------------------------
DEVICE_COSTS: Dict[str, float] = {
    "tpu_v5e1": 0.36,
    "tpu_v6e1": 0.75,
    "rtx3080": 0.0,
    "rtx4090": 0.0,
    "b200": 0.0,
    "cpu": 0.0,
    "tpu": 0.36,
    "gpu": 0.0,
}


# ---------------------------------------------------------------------------
# Shared YAML registry loader
# ---------------------------------------------------------------------------

def _load_yaml_registry(
    path: pathlib.Path,
    top_level_key: str,
) -> List[Dict[str, Any]]:
    """
    Open a registry YAML file and return the list stored at ``top_level_key``.

    Args:
        path:           Absolute path to the YAML file.
        top_level_key:  The mapping key whose value is the list of entries
                        (e.g. ``"models"`` or ``"tasks"``).

    Returns:
        List of registry entry dicts.

    Raises:
        ImportError:  If ``pyyaml`` is not installed.
        FileNotFoundError / KeyError:  Propagated from the file system / YAML
                        parsing — callers are expected to let these surface.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required. Install with: pip install pyyaml")

    with path.open() as fh:
        data = yaml.safe_load(fh)
    return data[top_level_key]
