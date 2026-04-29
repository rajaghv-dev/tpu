"""
Experiment lineage capture for full reproducibility.

Records the git SHA, package versions, HF model revision, and input seed
so any result row in runs.jsonl can be reproduced exactly.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from typing import Optional, Tuple


# Packages hashed into the environment fingerprint.
_ENV_PACKAGES: Tuple[str, ...] = ("jax", "torch", "transformers", "numpy", "flax")


def get_git_sha() -> str:
    """Return the current HEAD commit SHA, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def get_package_version(name: str) -> str:
    """Return the installed version of *name*, or 'not_installed'."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def build_environment_hash(
    packages: Tuple[str, ...] = _ENV_PACKAGES,
) -> str:
    """
    SHA-256 of all key package version strings concatenated.

    Produces a 16-hex-char fingerprint that changes whenever any
    of the measurement-relevant packages is updated.
    """
    tokens = [f"{p}={get_package_version(p)}" for p in packages]
    combined = ";".join(tokens)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def build_lineage(
    model_id: str,
    hf_revision: Optional[str] = None,
    input_seed: int = 42,
) -> dict:
    """
    Build the lineage sub-dict for one result row.

    Args:
        model_id: Registry model id (e.g. 'bert_base').
        hf_revision: HuggingFace model commit hash, if available.
        input_seed: Random seed used to generate synthetic inputs.

    Returns:
        Dict with all lineage fields matching the JSONL schema.
    """
    return {
        "git_sha": get_git_sha(),
        "jax_version": get_package_version("jax"),
        "torch_version": get_package_version("torch"),
        "transformers_version": get_package_version("transformers"),
        "hf_model_revision": hf_revision or "unknown",
        "input_seed": input_seed,
        "n_independent_runs": 3,
        "environment_hash": build_environment_hash(),
    }
