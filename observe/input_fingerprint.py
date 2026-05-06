"""
observe/input_fingerprint.py — SHA-256 of synthetic latency-phase inputs.

Two runs with identical configs MUST produce identical synthetic inputs;
otherwise a "stable seed" change has crept in and apples-to-apples
comparisons across runs are meaningless.

This probe regenerates the bs=1 inputs that `make_synthetic_inputs` would
have produced for the latency phase, hashes the raw bytes (sorted by key
for determinism), and writes a 16-hex-char digest. Two runs with the same
fingerprint are guaranteed to drive the same compute pattern through the
forward function — modulo any non-determinism introduced by the model
itself, which is precisely what other probes are for.

Output schema:

    {
      "input_seed": 42,
      "fingerprint_sha256_16": "a1b2c3d4e5f6...",
      "input_shapes":  {"input_ids": [1, 128], "attention_mask": [1, 128], ...},
      "input_dtypes":  {"input_ids": "int32",  "attention_mask": "int32",  ...}
    }

We deliberately don't store the array bytes — even for bs=1 they balloon
the run-log directory (a single 1×3×224×224 fp32 tensor is 600 KB, and
we run thousands of experiments). The fingerprint is enough to detect
drift; if you need the actual inputs, regenerate from the seed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from observe.probe import Probe


class InputFingerprintProbe(Probe):
    """
    Compute a SHA-256 of the synthetic latency-phase inputs.

    The probe stores `config` from `before_run` (the only hook with cfg
    access) and uses it later in `before_phase("latency")` to regenerate
    the inputs. We hash inside `before_phase` rather than `before_run`
    because the latency phase is what the result dict's headline numbers
    are computed from — that's the slice of inputs we care about pinning.
    """

    name = "input_fingerprint"

    def __init__(self) -> None:
        self._config: Any = None
        self._input_seed: Optional[int] = None
        self._fingerprint: Optional[str] = None
        self._fingerprint_inputs: Dict[str, Dict[str, Any]] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def before_run(self, run_id: str, config: Any, log_dir: Path) -> None:
        # Stash the config so `before_phase("latency")` can rebuild inputs.
        self._config = config
        self._input_seed = getattr(config, "input_seed", None)

    def before_phase(self, phase_name: str) -> None:
        if phase_name != "latency":
            return
        if self._config is None:
            return
        # Already fingerprinted this run — don't recompute.
        if self._fingerprint is not None:
            return

        # Local import to break the import cycle: runner.py imports from
        # observe.probe, observe.* imports from runner would loop.
        try:
            from benchmarks.runner import make_synthetic_inputs
        except Exception:  # noqa: BLE001
            return

        try:
            import numpy as np  # noqa: F401  (used implicitly by make_synthetic_inputs)
        except Exception:  # noqa: BLE001
            return

        try:
            inputs = make_synthetic_inputs(self._config, batch_size=1)
        except Exception:  # noqa: BLE001
            return

        # Deterministic hash: iterate keys in sorted order so dict-iteration
        # order can never affect the digest.
        h = hashlib.sha256()
        shape_map: Dict[str, list] = {}
        dtype_map: Dict[str, str] = {}
        for k in sorted(inputs.keys()):
            arr = inputs[k]
            shape_map[k] = list(arr.shape)
            dtype_map[k] = str(arr.dtype)
            # Mix the key + dtype + shape into the hash so two arrays with
            # identical bytes but different metadata still produce different
            # fingerprints (defensive — unlikely in practice but cheap).
            h.update(k.encode("utf-8"))
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(repr(arr.shape).encode("utf-8"))
            # tobytes() materialises a contiguous byte buffer regardless of
            # the array's stride layout, so we hash semantic content.
            h.update(arr.tobytes())

        # 16 hex chars = 64 bits of entropy — plenty to detect accidental
        # drift, and short enough to eyeball in the dashboard.
        self._fingerprint = h.hexdigest()[:16]
        self._fingerprint_inputs = {"shapes": shape_map, "dtypes": dtype_map}

    # ── output ───────────────────────────────────────────────────────────────

    def write_log(self) -> Optional[Dict[str, Any]]:
        return {
            "input_seed": self._input_seed,
            "fingerprint_sha256_16": self._fingerprint,
            "input_shapes": self._fingerprint_inputs.get("shapes", {}),
            "input_dtypes": self._fingerprint_inputs.get("dtypes", {}),
        }
