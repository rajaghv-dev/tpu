"""
observe/hlo_dump_probe.py — HLO IR dump probe.

Sets `XLA_FLAGS` so that XLA dumps its HLO IR (text form) for every JIT
compile that happens during the run, then walks the dump directory at
the end of the run and produces a small summary suitable for inclusion
in the per-run probe log.

## Caveats

XLA reads `XLA_FLAGS` once when the JAX/XLA backend is initialised. If
JAX has already compiled anything in the current Python process — for
example because a previous `run_experiment` ran in the same process, or
because some import-time code triggered a `jax.jit` — then setting
`XLA_FLAGS` here is a no-op and no dump files will appear. To work
correctly this probe MUST be registered (and `before_run` must fire)
BEFORE any JIT compilation happens in the process. The simplest way to
guarantee that is to register the probe at the top of your harness
script, before the first `run_experiment` call.

When the dump directory does not exist after the run (compile didn't
run, the env var was set too late, or XLA decided not to dump for some
other reason), `write_log` returns `{"available": False, "reason": ...}`
rather than raising — observability code must never break a benchmark.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from observe.probe import Probe

if TYPE_CHECKING:  # avoid runtime import cycle with runner
    from benchmarks.runner import ExperimentConfig

_log = logging.getLogger(__name__)


# Matches an HLO instruction definition like:
#     %add.5 = f32[1024,768]{1,0} add(%lhs, %rhs)
# We only need the LHS up through the type tag to count an instruction;
# the wide bracket of dtypes here is not exhaustive but covers the
# common cases (f32, bf16, f16, f64, s32, s64, u32, u64, pred, c64, c128).
_HLO_INSTR_RE = re.compile(
    r"=\s+(?:f16|f32|f64|bf16|s8|s16|s32|s64|u8|u16|u32|u64|pred|c64|c128)\["
)

# Capture the op kind from a line like `... = f32[...] add(...)` —
# the token immediately after `]` and (optional) layout `{...}`, before
# the `(` of the operand list.
_HLO_OP_RE = re.compile(
    r"=\s+(?:f16|f32|f64|bf16|s8|s16|s32|s64|u8|u16|u32|u64|pred|c64|c128)"
    r"\[[^\]]*\](?:\{[^}]*\})?\s+([A-Za-z_][A-Za-z0-9_-]*)\("
)


class HloDumpProbe(Probe):
    """
    Probe that asks XLA to dump HLO IR text and summarises it.

    Outputs a `hlo_dump.json` file alongside other probe logs.
    """

    name = "hlo_dump"

    def __init__(self) -> None:
        self._dump_dir: Optional[Path] = None
        self._original_xla_flags: Optional[str] = None
        self._xla_flags_set: Optional[str] = None
        # Populated incrementally:
        self._snapshot_after_compile: Optional[Dict[str, Any]] = None
        self._summary: Optional[Dict[str, Any]] = None

    # ── Lifecycle ────────────────────────────────────────────────────────
    def before_run(
        self,
        run_id: str,
        config: "ExperimentConfig",
        log_dir: Path,
    ) -> None:
        """
        Set XLA_FLAGS to enable HLO text dump into <log_dir>/hlo.

        Saves the prior XLA_FLAGS value so `after_run` can restore it; the
        test harness may reuse the same Python process to run several
        experiments and we do not want our flags to leak.
        """
        self._dump_dir = (log_dir / "hlo").resolve()
        self._dump_dir.mkdir(parents=True, exist_ok=True)

        existing = os.environ.get("XLA_FLAGS", "")
        self._original_xla_flags = existing  # may be ""
        added = (
            f"--xla_dump_to={self._dump_dir} "
            f"--xla_dump_hlo_as_text "
            f"--xla_dump_hlo_pass_re=.*"
        )
        merged = (existing + " " + added).strip() if existing else added
        os.environ["XLA_FLAGS"] = merged
        self._xla_flags_set = merged

    def after_phase(self, phase_name: str, duration_s: float) -> None:
        """After the compile phase, snapshot file count + total size."""
        if phase_name != "compile" or self._dump_dir is None:
            return
        try:
            files = self._list_txt_files()
            total_bytes = sum(f.stat().st_size for f in files)
            self._snapshot_after_compile = {
                "n_files": len(files),
                "total_bytes": total_bytes,
            }
        except OSError as exc:
            _log.warning("hlo_dump after_phase(compile) failed: %s", exc)
            self._snapshot_after_compile = {"error": str(exc)}

    def after_run(
        self,
        run_id: str,
        result: Optional[Dict[str, Any]],
    ) -> None:
        """Walk the dump dir and build the summary. Restore XLA_FLAGS."""
        try:
            self._summary = self._build_summary()
        finally:
            # Always restore env so the next experiment in the same
            # process gets a clean slate.
            if self._original_xla_flags is None:
                pass
            elif self._original_xla_flags == "":
                os.environ.pop("XLA_FLAGS", None)
            else:
                os.environ["XLA_FLAGS"] = self._original_xla_flags

    # ── Output ───────────────────────────────────────────────────────────
    def write_log(self) -> Optional[Dict[str, Any]]:
        if self._summary is None:
            # after_run wasn't called, or before_run wasn't called. Either
            # way return a degraded record so the file at least exists.
            return {
                "dump_dir": str(self._dump_dir) if self._dump_dir else None,
                "available": False,
                "reason": "after_run did not populate summary",
                "xla_flags_set": self._xla_flags_set,
            }
        return self._summary

    # ── Internals ────────────────────────────────────────────────────────
    def _list_txt_files(self) -> list[Path]:
        if self._dump_dir is None or not self._dump_dir.exists():
            return []
        return [p for p in self._dump_dir.rglob("*.txt") if p.is_file()]

    def _build_summary(self) -> Dict[str, Any]:
        if self._dump_dir is None or not self._dump_dir.exists():
            return {
                "dump_dir": str(self._dump_dir) if self._dump_dir else None,
                "available": False,
                "reason": "dump dir does not exist (compile may not have run, or XLA_FLAGS was set too late)",
                "xla_flags_set": self._xla_flags_set,
            }

        try:
            files = self._list_txt_files()
        except OSError as exc:
            return {
                "dump_dir": str(self._dump_dir),
                "available": False,
                "reason": f"could not list dump dir: {exc}",
                "xla_flags_set": self._xla_flags_set,
            }

        if not files:
            return {
                "dump_dir": str(self._dump_dir),
                "available": False,
                "reason": "no .txt files found (XLA_FLAGS likely set too late — register HloDumpProbe before any jax.jit runs)",
                "n_files": 0,
                "total_bytes": 0,
                "xla_flags_set": self._xla_flags_set,
            }

        total_bytes = 0
        for f in files:
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass

        # Largest file is usually the optimized HLO — most informative
        # for an op-frequency histogram.
        largest = max(files, key=lambda f: self._safe_size(f))
        instruction_count, fusion_count, top_ops = self._parse_hlo_file(largest)

        return {
            "dump_dir": str(self._dump_dir),
            "available": True,
            "n_files": len(files),
            "total_bytes": total_bytes,
            "largest_file": str(largest),
            "largest_file_bytes": self._safe_size(largest),
            "instruction_count_estimate": instruction_count,
            "fusion_count_estimate": fusion_count,
            "top_ops": dict(top_ops),
            "xla_flags_set": self._xla_flags_set,
        }

    @staticmethod
    def _safe_size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _parse_hlo_file(path: Path) -> tuple[int, int, list[tuple[str, int]]]:
        """
        Return (instruction_count, fusion_count, top_10_op_kinds).

        Parsing is line-based and tolerant — HLO text is large, and our
        goal is a coarse summary, not an authoritative parse.
        """
        instruction_count = 0
        fusion_count = 0
        op_counter: Counter[str] = Counter()

        try:
            with path.open("r", errors="replace") as f:
                for line in f:
                    if _HLO_INSTR_RE.search(line):
                        instruction_count += 1
                        m = _HLO_OP_RE.search(line)
                        if m:
                            op_counter[m.group(1)] += 1
                    # `kFusion` shows up in compiler-pass output;
                    # `fusion(` in instruction lists. Either signals a
                    # fusion group.
                    if "kFusion" in line or "fusion(" in line:
                        fusion_count += 1
        except OSError as exc:
            _log.warning("could not read %s: %s", path, exc)
            return 0, 0, []

        return instruction_count, fusion_count, op_counter.most_common(10)
