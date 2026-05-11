"""
tests/test_compiler_probes.py — unit tests for the compiler-level probes.

We exercise the no-op and degraded paths only — actually starting a JAX
trace or invoking XLA requires real JAX + a TPU and is out of scope for
unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from observe.hlo_dump_probe import HloDumpProbe
from observe.jax_profiler_probe import JaxProfilerProbe


# ── HloDumpProbe ─────────────────────────────────────────────────────────


def test_hlo_dump_probe_missing_dump_dir(tmp_path: Path) -> None:
    """If the dump dir was never created, write_log reports unavailable."""
    probe = HloDumpProbe()
    # Point dump dir at a path that does NOT exist on disk.
    probe._dump_dir = tmp_path / "does-not-exist"
    probe._summary = probe._build_summary()

    payload = probe.write_log()
    assert payload is not None
    assert payload["available"] is False
    assert "dump dir does not exist" in payload["reason"]


def test_hlo_dump_probe_empty_dump_dir(tmp_path: Path) -> None:
    """Existing dir but no .txt files → unavailable, n_files=0."""
    dump = tmp_path / "hlo"
    dump.mkdir()
    probe = HloDumpProbe()
    probe._dump_dir = dump
    probe._summary = probe._build_summary()

    payload = probe.write_log()
    assert payload is not None
    assert payload["available"] is False
    assert payload["n_files"] == 0
    assert payload["total_bytes"] == 0


def test_hlo_dump_probe_parses_fake_hlo(tmp_path: Path) -> None:
    """A fake HLO text file is summarised correctly."""
    dump = tmp_path / "hlo"
    dump.mkdir()
    fake = dump / "module_0001.optimized_module.txt"
    fake.write_text(
        "HloModule fake_module\n"
        "ENTRY main {\n"
        "  %p0 = f32[128,768]{1,0} parameter(0)\n"
        "  %p1 = f32[768,768]{1,0} parameter(1)\n"
        "  %dot.1 = f32[128,768]{1,0} dot(%p0, %p1)\n"
        "  %add.2 = f32[128,768]{1,0} add(%dot.1, %p0)\n"
        "  %add.3 = bf16[128,768]{1,0} add(%add.2, %p0)\n"
        "  %bcast.4 = f32[128,768,4]{2,1,0} broadcast(%add.2)\n"
        "  %fusion.5 = f32[128,768] fusion(%add.2), kind=kLoop\n"
        "  ROOT %tuple = (f32[128,768]) tuple(%fusion.5)\n"
        "}\n"
        "Pass: kFusion ran on this module.\n"
    )

    probe = HloDumpProbe()
    probe._dump_dir = dump
    probe._summary = probe._build_summary()

    payload = probe.write_log()
    assert payload is not None
    assert payload["available"] is True
    assert payload["n_files"] == 1
    assert payload["total_bytes"] > 0
    # 6 instruction lines (parameter, dot, add, add, broadcast, fusion):
    # 2 parameters + 1 dot + 2 add + 1 broadcast + 1 fusion = 7
    assert payload["instruction_count_estimate"] >= 6
    # `kFusion` (1 line) + `fusion(` in fusion.5 line (1) ≥ 2.
    assert payload["fusion_count_estimate"] >= 2
    # top_ops should include add (twice) and the unique kinds
    top = payload["top_ops"]
    assert "add" in top
    assert top["add"] >= 2
    # xla_flags_set captured (may be None since we didn't run before_run)
    assert "xla_flags_set" in payload


def test_hlo_dump_probe_write_log_without_after_run(tmp_path: Path) -> None:
    """write_log called before after_run still produces a degraded record."""
    probe = HloDumpProbe()
    probe._dump_dir = tmp_path / "hlo"
    payload = probe.write_log()
    assert payload is not None
    assert payload["available"] is False
    assert "after_run" in payload["reason"]


# ── JaxProfilerProbe ─────────────────────────────────────────────────────


def test_jax_profiler_probe_unavailable(tmp_path: Path, monkeypatch) -> None:
    """If jax.profiler import fails, write_log says available=False."""
    # Force the import inside before_run to fail.
    monkeypatch.setitem(sys.modules, "jax", None)

    probe = JaxProfilerProbe()

    class _FakeCfg:
        pass

    probe.before_run("run-xyz", _FakeCfg(), tmp_path)
    # before_phase / after_phase should be safe no-ops here:
    probe.before_phase("latency")
    probe.after_phase("latency", 0.5)

    payload = probe.write_log()
    assert payload is not None
    assert payload["available"] is False
    assert payload["started"] is False
    assert payload["stopped"] is False
    assert payload["reason"] is not None
    assert "jax.profiler unavailable" in payload["reason"]
    assert payload["trace_dir"] is not None


def test_jax_profiler_probe_skips_non_latency_phases(tmp_path: Path, monkeypatch) -> None:
    """Phase hooks for non-latency phases must be no-ops even if profiler exists."""
    monkeypatch.setitem(sys.modules, "jax", None)
    probe = JaxProfilerProbe()

    class _FakeCfg:
        pass

    probe.before_run("run-xyz", _FakeCfg(), tmp_path)
    probe.before_phase("compile")  # should not start a trace
    probe.after_phase("compile", 0.1)
    probe.on_error("compile", RuntimeError("boom"))

    payload = probe.write_log()
    assert payload["started"] is False
    assert payload["stopped"] is False


def test_jax_profiler_probe_on_error_stops_started_trace(tmp_path: Path) -> None:
    """If a trace was started and the phase errors, on_error must stop it."""
    probe = JaxProfilerProbe()
    probe._trace_dir = tmp_path / "jax_profiler"
    probe._trace_dir.mkdir()
    probe._available = True
    probe._started = True

    class _StubProfiler:
        stop_called = 0

        def stop_trace(self) -> None:  # noqa: D401
            type(self).stop_called += 1

    stub = _StubProfiler()
    probe._jax_profiler = stub

    probe.on_error("latency", RuntimeError("kaboom"))
    assert _StubProfiler.stop_called == 1
    assert probe._stopped is True

    # Calling again must not double-stop.
    probe.on_error("latency", RuntimeError("again"))
    assert _StubProfiler.stop_called == 1


def test_jax_profiler_probe_walks_trace_dir(tmp_path: Path) -> None:
    """write_log walks the trace dir and counts files / bytes."""
    probe = JaxProfilerProbe()
    probe._trace_dir = tmp_path / "jax_profiler"
    probe._trace_dir.mkdir()
    (probe._trace_dir / "xspace.pb").write_bytes(b"\x00" * 128)
    nested = probe._trace_dir / "plugins" / "profile" / "host"
    nested.mkdir(parents=True)
    (nested / "events.json.gz").write_bytes(b"\x00" * 64)

    payload = probe.write_log()
    assert payload["n_files"] == 2
    assert payload["total_bytes"] == 128 + 64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
