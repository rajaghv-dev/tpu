"""
Tests for scripts/render_results.py.

We don't import the script as a module — it lives under scripts/ which isn't a
package. Instead we load it via importlib.util so the tests work regardless of
sys.path config. The tests build a tmp_path-rooted fake `results/` tree and
assert the rendered MD contains the substrings we expect.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "render_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_results", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rr():
    return _load_module()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_success_row(run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": "2026-05-06T00:00:00Z",
        "model": "bert_base",
        "device": "tpu",
        "precision": "bf16",
        "framework": "jax",
        "first_compile_s": 5.0,
        "latency_p50_ms": 0.5,
        "latency_p95_ms": 0.6,
        "latency_p99_ms": 0.7,
        "latency_cv_pct": 1.0,
        "throughput_mean_samples_sec": 1000.0,
        "cost_per_1k_samples_usd": 1e-5,
    }


# ── _read_run_log loads every JSON file ──────────────────────────────────────

def test_read_run_log_loads_all_json(tmp_path, rr):
    run_id = "abc"
    log_dir = tmp_path / "run_logs" / run_id
    _write_json(log_dir / "lineage.json", {"git_sha": "deadbeef"})
    _write_json(log_dir / "timing.json", {"phase_summary": {"compile": 1.0}})
    _write_json(log_dir / "memory.json", {"baseline_rss_mb": 200})
    out = rr._read_run_log(tmp_path / "run_logs", run_id)
    assert set(out.keys()) == {"lineage.json", "timing.json", "memory.json"}
    assert out["timing.json"]["phase_summary"]["compile"] == 1.0


def test_read_run_log_skips_corrupt(tmp_path, rr):
    run_id = "abc"
    log_dir = tmp_path / "run_logs" / run_id
    log_dir.mkdir(parents=True)
    (log_dir / "lineage.json").write_text('{"ok": true}')
    (log_dir / "broken.json").write_text("{not valid json")
    out = rr._read_run_log(tmp_path / "run_logs", run_id)
    assert "lineage.json" in out
    assert "broken.json" not in out


def test_read_run_log_handles_missing_dir(tmp_path, rr):
    out = rr._read_run_log(tmp_path / "run_logs", "no-such-run")
    assert out == {}


def test_read_run_log_handles_missing_run_id(tmp_path, rr):
    assert rr._read_run_log(tmp_path / "run_logs", None) == {}
    assert rr._read_run_log(tmp_path / "run_logs", "") == {}


# ── timing.json summariser ────────────────────────────────────────────────────

def test_render_timing_orders_by_phase(rr):
    # Insert phases out-of-order; renderer should reorder by _PHASE_ORDER.
    payload = {
        "phase_summary": {
            "throughput": 4.0,
            "compile": 2.0,
            "preflight": 1.0,
            "latency": 3.0,
            "unknown_phase": 9.9,
        }
    }
    md = "\n".join(rr._summarise_timing(payload))
    assert "| phase | duration_s |" in md
    # Known phases appear before unknown.
    pre = md.index("preflight")
    cmp_idx = md.index("compile")
    lat = md.index("latency")
    tput = md.index("throughput")
    unk = md.index("unknown_phase")
    assert pre < cmp_idx < lat < tput < unk


# ── cloud_monitoring summariser ──────────────────────────────────────────────

def test_render_cloud_monitoring_unavailable(rr):
    md = "\n".join(rr._summarise_cloud_monitoring(
        {"available": False, "reason": "no perms"}
    ))
    assert "not available" in md
    assert "no perms" in md


def test_render_cloud_monitoring_per_phase(rr):
    payload = {
        "available": True,
        "per_phase_summary": {
            "latency": {
                "mxu": {"min": 10, "mean": 30, "max": 50},
                "hbm_util_pct": {"min": 20, "mean": 40, "max": 60},
            }
        },
    }
    md = "\n".join(rr._summarise_cloud_monitoring(payload))
    assert "mxu min" in md and "hbm util %" in md
    assert "latency" in md
    # Each numeric value should appear in the rendered table.
    for v in (10, 30, 50, 20, 40, 60):
        assert str(v) in md


# ── hlo_dump summariser ──────────────────────────────────────────────────────

def test_render_hlo_dump_top_ops_list(rr):
    payload = {
        "instruction_count_estimate": 1234,
        "fusion_count_estimate": 56,
        "top_ops": ["dot", "add", "mul", "transpose", "reshape", "extra_op"],
    }
    md = "\n".join(rr._summarise_hlo_dump(payload))
    assert "1234" in md and "56" in md
    # Only first 5 ops listed.
    assert "dot" in md and "reshape" in md
    assert "extra_op" not in md


def test_render_hlo_dump_top_ops_dict(rr):
    payload = {
        "instruction_count_estimate": 1,
        "fusion_count_estimate": 2,
        "top_ops": {"dot": 100, "add": 50},
    }
    md = "\n".join(rr._summarise_hlo_dump(payload))
    assert "dot" in md and "100" in md


# ── memory summariser ────────────────────────────────────────────────────────

def test_render_memory(rr):
    payload = {
        "baseline_rss_mb": 250.5,
        "delta_from_baseline_mb": {
            "model_load": 1200.0,
            "compile": 1500.0,
        },
    }
    md = "\n".join(rr._summarise_memory(payload))
    assert "Baseline RSS" in md and "250.5" in md
    assert "model_load" in md and "1200" in md
    assert "compile" in md and "1500" in md


# ── End-to-end run report rendering ──────────────────────────────────────────

def test_render_run_report_includes_probes(tmp_path, rr):
    row = _make_success_row("run-1")
    log_files = {
        "lineage.json": {"git_sha": "abc"},
        "timing.json": {"phase_summary": {"compile": 2.5, "latency": 0.5}},
        "memory.json": {
            "baseline_rss_mb": 100,
            "delta_from_baseline_mb": {"compile": 500.0},
        },
        "input_fingerprint.json": {"hash": "0xdeadbeef", "shape": [1, 128]},
    }
    md = rr._render_run_report(row, log_files)
    # Section header.
    assert "## Probes" in md
    # All non-lineage/error probes show up.
    assert "### timing" in md
    assert "### memory" in md
    assert "### input_fingerprint" in md
    # lineage isn't rendered as a probe.
    assert "### lineage" not in md
    # Numbers from the probes are present.
    assert "2.5" in md
    assert "100" in md
    assert "0xdeadbeef" in md


def test_render_run_report_no_probes_still_works(rr):
    row = _make_success_row("run-2")
    md = rr._render_run_report(row, {})
    # No probes section when there are no probe files.
    assert "## Probes" not in md
    # Standard groups still render.
    assert "## Identity" in md
    assert "## Latency" in md


def test_render_run_report_failure_with_probes(rr):
    row = {
        "run_id": "run-3",
        "status": "failed",
        "model": "bert_base",
        "device": "tpu",
        "precision": "bf16",
        "phase": "compile",
        "exception_type": "RuntimeError",
        "exception_message": "boom",
        "error_category": "compile_error",
    }
    log_files = {
        "error.json": {
            "traceback": "Traceback...\nRuntimeError: boom\n",
            "lineage": {"git_sha": "abc"},
        },
        "timing.json": {"phase_summary": {"preflight": 0.1}},
    }
    md = rr._render_run_report(row, log_files)
    assert "Failure summary" in md
    assert "Traceback..." in md
    # Probes section still appears on failure.
    assert "## Probes" in md
    assert "### timing" in md


# ── End-to-end RESULTS.md rendering ──────────────────────────────────────────

def test_render_results_index_probe_columns_and_coverage(tmp_path, rr):
    run_logs_dir = tmp_path / "run_logs"
    # Row 1: has timing+memory.
    _write_json(run_logs_dir / "run-A" / "timing.json", {"phase_summary": {}})
    _write_json(run_logs_dir / "run-A" / "memory.json", {"baseline_rss_mb": 1})
    _write_json(run_logs_dir / "run-A" / "lineage.json", {})
    # Row 2: just lineage, no probes.
    _write_json(run_logs_dir / "run-B" / "lineage.json", {})

    rows = [_make_success_row("run-A"), _make_success_row("run-B")]
    rows[1]["model"] = "gpt2_small"

    md = rr._render_results_index(rows, run_logs_dir=run_logs_dir)
    # Probes-attached column includes both probe names for run-A.
    assert "timing, memory" in md or "memory, timing" in md
    # Coverage section appears.
    assert "## Probe coverage" in md
    # timing covers 1 of 2 rows; memory covers 1 of 2 rows.
    assert "timing | 1 / 2" in md
    assert "memory | 1 / 2" in md


def test_render_results_index_no_probes_no_coverage_section(tmp_path, rr):
    run_logs_dir = tmp_path / "run_logs"
    _write_json(run_logs_dir / "run-A" / "lineage.json", {})
    rows = [_make_success_row("run-A")]
    md = rr._render_results_index(rows, run_logs_dir=run_logs_dir)
    # No probe files exist, so no coverage section.
    assert "## Probe coverage" not in md
    # "Probes attached" column shows em-dash placeholder.
    assert "| — |" in md


def test_render_results_index_backwards_compat_no_run_logs_dir(rr):
    # When called without run_logs_dir (older callers), it shouldn't crash.
    rows = [_make_success_row("run-X")]
    md = rr._render_results_index(rows)
    assert "## Successful runs" in md
    # No coverage section since we couldn't introspect probes.
    assert "## Probe coverage" not in md
