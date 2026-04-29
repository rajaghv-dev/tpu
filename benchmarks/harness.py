#!/usr/bin/env python3
"""
Benchmark harness — Stage 1 CLI entry point.

Usage examples:
  python benchmarks/harness.py --suite smoke --device tpu
  python benchmarks/harness.py --suite quick --device cpu --precision bf16
  python benchmarks/harness.py --model bert_base --device gpu --dry-run

Suites:
  smoke  — 1 model (BERT-base), BF16 only,  ~8 min on v5e-1
  quick  — 5 models, BF16 only,             ~50 min on v5e-1

Results append to results/runs.jsonl (one JSON line per experiment).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.runner import ExperimentConfig, run_experiment

# ── Suite definitions ─────────────────────────────────────────────────────────
SUITES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "model_ids": ["bert_base"],
        "precisions": ["bf16"],
        "description": "1 model · BF16 · ~8 min on v5e-1",
    },
    "quick": {
        "model_ids": ["bert_base", "vit_b16", "gpt2", "whisper_base", "clip_vit_b32"],
        "precisions": ["bf16"],
        "description": "5 models · BF16 · ~50 min on v5e-1",
    },
}

# Hourly preemptible rates (USD)
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

_REPO_ROOT = Path(__file__).parent.parent


# ── Registry loading ──────────────────────────────────────────────────────────

def load_registry(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load the model registry from YAML.

    Args:
        path: Override registry file path. Defaults to models/registry.yaml.

    Returns:
        List of model entry dicts.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required. Install with: pip install pyyaml")

    registry_path = Path(path) if path else _REPO_ROOT / "models" / "registry.yaml"
    with registry_path.open() as fh:
        data = yaml.safe_load(fh)
    return data["models"]


def filter_registry(
    registry: List[Dict[str, Any]],
    model_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return only models whose id is in model_ids (or all if None)."""
    if model_ids is None:
        return registry
    return [m for m in registry if m["id"] in model_ids]


# ── Config building ───────────────────────────────────────────────────────────

def build_config(
    entry: Dict[str, Any],
    precision: str,
    device: str,
    framework: str = "jax",
) -> ExperimentConfig:
    return ExperimentConfig(
        model_id=entry["id"],
        hf_id=entry["hf_id"],
        task=entry["task"],
        domain=entry["domain"],
        architecture_family=entry["architecture_family"],
        attention_variant=entry["attention_variant"],
        positional_encoding=entry["positional_encoding"],
        is_moe=entry.get("is_moe", False),
        total_params_M=entry["total_params_M"],
        active_params_M=entry["active_params_M"],
        input_type=entry["input_type"],
        precision=precision,
        framework=framework,
        device=device,
        seq_len=entry.get("default_seq_len", 128),
        batch_size_throughput=entry.get("default_batch_size_throughput", 32),
        vocab_size=entry.get("vocab_size", 30522),
        image_size=entry.get("image_size"),
        n_mels=entry.get("n_mels"),
        n_frames=entry.get("n_frames"),
        device_cost_usd_per_hr=DEVICE_COSTS.get(device, 0.0),
    )


# ── Result persistence ────────────────────────────────────────────────────────

def append_result(result: dict, output_path: Path) -> None:
    """Append one result dict as a JSON line (atomic append)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as fh:
        fh.write(json.dumps(result, default=str) + "\n")


# ── Suite runner ──────────────────────────────────────────────────────────────

def run_suite(
    suite_name: Optional[str],
    model_id: Optional[str],
    device: str,
    framework: str,
    precision: str,
    output_path: Path,
    registry_path: Optional[str],
    dry_run: bool,
) -> int:
    """
    Execute a suite or a single model and append results to JSONL.

    Returns:
        Number of successfully completed experiments (≥0), or -1 on error.
    """
    registry = load_registry(registry_path)

    if model_id:
        models = [m for m in registry if m["id"] == model_id]
        if not models:
            print(f"[error] model '{model_id}' not in registry", file=sys.stderr)
            return -1
        precisions = [precision]
    elif suite_name:
        if suite_name not in SUITES:
            print(
                f"[error] unknown suite '{suite_name}'. "
                f"Available: {', '.join(SUITES)}",
                file=sys.stderr,
            )
            return -1
        suite = SUITES[suite_name]
        models = filter_registry(registry, suite["model_ids"])
        precisions = suite["precisions"]
        print(f"Suite: {suite_name} — {suite['description']}")
    else:
        print("[error] provide --suite or --model", file=sys.stderr)
        return -1

    configs = [
        build_config(entry, prec, device, framework)
        for entry in models
        for prec in precisions
    ]

    print(f"Experiments planned: {len(configs)}")
    if dry_run:
        for cfg in configs:
            cost = f"${cfg.device_cost_usd_per_hr:.2f}/hr" if cfg.device_cost_usd_per_hr > 0 else "local"
            print(f"  [dry-run] {cfg.model_id} | {cfg.precision} | {cfg.device} | {cost}")
        return 0

    completed = 0
    for cfg in configs:
        label = f"{cfg.model_id} | {cfg.precision} | {cfg.device}"
        print(f"\n▶ {label}", flush=True)
        t0 = time.time()
        try:
            result = run_experiment(cfg)
            append_result(result, output_path)
            elapsed = time.time() - t0
            lat = result.get("latency_p50_ms", "?")
            tp = result.get("throughput_mean_samples_sec", "?")
            cv = result.get("latency_cv_pct", "?")
            flags = result.get("flags", [])
            flag_str = f"  ⚠ {flags}" if flags else ""
            print(
                f"  ✓ {elapsed:.1f}s  "
                f"p50={lat}ms  tp={tp} smp/s  CV={cv}%{flag_str}"
            )
            completed += 1
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}", file=sys.stderr)

    print(f"\nDone: {completed}/{len(configs)} experiments written to {output_path}")
    return completed


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harness",
        description="TPU × GPU inference benchmark harness (Stage 1: Path 1 — JAX + XLA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k}: {v['description']}" for k, v in SUITES.items()
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--suite", choices=list(SUITES), help="Pre-defined experiment suite")
    g.add_argument("--model", dest="model_id", help="Single model id from registry")

    p.add_argument(
        "--device",
        default="tpu",
        help="Device: tpu | cpu | gpu | tpu_v5e1 | rtx4090 | b200  (default: tpu)",
    )
    p.add_argument(
        "--framework",
        default="jax",
        choices=["jax"],
        help="Framework (default: jax; Stage 1 supports jax only)",
    )
    p.add_argument(
        "--precision",
        default="bf16",
        choices=["fp32", "bf16"],
        help="Precision for --model runs (default: bf16)",
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "results" / "runs.jsonl"),
        help="Output JSONL file (default: results/runs.jsonl)",
    )
    p.add_argument(
        "--registry",
        default=None,
        help="Override path to models/registry.yaml",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned experiments with cost estimate — no model downloads",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    n = run_suite(
        suite_name=args.suite,
        model_id=args.model_id,
        device=args.device,
        framework=args.framework,
        precision=args.precision,
        output_path=Path(args.output),
        registry_path=args.registry,
        dry_run=args.dry_run,
    )
    # n == -1 on error, 0 on dry-run, >0 on success
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
