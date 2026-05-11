#!/usr/bin/env python3
"""
Training harness — Stage 1.6 CLI for train/runner.py.

Mirrors benchmarks/harness.py. Auto-registers the default-on training probe
set (Timing, Memory, InputFingerprint, TrainingMetrics, StepTiming, Checkpoint)
unless --probes none is passed.

Usage examples:
  python -m train.harness --suite smoke --device tpu
  python -m train.harness --task bert_finetune --device cpu --steps 5 --dry-run
  python -m train.harness --task bert_finetune --device tpu --probes full

Suites:
  smoke  — bert_finetune · 10 steps · BF16 · ~1 min on v5e-1
  quick  — bert_finetune · 200 steps · BF16 · ~5 min on v5e-1

Results append to results/training_runs.jsonl (separate index from inference).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.checkpoint_probe import CheckpointProbe
from observe.input_fingerprint import InputFingerprintProbe
from observe.memory_probe import MemoryProbe
from observe.probe import (
    clear_probes,
    register_probe,
)
from observe.step_timing_probe import StepTimingProbe
from observe.timing_probe import TimingProbe
from observe.training_metrics_probe import TrainingMetricsProbe

# Imported lazily inside main so --dry-run and --help don't pull in JAX.

SUITES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "task_ids": ["bert_finetune"],
        "precisions": ["bf16"],
        "n_steps": 10,
        "n_eval_steps": 2,
        "description": "1 task · 10 steps · BF16 · ~1 min on v5e-1",
    },
    "quick": {
        "task_ids": ["bert_finetune"],
        "precisions": ["bf16"],
        "n_steps": 200,
        "n_eval_steps": 10,
        "description": "1 task · 200 steps · BF16 · ~5 min on v5e-1",
    },
}

DEVICE_COSTS: Dict[str, float] = {
    "tpu_v5e1": 0.36, "tpu_v6e1": 0.75,
    "rtx3080": 0.0, "rtx4090": 0.0, "b200": 0.0,
    "cpu": 0.0, "tpu": 0.36, "gpu": 0.0,
}

_REPO_ROOT = Path(__file__).parent.parent


def load_registry(path: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required. Install with: pip install pyyaml")
    p = Path(path) if path else _REPO_ROOT / "train" / "registry.yaml"
    with p.open() as fh:
        data = yaml.safe_load(fh)
    return data["tasks"]


def build_config(
    entry: Dict[str, Any],
    precision: str,
    device: str,
    framework: str = "jax",
    n_steps_override: Optional[int] = None,
    n_eval_steps_override: Optional[int] = None,
    save_checkpoint: bool = False,
):
    from train.runner import TrainingExperimentConfig
    return TrainingExperimentConfig(
        task_id=entry["id"],
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
        batch_size=entry.get("default_batch_size", 32),
        vocab_size=entry.get("vocab_size", 30522),
        num_labels=entry.get("num_labels", 2),
        n_steps=n_steps_override
            if n_steps_override is not None
            else entry.get("default_steps", 200),
        n_eval_steps=n_eval_steps_override
            if n_eval_steps_override is not None
            else entry.get("default_eval_steps", 10),
        lr=entry.get("default_lr", 2.0e-5),
        lr_warmup_steps=entry.get("default_warmup_steps", 20),
        weight_decay=entry.get("default_weight_decay", 0.01),
        optimizer=entry.get("default_optimizer", "adamw"),
        save_checkpoint=save_checkpoint,
        device_cost_usd_per_hr=DEVICE_COSTS.get(device, 0.0),
    )


# ── Probe wiring ──────────────────────────────────────────────────────────────


def _register_probe_set(name: str) -> List[str]:
    """
    Register a named probe set onto the global probe registry.

    Sets:
      none    — clear all
      default — Timing + Memory + InputFingerprint + TrainingMetrics + StepTiming
      full    — default + CheckpointProbe + JaxProfilerProbe + HloDumpProbe
                + OTelProbe + CloudMonitoringProbe
    """
    clear_probes()
    if name == "none":
        return []

    probes = [
        TimingProbe(),
        MemoryProbe(),
        InputFingerprintProbe(),
        TrainingMetricsProbe(),
        StepTimingProbe(),
    ]
    if name == "full":
        # Heavy / opt-in probes. Imported lazily so a missing optional dep
        # (otel, google-cloud-monitoring) only matters when "full" is asked.
        try:
            from observe.checkpoint_probe import CheckpointProbe as _Cp
            probes.append(_Cp())
        except Exception:
            pass
        try:
            from observe.jax_profiler_probe import JaxProfilerProbe
            probes.append(JaxProfilerProbe())
        except Exception:
            pass
        try:
            from observe.hlo_dump_probe import HloDumpProbe
            probes.append(HloDumpProbe())
        except Exception:
            pass
        try:
            from observe.otel_probe import OTelProbe
            probes.append(OTelProbe())
        except Exception:
            pass
        try:
            from observe.cloud_monitoring_probe import CloudMonitoringProbe
            probes.append(CloudMonitoringProbe())
        except Exception:
            pass
    elif name == "default":
        # Default already includes CheckpointProbe — small overhead, useful.
        probes.append(CheckpointProbe())

    names = []
    for p in probes:
        try:
            register_probe(p)
            names.append(p.name)
        except ValueError:
            pass  # already registered
    return names


def append_result(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as fh:
        fh.write(json.dumps(result, default=str) + "\n")


def run_suite(
    suite_name: Optional[str],
    task_id: Optional[str],
    device: str,
    framework: str,
    precision: str,
    n_steps_override: Optional[int],
    n_eval_steps_override: Optional[int],
    save_checkpoint: bool,
    probes_set: str,
    output_path: Path,
    registry_path: Optional[str],
    dry_run: bool,
) -> int:
    registry = load_registry(registry_path)

    if task_id:
        tasks = [t for t in registry if t["id"] == task_id]
        if not tasks:
            print(f"[error] task '{task_id}' not in registry", file=sys.stderr)
            return -1
        precisions = [precision]
        n_steps = n_steps_override
        n_eval = n_eval_steps_override
    elif suite_name:
        if suite_name not in SUITES:
            print(
                f"[error] unknown suite '{suite_name}'. Available: {', '.join(SUITES)}",
                file=sys.stderr,
            )
            return -1
        suite = SUITES[suite_name]
        tasks = [t for t in registry if t["id"] in suite["task_ids"]]
        precisions = suite["precisions"]
        n_steps = suite["n_steps"]
        n_eval = suite["n_eval_steps"]
        print(f"Suite: {suite_name} — {suite['description']}")
    else:
        print("[error] provide --suite or --task", file=sys.stderr)
        return -1

    configs = [
        build_config(
            t, prec, device, framework,
            n_steps_override=n_steps,
            n_eval_steps_override=n_eval,
            save_checkpoint=save_checkpoint,
        )
        for t in tasks for prec in precisions
    ]

    print(f"Experiments planned: {len(configs)}")
    if dry_run:
        for cfg in configs:
            print(
                f"  [dry-run] {cfg.task_id} | {cfg.precision} | {cfg.device} | "
                f"steps={cfg.n_steps} bs={cfg.batch_size} seq={cfg.seq_len}"
            )
        return 0

    registered = _register_probe_set(probes_set)
    if registered:
        print(f"Probes ({probes_set}): {', '.join(registered)}")
    else:
        print("Probes: none")

    # Lazy import — see top-of-file note.
    from benchmarks.runner import BenchmarkError
    from train.runner import run_training

    completed = 0
    failed = 0
    for cfg in configs:
        label = f"{cfg.task_id} | {cfg.precision} | {cfg.device} | steps={cfg.n_steps}"
        print(f"\n▶ {label}", flush=True)
        t0 = time.time()
        try:
            result = run_training(cfg)
            append_result(result, output_path)
            elapsed = time.time() - t0
            print(
                f"  ✓ {elapsed:.1f}s  "
                f"final_loss={result.get('final_train_loss'):.4f}  "
                f"eval_loss={result.get('eval_loss'):.4f}  "
                f"throughput={result.get('throughput_samples_sec'):.0f} smp/s"
            )
            completed += 1
        except BenchmarkError as exc:
            print(
                f"  ✗ FAILED [{exc.phase}/{exc.error_category}] "
                f"{exc.original_type}: {exc.original_message}",
                file=sys.stderr,
            )
            try:
                run_logs = Path(_REPO_ROOT) / "results" / "run_logs"
                latest = max(
                    (p for p in run_logs.iterdir() if p.is_dir()),
                    key=lambda p: p.stat().st_mtime,
                )
                run_id_for_stub = latest.name
            except (ValueError, OSError, FileNotFoundError):
                run_id_for_stub = None
            append_result({
                "kind": "training",
                "status": "failed",
                "run_id": run_id_for_stub,
                "task_id": cfg.task_id,
                "device": cfg.device,
                "precision": cfg.precision,
                "phase": exc.phase,
                "exception_type": exc.original_type,
                "exception_message": exc.original_message,
                "error_category": exc.error_category,
            }, output_path)
            failed += 1
        except KeyboardInterrupt:
            print(f"  ✗ INTERRUPTED — partial results in {output_path}", file=sys.stderr)
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ FAILED (unhandled): {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1

    summary = f"Done: {completed}/{len(configs)} succeeded"
    if failed > 0:
        summary += f", {failed} failed (see results/run_logs/<run_id>/error.json)"
    summary += f" → {output_path}"
    print(f"\n{summary}")
    return completed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.harness",
        description="TPU × GPU training observability harness (Stage 1.6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k}: {v['description']}" for k, v in SUITES.items()),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--suite", choices=list(SUITES))
    g.add_argument("--task", dest="task_id", help="Single task id from train/registry.yaml")

    p.add_argument("--device", default="tpu",
                   help="tpu | cpu | gpu | tpu_v5e1 | rtx4090 | b200 (default: tpu)")
    p.add_argument("--framework", default="jax", choices=["jax"])
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--steps", dest="n_steps", type=int, default=None,
                   help="Override training steps")
    p.add_argument("--eval-steps", dest="n_eval_steps", type=int, default=None)
    p.add_argument("--save-checkpoint", action="store_true",
                   help="Write a final-state checkpoint to run_logs/<run_id>/checkpoints/")
    p.add_argument("--probes", default="default", choices=["none", "default", "full"],
                   help="Probe set to auto-register (default: default)")
    p.add_argument("--output", default=str(_REPO_ROOT / "results" / "training_runs.jsonl"),
                   help="Output JSONL (default: results/training_runs.jsonl)")
    p.add_argument("--registry", default=None, help="Override path to train/registry.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned experiments — no model downloads")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    n = run_suite(
        suite_name=args.suite,
        task_id=args.task_id,
        device=args.device,
        framework=args.framework,
        precision=args.precision,
        n_steps_override=args.n_steps,
        n_eval_steps_override=args.n_eval_steps,
        save_checkpoint=args.save_checkpoint,
        probes_set=args.probes,
        output_path=Path(args.output),
        registry_path=args.registry,
        dry_run=args.dry_run,
    )
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
