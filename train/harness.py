#!/usr/bin/env python3
"""
Training harness — Stage 1.6+ CLI for train/runner.py.

Mirrors benchmarks/harness.py. Auto-registers a probe set (minimal/default/full)
unless --probes none is passed.

## Usage examples

  python -m train.harness --suite smoke --device tpu
  python -m train.harness --task bert_finetune --device cpu --steps 5 --dry-run
  python -m train.harness --task gpt2_lm --device tpu --probes full
  python -m train.harness --suite diverse --device tpu --probes default
  python -m train.harness --task gpt2_medium_lm --grad-accum 8 --max-grad-norm 1.0

## Suites

  smoke         — bert · 10 steps · BF16 · ~1 min on v5e-1
  quick         — bert · 200 steps · BF16 · ~5 min on v5e-1
  causal_smoke  — distilgpt2 · 10 steps · BF16 · ~1 min
  causal_quick  — gpt2 · 100 steps · BF16 · ~5 min
  vit_smoke     — vit-base · 10 steps · BF16 · ~1 min
  vit_quick     — vit-base · 100 steps · BF16 · ~5 min
  diverse       — one tiny task from each domain (bert + distilgpt2 + resnet50)
                  · 20 steps each · sanity-check the whole task dispatch
  scaling       — distilgpt2 → gpt2 → gpt2-medium · 50 steps each · shows
                  how per-step time grows with parameter count

## Probe sets

  none     — no probes
  minimal  — Timing, Memory, TrainingMetrics, StepTiming (≈zero overhead)
  default  — minimal + InputFingerprint, Checkpoint, DeviceInfo,
             Determinism, XlaCompile (one-shot setup, no per-step cost)
  full     — default + PowerThermal (1 Hz background sampler) + heavy
             opt-in probes (JaxProfiler, HloDump, OTel, CloudMonitoring)

Results append to results/training_runs.jsonl (separate index from inference).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "causal_smoke": {
        "task_ids": ["distilgpt2_lm"],
        "precisions": ["bf16"],
        "n_steps": 10,
        "n_eval_steps": 2,
        "description": "distilgpt2 · 10 steps · BF16 · ~1 min on v5e-1",
    },
    "causal_quick": {
        "task_ids": ["gpt2_lm"],
        "precisions": ["bf16"],
        "n_steps": 100,
        "n_eval_steps": 5,
        "description": "gpt2 · 100 steps · BF16 · ~5 min on v5e-1",
    },
    "vit_smoke": {
        "task_ids": ["vit_b16_finetune"],
        "precisions": ["bf16"],
        "n_steps": 10,
        "n_eval_steps": 2,
        "description": "vit-base · 10 steps · BF16 · ~1 min on v5e-1",
    },
    "vit_quick": {
        "task_ids": ["vit_b16_finetune"],
        "precisions": ["bf16"],
        "n_steps": 100,
        "n_eval_steps": 5,
        "description": "vit-base · 100 steps · BF16 · ~5 min on v5e-1",
    },
    "diverse": {
        # One small task per domain — exercises every task dispatch path in
        # the runner. Useful as a CI gate after touching _build_train_step.
        "task_ids": ["bert_finetune", "distilgpt2_lm", "resnet50_finetune"],
        "precisions": ["bf16"],
        "n_steps": 20,
        "n_eval_steps": 3,
        "description": "3 tasks (bert + distilgpt2 + resnet50) · 20 steps each",
    },
    "scaling": {
        # Same task family at three different scales — for per-step-time
        # vs. parameter-count plots. distilgpt2 → gpt2 → gpt2-medium.
        "task_ids": ["distilgpt2_lm", "gpt2_lm", "gpt2_medium_lm"],
        "precisions": ["bf16"],
        "n_steps": 50,
        "n_eval_steps": 3,
        "description": "scaling sweep: distilgpt2 (82M) → gpt2 (124M) → gpt2-medium (355M)",
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
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Build a TrainingExperimentConfig from a registry entry.

    `overrides` is a dict of CLI-provided knobs that beat the registry default
    (e.g. {"max_grad_norm": 0.5, "optimizer": "lion"}). Keys not present in
    `overrides` fall through to the registry value, then to the dataclass
    default.
    """
    from train.runner import TrainingExperimentConfig
    overrides = overrides or {}

    def pick(cli_key: str, registry_key: str, fallback: Any) -> Any:
        if cli_key in overrides and overrides[cli_key] is not None:
            return overrides[cli_key]
        return entry.get(registry_key, fallback)

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
        image_size=entry.get("image_size"),
        n_steps=(
            n_steps_override
            if n_steps_override is not None
            else entry.get("default_steps", 200)
        ),
        n_eval_steps=(
            n_eval_steps_override
            if n_eval_steps_override is not None
            else entry.get("default_eval_steps", 10)
        ),
        lr=entry.get("default_lr", 2.0e-5),
        lr_warmup_steps=entry.get("default_warmup_steps", 20),
        lr_schedule=pick("lr_schedule", "default_lr_schedule", "linear"),
        weight_decay=entry.get("default_weight_decay", 0.01),
        optimizer=pick("optimizer", "default_optimizer", "adamw"),
        max_grad_norm=pick("max_grad_norm", "default_max_grad_norm", 1.0),
        grad_accum_steps=pick("grad_accum_steps", "default_grad_accum_steps", 1),
        eval_seed=overrides.get("eval_seed", 1337),
        deterministic=overrides.get("deterministic", False),
        save_checkpoint=save_checkpoint,
        device_cost_usd_per_hr=DEVICE_COSTS.get(device, 0.0),
    )


# ── Probe wiring ──────────────────────────────────────────────────────────────


def _try_import_optional(module_path: str, class_name: str) -> Optional[Any]:
    """Import a probe class lazily; return None on any failure."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except Exception:  # noqa: BLE001 — any failure → optional probe is skipped
        return None


def _register_probe_set(name: str) -> Tuple[List[str], List[str]]:
    """
    Register a named probe set onto the global probe registry.

    Returns (registered_names, skipped_optional_names). The skipped list lets
    the harness print "PowerThermal skipped (missing nvidia-smi/psutil)" so
    users know what's missing rather than wondering why the file isn't there.

    Sets:
      none     — clear all
      minimal  — Timing + Memory + TrainingMetrics + StepTiming
      default  — minimal + InputFingerprint + Checkpoint + DeviceInfo +
                 Determinism + XlaCompile
      full     — default + PowerThermal + JaxProfiler + HloDump +
                 OTel + CloudMonitoring
    """
    clear_probes()
    if name == "none":
        return [], []

    # Always-on baseline (these have no optional deps beyond stdlib + psutil
    # which both degrade gracefully).
    probes: List[Any] = [
        TimingProbe(),
        MemoryProbe(),
        TrainingMetricsProbe(),
        StepTimingProbe(),
    ]
    skipped: List[str] = []

    if name in ("default", "full"):
        probes.extend([
            InputFingerprintProbe(),
            CheckpointProbe(),
        ])
        # The new Stage-1.6 observability probes — one-shot setup, no
        # per-step cost. We treat them as optional in case the file fails
        # to import on an older codebase.
        for module_path, cls_name in (
            ("observe.device_info_probe", "DeviceInfoProbe"),
            ("observe.determinism_probe", "DeterminismProbe"),
            ("observe.xla_compile_probe", "XlaCompileProbe"),
        ):
            cls = _try_import_optional(module_path, cls_name)
            if cls is None:
                skipped.append(cls_name)
            else:
                probes.append(cls())

    if name == "full":
        # Heavy / opt-in probes. Imported lazily so a missing optional dep
        # (otel, google-cloud-monitoring, nvidia-smi) only matters when
        # "full" is asked.
        for module_path, cls_name in (
            ("observe.power_thermal_probe", "PowerThermalProbe"),
            ("observe.jax_profiler_probe", "JaxProfilerProbe"),
            ("observe.hlo_dump_probe", "HloDumpProbe"),
            ("observe.otel_probe", "OTelProbe"),
            ("observe.cloud_monitoring_probe", "CloudMonitoringProbe"),
        ):
            cls = _try_import_optional(module_path, cls_name)
            if cls is None:
                skipped.append(cls_name)
            else:
                probes.append(cls())

    names: List[str] = []
    for p in probes:
        try:
            register_probe(p)
            names.append(p.name)
        except ValueError:
            pass  # already registered
    return names, skipped


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
    overrides: Optional[Dict[str, Any]] = None,
) -> int:
    registry = load_registry(registry_path)
    overrides = overrides or {}

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
        n_steps = n_steps_override if n_steps_override is not None else suite["n_steps"]
        n_eval = (
            n_eval_steps_override
            if n_eval_steps_override is not None
            else suite["n_eval_steps"]
        )
        print(f"Suite: {suite_name} — {suite['description']}")
        # If the suite references tasks not yet in the registry, warn — the
        # next refactor will skip them rather than silently shrink the suite.
        missing = set(suite["task_ids"]) - {t["id"] for t in tasks}
        if missing:
            print(
                f"[warn] suite references unknown tasks: {sorted(missing)}",
                file=sys.stderr,
            )
    else:
        print("[error] provide --suite or --task", file=sys.stderr)
        return -1

    configs = [
        build_config(
            t, prec, device, framework,
            n_steps_override=n_steps,
            n_eval_steps_override=n_eval,
            save_checkpoint=save_checkpoint,
            overrides=overrides,
        )
        for t in tasks for prec in precisions
    ]

    print(f"Experiments planned: {len(configs)}")
    if dry_run:
        for cfg in configs:
            print(
                f"  [dry-run] {cfg.task_id} | {cfg.task} | {cfg.precision} | "
                f"{cfg.device} | steps={cfg.n_steps} bs={cfg.batch_size} "
                f"seq={cfg.seq_len} opt={cfg.optimizer} "
                f"clip={cfg.max_grad_norm} accum={cfg.grad_accum_steps} "
                f"sched={cfg.lr_schedule}"
                + (" [deterministic]" if cfg.deterministic else "")
            )
        return 0

    registered, skipped = _register_probe_set(probes_set)
    if registered:
        print(f"Probes ({probes_set}): {', '.join(registered)}")
    else:
        print("Probes: none")
    if skipped:
        print(
            f"Probes skipped ({probes_set}, optional dep missing): "
            f"{', '.join(skipped)}",
            file=sys.stderr,
        )

    # Lazy import — see top-of-file note.
    from benchmarks.runner import BenchmarkError
    from train.runner import run_training

    completed = 0
    failed = 0
    for cfg in configs:
        label = (
            f"{cfg.task_id} | {cfg.task} | {cfg.precision} | {cfg.device} | "
            f"steps={cfg.n_steps}"
        )
        print(f"\n▶ {label}", flush=True)
        t0 = time.time()
        try:
            result = run_training(cfg)
            append_result(result, output_path)
            elapsed = time.time() - t0
            extra = ""
            if result.get("eval_perplexity") is not None:
                extra = f"  ppl={result.get('eval_perplexity'):.2f}"
            print(
                f"  ✓ {elapsed:.1f}s  "
                f"final_loss={result.get('final_train_loss'):.4f}  "
                f"eval_loss={result.get('eval_loss'):.4f}  "
                f"throughput={result.get('throughput_samples_sec'):.0f} smp/s"
                + extra
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
        description="TPU × GPU training observability harness (Stage 1.6+).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Suites:\n"
        + "\n".join(f"  {k}: {v['description']}" for k, v in SUITES.items()),
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

    # ── Training controllability flags (override registry defaults) ───────
    p.add_argument("--optimizer", choices=["adamw", "sgd", "lion", "adafactor"],
                   default=None, help="Override registry optimizer choice")
    p.add_argument("--lr-schedule", choices=["linear", "cosine", "constant"],
                   dest="lr_schedule", default=None,
                   help="Override registry LR schedule")
    p.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=None,
                   help="Global-norm clip threshold (0.0 disables; overrides registry)")
    p.add_argument("--grad-accum", dest="grad_accum_steps", type=int, default=None,
                   help="Gradient accumulation steps (1 = no accumulation)")
    p.add_argument("--eval-seed", dest="eval_seed", type=int, default=None,
                   help="RNG seed for eval batches (default: 1337)")
    p.add_argument("--deterministic", action="store_true",
                   help="Toggle XLA + matmul-precision flags for bit-reproducibility "
                        "(slower; requires CUBLAS_WORKSPACE_CONFIG and XLA_FLAGS set "
                        "externally for full effect)")

    # ── Probe / output flags ──────────────────────────────────────────────
    p.add_argument(
        "--probes", default="default",
        choices=["none", "minimal", "default", "full"],
        help="Probe set to auto-register (default: default)",
    )
    p.add_argument("--output", default=str(_REPO_ROOT / "results" / "training_runs.jsonl"),
                   help="Output JSONL (default: results/training_runs.jsonl)")
    p.add_argument("--registry", default=None, help="Override path to train/registry.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned experiments — no model downloads")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides = {
        "optimizer": args.optimizer,
        "lr_schedule": args.lr_schedule,
        "max_grad_norm": args.max_grad_norm,
        "grad_accum_steps": args.grad_accum_steps,
        "eval_seed": args.eval_seed,
        "deterministic": bool(args.deterministic),
    }
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
        overrides=overrides,
    )
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
