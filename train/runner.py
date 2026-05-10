"""
train/runner.py — single-experiment training runner.

Mirrors benchmarks/runner.py but for training. Same probe layer (observe/),
same per-phase context manager, same BenchmarkError contract — only the body
of the run changes (forward+backward+optimizer step in a loop instead of a
forward-only timing sweep).

## Phases

  1. preflight     — device reachable
  2. data_load     — generate synthetic inputs + labels from seed
  3. model_load    — Flax model + initialise optimizer state
  4. compile       — cold + warm compile of the train_step function
  5. warmup        — 5 discarded steps to stabilise the JIT cache and
                     allocator behaviour. Step probes treat these as
                     warmup (excluded from steady-state percentiles).
  6. train_loop    — N training steps; per-step probe fan-out emits
                     loss / lr / grad_norm / samples_in_batch.
  7. eval          — M eval batches under the no-grad function;
                     emits eval_loss / eval_accuracy via record_metric.
  8. checkpoint    — optionally writes the final state to
                     results/run_logs/<run_id>/checkpoints/final/
  9. postflight    — device still responds

Each phase is wrapped in `phase("name")` from benchmarks.runner so the
existing exception capture and probe fan-out are reused.
"""
from __future__ import annotations

import datetime
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Reuse the inference runner's phase() context-manager and BenchmarkError —
# both wrap probe fan-out and structured exception capture, both of which
# we want unchanged for training.
from benchmarks.runner import BenchmarkError, phase
from observe.compile_controller import clear_xla_cache, timed_call
from observe.lineage import build_lineage
from observe.probe import (
    fanout_after_run,
    fanout_after_step,
    fanout_before_run,
    fanout_before_step,
    fanout_record_metric,
)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class TrainingExperimentConfig:
    """Complete spec for one training-observability experiment."""
    # Identity
    task_id: str
    hf_id: str
    task: str
    domain: str
    architecture_family: str
    attention_variant: str
    positional_encoding: str
    is_moe: bool
    total_params_M: int
    active_params_M: int
    input_type: str

    # Variant
    precision: str = "bf16"
    framework: str = "jax"
    device: str = "tpu"

    # Input / training shape
    seq_len: int = 128
    batch_size: int = 32
    vocab_size: int = 30522
    num_labels: int = 2

    # Training hyperparameters
    n_steps: int = 200
    n_eval_steps: int = 10
    n_warmup_steps_train: int = 5  # warmup steps (discarded for percentiles)
    lr: float = 2.0e-5
    lr_warmup_steps: int = 20
    weight_decay: float = 0.01
    optimizer: str = "adamw"

    # Determinism
    input_seed: int = 42
    init_seed: int = 0

    # Checkpointing — off by default to keep smoke runs lean.
    save_checkpoint: bool = False

    # Cost reference
    device_cost_usd_per_hr: float = 0.36


# ── Synthetic data ────────────────────────────────────────────────────────────


def make_synthetic_train_batch(
    cfg: TrainingExperimentConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """One synthetic batch: same input_type semantics as benchmarks/runner."""
    if cfg.input_type == "text":
        return {
            "input_ids": rng.integers(
                1, cfg.vocab_size, size=(cfg.batch_size, cfg.seq_len)
            ).astype(np.int32),
            "attention_mask": np.ones((cfg.batch_size, cfg.seq_len), dtype=np.int32),
            "token_type_ids": np.zeros((cfg.batch_size, cfg.seq_len), dtype=np.int32),
            "labels": rng.integers(0, cfg.num_labels, size=(cfg.batch_size,)).astype(
                np.int32
            ),
        }
    raise ValueError(
        f"Stage 1.6 supports input_type='text' only; got {cfg.input_type!r}"
    )


# ── Model + optimizer ─────────────────────────────────────────────────────────


def _load_flax_train_model(cfg: TrainingExperimentConfig) -> Tuple[Any, Any, str]:
    """
    Load Flax model + cast to target precision; return (model, params, hf_revision).
    """
    import jax
    import jax.numpy as jnp

    if cfg.task == "sequence-classification":
        from transformers import FlaxAutoModelForSequenceClassification
        model = FlaxAutoModelForSequenceClassification.from_pretrained(
            cfg.hf_id,
            num_labels=cfg.num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        raise ValueError(f"Stage 1.6 supports sequence-classification only; got {cfg.task!r}")

    hf_revision = (
        getattr(getattr(model, "config", None), "_commit_hash", None) or "unknown"
    )

    if cfg.precision == "bf16":
        params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, "astype") else x,
            model.params,
        )
    else:
        params = model.params

    return model, params, hf_revision


def _build_optimizer(cfg: TrainingExperimentConfig):
    """Build an optax optimizer with linear warmup + linear decay."""
    import optax

    schedule = optax.warmup_linear_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.lr_warmup_steps,
        decay_steps=max(cfg.n_steps - cfg.lr_warmup_steps, 1),
        end_value=0.0,
    )
    if cfg.optimizer == "adamw":
        tx = optax.adamw(schedule, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        tx = optax.sgd(schedule, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")
    return tx, schedule


def _build_train_step(model: Any, tx: Any) -> Callable:
    """JIT-compiled train_step(params, opt_state, batch) → (params, opt_state, metrics)."""
    import jax
    import jax.numpy as jnp
    import optax

    @jax.jit
    def _step(params, opt_state, input_ids, attention_mask, token_type_ids, labels):
        def loss_fn(p):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                params=p,
                train=True,
            )
            logits = out.logits.astype(jnp.float32)
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            )
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        # grad_norm — useful for spotting blow-ups; computed in fp32.
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in jax.tree_util.tree_leaves(grads))
        )
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return new_params, new_opt_state, {
            "loss": loss,
            "grad_norm": grad_norm,
            "accuracy": accuracy,
        }

    return _step


def _build_eval_step(model: Any) -> Callable:
    """JIT-compiled eval_step(params, batch) → metrics. No optimizer update."""
    import jax
    import jax.numpy as jnp
    import optax

    @jax.jit
    def _step(params, input_ids, attention_mask, token_type_ids, labels):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            params=params,
            train=False,
        )
        logits = out.logits.astype(jnp.float32)
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        )
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return {"loss": loss, "accuracy": accuracy}

    return _step


# ── Per-run log writer ────────────────────────────────────────────────────────


def _write_run_log(run_id: str, result: dict, results_dir: str) -> None:
    """Write per-run lineage.json + final metrics summary."""
    log_dir = Path(results_dir) / "run_logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    lineage_keys = (
        "git_sha", "jax_version", "transformers_version",
        "hf_model_revision", "input_seed", "environment_hash",
    )
    lineage = {k: result.get(k) for k in lineage_keys}
    lineage["task_id"] = result.get("task_id")
    lineage["precision"] = result.get("precision")
    lineage["timestamp"] = result.get("timestamp")
    (log_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))


def _write_error_log(run_id: str, results_dir: str, payload: dict) -> None:
    log_dir = Path(results_dir) / "run_logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "error.json").write_text(json.dumps(payload, indent=2, default=str))


# ── Main runner ───────────────────────────────────────────────────────────────


def run_training(
    config: TrainingExperimentConfig,
    results_dir: str = "results",
    _loader: Optional[Callable] = None,
) -> dict:
    """
    Run one training experiment end-to-end with full per-step observability.

    Returns a result dict with the same identity/lineage fields as inference
    runs, plus training-specific metrics (final_loss, mean_step_s, etc.).
    Per-step history is in `results/run_logs/<run_id>/training_metrics.json`
    via the registered probes.
    """
    import jax
    import jax.numpy as jnp

    run_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    sync = jax.block_until_ready

    lineage = build_lineage(
        model_id=config.task_id,
        hf_revision=None,
        input_seed=config.input_seed,
    )

    log_dir = Path(results_dir) / "run_logs" / run_id
    fanout_before_run(run_id, config, log_dir)

    train_metrics_history: List[Dict[str, float]] = []
    eval_metrics: Dict[str, float] = {}
    final_loss: Optional[float] = None
    cold_compile_s: Optional[float] = None
    warm_compile_s: Optional[float] = None
    train_loop_wall_s: Optional[float] = None

    try:
        # ── Phase 1 ────────────────────────────────────────────────────────
        with phase("preflight"):
            _ = jax.local_devices()

        # ── Phase 2: data ──────────────────────────────────────────────────
        with phase("data_load"):
            rng = np.random.default_rng(config.input_seed)
            # We pre-generate the steady-state batch shape once; each step
            # gets a freshly-rolled batch from the same RNG so values vary
            # but shapes are constant (drop_remainder semantics — R16).
            sample_batch = make_synthetic_train_batch(config, rng)
            batch_shape_keys = sorted(sample_batch.keys())

        # ── Phase 3: model + optimizer ─────────────────────────────────────
        with phase("model_load"):
            if _loader is not None:
                model, params, hf_revision = _loader(config)
            else:
                model, params, hf_revision = _load_flax_train_model(config)
            lineage["hf_model_revision"] = hf_revision

            tx, schedule = _build_optimizer(config)
            opt_state = tx.init(params)
            train_step = _build_train_step(model, tx)
            eval_step = _build_eval_step(model)

        # ── Phase 4: compile ───────────────────────────────────────────────
        with phase("compile"):
            clear_xla_cache()
            cb = sample_batch
            args = (
                params, opt_state,
                jnp.asarray(cb["input_ids"]),
                jnp.asarray(cb["attention_mask"]),
                jnp.asarray(cb["token_type_ids"]),
                jnp.asarray(cb["labels"]),
            )
            cold_compile_s, (params, opt_state, _m) = timed_call(
                lambda *a: train_step(*a), args, sync,
            )
            args = (
                params, opt_state,
                jnp.asarray(cb["input_ids"]),
                jnp.asarray(cb["attention_mask"]),
                jnp.asarray(cb["token_type_ids"]),
                jnp.asarray(cb["labels"]),
            )
            warm_compile_s, (params, opt_state, _m) = timed_call(
                lambda *a: train_step(*a), args, sync,
            )

        # ── Phase 5: warmup ────────────────────────────────────────────────
        with phase("warmup"):
            for _ in range(config.n_warmup_steps_train):
                wbatch = make_synthetic_train_batch(config, rng)
                params, opt_state, m = train_step(
                    params, opt_state,
                    jnp.asarray(wbatch["input_ids"]),
                    jnp.asarray(wbatch["attention_mask"]),
                    jnp.asarray(wbatch["token_type_ids"]),
                    jnp.asarray(wbatch["labels"]),
                )
                sync(m["loss"])

        # ── Phase 6: train loop ────────────────────────────────────────────
        with phase("train_loop"):
            t_loop = time.perf_counter()
            for step in range(config.n_steps):
                fanout_before_step(step)
                tbatch = make_synthetic_train_batch(config, rng)
                params, opt_state, m = train_step(
                    params, opt_state,
                    jnp.asarray(tbatch["input_ids"]),
                    jnp.asarray(tbatch["attention_mask"]),
                    jnp.asarray(tbatch["token_type_ids"]),
                    jnp.asarray(tbatch["labels"]),
                )
                # Force the device→host sync so the timing this step measures
                # is real wall-clock, not latency-of-dispatch. Doing it on a
                # single scalar rather than the entire param tree minimises
                # the host-side cost.
                loss_val = float(sync(m["loss"]))
                step_metrics = {
                    "loss": loss_val,
                    "grad_norm": float(m["grad_norm"]),
                    "accuracy": float(m["accuracy"]),
                    "lr": float(schedule(step)),
                    "samples_in_batch": config.batch_size,
                    "tokens_in_batch": config.batch_size * config.seq_len,
                }
                train_metrics_history.append(step_metrics)
                fanout_after_step(step, step_metrics)
                final_loss = loss_val
            train_loop_wall_s = time.perf_counter() - t_loop

        # ── Phase 7: eval ──────────────────────────────────────────────────
        with phase("eval"):
            losses, accs = [], []
            for ei in range(config.n_eval_steps):
                ebatch = make_synthetic_train_batch(config, rng)
                m = eval_step(
                    params,
                    jnp.asarray(ebatch["input_ids"]),
                    jnp.asarray(ebatch["attention_mask"]),
                    jnp.asarray(ebatch["token_type_ids"]),
                    jnp.asarray(ebatch["labels"]),
                )
                losses.append(float(sync(m["loss"])))
                accs.append(float(m["accuracy"]))
            eval_metrics = {
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "accuracy": float(np.mean(accs)) if accs else float("nan"),
            }
            fanout_record_metric("eval_loss", eval_metrics["loss"], step=config.n_steps)
            fanout_record_metric(
                "eval_accuracy", eval_metrics["accuracy"], step=config.n_steps
            )

        # ── Phase 8: checkpoint (optional) ────────────────────────────────
        with phase("checkpoint"):
            if config.save_checkpoint:
                ckpt_dir = log_dir / "checkpoints" / "final"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                t_ckpt = time.perf_counter()
                # Minimal-overhead checkpoint: write the param tree as a
                # numpy npz. We deliberately avoid orbax / msgpack here to
                # keep the runner dependency-light. Stage 2 can swap in a
                # real checkpoint manager.
                flat = {
                    f"p_{i}": np.asarray(leaf)
                    for i, leaf in enumerate(jax.tree_util.tree_leaves(params))
                }
                ckpt_path = ckpt_dir / "params.npz"
                np.savez(ckpt_path, **flat)
                ckpt_dur = time.perf_counter() - t_ckpt
                ckpt_size = ckpt_path.stat().st_size
                fanout_record_metric(
                    "checkpoint_write", ckpt_dur, step=config.n_steps
                )
                fanout_record_metric(
                    "checkpoint_size_bytes", ckpt_size, step=config.n_steps
                )
                fanout_record_metric(
                    "checkpoint_path", str(ckpt_path), step=config.n_steps
                )

        # ── Phase 9: postflight ────────────────────────────────────────────
        with phase("postflight"):
            _ = jax.local_devices()

    except BenchmarkError as exc:
        _write_error_log(run_id, results_dir, {
            "run_id": run_id,
            "timestamp": timestamp,
            "task_id": config.task_id,
            "device": config.device,
            "precision": config.precision,
            "phase": exc.phase,
            "exception_type": exc.original_type,
            "exception_message": exc.original_message,
            "error_category": exc.error_category,
            "traceback": exc.traceback_str,
            "lineage": lineage,
        })
        fanout_after_run(run_id, None, log_dir)
        raise

    # ── Summary metrics ────────────────────────────────────────────────────
    mean_step_s = (
        train_loop_wall_s / max(config.n_steps, 1)
        if train_loop_wall_s is not None
        else None
    )
    samples_per_sec = (
        (config.batch_size * config.n_steps) / train_loop_wall_s
        if train_loop_wall_s
        else None
    )
    tokens_per_sec = (
        (config.batch_size * config.seq_len * config.n_steps) / train_loop_wall_s
        if train_loop_wall_s
        else None
    )

    flags: List[str] = []
    if final_loss is not None and not np.isfinite(final_loss):
        flags.append("nonfinite_loss")
    if any(not np.isfinite(m["grad_norm"]) for m in train_metrics_history):
        flags.append("nonfinite_grad_norm")

    result = {
        "kind": "training",
        "run_id": run_id,
        "timestamp": timestamp,
        **lineage,
        "device": config.device,
        "framework": config.framework,
        "path": 1,
        "task_id": config.task_id,
        "domain": config.domain,
        "architecture_family": config.architecture_family,
        "attention_variant": config.attention_variant,
        "positional_encoding": config.positional_encoding,
        "is_moe": config.is_moe,
        "total_params_M": config.total_params_M,
        "active_params_M": config.active_params_M,
        "precision": config.precision,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "lr": config.lr,
        "lr_warmup_steps": config.lr_warmup_steps,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer,
        "first_compile_s": round(cold_compile_s or 0.0, 4),
        "subsequent_compile_s": round(warm_compile_s or 0.0, 4),
        "train_loop_wall_s": round(train_loop_wall_s or 0.0, 4),
        "mean_step_s": round(mean_step_s or 0.0, 6),
        "throughput_samples_sec": round(samples_per_sec or 0.0, 2),
        "throughput_tokens_sec": round(tokens_per_sec or 0.0, 2),
        "final_train_loss": final_loss,
        "eval_loss": eval_metrics.get("loss"),
        "eval_accuracy": eval_metrics.get("accuracy"),
        "flags": flags,
        "device_cost_usd_per_hr": config.device_cost_usd_per_hr,
        "experiment_cost_usd": round(
            (train_loop_wall_s or 0.0) / 3600.0 * config.device_cost_usd_per_hr, 6
        ),
    }

    _write_run_log(run_id, result, results_dir)
    fanout_after_run(run_id, result, log_dir)
    return result
