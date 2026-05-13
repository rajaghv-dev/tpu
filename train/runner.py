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
  5. warmup        — `n_warmup_steps_train` discarded steps to stabilise the
                     JIT cache and allocator behaviour. Step probes treat
                     these as warmup (excluded from steady-state percentiles).
  6. train_loop    — N training steps; per-step probe fan-out emits
                     loss / lr / grad_norm / samples_in_batch / tokens_in_batch.
  7. eval          — M eval batches under the no-grad function; uses a SEPARATE
                     RNG seeded from cfg.eval_seed so eval is reproducible
                     independent of how many train steps ran. Emits
                     eval_loss / eval_accuracy via record_metric.
  8. checkpoint    — optionally writes the final state to
                     results/run_logs/<run_id>/checkpoints/final/
  9. postflight    — device still responds

Each phase is wrapped in `phase("name")` from benchmarks.runner so the
existing exception capture and probe fan-out are reused.

## Multi-task support

Three task families are supported:

  * `sequence-classification` — BERT-style encoder + classification head.
    Loss = softmax cross-entropy on `[CLS]` over `num_labels`.

  * `causal-lm` — GPT-style decoder. Loss = next-token softmax
    cross-entropy on shifted logits, masked by `attention_mask[:, 1:]`
    so padding positions contribute zero loss. Accuracy = masked top-1
    match rate of predicted next token.

  * `image-classification` — ViT / ResNet / Swin classifiers. Input is
    a (B, C, H, W) random tensor (mean-0, std-1 — *not* normalised to
    [0,1] because pretrained classifiers expect normalised pixels and
    we want to keep the runner data-pipeline-free). Loss = softmax CE.

Each task has its own JIT-compiled `train_step` and `eval_step`. The
runner dispatches via `_build_train_step(model, tx, cfg)` which returns
both the step function and a `batch_to_args(batch_dict) -> tuple` adapter,
so the inner training loop is task-agnostic.

## Controllability knobs

The TrainingExperimentConfig surfaces every commonly-tuned training
control as a field — none of these require touching the runner body:

  * `optimizer`            — adamw | sgd | lion | adafactor
  * `lr_schedule`          — linear | cosine | constant (all with warmup)
  * `max_grad_norm`        — global-norm clipping threshold (0.0 disables)
  * `grad_accum_steps`     — gradient accumulation (1 = no accumulation,
                             implemented via `optax.MultiSteps`)
  * `eval_seed`            — separate RNG so eval is independent of train
                             RNG consumption
  * `deterministic`        — toggle XLA / matmul-precision flags that make
                             results bit-reproducible at the cost of speed
  * `save_checkpoint`      — write final params as `.npz`

The probes (observe/) and harness CLI (train/harness.py) surface the
remaining knobs (probe set selection, output paths, suite definitions).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
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

_log = logging.getLogger(__name__)


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
    # Image inputs only. Stored as a list to keep the dataclass YAML-friendly.
    image_size: Optional[List[int]] = None  # [C, H, W]

    # Training hyperparameters
    n_steps: int = 200
    n_eval_steps: int = 10
    n_warmup_steps_train: int = 5  # warmup steps (discarded for percentiles)
    lr: float = 2.0e-5
    lr_warmup_steps: int = 20
    lr_schedule: str = "linear"  # linear | cosine | constant
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # adamw | sgd | lion | adafactor
    max_grad_norm: float = 1.0  # global-norm clip; 0.0 disables
    grad_accum_steps: int = 1  # 1 = no accumulation

    # Determinism
    input_seed: int = 42
    init_seed: int = 0
    eval_seed: int = 1337  # separate from input_seed: eval should not depend on n_steps
    deterministic: bool = False  # toggle XLA + matmul-precision flags

    # Checkpointing — off by default to keep smoke runs lean.
    save_checkpoint: bool = False

    # Cost reference
    device_cost_usd_per_hr: float = 0.36


# ── Synthetic data ────────────────────────────────────────────────────────────


def _make_text_seq_cls_batch(
    cfg: TrainingExperimentConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
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


def _make_text_causal_lm_batch(
    cfg: TrainingExperimentConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    # input_ids are the only data — labels are derived in-step by shifting,
    # so we don't carry a separate `labels` key. attention_mask is all ones
    # for synthetic inputs (no padding), but we keep the field so the
    # train_step body matches what a real batch would look like.
    return {
        "input_ids": rng.integers(
            1, cfg.vocab_size, size=(cfg.batch_size, cfg.seq_len)
        ).astype(np.int32),
        "attention_mask": np.ones((cfg.batch_size, cfg.seq_len), dtype=np.int32),
    }


def _make_image_cls_batch(
    cfg: TrainingExperimentConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if not cfg.image_size or len(cfg.image_size) != 3:
        raise ValueError(
            f"image-classification requires image_size=[C,H,W]; got {cfg.image_size!r}"
        )
    c, h, w = cfg.image_size
    # standard_normal → mean 0, std 1. Most pretrained vision models expect
    # ImageNet-normalised input (mean≈0, std≈1) so this is closer to what
    # they were trained on than uniform [0,1] would be.
    return {
        "pixel_values": rng.standard_normal((cfg.batch_size, c, h, w)).astype(
            np.float32
        ),
        "labels": rng.integers(0, cfg.num_labels, size=(cfg.batch_size,)).astype(
            np.int32
        ),
    }


def make_synthetic_train_batch(
    cfg: TrainingExperimentConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """One synthetic batch dispatched on (input_type, task)."""
    if cfg.task == "sequence-classification":
        if cfg.input_type != "text":
            raise ValueError(
                f"sequence-classification expects input_type='text', got "
                f"{cfg.input_type!r}"
            )
        return _make_text_seq_cls_batch(cfg, rng)
    if cfg.task == "causal-lm":
        if cfg.input_type != "text":
            raise ValueError(
                f"causal-lm expects input_type='text', got {cfg.input_type!r}"
            )
        return _make_text_causal_lm_batch(cfg, rng)
    if cfg.task == "image-classification":
        if cfg.input_type != "image":
            raise ValueError(
                f"image-classification expects input_type='image', got "
                f"{cfg.input_type!r}"
            )
        return _make_image_cls_batch(cfg, rng)
    raise ValueError(
        f"Unsupported task {cfg.task!r}; expected one of "
        f"sequence-classification | causal-lm | image-classification"
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
    elif cfg.task == "causal-lm":
        from transformers import FlaxAutoModelForCausalLM
        model = FlaxAutoModelForCausalLM.from_pretrained(cfg.hf_id)
    elif cfg.task == "image-classification":
        from transformers import FlaxAutoModelForImageClassification
        try:
            model = FlaxAutoModelForImageClassification.from_pretrained(
                cfg.hf_id,
                num_labels=cfg.num_labels,
                ignore_mismatched_sizes=True,
            )
        except TypeError:
            # Some image models don't accept num_labels — fall back to the
            # checkpoint's native head and trust the registry to set
            # cfg.num_labels accordingly.
            model = FlaxAutoModelForImageClassification.from_pretrained(cfg.hf_id)
    else:
        raise ValueError(
            f"Stage 1.6 supports sequence-classification | causal-lm | "
            f"image-classification; got {cfg.task!r}"
        )

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


def _build_schedule(cfg: TrainingExperimentConfig):
    """Build an optax LR schedule with linear warmup + chosen decay shape."""
    import optax

    warmup = max(cfg.lr_warmup_steps, 1)
    # n_steps in the schedule is the *optimizer* step count, which equals
    # cfg.n_steps when no accumulation is used. With accumulation, only
    # every k-th forward-backward triggers an optimizer step, so the
    # schedule should be sized accordingly. optax.MultiSteps internally
    # skips the schedule on accumulation steps, but the schedule we pass
    # is indexed by gradient-update count — so the math works out without
    # adjusting decay_steps. We still expose this as a clear formula:
    decay_steps = max(cfg.n_steps - warmup, 1)

    if cfg.lr_schedule == "linear":
        return optax.warmup_linear_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=warmup,
            decay_steps=decay_steps,
            end_value=0.0,
        )
    if cfg.lr_schedule == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=warmup,
            decay_steps=decay_steps,
            end_value=0.0,
        )
    if cfg.lr_schedule == "constant":
        # Linear warmup → flat at peak_value. Join lets us name the
        # boundary explicitly rather than relying on schedule arithmetic.
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0, end_value=cfg.lr,
                    transition_steps=warmup,
                ),
                optax.constant_schedule(cfg.lr),
            ],
            boundaries=[warmup],
        )
    raise ValueError(
        f"Unknown lr_schedule: {cfg.lr_schedule!r}; "
        f"expected one of linear | cosine | constant"
    )


def _build_optimizer(cfg: TrainingExperimentConfig):
    """
    Build an optax optimizer with:
      * LR schedule (linear / cosine / constant warmup-then-decay)
      * optional global-norm gradient clipping (max_grad_norm > 0)
      * optional gradient accumulation (grad_accum_steps > 1)

    Returns (tx, schedule) where `schedule` is the bare LR schedule (used by
    the runner for logging `lr` at each step).
    """
    import optax

    schedule = _build_schedule(cfg)

    if cfg.optimizer == "adamw":
        base = optax.adamw(schedule, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        base = optax.sgd(schedule, momentum=0.9)
    elif cfg.optimizer == "lion":
        # Lion typically wants a 3–10× smaller LR than AdamW at the same
        # batch size. We don't adjust here — the registry encodes that.
        base = optax.lion(schedule, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adafactor":
        # Adafactor doesn't take a separate weight_decay arg; it's baked
        # into the per-parameter scale. Acceptable for very large models
        # because it has O(d) memory instead of O(d) per moment.
        base = optax.adafactor(schedule)
    else:
        raise ValueError(
            f"Unknown optimizer: {cfg.optimizer!r}; "
            f"expected one of adamw | sgd | lion | adafactor"
        )

    chain: List[Any] = []
    if cfg.max_grad_norm and cfg.max_grad_norm > 0:
        chain.append(optax.clip_by_global_norm(cfg.max_grad_norm))
    chain.append(base)
    tx = optax.chain(*chain) if len(chain) > 1 else base

    if cfg.grad_accum_steps and cfg.grad_accum_steps > 1:
        # MultiSteps accumulates grads in opt_state for `every_k_schedule`
        # micro-batches before applying the chained tx. The forward+backward
        # still runs on every micro-batch — only the optimizer state update
        # is skipped — so we still get a `loss` value per micro-batch for
        # the probes to record.
        tx = optax.MultiSteps(tx, every_k_schedule=cfg.grad_accum_steps)

    return tx, schedule


# ── Per-task step builders ────────────────────────────────────────────────────
#
# Each builder returns (jit'd step fn, batch_to_args fn). The runner inner
# loop calls batch_to_args(batch_dict) to produce a positional tuple,
# then splats it into the step. Keeping signatures explicit-positional
# lets XLA trace each tensor separately for the cleanest memory layout —
# the same convention benchmarks/runner.py uses for forward-only paths.


def _global_grad_norm(grads, jnp, jax):
    """Sum-of-squares norm computed in fp32 regardless of param dtype."""
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.square(g.astype(jnp.float32)))
            for g in jax.tree_util.tree_leaves(grads)
        )
    )


def _build_train_step(model: Any, tx: Any, cfg: TrainingExperimentConfig) -> Tuple[Callable, Callable]:
    """JIT'd train step + a batch_to_args adapter, both task-aware."""
    import jax
    import jax.numpy as jnp
    import optax

    task = cfg.task

    if task == "sequence-classification":
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
            grad_norm = _global_grad_norm(grads, jnp, jax)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            return new_params, new_opt_state, {
                "loss": loss, "grad_norm": grad_norm, "accuracy": accuracy,
            }

        def _prep(batch: Dict[str, np.ndarray]) -> Tuple[Any, ...]:
            return (
                jnp.asarray(batch["input_ids"]),
                jnp.asarray(batch["attention_mask"]),
                jnp.asarray(batch["token_type_ids"]),
                jnp.asarray(batch["labels"]),
            )

    elif task == "causal-lm":
        @jax.jit
        def _step(params, opt_state, input_ids, attention_mask):
            def loss_fn(p):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    params=p,
                    train=True,
                )
                logits = out.logits.astype(jnp.float32)
                # Next-token prediction: shift logits left, shift labels right.
                shifted_logits = logits[:, :-1, :]
                shifted_labels = input_ids[:, 1:]
                shifted_mask = attention_mask[:, 1:].astype(jnp.float32)
                per_tok = optax.softmax_cross_entropy_with_integer_labels(
                    shifted_logits, shifted_labels,
                )
                # Masked mean — padding tokens (mask=0) contribute zero.
                denom = jnp.maximum(jnp.sum(shifted_mask), 1.0)
                loss = jnp.sum(per_tok * shifted_mask) / denom
                return loss, (shifted_logits, shifted_labels, shifted_mask)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            shifted_logits, shifted_labels, shifted_mask = aux
            grad_norm = _global_grad_norm(grads, jnp, jax)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            # Top-1 next-token accuracy, masked.
            correct = (
                (jnp.argmax(shifted_logits, -1) == shifted_labels).astype(jnp.float32)
                * shifted_mask
            )
            denom = jnp.maximum(jnp.sum(shifted_mask), 1.0)
            accuracy = jnp.sum(correct) / denom
            # Perplexity is exp(loss) — emit it as a derived metric so the
            # dashboard doesn't need to recompute on every plot refresh.
            perplexity = jnp.exp(loss)
            return new_params, new_opt_state, {
                "loss": loss,
                "grad_norm": grad_norm,
                "accuracy": accuracy,
                "perplexity": perplexity,
            }

        def _prep(batch: Dict[str, np.ndarray]) -> Tuple[Any, ...]:
            return (
                jnp.asarray(batch["input_ids"]),
                jnp.asarray(batch["attention_mask"]),
            )

    elif task == "image-classification":
        @jax.jit
        def _step(params, opt_state, pixel_values, labels):
            def loss_fn(p):
                out = model(pixel_values=pixel_values, params=p, train=True)
                logits = out.logits.astype(jnp.float32)
                loss = jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(logits, labels)
                )
                return loss, logits

            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            grad_norm = _global_grad_norm(grads, jnp, jax)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            return new_params, new_opt_state, {
                "loss": loss, "grad_norm": grad_norm, "accuracy": accuracy,
            }

        def _prep(batch: Dict[str, np.ndarray]) -> Tuple[Any, ...]:
            return (
                jnp.asarray(batch["pixel_values"]),
                jnp.asarray(batch["labels"]),
            )

    else:
        raise ValueError(f"Unsupported task: {task!r}")

    return _step, _prep


def _build_eval_step(model: Any, cfg: TrainingExperimentConfig) -> Tuple[Callable, Callable]:
    """JIT'd eval step + batch_to_args adapter. No optimizer update."""
    import jax
    import jax.numpy as jnp
    import optax

    task = cfg.task

    if task == "sequence-classification":
        @jax.jit
        def _step(params, input_ids, attention_mask, token_type_ids, labels):
            out = model(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, params=params, train=False,
            )
            logits = out.logits.astype(jnp.float32)
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            )
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            return {"loss": loss, "accuracy": accuracy}

        def _prep(batch):
            return (
                jnp.asarray(batch["input_ids"]),
                jnp.asarray(batch["attention_mask"]),
                jnp.asarray(batch["token_type_ids"]),
                jnp.asarray(batch["labels"]),
            )

    elif task == "causal-lm":
        @jax.jit
        def _step(params, input_ids, attention_mask):
            out = model(
                input_ids=input_ids, attention_mask=attention_mask,
                params=params, train=False,
            )
            logits = out.logits.astype(jnp.float32)
            shifted_logits = logits[:, :-1, :]
            shifted_labels = input_ids[:, 1:]
            shifted_mask = attention_mask[:, 1:].astype(jnp.float32)
            per_tok = optax.softmax_cross_entropy_with_integer_labels(
                shifted_logits, shifted_labels,
            )
            denom = jnp.maximum(jnp.sum(shifted_mask), 1.0)
            loss = jnp.sum(per_tok * shifted_mask) / denom
            correct = (
                (jnp.argmax(shifted_logits, -1) == shifted_labels).astype(jnp.float32)
                * shifted_mask
            )
            accuracy = jnp.sum(correct) / denom
            return {"loss": loss, "accuracy": accuracy, "perplexity": jnp.exp(loss)}

        def _prep(batch):
            return (
                jnp.asarray(batch["input_ids"]),
                jnp.asarray(batch["attention_mask"]),
            )

    elif task == "image-classification":
        @jax.jit
        def _step(params, pixel_values, labels):
            out = model(pixel_values=pixel_values, params=params, train=False)
            logits = out.logits.astype(jnp.float32)
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            )
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            return {"loss": loss, "accuracy": accuracy}

        def _prep(batch):
            return (
                jnp.asarray(batch["pixel_values"]),
                jnp.asarray(batch["labels"]),
            )

    else:
        raise ValueError(f"Unsupported task: {task!r}")

    return _step, _prep


# ── Determinism ───────────────────────────────────────────────────────────────


_DETERMINISTIC_ENV_HINTS = {
    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
    "NCCL_DETERMINISTIC": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TF_CUDNN_DETERMINISTIC": "1",
    "PYTHONHASHSEED": "0",
}


def _apply_determinism(cfg: TrainingExperimentConfig) -> Dict[str, Any]:
    """
    Apply deterministic-mode flags that can still take effect at runtime.

    Returns a small report dict that the runner can stash in `result` for
    the dashboard (so users know what was active and what was missed).
    Env vars must be set BEFORE jax/cuda init for full effect — at runtime
    we can only nudge the matmul precision and emit warnings.
    """
    if not cfg.deterministic:
        return {"requested": False}

    report: Dict[str, Any] = {"requested": True, "missing_env": []}
    for env_var, recommended in _DETERMINISTIC_ENV_HINTS.items():
        actual = os.environ.get(env_var)
        if not actual or recommended not in actual:
            report["missing_env"].append(
                {"var": env_var, "recommended": recommended, "current": actual}
            )

    try:
        import jax
        jax.config.update("jax_default_matmul_precision", "highest")
        report["jax_default_matmul_precision"] = "highest"
    except Exception as exc:  # noqa: BLE001
        report["jax_config_update_error"] = f"{type(exc).__name__}: {exc}"

    if report["missing_env"]:
        _log.warning(
            "deterministic=True but %d env var(s) not set: %s",
            len(report["missing_env"]),
            [m["var"] for m in report["missing_env"]],
        )
    return report


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
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    sync = jax.block_until_ready

    determinism_report = _apply_determinism(config)

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
            sample_batch = make_synthetic_train_batch(config, rng)

        # ── Phase 3: model + optimizer ─────────────────────────────────────
        with phase("model_load"):
            if _loader is not None:
                model, params, hf_revision = _loader(config)
            else:
                model, params, hf_revision = _load_flax_train_model(config)
            lineage["hf_model_revision"] = hf_revision

            tx, schedule = _build_optimizer(config)
            opt_state = tx.init(params)
            train_step, prep_train_args = _build_train_step(model, tx, config)
            eval_step, prep_eval_args = _build_eval_step(model, config)

        # ── Phase 4: compile ───────────────────────────────────────────────
        with phase("compile"):
            clear_xla_cache()
            # Two timed_call invocations: first is cold compile, second is the
            # warm hit (everything is already in the in-memory JIT cache).
            # On the warm call we feed the SAME tensors so the cache key
            # matches exactly — any difference (even one shape) re-compiles.
            args = (params, opt_state) + prep_train_args(sample_batch)
            cold_compile_s, (params, opt_state, _m) = timed_call(
                lambda *a: train_step(*a), args, sync,
            )
            args = (params, opt_state) + prep_train_args(sample_batch)
            warm_compile_s, (params, opt_state, _m) = timed_call(
                lambda *a: train_step(*a), args, sync,
            )

        # ── Phase 5: warmup ────────────────────────────────────────────────
        with phase("warmup"):
            for _ in range(config.n_warmup_steps_train):
                wbatch = make_synthetic_train_batch(config, rng)
                wargs = (params, opt_state) + prep_train_args(wbatch)
                params, opt_state, m = train_step(*wargs)
                sync(m["loss"])

        # ── Phase 6: train loop ────────────────────────────────────────────
        with phase("train_loop"):
            t_loop = time.perf_counter()
            for step in range(config.n_steps):
                fanout_before_step(step)
                tbatch = make_synthetic_train_batch(config, rng)
                targs = (params, opt_state) + prep_train_args(tbatch)
                params, opt_state, m = train_step(*targs)
                # Force the device→host sync so the timing this step measures
                # is real wall-clock, not latency-of-dispatch. Doing it on a
                # single scalar rather than the entire param tree minimises
                # the host-side cost (param tree sync would block on writes
                # to every leaf).
                loss_val = float(sync(m["loss"]))
                step_metrics: Dict[str, float] = {
                    "loss": loss_val,
                    "grad_norm": float(m["grad_norm"]),
                    "accuracy": float(m["accuracy"]),
                    "lr": float(schedule(step)),
                    "samples_in_batch": config.batch_size,
                    "tokens_in_batch": config.batch_size * config.seq_len,
                }
                if "perplexity" in m:
                    step_metrics["perplexity"] = float(m["perplexity"])
                train_metrics_history.append(step_metrics)
                fanout_after_step(step, step_metrics)
                final_loss = loss_val
            train_loop_wall_s = time.perf_counter() - t_loop

        # ── Phase 7: eval ──────────────────────────────────────────────────
        with phase("eval"):
            # Separate RNG so eval is independent of how many train steps ran.
            # This is the difference between "eval changed because the model
            # got better" and "eval changed because we drew different inputs"
            # — only the first is meaningful, and a separate seed pins the
            # second to a constant.
            eval_rng = np.random.default_rng(config.eval_seed)
            losses, accs, perps = [], [], []
            for ei in range(config.n_eval_steps):
                ebatch = make_synthetic_train_batch(config, eval_rng)
                eargs = (params,) + prep_eval_args(ebatch)
                m = eval_step(*eargs)
                losses.append(float(sync(m["loss"])))
                accs.append(float(m["accuracy"]))
                if "perplexity" in m:
                    perps.append(float(m["perplexity"]))
            eval_metrics = {
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "accuracy": float(np.mean(accs)) if accs else float("nan"),
            }
            if perps:
                eval_metrics["perplexity"] = float(np.mean(perps))
            fanout_record_metric("eval_loss", eval_metrics["loss"], step=config.n_steps)
            fanout_record_metric(
                "eval_accuracy", eval_metrics["accuracy"], step=config.n_steps
            )
            if perps:
                fanout_record_metric(
                    "eval_perplexity", eval_metrics["perplexity"],
                    step=config.n_steps,
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
    # Throughput metrics — sample-based (always meaningful) and token-based
    # (only meaningful when seq_len > 0, which it always is for our tasks).
    samples_per_sec = (
        (config.batch_size * config.n_steps) / train_loop_wall_s
        if train_loop_wall_s
        else None
    )
    tokens_per_sec = (
        (config.batch_size * config.seq_len * config.n_steps) / train_loop_wall_s
        if train_loop_wall_s and config.input_type == "text"
        else None
    )

    flags: List[str] = []
    if final_loss is not None and not np.isfinite(final_loss):
        flags.append("nonfinite_loss")
    if any(not np.isfinite(m["grad_norm"]) for m in train_metrics_history):
        flags.append("nonfinite_grad_norm")
    if determinism_report.get("missing_env"):
        flags.append("determinism_env_incomplete")

    result = {
        "kind": "training",
        "run_id": run_id,
        "timestamp": timestamp,
        **lineage,
        "device": config.device,
        "framework": config.framework,
        "path": 1,
        "task_id": config.task_id,
        "task": config.task,
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
        "lr_schedule": config.lr_schedule,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer,
        "max_grad_norm": config.max_grad_norm,
        "grad_accum_steps": config.grad_accum_steps,
        "deterministic": config.deterministic,
        "determinism_report": determinism_report,
        "first_compile_s": round(cold_compile_s or 0.0, 4),
        "subsequent_compile_s": round(warm_compile_s or 0.0, 4),
        "train_loop_wall_s": round(train_loop_wall_s or 0.0, 4),
        "mean_step_s": round(mean_step_s or 0.0, 6),
        "throughput_samples_sec": round(samples_per_sec or 0.0, 2),
        "throughput_tokens_sec": (
            round(tokens_per_sec, 2) if tokens_per_sec else None
        ),
        "final_train_loss": final_loss,
        "eval_loss": eval_metrics.get("loss"),
        "eval_accuracy": eval_metrics.get("accuracy"),
        "eval_perplexity": eval_metrics.get("perplexity"),
        "flags": flags,
        "device_cost_usd_per_hr": config.device_cost_usd_per_hr,
        "experiment_cost_usd": round(
            (train_loop_wall_s or 0.0) / 3600.0 * config.device_cost_usd_per_hr, 6
        ),
    }

    _write_run_log(run_id, result, results_dir)
    fanout_after_run(run_id, result, log_dir)
    return result
