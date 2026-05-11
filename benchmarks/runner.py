"""
Single-experiment runner for Path 1 (JAX + XLA).

Implements the 9-phase protocol from context.md §6.
Stage 1 runs phases 1–5 and 9; phases 6–8 are scheduled for later stages.
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

from observe.compile_controller import clear_xla_cache, timed_call
from observe.lineage import build_lineage
from observe.otel import (
    get_instruments,
    get_tracer,
    init_otel,
    shutdown_otel,
)
from observe.stats import ThroughputStats, compute_timing_stats, throughput_stats

# ── Measurement constants ────────────────────────────────────────────────────
N_WARMUP = 20    # passes discarded before latency measurement
N_MEASURE = 100  # passes per independent block
N_BLOCKS = 3     # independent blocks (for CV check)


# ── Task dispatch ─────────────────────────────────────────────────────────────
# Single source of truth: task → ordered list of input keys for _inputs_to_args.
# "automatic-speech-recognition" is handled separately (decoder_ids injected).
_TASK_ARGS: Dict[str, List[str]] = {
    "sequence-classification":        ["input_ids", "attention_mask", "token_type_ids"],
    "image-classification":           ["pixel_values"],
    "causal-lm":                      ["input_ids", "attention_mask"],
    "zero-shot-image-classification": ["pixel_values", "input_ids", "attention_mask"],
}


# ── Experiment configuration ─────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Complete specification of one benchmark experiment."""
    # Identity (from registry)
    model_id: str
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

    # Input shape
    seq_len: int = 128
    batch_size_latency: int = 1
    batch_size_throughput: int = 32
    input_seed: int = 42

    # Optional per-domain fields
    vocab_size: int = 30522
    image_size: Optional[List[int]] = None
    n_mels: Optional[int] = None
    n_frames: Optional[int] = None

    # Cost reference
    device_cost_usd_per_hr: float = 0.36


# ── Synthetic input generation ───────────────────────────────────────────────

def make_synthetic_inputs(
    config: ExperimentConfig,
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Build a dict of synthetic numpy inputs for one forward pass.

    All inputs use config.input_seed for full reproducibility.
    No real data is loaded — hardware-only comparison (ADR-004).
    """
    if rng is None:
        rng = np.random.default_rng(config.input_seed)

    itype = config.input_type

    if itype == "text":
        input_ids = rng.integers(
            1, config.vocab_size, size=(batch_size, config.seq_len)
        ).astype(np.int32)
        return {
            "input_ids": input_ids,
            "attention_mask": np.ones((batch_size, config.seq_len), dtype=np.int32),
            "token_type_ids": np.zeros((batch_size, config.seq_len), dtype=np.int32),
        }

    if itype == "image":
        sz = config.image_size or [3, 224, 224]
        return {
            "pixel_values": rng.random(
                (batch_size, *sz), dtype=float
            ).astype(np.float32),
        }

    if itype == "audio":
        n_mels = config.n_mels or 80
        n_frames = config.n_frames or 3000
        return {
            "input_features": rng.random(
                (batch_size, n_mels, n_frames), dtype=float
            ).astype(np.float32),
        }

    if itype == "image_text":
        sz = config.image_size or [3, 224, 224]
        seq = config.seq_len or 77
        input_ids = rng.integers(
            1, config.vocab_size, size=(batch_size, seq)
        ).astype(np.int32)
        return {
            "pixel_values": rng.random(
                (batch_size, *sz), dtype=float
            ).astype(np.float32),
            "input_ids": input_ids,
            "attention_mask": np.ones((batch_size, seq), dtype=np.int32),
        }

    raise ValueError(f"Unknown input_type: {itype!r}")


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_flax_model(config: ExperimentConfig) -> Tuple[Any, Any, str]:
    """
    Load a HuggingFace Flax model and cast params to target precision.

    Returns:
        (model, params, hf_revision)
    """
    import jax
    import jax.numpy as jnp

    task = config.task

    if task == "sequence-classification":
        from transformers import FlaxAutoModelForSequenceClassification
        model = FlaxAutoModelForSequenceClassification.from_pretrained(
            config.hf_id, ignore_mismatched_sizes=True
        )
    elif task == "image-classification":
        from transformers import FlaxAutoModelForImageClassification
        model = FlaxAutoModelForImageClassification.from_pretrained(config.hf_id)
    elif task == "causal-lm":
        from transformers import FlaxAutoModelForCausalLM
        model = FlaxAutoModelForCausalLM.from_pretrained(config.hf_id)
    elif task == "automatic-speech-recognition":
        from transformers import FlaxWhisperForConditionalGeneration
        model = FlaxWhisperForConditionalGeneration.from_pretrained(config.hf_id)
    elif task == "zero-shot-image-classification":
        from transformers import FlaxCLIPModel
        model = FlaxCLIPModel.from_pretrained(config.hf_id)
    else:
        from transformers import FlaxAutoModel
        model = FlaxAutoModel.from_pretrained(config.hf_id)

    hf_revision = getattr(
        getattr(model, "config", None), "_commit_hash", None
    ) or "unknown"

    if config.precision == "bf16":
        params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, "astype") else x,
            model.params,
        )
    else:
        params = model.params

    return model, params, hf_revision


# ── Forward function construction ─────────────────────────────────────────────

def _build_forward_fn(model: Any, task: str) -> Callable:
    """
    Build and JIT a forward function for the given task.

    Explicit positional signatures let XLA trace each tensor separately
    for optimal memory-layout decisions — prefer over **kwargs.
    """
    import jax

    if task == "sequence-classification":
        @jax.jit
        def _fwd(params, input_ids, attention_mask, token_type_ids):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                params=params,
            )

    elif task == "image-classification":
        @jax.jit
        def _fwd(params, pixel_values):
            return model(pixel_values=pixel_values, params=params)

    elif task == "causal-lm":
        @jax.jit
        def _fwd(params, input_ids, attention_mask):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                params=params,
            )

    elif task == "automatic-speech-recognition":
        @jax.jit
        def _fwd(params, input_features, decoder_input_ids):
            return model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                params=params,
            )

    elif task == "zero-shot-image-classification":
        @jax.jit
        def _fwd(params, pixel_values, input_ids, attention_mask):
            return model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                params=params,
            )

    else:
        raise ValueError(f"No forward function for task: {task!r}")

    return _fwd


def _inputs_to_args(
    inputs: Dict[str, np.ndarray],
    task: str,
    decoder_start_id: Optional[int] = None,
) -> tuple:
    """
    Convert inputs dict → positional args matching _build_forward_fn signature.

    Uses _TASK_ARGS as the single source of truth for key ordering, so adding
    a new task only requires one edit (to _TASK_ARGS + _build_forward_fn).
    """
    if task == "automatic-speech-recognition":
        bs = inputs["input_features"].shape[0]
        tok = decoder_start_id if decoder_start_id is not None else 50258
        decoder_ids = np.full((bs, 1), tok, dtype=np.int32)
        return (inputs["input_features"], decoder_ids)

    if task not in _TASK_ARGS:
        raise ValueError(f"Unknown task: {task!r}")

    return tuple(inputs[k] for k in _TASK_ARGS[task])


# ── Timing loop ───────────────────────────────────────────────────────────────

def _time_passes(
    forward_fn: Callable,
    params: Any,
    args: tuple,
    n_passes: int,
    sync_fn: Callable,
) -> List[float]:
    """Run forward_fn n_passes times and return per-pass timings in ms."""
    timings: List[float] = []
    for _ in range(n_passes):
        t0 = time.perf_counter()
        out = forward_fn(params, *args)
        sync_fn(out)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


# ── Per-run log writer ────────────────────────────────────────────────────────

def _write_run_log(run_id: str, result: dict, results_dir: str) -> None:
    """
    Write per-run evidence files to results_dir/run_logs/<run_id>/.

    Stage 1 writes lineage.json. Later stages add profiles/, raw_timings.jsonl, etc.
    """
    log_dir = Path(results_dir) / "run_logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    lineage_keys = (
        "git_sha", "jax_version", "transformers_version",
        "hf_model_revision", "input_seed", "environment_hash",
    )
    lineage = {k: result.get(k) for k in lineage_keys}
    lineage["model"] = result.get("model")
    lineage["precision"] = result.get("precision")
    lineage["timestamp"] = result.get("timestamp")

    (log_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    config: ExperimentConfig,
    results_dir: str = "results",
    _loader: Optional[Callable] = None,
) -> dict:
    """
    Run one complete benchmark experiment for Path 1 (JAX + XLA).

    Phases implemented in Stage 1:
      1 Pre-flight   — verify device is reachable
      2 Compile      — cold + warm compilation timing
      3 Warmup       — 20 discarded passes to stabilise kernels
      4 Latency      — 3 × 100 passes at bs=1 → p50/p95/p99/CV
      5 Throughput   — 3 × 100 passes at bs=max → samples/sec
      9 Post-flight  — verify device still responds

    Phases 6–8 (profiler, memory sweep, numerics) scheduled for Stage 3.

    Per-run evidence written to results_dir/run_logs/<run_id>/lineage.json.

    Args:
        config: Full experiment specification.
        results_dir: Root for per-run log files.
        _loader: Optional model loader override (for testing).

    Returns:
        Result dict matching the JSONL schema in context.md §8.
    """
    import jax

    run_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    sync = jax.block_until_ready

    # ── OTel bootstrap ─────────────────────────────────────────────────────
    init_otel(resource_attrs={
        "run_id": run_id,
        "model.id": config.model_id,
        "device": config.device,
        "precision": config.precision,
    })
    tracer = get_tracer()
    instruments = get_instruments()
    metric_attrs = {
        "model_id": config.model_id,
        "device": config.device,
        "precision": config.precision,
    }

    try:
        with tracer.start_as_current_span("experiment") as exp_span:
            exp_span.set_attribute("run_id", run_id)
            exp_span.set_attribute("model.id", config.model_id)
            exp_span.set_attribute("model.hf_id", config.hf_id)
            exp_span.set_attribute("model.task", config.task)
            exp_span.set_attribute("model.domain", config.domain)
            exp_span.set_attribute("model.architecture_family", config.architecture_family)
            exp_span.set_attribute("model.total_params_M", config.total_params_M)
            exp_span.set_attribute("model.active_params_M", config.active_params_M)
            exp_span.set_attribute("model.is_moe", bool(config.is_moe))
            exp_span.set_attribute("device", config.device)
            exp_span.set_attribute("precision", config.precision)
            exp_span.set_attribute("framework", config.framework)
            exp_span.set_attribute("path", 1)
            exp_span.set_attribute("batch_size.latency", 1)
            exp_span.set_attribute("batch_size.throughput", config.batch_size_throughput)
            exp_span.set_attribute("seq_len", config.seq_len)

            # ── Phase 1: Pre-flight ────────────────────────────────────────
            with tracer.start_as_current_span("phase.preflight"):
                _ = jax.local_devices()

            # ── Load model ────────────────────────────────────────────────
            with tracer.start_as_current_span("phase.model_load"):
                if _loader is not None:
                    model, params, hf_revision = _loader(config)
                else:
                    model, params, hf_revision = _load_flax_model(config)
                forward_fn = _build_forward_fn(model, config.task)

            decoder_start_id: Optional[int] = None
            if config.task == "automatic-speech-recognition":
                decoder_start_id = getattr(
                    getattr(model, "config", None), "decoder_start_token_id", 50258
                ) or 50258

            rng = np.random.default_rng(config.input_seed)
            inputs_bs1 = make_synthetic_inputs(config, batch_size=1, rng=rng)
            args_bs1 = _inputs_to_args(inputs_bs1, config.task, decoder_start_id)

            # ── Phase 2: Compile ───────────────────────────────────────────
            clear_xla_cache()
            with tracer.start_as_current_span("phase.compile.cold") as _cs:
                first_compile_s, _ = timed_call(forward_fn, (params, *args_bs1), sync)
                _cs.set_attribute("elapsed_s", first_compile_s)
            with tracer.start_as_current_span("phase.compile.warm") as _ws:
                subsequent_compile_s, _ = timed_call(forward_fn, (params, *args_bs1), sync)
                _ws.set_attribute("elapsed_s", subsequent_compile_s)

            # ── Phase 3: Warmup ────────────────────────────────────────────
            with tracer.start_as_current_span("phase.warmup"):
                for _ in range(N_WARMUP):
                    out = forward_fn(params, *args_bs1)
                sync(out)

            # ── Phase 4: Latency (bs=1, N_BLOCKS × N_MEASURE passes) ──────
            all_latency_ms: List[float] = []
            with tracer.start_as_current_span("phase.latency"):
                for blk in range(N_BLOCKS):
                    blk_rng = np.random.default_rng(config.input_seed + blk)
                    blk_inputs = make_synthetic_inputs(config, batch_size=1, rng=blk_rng)
                    blk_args = _inputs_to_args(blk_inputs, config.task, decoder_start_id)
                    with tracer.start_as_current_span("phase.latency.block") as bspan:
                        bspan.set_attribute("block_index", blk)
                        bspan.set_attribute("n_passes", N_MEASURE)
                        block_timings = _time_passes(
                            forward_fn, params, blk_args, N_MEASURE, sync
                        )
                        for t_ms in block_timings:
                            instruments["latency_ms"].record(t_ms, metric_attrs)
                        all_latency_ms.extend(block_timings)
                        bspan.set_attribute(
                            "mean_ms",
                            float(sum(block_timings) / len(block_timings))
                            if block_timings else 0.0,
                        )

            lat = compute_timing_stats(all_latency_ms)

            # ── Phase 5: Throughput (bs=max, N_BLOCKS × N_MEASURE passes) ─
            bs_tp = config.batch_size_throughput
            all_tp_ms: List[float] = []
            with tracer.start_as_current_span("phase.throughput"):
                for blk in range(N_BLOCKS):
                    blk_rng = np.random.default_rng(config.input_seed + blk + 100)
                    blk_inputs = make_synthetic_inputs(config, batch_size=bs_tp, rng=blk_rng)
                    blk_args = _inputs_to_args(blk_inputs, config.task, decoder_start_id)
                    with tracer.start_as_current_span("phase.throughput.block") as bspan:
                        bspan.set_attribute("block_index", blk)
                        bspan.set_attribute("n_passes", N_MEASURE)
                        block_timings = _time_passes(
                            forward_fn, params, blk_args, N_MEASURE, sync
                        )
                        all_tp_ms.extend(block_timings)
                        if block_timings:
                            mean_ms_blk = sum(block_timings) / len(block_timings)
                            mean_sps = (bs_tp * 1000.0 / mean_ms_blk) if mean_ms_blk > 0 else 0.0
                        else:
                            mean_sps = 0.0
                        bspan.set_attribute("mean_samples_sec", mean_sps)
                        instruments["throughput_samples_sec"].record(mean_sps, metric_attrs)

            tp = throughput_stats(all_tp_ms, bs_tp)

            # ── Phase 9: Post-flight ───────────────────────────────────────
            with tracer.start_as_current_span("phase.postflight"):
                _post = forward_fn(params, *args_bs1)
                sync(_post)

            # ── Lineage ────────────────────────────────────────────────────
            lineage = build_lineage(
                model_id=config.model_id,
                hf_revision=hf_revision,
                input_seed=config.input_seed,
            )
            exp_span.set_attribute("git_sha", lineage.get("git_sha", ""))
            exp_span.set_attribute("jax_version", lineage.get("jax_version", ""))
            exp_span.set_attribute(
                "transformers_version", lineage.get("transformers_version", "")
            )
            exp_span.set_attribute(
                "environment_hash", lineage.get("environment_hash", "")
            )
            exp_span.set_attribute(
                "hf_model_revision", lineage.get("hf_model_revision", "")
            )
            exp_span.set_attribute("input_seed", lineage.get("input_seed", 0))

            # ── Summary metrics ────────────────────────────────────────────
            instruments["compile_cold_s"].record(first_compile_s, metric_attrs)
            instruments["compile_warm_s"].record(subsequent_compile_s, metric_attrs)
            instruments["latency_cv_pct"].record(lat.cv_pct, metric_attrs)

        # ── Cost estimate ──────────────────────────────────────────────────
        total_s = (
            first_compile_s
            + N_WARMUP * lat.mean_ms / 1000.0
            + N_BLOCKS * N_MEASURE * 2 * lat.mean_ms / 1000.0
        )
        experiment_cost_usd = (total_s / 3600.0) * config.device_cost_usd_per_hr
        cost_per_1k = (
            (config.device_cost_usd_per_hr / 3600.0) / (tp.mean_samples_sec / 1000.0)
            if tp.mean_samples_sec > 0
            else None
        )

        flags: List[str] = []
        if lat.high_variance:
            flags.append("high_variance")

        result = {
            # ── Identity ──────────────────────────────────────────────────
            "run_id": run_id,
            "timestamp": timestamp,
            # ── Lineage ───────────────────────────────────────────────────
            **lineage,
            # ── Hardware ──────────────────────────────────────────────────
            "device": config.device,
            "framework": config.framework,
            "path": 1,
            # ── Model ─────────────────────────────────────────────────────
            "model": config.model_id,
            "domain": config.domain,
            "architecture_family": config.architecture_family,
            "attention_variant": config.attention_variant,
            "positional_encoding": config.positional_encoding,
            "is_moe": config.is_moe,
            "total_params_M": config.total_params_M,
            "active_params_M": config.active_params_M,
            # ── Variant ───────────────────────────────────────────────────
            "precision": config.precision,
            "pruning": "dense",
            "compiled": True,
            "compile_mode": "default",
            "inference_mode": "combined",
            "kv_cache_tokens": 0,
            # ── Input ─────────────────────────────────────────────────────
            "batch_size": config.batch_size_latency,
            "batch_size_throughput": bs_tp,
            "seq_len": config.seq_len,
            # ── Compile ───────────────────────────────────────────────────
            "first_compile_s": round(first_compile_s, 4),
            "subsequent_compile_s": round(subsequent_compile_s, 4),
            "compile_cache_hit": False,
            # ── Latency ───────────────────────────────────────────────────
            "latency_mean_ms": lat.mean_ms,
            "latency_std_ms": lat.std_ms,
            "latency_cv_pct": lat.cv_pct,
            "latency_p50_ms": lat.p50_ms,
            "latency_p95_ms": lat.p95_ms,
            "latency_p99_ms": lat.p99_ms,
            # ── Throughput ────────────────────────────────────────────────
            "throughput_mean_samples_sec": tp.mean_samples_sec,
            "throughput_std_samples_sec": tp.std_samples_sec,
            # ── Memory (Stage 3) ──────────────────────────────────────────
            "peak_memory_gb": None,
            "weight_memory_gb": None,
            "activation_memory_gb": None,
            "max_batch_before_oom": None,
            # ── Compute analysis (Stage 3) ────────────────────────────────
            "flops_per_sample_G": None,
            "arithmetic_intensity_flops_per_byte": None,
            "achieved_tflops": None,
            "mfu_pct": None,
            # ── Hardware utilisation (Stage 2) ────────────────────────────
            "mxu_utilization_pct": None,
            "sm_utilization_pct": None,
            # ── Numerical correctness (Stage 6) ───────────────────────────
            "output_cosine_sim_vs_fp32": None,
            # ── Flags ─────────────────────────────────────────────────────
            "flags": flags,
            # ── Cost ──────────────────────────────────────────────────────
            "device_cost_usd_per_hr": config.device_cost_usd_per_hr,
            "experiment_cost_usd": round(experiment_cost_usd, 6),
            "cost_per_1k_samples_usd": round(cost_per_1k, 8) if cost_per_1k else None,
        }

        _write_run_log(run_id, result, results_dir)
        return result
    finally:
        shutdown_otel()
