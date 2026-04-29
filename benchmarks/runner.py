"""
Single-experiment runner for Path 1 (JAX + XLA).

Implements the 9-phase protocol from context.md §6.
Stage 1 runs phases 1–5 and 9; phases 6–8 are scheduled for later stages.
"""
from __future__ import annotations

import datetime
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from observe.compile_controller import clear_xla_cache, timed_call
from observe.lineage import build_lineage
from observe.stats import compute_timing_stats, throughput_stats

# ── Measurement constants ────────────────────────────────────────────────────
N_WARMUP = 20    # passes discarded before latency measurement
N_MEASURE = 100  # passes per independent block
N_BLOCKS = 3     # independent blocks (for CV check across blocks)


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

    All inputs use seed = config.input_seed for full reproducibility.
    No real data is loaded — hardware-only comparison.
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

    # Cast to target precision
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

    Each task has different required keyword arguments; explicit signatures
    let XLA trace each tensor separately for optimal layout decisions.
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
    """Convert inputs dict → positional args matching _build_forward_fn signature."""
    if task == "sequence-classification":
        return (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
    if task == "image-classification":
        return (inputs["pixel_values"],)
    if task == "causal-lm":
        return (inputs["input_ids"], inputs["attention_mask"])
    if task == "automatic-speech-recognition":
        bs = inputs["input_features"].shape[0]
        tok = decoder_start_id if decoder_start_id is not None else 50258
        decoder_ids = np.full((bs, 1), tok, dtype=np.int32)
        return (inputs["input_features"], decoder_ids)
    if task == "zero-shot-image-classification":
        return (inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])
    raise ValueError(f"Unknown task: {task!r}")


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

    Args:
        config: Full experiment specification.
        results_dir: Where to write lineage files (future stages).
        _loader: Optional override for model loading (testing / injection).

    Returns:
        Result dict matching the JSONL schema in context.md §8.
    """
    import jax

    run_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    sync = jax.block_until_ready

    # ── Phase 1: Pre-flight ────────────────────────────────────────────────
    _ = jax.local_devices()

    # ── Load model ────────────────────────────────────────────────────────
    if _loader is not None:
        model, params, hf_revision = _loader(config)
    else:
        model, params, hf_revision = _load_flax_model(config)

    # Build JIT forward function
    forward_fn = _build_forward_fn(model, config.task)

    # Decoder start token for Whisper (fixed [BOS])
    decoder_start_id: Optional[int] = None
    if config.task == "automatic-speech-recognition":
        decoder_start_id = getattr(
            getattr(model, "config", None), "decoder_start_token_id", 50258
        ) or 50258

    # Build bs=1 inputs
    rng = np.random.default_rng(config.input_seed)
    inputs_bs1 = make_synthetic_inputs(config, batch_size=1, rng=rng)
    args_bs1 = _inputs_to_args(inputs_bs1, config.task, decoder_start_id)

    # ── Phase 2: Compile ───────────────────────────────────────────────────
    clear_xla_cache()
    first_compile_s, out = timed_call(forward_fn, (params, *args_bs1), sync)
    subsequent_compile_s, _ = timed_call(forward_fn, (params, *args_bs1), sync)

    # ── Phase 3: Warmup ────────────────────────────────────────────────────
    for _ in range(N_WARMUP):
        out = forward_fn(params, *args_bs1)
    sync(out)

    # ── Phase 4: Latency (bs=1, 3 × 100 passes) ───────────────────────────
    all_latency_ms: List[float] = []
    for blk in range(N_BLOCKS):
        blk_rng = np.random.default_rng(config.input_seed + blk)
        blk_inputs = make_synthetic_inputs(config, batch_size=1, rng=blk_rng)
        blk_args = _inputs_to_args(blk_inputs, config.task, decoder_start_id)
        all_latency_ms.extend(
            _time_passes(forward_fn, params, blk_args, N_MEASURE, sync)
        )

    lat = compute_timing_stats(all_latency_ms)

    # ── Phase 5: Throughput (bs=max, 3 × 100 passes) ──────────────────────
    bs_tp = config.batch_size_throughput
    all_tp_ms: List[float] = []
    for blk in range(N_BLOCKS):
        blk_rng = np.random.default_rng(config.input_seed + blk + 100)
        blk_inputs = make_synthetic_inputs(config, batch_size=bs_tp, rng=blk_rng)
        blk_args = _inputs_to_args(blk_inputs, config.task, decoder_start_id)
        all_tp_ms.extend(
            _time_passes(forward_fn, params, blk_args, N_MEASURE, sync)
        )

    tp = throughput_stats(all_tp_ms, bs_tp)

    # ── Phase 9: Post-flight ───────────────────────────────────────────────
    _post = forward_fn(params, *args_bs1)
    sync(_post)

    # ── Lineage ────────────────────────────────────────────────────────────
    lineage = build_lineage(
        model_id=config.model_id,
        hf_revision=hf_revision,
        input_seed=config.input_seed,
    )

    # ── Cost estimate ──────────────────────────────────────────────────────
    total_passes = N_WARMUP + N_BLOCKS * N_MEASURE * 2
    duration_s = (
        first_compile_s
        + total_passes * lat.mean_ms / 1000.0
    )
    experiment_cost_usd = (duration_s / 3600.0) * config.device_cost_usd_per_hr
    tp_mean = tp["throughput_mean_samples_sec"]
    cost_per_1k = (
        (config.device_cost_usd_per_hr / 3600.0) / (tp_mean / 1000.0)
        if tp_mean > 0
        else None
    )

    flags: List[str] = []
    if lat.high_variance:
        flags.append("high_variance")

    return {
        # ── Identity ──────────────────────────────────────────────────────
        "run_id": run_id,
        "timestamp": timestamp,
        # ── Lineage ───────────────────────────────────────────────────────
        **lineage,
        # ── Hardware ──────────────────────────────────────────────────────
        "device": config.device,
        "framework": config.framework,
        "path": 1,
        # ── Model ─────────────────────────────────────────────────────────
        "model": config.model_id,
        "domain": config.domain,
        "architecture_family": config.architecture_family,
        "attention_variant": config.attention_variant,
        "positional_encoding": config.positional_encoding,
        "is_moe": config.is_moe,
        "total_params_M": config.total_params_M,
        "active_params_M": config.active_params_M,
        # ── Variant ───────────────────────────────────────────────────────
        "precision": config.precision,
        "pruning": "dense",
        "compiled": True,
        "compile_mode": "default",
        "inference_mode": "combined",
        "kv_cache_tokens": 0,
        # ── Input ─────────────────────────────────────────────────────────
        "batch_size": config.batch_size_latency,
        "batch_size_throughput": bs_tp,
        "seq_len": config.seq_len,
        # ── Compile ───────────────────────────────────────────────────────
        "first_compile_s": round(first_compile_s, 4),
        "subsequent_compile_s": round(subsequent_compile_s, 4),
        "compile_cache_hit": False,
        # ── Latency ───────────────────────────────────────────────────────
        "latency_mean_ms": lat.mean_ms,
        "latency_std_ms": lat.std_ms,
        "latency_cv_pct": lat.cv_pct,
        "latency_p50_ms": lat.p50_ms,
        "latency_p95_ms": lat.p95_ms,
        "latency_p99_ms": lat.p99_ms,
        # ── Throughput ────────────────────────────────────────────────────
        "throughput_mean_samples_sec": tp["throughput_mean_samples_sec"],
        "throughput_std_samples_sec": tp["throughput_std_samples_sec"],
        # ── Memory (Stage 3) ──────────────────────────────────────────────
        "peak_memory_gb": None,
        "weight_memory_gb": None,
        "activation_memory_gb": None,
        "max_batch_before_oom": None,
        # ── Compute analysis (Stage 3) ────────────────────────────────────
        "flops_per_sample_G": None,
        "arithmetic_intensity_flops_per_byte": None,
        "achieved_tflops": None,
        "mfu_pct": None,
        # ── Hardware utilisation (Stage 2) ────────────────────────────────
        "mxu_utilization_pct": None,
        "sm_utilization_pct": None,
        # ── Numerical correctness (Stage 6) ───────────────────────────────
        "output_cosine_sim_vs_fp32": None,
        # ── Flags ─────────────────────────────────────────────────────────
        "flags": flags,
        # ── Cost ──────────────────────────────────────────────────────────
        "device_cost_usd_per_hr": config.device_cost_usd_per_hr,
        "experiment_cost_usd": round(experiment_cost_usd, 6),
        "cost_per_1k_samples_usd": round(cost_per_1k, 8) if cost_per_1k else None,
    }
