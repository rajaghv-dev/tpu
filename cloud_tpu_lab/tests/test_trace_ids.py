"""Trace-ID generators are deterministic-shaped and monotonic."""
import re

from cloud_tpu_lab.src.common.trace import (
    TraceContext, new_collective_id, new_device_event_id, new_executable_id,
    new_hlo_op_id, new_model_layer_id, new_shard_id, new_step_id,
    new_tensor_id, new_trace_id, reset_counters,
)


def test_prefixes_correct() -> None:
    reset_counters()
    assert new_trace_id().startswith("TRACE-")
    assert new_step_id().startswith("STEP-")
    assert new_model_layer_id().startswith("LAYER-")
    assert new_hlo_op_id().startswith("HLO-")
    assert new_executable_id().startswith("EXE-")
    assert new_device_event_id().startswith("DEV-")
    assert new_tensor_id().startswith("TENSOR-")
    assert new_shard_id().startswith("SHARD-")
    assert new_collective_id().startswith("COLL-")


def test_format_is_prefix_dash_four_digits() -> None:
    reset_counters()
    tid = new_trace_id()
    assert re.fullmatch(r"TRACE-\d{4,}", tid)


def test_counters_are_monotonic() -> None:
    reset_counters()
    ids = [new_trace_id() for _ in range(3)]
    nums = [int(t.split("-")[1]) for t in ids]
    assert nums == sorted(nums)
    assert len(set(nums)) == 3


def test_with_step_returns_new_context() -> None:
    reset_counters()
    ctx = TraceContext()
    next_ctx = ctx.with_step("STEP-9999")
    assert next_ctx.step_id == "STEP-9999"
    assert ctx.step_id is None  # original untouched


def test_as_log_fields_drops_only_unset_keys_implicitly() -> None:
    reset_counters()
    ctx = TraceContext()
    fields = ctx.as_log_fields()
    # trace_id is set, others may be None — keys still present.
    assert "trace_id" in fields and fields["trace_id"].startswith("TRACE-")
