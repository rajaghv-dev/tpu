"""Lowering produces HLO ops for each known layer kind."""
from cloud_tpu_lab.src.xla_sim.fake_hlo import HloOpKind
from cloud_tpu_lab.src.xla_sim.lowering import Layer, ModelGraph, lower_to_hlo
from cloud_tpu_lab.src.common.trace import reset_counters


def test_linear_lowers_to_dot_general_plus_add() -> None:
    reset_counters()
    g = ModelGraph(
        name="t", dtype="bf16",
        layers=[Layer(name="l", kind="linear",
                      shape_in=(4, 8), shape_out=(4, 16))],
    )
    mod = lower_to_hlo(g)
    kinds = [op.kind for op in mod.ops]
    assert kinds == [HloOpKind.DOT_GENERAL, HloOpKind.ADD]
    assert all(op.model_layer_id is not None for op in mod.ops)


def test_attention_lowers_to_dot_softmax_dot() -> None:
    reset_counters()
    g = ModelGraph(
        name="t", dtype="bf16",
        layers=[Layer(name="a", kind="attention",
                      shape_in=(2, 16, 32), shape_out=(2, 16, 32))],
    )
    mod = lower_to_hlo(g)
    kinds = [op.kind for op in mod.ops]
    assert kinds == [HloOpKind.DOT_GENERAL, HloOpKind.SOFTMAX, HloOpKind.DOT_GENERAL]


def test_layernorm_lowers_to_reduce_variance_normalize() -> None:
    reset_counters()
    g = ModelGraph(
        name="t", dtype="bf16",
        layers=[Layer(name="ln", kind="layernorm",
                      shape_in=(2, 16, 32), shape_out=(2, 16, 32))],
    )
    mod = lower_to_hlo(g)
    kinds = [op.kind for op in mod.ops]
    assert kinds == [HloOpKind.REDUCE_MEAN, HloOpKind.VARIANCE, HloOpKind.NORMALIZE]


def test_flops_and_bytes_are_positive() -> None:
    reset_counters()
    g = ModelGraph(
        name="t", dtype="bf16",
        layers=[Layer(name="l", kind="linear",
                      shape_in=(8, 64), shape_out=(8, 128))],
    )
    mod = lower_to_hlo(g)
    assert mod.total_flops() > 0
    assert mod.total_bytes_moved() > 0


def test_unknown_kind_raises() -> None:
    import pytest
    g = ModelGraph(
        name="t", dtype="bf16",
        layers=[Layer(name="x", kind="not_a_real_kind",
                      shape_in=(1,), shape_out=(1,))],
    )
    with pytest.raises(ValueError, match="Unknown layer kind"):
        lower_to_hlo(g)
