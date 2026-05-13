"""
Model → fake HLO lowering.

Mirrors the lowering rules the user asked for:

    Linear     → DotGeneral + Add
    Conv       → Convolution
    Attention  → DotGeneral + Softmax + DotGeneral
    LayerNorm  → ReduceMean + Variance + Normalize

The lowering produces an `HloModule`. Each `HloOp` carries the
`model_layer_id` of the layer that produced it, so the profiler can join
HLO ops back to model layers.

The byte / flop estimates assume bf16 (2 bytes / element) when the model
config asks for bf16, else fp32 (4 bytes / element).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..common.trace import new_hlo_op_id, new_model_layer_id
from .fake_hlo import HloModule, HloOp, HloOpKind


# ── Model graph types (deliberately minimal) ─────────────────────────────────


@dataclass
class Layer:
    """One node in the user-facing model graph."""
    name: str
    kind: str          # linear | conv | attention | layernorm | embedding
    shape_in: Tuple[int, ...]
    shape_out: Tuple[int, ...]
    extras: dict = field(default_factory=dict)
    # Populated by the lowering.
    model_layer_id: Optional[str] = None


@dataclass
class ModelGraph:
    name: str
    layers: List[Layer]
    dtype: str = "bf16"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _bytes_per_elem(dtype: str) -> int:
    return 2 if dtype == "bf16" else 4


def _prod(xs: Tuple[int, ...]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


# ── Per-layer lowerings ───────────────────────────────────────────────────────


def _lower_linear(layer: Layer, dtype: str) -> List[HloOp]:
    # Linear(in_dim → out_dim) → DotGeneral + Add (bias)
    in_dim = int(layer.shape_in[-1])
    out_dim = int(layer.shape_out[-1])
    # Output shape — preserve leading batch dims.
    batch_dims = tuple(layer.shape_in[:-1])
    out_shape = (*batch_dims, out_dim)
    elem = _bytes_per_elem(dtype)

    dot = HloOp(
        op_id=new_hlo_op_id(),
        kind=HloOpKind.DOT_GENERAL,
        shape=out_shape,
        dtype=dtype,
        model_layer_id=layer.model_layer_id,
        layer_name=layer.name,
        flops=2 * _prod(layer.shape_in) * out_dim,
        bytes_in=_prod(layer.shape_in) * elem + in_dim * out_dim * elem,
        bytes_out=_prod(out_shape) * elem,
        metadata={"in_dim": in_dim, "out_dim": out_dim},
    )
    add = HloOp(
        op_id=new_hlo_op_id(),
        kind=HloOpKind.ADD,
        shape=out_shape,
        dtype=dtype,
        inputs=[dot.op_id],
        model_layer_id=layer.model_layer_id,
        layer_name=layer.name,
        flops=_prod(out_shape),
        bytes_in=_prod(out_shape) * elem,
        bytes_out=_prod(out_shape) * elem,
    )
    return [dot, add]


def _lower_conv(layer: Layer, dtype: str) -> List[HloOp]:
    # Conv(C_in, H, W → C_out, H', W') — simplified single op.
    elem = _bytes_per_elem(dtype)
    kernel = int(layer.extras.get("kernel", 3))
    c_in = int(layer.shape_in[1]) if len(layer.shape_in) >= 2 else 1
    c_out = int(layer.shape_out[1]) if len(layer.shape_out) >= 2 else 1
    spatial_out = _prod(layer.shape_out[2:]) if len(layer.shape_out) > 2 else 1
    flops = 2 * c_in * c_out * kernel * kernel * spatial_out * int(layer.shape_in[0])
    return [HloOp(
        op_id=new_hlo_op_id(),
        kind=HloOpKind.CONVOLUTION,
        shape=tuple(layer.shape_out),
        dtype=dtype,
        model_layer_id=layer.model_layer_id,
        layer_name=layer.name,
        flops=flops,
        bytes_in=_prod(layer.shape_in) * elem + c_in * c_out * kernel * kernel * elem,
        bytes_out=_prod(layer.shape_out) * elem,
        metadata={"kernel": kernel, "c_in": c_in, "c_out": c_out},
    )]


def _lower_attention(layer: Layer, dtype: str) -> List[HloOp]:
    # Attention(Q,K,V) → DotGeneral(QK^T) → Softmax → DotGeneral(...V)
    elem = _bytes_per_elem(dtype)
    b, s, d = layer.shape_in  # (batch, seq, hidden)
    out_shape = (b, s, d)
    qkt_shape = (b, s, s)
    qkt = HloOp(
        op_id=new_hlo_op_id(),
        kind=HloOpKind.DOT_GENERAL,
        shape=qkt_shape, dtype=dtype,
        model_layer_id=layer.model_layer_id, layer_name=layer.name,
        flops=2 * b * s * s * d,
        bytes_in=2 * b * s * d * elem,
        bytes_out=b * s * s * elem,
        metadata={"role": "QK^T"},
    )
    sm = HloOp(
        op_id=new_hlo_op_id(),
        kind=HloOpKind.SOFTMAX,
        shape=qkt_shape, dtype=dtype,
        inputs=[qkt.op_id],
        model_layer_id=layer.model_layer_id, layer_name=layer.name,
        flops=3 * b * s * s,  # exp + sum + div
        bytes_in=b * s * s * elem,
        bytes_out=b * s * s * elem,
    )
    av = HloOp(
        op_id=new_hlo_op_id(),
        kind=HloOpKind.DOT_GENERAL,
        shape=out_shape, dtype=dtype,
        inputs=[sm.op_id],
        model_layer_id=layer.model_layer_id, layer_name=layer.name,
        flops=2 * b * s * s * d,
        bytes_in=b * s * s * elem + b * s * d * elem,
        bytes_out=b * s * d * elem,
        metadata={"role": "(A)V"},
    )
    return [qkt, sm, av]


def _lower_layernorm(layer: Layer, dtype: str) -> List[HloOp]:
    elem = _bytes_per_elem(dtype)
    n = _prod(layer.shape_in)
    base = dict(shape=tuple(layer.shape_out), dtype=dtype,
                model_layer_id=layer.model_layer_id, layer_name=layer.name)
    rmean = HloOp(op_id=new_hlo_op_id(), kind=HloOpKind.REDUCE_MEAN,
                  flops=n, bytes_in=n*elem, bytes_out=n*elem, **base)
    var = HloOp(op_id=new_hlo_op_id(), kind=HloOpKind.VARIANCE,
                inputs=[rmean.op_id], flops=2*n, bytes_in=n*elem, bytes_out=n*elem, **base)
    norm = HloOp(op_id=new_hlo_op_id(), kind=HloOpKind.NORMALIZE,
                 inputs=[var.op_id], flops=3*n, bytes_in=n*elem, bytes_out=n*elem, **base)
    return [rmean, var, norm]


_LOWERINGS = {
    "linear": _lower_linear,
    "conv": _lower_conv,
    "attention": _lower_attention,
    "layernorm": _lower_layernorm,
}


# ── Public entry point ───────────────────────────────────────────────────────


def lower_to_hlo(graph: ModelGraph) -> HloModule:
    """Walk the model graph and emit a fake HLO module."""
    ops: List[HloOp] = []
    last_id = ""
    for layer in graph.layers:
        layer.model_layer_id = layer.model_layer_id or new_model_layer_id()
        lower = _LOWERINGS.get(layer.kind)
        if lower is None:
            raise ValueError(
                f"Unknown layer kind: {layer.kind!r} "
                f"(supported: {sorted(_LOWERINGS)})"
            )
        layer_ops = lower(layer, graph.dtype)
        ops.extend(layer_ops)
        last_id = layer_ops[-1].op_id

    return HloModule(
        name=f"{graph.name}.module",
        ops=ops,
        entry_op_id=last_id,
        model_name=graph.name,
    )
