"""
Tiny transformer block — defined as a ModelGraph for the simulator.
"""
from __future__ import annotations

from typing import List

from ..xla_sim.lowering import Layer, ModelGraph


def build_tiny_transformer_graph(
    batch_size: int = 4,
    seq_len: int = 32,
    hidden_size: int = 128,
    num_layers: int = 2,
    dtype: str = "bf16",
) -> ModelGraph:
    """One repeated (LayerNorm → Attention → LayerNorm → MLP) block."""
    layers: List[Layer] = []
    shape = (batch_size, seq_len, hidden_size)
    for i in range(num_layers):
        layers.append(Layer(name=f"block{i}.ln1", kind="layernorm",
                            shape_in=shape, shape_out=shape))
        layers.append(Layer(name=f"block{i}.attn", kind="attention",
                            shape_in=shape, shape_out=shape))
        layers.append(Layer(name=f"block{i}.ln2", kind="layernorm",
                            shape_in=shape, shape_out=shape))
        # MLP = Linear(H→4H) + Linear(4H→H).
        layers.append(Layer(name=f"block{i}.mlp.fc1", kind="linear",
                            shape_in=shape,
                            shape_out=(batch_size, seq_len, 4 * hidden_size)))
        layers.append(Layer(name=f"block{i}.mlp.fc2", kind="linear",
                            shape_in=(batch_size, seq_len, 4 * hidden_size),
                            shape_out=shape))
    return ModelGraph(name="tiny_transformer", layers=layers, dtype=dtype)
