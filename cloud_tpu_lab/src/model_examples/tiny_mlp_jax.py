"""
Tiny MLP — defined as a ModelGraph (the lowering target).

We intentionally do NOT import jax here. The whole vertical slice runs on
plain Python + NumPy in simulation mode; jax is only needed for the
notebooks marked "TPU-ready".
"""
from __future__ import annotations

from typing import List

from ..xla_sim.lowering import Layer, ModelGraph


def build_tiny_mlp_graph(
    batch_size: int = 8,
    input_dim: int = 64,
    hidden_size: int = 128,
    output_dim: int = 10,
    num_layers: int = 2,
    dtype: str = "bf16",
) -> ModelGraph:
    """Linear → ReLU(skipped) → Linear → ... → Linear(output_dim)"""
    layers: List[Layer] = []
    in_dim = input_dim
    for i in range(num_layers):
        out_dim = hidden_size if i < num_layers - 1 else output_dim
        layers.append(Layer(
            name=f"mlp.layer{i}",
            kind="linear",
            shape_in=(batch_size, in_dim),
            shape_out=(batch_size, out_dim),
        ))
        in_dim = out_dim
    return ModelGraph(name="tiny_mlp", layers=layers, dtype=dtype)
