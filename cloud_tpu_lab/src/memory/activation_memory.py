"""
Activation-memory estimator.

For an MLP layer with input shape (B, D_in) and output shape (B, D_out)
the forward activation is B × D_out elements. For a transformer block
with shape (B, S, H), the dominant activations are the attention scores
(B × S × S) plus the residual stream (B × S × H).

This module exposes a single helper used by the demo to plot
activation-memory growth with batch size & sequence length, which is the
clearest way to teach why TPU memory pressure scales the way it does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ActivationEstimate:
    layer_name: str
    bytes: int
    dominant_factor: str  # "batch×hidden" | "batch×seq^2" | etc.


def _bytes_per_elem(dtype: str) -> int:
    return 2 if dtype == "bf16" else 4


def estimate_mlp_activations(
    batch_size: int, hidden_size: int, num_layers: int, dtype: str = "bf16"
) -> List[ActivationEstimate]:
    elem = _bytes_per_elem(dtype)
    out: List[ActivationEstimate] = []
    for i in range(num_layers):
        out.append(ActivationEstimate(
            layer_name=f"mlp.layer{i}",
            bytes=batch_size * hidden_size * elem,
            dominant_factor="batch×hidden",
        ))
    return out


def estimate_transformer_activations(
    batch_size: int, seq_len: int, hidden_size: int,
    num_layers: int, dtype: str = "bf16",
) -> List[ActivationEstimate]:
    elem = _bytes_per_elem(dtype)
    out: List[ActivationEstimate] = []
    for i in range(num_layers):
        # Attention scores — quadratic in seq_len.
        out.append(ActivationEstimate(
            layer_name=f"transformer.layer{i}.attn_scores",
            bytes=batch_size * seq_len * seq_len * elem,
            dominant_factor="batch×seq^2",
        ))
        # Residual stream + MLP intermediate — linear in seq_len.
        out.append(ActivationEstimate(
            layer_name=f"transformer.layer{i}.residual",
            bytes=batch_size * seq_len * hidden_size * elem,
            dominant_factor="batch×seq×hidden",
        ))
    return out


def total_bytes(estimates: List[ActivationEstimate]) -> int:
    return sum(e.bytes for e in estimates)
