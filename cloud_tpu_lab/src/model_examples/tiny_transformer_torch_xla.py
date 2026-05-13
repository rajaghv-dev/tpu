"""
PyTorch/XLA-ready tiny transformer block.

CPU-safe wrapper — the actual forward pass needs torch. See notebook
`05_pytorch_xla_cpu_to_tpu_ready.ipynb` for the TPU-runtime instructions
(xm.xla_device(), xm.mark_step()).
"""
from __future__ import annotations

from typing import Tuple


def build_tiny_transformer_torch(
    hidden: int = 128, n_heads: int = 4, n_layers: int = 2,
) -> Tuple["torch.nn.Module", str]:  # noqa: F821
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch not installed; see TPU-ready notebook 05") from exc

    layer = nn.TransformerEncoderLayer(
        d_model=hidden, nhead=n_heads, batch_first=True,
    )
    model = nn.TransformerEncoder(layer, num_layers=n_layers)
    return model, "Move the model to xm.xla_device() and call xm.mark_step() per step."
