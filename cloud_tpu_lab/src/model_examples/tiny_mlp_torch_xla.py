"""
PyTorch/XLA-ready tiny MLP.

CPU-safe: the import below uses torch on CPU. If torch isn't installed the
module degrades to a NotImplementedError when called — the simulator path
doesn't need this file, only the optional Cloud-TPU-ready notebook does.
"""
from __future__ import annotations

from typing import Tuple


def build_tiny_mlp_torch(
    input_dim: int = 64, hidden: int = 128, output_dim: int = 10,
) -> Tuple["torch.nn.Module", str]:  # noqa: F821
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "torch is not installed — install torch+torch-xla for the "
            "TPU-ready path."
        ) from exc

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden)
            self.l2 = nn.Linear(hidden, output_dim)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    note = (
        "TPU-ready: set device = xm.xla_device(); call xm.mark_step() each "
        "training step; see notebooks/05_pytorch_xla_cpu_to_tpu_ready.ipynb."
    )
    return TinyMLP(), note
