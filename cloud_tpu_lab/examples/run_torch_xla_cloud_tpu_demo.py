#!/usr/bin/env python3
"""
PyTorch/XLA Cloud TPU demo — CPU-safe.

If torch_xla is installed: runs a tiny mm on the XLA device.
Else: prints install instructions and exits 0.

Referenced by docs/05_pytorch_xla_on_tpu.md.
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"[torch-xla-demo] torch not installed ({type(exc).__name__}).")
        print("  See https://pytorch.org/xla/")
        return 0

    try:
        import torch_xla.core.xla_model as xm
    except Exception as exc:  # noqa: BLE001
        print(f"[torch-xla-demo] torch_xla not installed ({type(exc).__name__}).")
        print("  Install on a Cloud TPU VM per https://pytorch.org/xla/")
        # Fall back to plain CPU torch demo so the script still does something.
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        t0 = time.perf_counter()
        for _ in range(10):
            c = a @ b
        dt = time.perf_counter() - t0
        print(f"[torch-xla-demo] CPU-only fallback — 10× mm: {dt*1000:.2f} ms")
        return 0

    device = xm.xla_device()
    print(f"[torch-xla-demo] device: {device}")
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)

    # Compile + warmup.
    c = a @ b
    xm.mark_step()
    _ = c.cpu()

    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    xm.mark_step()
    _ = c.cpu()
    dt = time.perf_counter() - t0
    print(f"[torch-xla-demo] 10× 1024×1024 mm: {dt*1000:.2f} ms total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
