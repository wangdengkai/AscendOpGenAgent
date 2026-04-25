import sys
from pathlib import Path

import torch
import torch.nn as nn


_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _linear_index_v2_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, indices, stride: torch.Tensor, value_size: torch.Tensor):
        contiguous_indices = [tensor.contiguous() for tensor in indices]
        return _ext.run_linear_index_v2(contiguous_indices, stride.contiguous(), value_size.contiguous())
