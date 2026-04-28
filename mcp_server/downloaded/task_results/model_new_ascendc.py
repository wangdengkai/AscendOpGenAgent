import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import gelu_op as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
        assert x.is_contiguous(), "Input tensor must be contiguous"
        assert approximate == "tanh", "Only tanh approximation is supported"
        assert x.dtype in (torch.float16, torch.float32, torch.bfloat16), "unsupported dtype"
        return _ext.run_gelu(x)


def get_input_groups():
    from model import SCENARIOS

    input_groups = []
    for scenario in SCENARIOS:
        shape = scenario["shape"]
        dtype = scenario["dtype"]
        approximate = scenario.get("approximate", "tanh")

        if "neg_large" in scenario["name"]:
            x = torch.randn(shape, dtype=dtype) * 10 - 5
        elif "pos_large" in scenario["name"]:
            x = torch.randn(shape, dtype=dtype) * 10 + 5
        else:
            x = torch.randn(shape, dtype=dtype)

        input_groups.append([x, approximate])
    return input_groups


def get_init_inputs():
    return []
