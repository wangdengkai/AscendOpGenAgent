import torch
import torch.nn as nn
import torch.nn.functional as F

SCENARIOS = [
    # FP16 scenarios
    {"name": "gelu_fp16_small", "shape": (128,), "dtype": torch.float16, "approximate": "tanh"},
    {"name": "gelu_fp16_1d", "shape": (1024,), "dtype": torch.float16, "approximate": "tanh"},
    {"name": "gelu_fp16_2d", "shape": (64, 128), "dtype": torch.float16, "approximate": "tanh"},
    {"name": "gelu_fp16_3d", "shape": (4, 64, 128), "dtype": torch.float16, "approximate": "tanh"},
    {"name": "gelu_fp16_large", "shape": (1024, 1024), "dtype": torch.float16, "approximate": "tanh"},
    {"name": "gelu_fp16_scalar", "shape": (1,), "dtype": torch.float16, "approximate": "tanh"},
    # BF16 scenarios
    {"name": "gelu_bf16_small", "shape": (128,), "dtype": torch.bfloat16, "approximate": "tanh"},
    {"name": "gelu_bf16_1d", "shape": (1024,), "dtype": torch.bfloat16, "approximate": "tanh"},
    {"name": "gelu_bf16_2d", "shape": (64, 128), "dtype": torch.bfloat16, "approximate": "tanh"},
    {"name": "gelu_bf16_3d", "shape": (4, 64, 128), "dtype": torch.bfloat16, "approximate": "tanh"},
    {"name": "gelu_bf16_large", "shape": (1024, 1024), "dtype": torch.bfloat16, "approximate": "tanh"},
    {"name": "gelu_bf16_scalar", "shape": (1,), "dtype": torch.bfloat16, "approximate": "tanh"},
    # FP32 scenarios
    {"name": "gelu_fp32_small", "shape": (128,), "dtype": torch.float32, "approximate": "tanh"},
    {"name": "gelu_fp32_1d", "shape": (1024,), "dtype": torch.float32, "approximate": "tanh"},
    {"name": "gelu_fp32_2d", "shape": (64, 128), "dtype": torch.float32, "approximate": "tanh"},
    {"name": "gelu_fp32_3d", "shape": (4, 64, 128), "dtype": torch.float32, "approximate": "tanh"},
    {"name": "gelu_fp32_large", "shape": (1024, 1024), "dtype": torch.float32, "approximate": "tanh"},
    {"name": "gelu_fp32_scalar", "shape": (1,), "dtype": torch.float32, "approximate": "tanh"},
    # Edge cases
    {"name": "gelu_fp32_neg_large", "shape": (128,), "dtype": torch.float32, "approximate": "tanh"},
    {"name": "gelu_fp32_pos_large", "shape": (128,), "dtype": torch.float32, "approximate": "tanh"},
    {"name": "gelu_fp16_neg_large", "shape": (128,), "dtype": torch.float16, "approximate": "tanh"},
    {"name": "gelu_fp16_pos_large", "shape": (128,), "dtype": torch.float16, "approximate": "tanh"},
]


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
        return F.gelu(x, approximate=approximate)


def get_input_groups():
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
