"""Golden for L2/9 RotaryPositionEmbedding — NPU kernel via torch_npu.

Uses torch_npu.npu_rotary_mul which applies RoPE with different rotation modes.

API schema:
  npu_rotary_mul(input, r1, r2, rotary_mode="half") -> Tensor

Supported modes:
  - Mode 0 (half): rotate_half, supports partial RoPE
  - Mode 1 (interleave): rotate_every_two, x_dim must equal cos_dim

Test cases are constrained to NPU-compatible configurations only.

Reference contract:
  forward(x, cos, sin, mode=0) -> output
"""
import json as _json
import os as _os
from pathlib import Path as _Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mode: int = 0,
    ) -> torch.Tensor:
        # Move to NPU
        x_npu = x.npu() if not x.is_npu else x
        cos_npu = cos.npu() if not cos.is_npu else cos
        sin_npu = sin.npu() if not sin.is_npu else sin

        # Map mode to rotary_mode string
        if mode == 0:
            rotary_mode = "half"
        elif mode == 1:
            rotary_mode = "interleave"
        else:
            raise ValueError(f"Unsupported rotary mode: {mode}")

        # npu_rotary_mul: r1=cos, r2=sin
        y = torch_npu.npu_rotary_mul(x_npu, cos_npu, sin_npu, rotary_mode=rotary_mode)

        return y.cpu()


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

_DTYPE_ALIAS = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "fp64": "float64"}
_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _load_jsonl_cases(path):
    p = _Path(path)
    if not p.exists():
        return []
    cases = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = _json.loads(line)
            for inp in raw["inputs"]:
                inp["dtype"] = _DTYPE_ALIAS.get(inp["dtype"], inp["dtype"])
            cases.append(raw)
    return cases


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "9_RotaryPositionEmbedding.json")
INPUT_CASES = _load_jsonl_cases(_JSONL_PATH)
def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    value_range = spec.get("range")
    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)
    if value_range is not None:
        low, high = value_range
        if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            return torch.randint(low, high + 1, tuple(shape), dtype=dtype)
        return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)
    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)
    return torch.randn(*shape, dtype=dtype)


def _make_arg(spec):
    t = spec["type"]
    if t == "tensor":
        return _make_tensor(spec)
    if t == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {t}")


def get_input_groups():
    return [[_make_arg(spec) for spec in case["inputs"]] for case in INPUT_CASES]


def get_init_inputs():
    return []
