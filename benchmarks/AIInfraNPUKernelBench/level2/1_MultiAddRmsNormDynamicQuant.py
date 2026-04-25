"""Golden for L2/1 MultiAddRmsNormDynamicQuant — single-op NPU calls.

Uses torch_npu.npu_add_rms_norm + torch_npu.npu_dynamic_quant in eager mode.

API schema:
  npu_add_rms_norm(x1, x2, gamma, epsilon) -> (y_norm, rstd, x_sum)
  npu_dynamic_quant(input, smooth_scales=None) -> (y_quant, scale)
  Note: npu_dynamic_quant requires smooth_scales to be 1D.
        For multi-dim smooth_scales, pre-multiply then pass None.

Reference contract:
  forward(x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, epsilon=1e-5)
  -> [x_sum, y_norm, y1_quant, scale1, y2_quant, scale2]
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


def _dynamic_quant(y_norm, smooth_scale):
    """Run npu_dynamic_quant; handle multi-dim smooth_scales by pre-multiply."""
    if smooth_scale is None:
        return torch_npu.npu_dynamic_quant(y_norm, smooth_scales=None)
    if smooth_scale.dim() == 1:
        return torch_npu.npu_dynamic_quant(y_norm, smooth_scales=smooth_scale)
    # multi-dim: pre-multiply then quant without smooth
    return torch_npu.npu_dynamic_quant(y_norm * smooth_scale, smooth_scales=None)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: torch.Tensor,
        smooth_scale1: torch.Tensor = None,
        smooth_scale2: torch.Tensor = None,
        epsilon: float = 1e-5,
    ) -> List[torch.Tensor]:
        x1_npu = x1.npu() if not x1.is_npu else x1
        x2_npu = x2.npu() if not x2.is_npu else x2
        gamma_npu = gamma.npu() if not gamma.is_npu else gamma

        ss1 = smooth_scale1.npu() if smooth_scale1 is not None else None
        ss2 = smooth_scale2.npu() if smooth_scale2 is not None else None

        y_norm, rstd, x_sum = torch_npu.npu_add_rms_norm(x1_npu, x2_npu, gamma_npu, epsilon)

        y1, scale1 = _dynamic_quant(y_norm, ss1)

        if ss2 is not None:
            y2, scale2 = _dynamic_quant(y_norm, ss2)
        elif ss1 is not None:
            y2, scale2 = y1, scale1
        else:
            y2, scale2 = y1, scale1

        torch.npu.synchronize()
        return [x_sum.cpu(), y_norm.cpu(), y1.cpu(), scale1.cpu(), y2.cpu(), scale2.cpu()]


# ---------------------------------------------------------------------------
# Case loading (mirrors prompt_reference.py contract for test_golden.py)
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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "1_MultiAddRmsNormDynamicQuant.json")
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


def _make_multi_add_inputs(case):
    """Custom input generation that maps tensor/attr inputs to correct forward() positions."""
    inputs = case["inputs"]
    tensors = [inp for inp in inputs if inp["type"] == "tensor"]
    attrs = {inp["name"]: inp["value"] for inp in inputs if inp["type"] == "attr"}

    args = []
    for t in tensors:
        args.append(_make_tensor(t))

    while len(args) < 5:
        args.append(None)

    epsilon = attrs.get("epsilon", 1e-5)
    args.append(epsilon)

    return args


def get_input_groups():
    results = []
    for case in INPUT_CASES:
        results.append(_make_multi_add_inputs(case))


    return results

def get_init_inputs():
    return []
