"""Golden for L1/4 AggregateHidden — direct wrapper around
`torch.ops.custom.npu_aggregate_hidden` and `torch.ops.custom.npu_aggregate_hidden_grad`.

Schema (runtime-confirmed):
    custom::npu_aggregate_hidden(Tensor input, Tensor weight, *, Tensor? mask=None) -> Tensor
    custom::npu_aggregate_hidden_grad(Tensor grad_output, Tensor input, Tensor weight, *, Tensor? mask=None) -> (Tensor, Tensor)

Reference signature: forward(grad_out, input, weight, mask) -> [output, grad_input, grad_weight]

Cases come from neighboring `4_AggregateHidden.json` (one JSON record per line).
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import omni_training_custom_ops  # noqa: F401  (registers torch.ops.custom.npu_aggregate_hidden*)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        inp_npu = input.npu()
        w_npu = weight.npu()
        m_npu = mask.npu() if mask is not None else None
        go_npu = grad_out.npu()

        output = torch.ops.custom.npu_aggregate_hidden(inp_npu, w_npu, mask=m_npu).cpu()
        gi, gw = torch.ops.custom.npu_aggregate_hidden_grad(go_npu, inp_npu, w_npu, mask=m_npu)
        return [output, gi.cpu(), gw.cpu()]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "4_AggregateHidden.json")
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


def _make_tensor_list(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    return [torch.randn(*shape, dtype=dtype) for shape in spec["shapes"]]


def _make_arg(spec):
    t = spec["type"]
    if t == "tensor":
        return _make_tensor(spec)
    if t == "tensor_list":
        return _make_tensor_list(spec)
    if t == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {t}")


def get_input_groups():
    return [[_make_arg(spec) for spec in case["inputs"]] for case in INPUT_CASES]


def get_init_inputs():
    return []
