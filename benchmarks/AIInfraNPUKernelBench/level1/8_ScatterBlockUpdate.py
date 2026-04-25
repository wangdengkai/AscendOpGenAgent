"""Golden for L1/8 ScatterBlockUpdate — direct wrapper around
`torch.ops.custom.npu_ai_infra_scatter_block_update_`.

Schema (runtime-confirmed):
    custom::npu_ai_infra_scatter_block_update_(Tensor(a!) input, Tensor indices, Tensor update) -> ()

The op is IN-PLACE. Golden clones input, moves to NPU, calls the op, returns the clone.

Reference signature: forward(input, indices, update) -> Tensor

Cases come from neighboring `8_ScatterBlockUpdate.json` (one JSON record per line).
"""
import json as _json
import os as _os
from pathlib import Path as _Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import omni_custom_ops  # noqa: F401  (registers torch.ops.custom.npu_ai_infra_scatter_block_update_)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        indices: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        output = input.clone().npu()
        idx = indices.npu()
        upd = update.npu()
        # Returns () — in-place op
        torch.ops.custom.npu_ai_infra_scatter_block_update_(output, idx, upd)
        return output.cpu()


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "8_ScatterBlockUpdate.json")
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
    results = []
    for case in INPUT_CASES:
        specs = case["inputs"]
        input_spec = specs[0]
        indices_spec = specs[1]
        update_spec = specs[2]
        input_tensor = _make_tensor(input_spec)
        K = indices_spec["shape"][0]
        d0, d1 = input_spec["shape"][0], input_spec["shape"][1]
        idx_dtype = _DTYPE_MAP[indices_spec["dtype"]]
        flat = torch.arange(K, dtype=idx_dtype)
        col0 = (flat // d1) % d0
        col1 = flat % d1
        indices_tensor = torch.stack([col0, col1], dim=1)
        update_tensor = _make_tensor(update_spec)
        results.append([input_tensor, indices_tensor, update_tensor])


    return results

def get_init_inputs():
    return []
