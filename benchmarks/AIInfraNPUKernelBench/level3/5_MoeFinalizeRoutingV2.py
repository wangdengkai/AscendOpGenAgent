"""Golden for L3/5 MoeFinalizeRoutingV2 -- direct wrapper around `torch_npu.npu_moe_finalize_routing`.

Schema (runtime-confirmed via `torch.ops.npu.npu_moe_finalize_routing.default._schema`):
    npu_moe_finalize_routing(
        Tensor expanded_permuted_rows,
        Tensor? skip1, Tensor? skip2, Tensor? bias, Tensor? scales,
        Tensor expanded_src_to_dst_row,
        Tensor? export_for_source_row,
        int? drop_pad_mode=0
    ) -> Tensor

Parameter name mapping (reference -> schema):
    expanded_x         -> expanded_permuted_rows
    expanded_row_idx   -> expanded_src_to_dst_row
    x1                 -> skip1
    x2                 -> skip2
    expert_idx         -> export_for_source_row

Note: Schema positional order differs from reference forward() — we remap.
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Optional

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        expanded_x: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        expert_idx: Optional[torch.Tensor] = None,
        drop_pad_mode: int = 0,
    ) -> torch.Tensor:
        _to_npu = lambda t: t.npu() if isinstance(t, torch.Tensor) else t
        result = torch_npu.npu_moe_finalize_routing(
            _to_npu(expanded_x),            # expanded_permuted_rows
            _to_npu(x1),                    # skip1
            _to_npu(x2),                    # skip2
            _to_npu(bias),                  # bias
            _to_npu(scales),                # scales
            _to_npu(expanded_row_idx),      # expanded_src_to_dst_row
            _to_npu(expert_idx),            # export_for_source_row
            drop_pad_mode=drop_pad_mode,
        )
        return result.cpu() if isinstance(result, torch.Tensor) else result


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

_DTYPE_ALIAS = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "fp64": "float64"}
_DTYPE_MAP = {
    "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    "bfloat16": torch.bfloat16, "int8": torch.int8, "int16": torch.int16,
    "int32": torch.int32, "int64": torch.int64, "uint8": torch.uint8, "bool": torch.bool,
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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "5_MoeFinalizeRoutingV2.json")
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
    _PARAM_ORDER = [
        "expanded_x", "expanded_row_idx", "x1", "x2",
        "bias", "scales", "expert_idx", "drop_pad_mode",
    ]
    _PARAM_DEFAULTS = {
        "x2": None, "bias": None, "scales": None,
        "expert_idx": None, "drop_pad_mode": 0,
    }
    results = []
    for case in INPUT_CASES:
        kwargs = {}
        for spec in case["inputs"]:
            kwargs[spec["name"]] = _make_arg(spec)
        args = []
        for p in _PARAM_ORDER:
            if p in kwargs:
                args.append(kwargs[p])
            elif p in _PARAM_DEFAULTS:
                args.append(_PARAM_DEFAULTS[p])
        results.append(args)


    return results

def get_init_inputs():
    return []
