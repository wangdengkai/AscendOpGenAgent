"""Golden for L3/6 MoeGatingTopK -- direct wrapper around `torch_npu.npu_moe_gating_top_k`.

Schema (runtime-confirmed via `torch.ops.npu.npu_moe_gating_top_k.default._schema`):
    npu_moe_gating_top_k(
        Tensor x, int k, *,
        Tensor? bias=None,
        int k_group=1, int group_count=1, int group_select_mode=0,
        int renorm=0, int norm_type=0, bool out_flag=False,
        float routed_scaling_factor=1., float eps=1e-21
    ) -> (Tensor, Tensor, Tensor)

Parameter order mapping (reference -> schema):
    Reference: forward(self, x, bias, k=8, ...)
    Schema:    npu_moe_gating_top_k(x, k, *, bias=None, ...)
    `k` is positional in schema, keyword with default 8 in reference.
    `bias` is positional in reference, keyword-only in schema.

Default divergences (reference vs schema):
    eps: 1e-20 vs 1e-21
    (golden passes reference values through, so kernel sees reference defaults)
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        k: int = 8,
        k_group: int = 1,
        group_count: int = 1,
        group_select_mode: int = 0,
        renorm: int = 0,
        norm_type: int = 0,
        out_flag: bool = False,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
    ) -> List[torch.Tensor]:
        _to_npu = lambda t: t.npu() if isinstance(t, torch.Tensor) else t
        result = torch_npu.npu_moe_gating_top_k(
            _to_npu(x), k,
            bias=_to_npu(bias),
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
        )
        # Schema returns 3-tuple: (y, expert_idx, y2)
        out = [t.cpu() if isinstance(t, torch.Tensor) else t for t in result]
        if not out_flag:
            out[2] = torch.empty(0)
        return out


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "6_MoeGatingTopK.json")
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
        "x", "bias", "k", "k_group", "group_count",
        "group_select_mode", "renorm", "norm_type",
        "out_flag", "routed_scaling_factor", "eps",
    ]
    _PARAM_DEFAULTS = {
        "k": 8, "k_group": 1, "group_count": 1,
        "group_select_mode": 0, "renorm": 0, "norm_type": 0,
        "out_flag": False, "routed_scaling_factor": 1.0, "eps": 1e-20,
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
