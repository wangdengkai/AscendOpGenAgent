"""Golden for L3/1 DequantSwigluQuant -- direct wrapper around `torch_npu.npu_dequant_swiglu_quant`.

Schema (runtime-confirmed via `torch.ops.npu.npu_dequant_swiglu_quant.default._schema`):
    npu_dequant_swiglu_quant(
        Tensor x, *,
        Tensor? weight_scale=None, Tensor? activation_scale=None,
        Tensor? bias=None, Tensor? quant_scale=None, Tensor? quant_offset=None,
        Tensor? group_index=None,
        bool activate_left=False, int quant_mode=0,
        int? dst_type=None, int? round_mode=None, int? activate_dim=None,
        int swiglu_mode=0, float clamp_limit=7., float glu_alpha=1.702, float glu_bias=1.
    ) -> (Tensor, Tensor)

Parameter name mapping (reference -> schema):
    activate_scale -> activation_scale
    group_num      -> group_index
    quant_mode str -> quant_mode int  ("dynamic" -> 0)

Default divergences (reference vs schema):
    activate_left: True  vs False
    clamp_limit:   5.0   vs 7.0
    glu_alpha:     1.0   vs 1.702
    glu_bias:      0.0   vs 1.0
    (golden passes reference values through, so kernel sees reference defaults)
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


_QUANT_MODE_MAP = {"static": 0, "dynamic": 1}


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
        activate_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_scale: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
        group_num: Optional[torch.Tensor] = None,
        activate_left: bool = True,
        quant_mode: str = "dynamic",
        swiglu_mode: int = 0,
        clamp_limit: float = 5.0,
        glu_alpha: float = 1.0,
        glu_bias: float = 0.0,
    ) -> List[torch.Tensor]:
        qm_int = _QUANT_MODE_MAP.get(quant_mode, 0) if isinstance(quant_mode, str) else quant_mode
        # Kernel requires non-None quant_offset in non-group mode; supply zeros.
        # In dynamic + group mode, kernel requires quant_offset=None.
        if quant_offset is None and quant_scale is not None and group_num is None:
            quant_offset = torch.zeros_like(quant_scale, dtype=torch.float32)
        _to_npu = lambda t: t.npu() if isinstance(t, torch.Tensor) else t
        result = torch_npu.npu_dequant_swiglu_quant(
            _to_npu(x),
            weight_scale=_to_npu(weight_scale),
            activation_scale=_to_npu(activate_scale),
            bias=_to_npu(bias),
            quant_scale=_to_npu(quant_scale),
            quant_offset=_to_npu(quant_offset),
            group_index=_to_npu(group_num),
            activate_left=activate_left,
            quant_mode=qm_int,
            swiglu_mode=swiglu_mode,
            clamp_limit=clamp_limit,
            glu_alpha=glu_alpha,
            glu_bias=glu_bias,
        )
        return [t.cpu() if isinstance(t, torch.Tensor) else t for t in result]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "1_DequantSwigluQuant.json")
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
        "x", "weight_scale", "activate_scale", "bias",
        "quant_scale", "quant_offset", "group_num",
        "activate_left", "quant_mode", "swiglu_mode",
        "clamp_limit", "glu_alpha", "glu_bias",
    ]
    _PARAM_DEFAULTS = {
        "weight_scale": None, "activate_scale": None, "bias": None,
        "quant_scale": None, "quant_offset": None, "group_num": None,
        "activate_left": True, "quant_mode": "dynamic", "swiglu_mode": 0,
        "clamp_limit": 5.0, "glu_alpha": 1.0, "glu_bias": 0.0,
    }
    results = []
    for case in INPUT_CASES:
        kwargs = {}
        x_rows = None
        for spec in case["inputs"]:
            name = spec["name"]
            if name == "x":
                t = _make_tensor(spec)
                x_rows = spec["shape"][0]
                kwargs["x"] = t
            elif name == "group_num":
                group_count = spec["shape"][0]
                if x_rows is not None and group_count > 0:
                    if group_count <= x_rows:
                        base = x_rows // group_count
                        remainder = x_rows % group_count
                        vals = [base + (1 if i < remainder else 0) for i in range(group_count)]
                    else:
                        vals = [1] * x_rows + [0] * (group_count - x_rows)
                    kwargs["group_num"] = torch.tensor(vals, dtype=torch.int64)
                else:
                    kwargs["group_num"] = _make_tensor(spec)
            else:
                kwargs[name] = _make_arg(spec)
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
