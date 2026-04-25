"""Golden for L1/2 ClippedSwiglu — direct wrapper around `torch_npu.npu_clipped_swiglu`.

Schema (runtime-confirmed via `torch.ops.npu.npu_clipped_swiglu.default._schema`):
    npu_clipped_swiglu(
        Tensor x, *,
        Tensor? group_index=None,
        int dim=-1, float alpha=1.702, float limit=7., float bias=1.,
        bool interleaved=True
    ) -> Tensor

Cases come from neighboring `2_ClippedSwiglu.json` (one JSON record per line).
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Optional

import torch
import torch.nn as nn
import torch_npu  # noqa: F401  (registers torch.ops.npu.npu_clipped_swiglu)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        group_index: Optional[torch.Tensor],
        dim: int,
        alpha: float,
        limit: float,
        bias: float,
        interleaved: bool,
    ) -> torch.Tensor:
        x_npu = x.npu()
        gi_npu = group_index.npu() if group_index is not None else None
        out = torch_npu.npu_clipped_swiglu(
            x_npu,
            group_index=gi_npu,
            dim=dim,
            alpha=alpha,
            limit=limit,
            bias=bias,
            interleaved=interleaved,
        )
        # NPU kernel does not write rows beyond `Σ group_index` in the merged
        # [pre, cut*after] layout, leaving uninitialized memory there. The
        # reference contract zero-fills those rows. Mask them here so the
        # padding region is well-defined and matches the reference bit-for-bit.
        if group_index is not None:
            ndim = x.dim()
            d = dim if dim >= 0 else dim + ndim
            pre = 1
            for s in x.shape[:d]:
                pre *= s
            after = 1
            for s in out.shape[d + 1:]:
                after *= s
            cut_half = out.shape[d]
            flat = out.reshape(pre, cut_half * after)
            group_sum = min(int(group_index.sum().item()), pre)
            if group_sum < pre:
                flat[group_sum:].zero_()
        return out.cpu()


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "2_ClippedSwiglu.json")
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
    results = []
    for case in INPUT_CASES:
        args = []
        has_group_index = False
        for spec in case["inputs"]:
            if spec["name"] == "group_index":
                has_group_index = True
            args.append(_make_arg(spec))
        if not has_group_index:
            args.insert(1, None)
        results.append(args)


    return results

def get_init_inputs():
    return []
