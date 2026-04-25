"""Golden for L1/6 ManifoldConstrainedHyperConnectionPost — direct wrapper around
`torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post`.

Schema (runtime-confirmed):
    custom::npu_ai_infra_manifold_constrained_hyper_connection_post(
        Tensor x, Tensor h_res, Tensor h_out, Tensor h_post
    ) -> Tensor

Reference signature: forward(x, h_res, h_out, h_post) -> Tensor

Cases come from neighboring `6_ManifoldConstrainedHyperConnectionPost.json` (one JSON record per line).
"""
import json as _json
import os as _os
from pathlib import Path as _Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import omni_training_custom_ops  # noqa: F401  (registers torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        x_npu = x.npu()
        h_res_npu = h_res.npu()
        h_out_npu = h_out.npu()
        h_post_npu = h_post.npu()

        out = torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
            x_npu, h_res_npu, h_out_npu, h_post_npu,
        )
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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "6_ManifoldConstrainedHyperConnectionPost.json")
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
