"""Golden for L2/8 ApplyRotaryPosEmb — NPU kernel via torch_npu.

Uses torch_npu.npu_apply_rotary_pos_emb which applies RoPE (Rotary Position
Embedding) to query and key tensors on NPU.

KNOWN PRECISION DIFFERENCE:
  - Reference: casts bf16/fp16 to float32 for computation, then back to original dtype
  - NPU kernel: computes directly in bf16/fp16
  - Result: NPU has slightly larger numerical error (max_abs ~0.03 for bf16, ~0.008 for fp16)
  - This is expected behavior, not a bug. The test tolerance may need adjustment for
    bf16 cases (current bf16_relaxed atol=0.005, but NPU produces ~0.03).

API schema:
  npu_apply_rotary_pos_emb(query, key, cos, sin, layout="BSH", rotary_mode="half")
  -> (Tensor, Tensor)

Reference contract:
  forward(query, key, cos, sin, layout="BSND")
  -> [q_embed, k_embed]
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: str = "BSND",
    ) -> List[torch.Tensor]:
        # Move to NPU
        query_npu = query.npu() if not query.is_npu else query
        key_npu = key.npu() if not key.is_npu else key
        cos_npu = cos.npu() if not cos.is_npu else cos
        sin_npu = sin.npu() if not sin.is_npu else sin

        # Map reference layout to NPU layout
        # Reference: BSND (batch, seq, num_heads, head_dim) or TND (tokens, num_heads, head_dim)
        # NPU API: BSH (batch, seq, hidden) or TND (tokens, num_heads, head_dim)
        # BSND is 4D batch-first, map to BSH
        # TND is 3D token-first, keep as TND
        if layout == "BSND":
            npu_layout = "BSH"
        elif layout == "TND":
            npu_layout = "TND"
        else:
            npu_layout = layout

        q_res, k_res = torch_npu.npu_apply_rotary_pos_emb(
            query_npu, key_npu, cos_npu, sin_npu,
            layout=npu_layout,
            rotary_mode="half"
        )

        return [q_res.cpu(), k_res.cpu()]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "8_ApplyRotaryPosEmb.json")
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
    """Yield input groups matching Model.forward(query, key, cos, sin, layout)."""
    _NAME_ORDER = {"queryRef": 0, "query": 0, "keyRef": 1, "key": 1,
                   "cos": 2, "sin": 3, "layout": 4}
    results = []
    for case in INPUT_CASES:
        slots = [None] * 5
        for spec in case["inputs"]:
            idx = _NAME_ORDER.get(spec["name"])
            if idx is not None:
                slots[idx] = _make_arg(spec)
        if slots[4] is None:
            slots[4] = "BSND"
        results.append(slots)


    return results

def get_init_inputs():
    return []
