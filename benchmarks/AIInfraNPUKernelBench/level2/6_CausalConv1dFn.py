"""Golden for L2/6 CausalConv1dFn — NPU kernel via omni_custom_ops.

Uses torch.ops.custom.npu_ai_infra_causal_conv1d_fn_add which performs
grouped depthwise causal conv1d with cached states on NPU.

API schema:
  npu_ai_infra_causal_conv1d_fn_add(x, weight, conv_states, bias=None,
      query_start_loc=None, cache_indices=None, initial_state_mode=None,
      activation=None, residual_connection=1, pad_slot_id=-1) -> Tensor

Note: conv_states is modified in-place. The API returns only the output tensor,
not the updated conv_states.

Reference contract:
  forward(x, weight, conv_states, query_start_loc, cache_indices,
          initial_state_mode, pad_slot_id=-1, residual_connection=1)
  -> [output, conv_states]
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import omni_custom_ops  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor,
        pad_slot_id: int = -1,
        residual_connection: int = 1,
    ) -> List[torch.Tensor]:
        # Move to NPU
        x_npu = x.npu() if not x.is_npu else x
        weight_npu = weight.npu() if not weight.is_npu else weight
        conv_states_npu = conv_states.clone().npu() if not conv_states.is_npu else conv_states.clone()
        query_start_loc_npu = query_start_loc.npu() if not query_start_loc.is_npu else query_start_loc
        cache_indices_npu = cache_indices.npu() if not cache_indices.is_npu else cache_indices
        initial_state_mode_npu = initial_state_mode.npu() if not initial_state_mode.is_npu else initial_state_mode

        # Call NPU kernel (conv_states_npu is modified in-place)
        output = torch.ops.custom.npu_ai_infra_causal_conv1d_fn_add(
            x_npu, weight_npu, conv_states_npu,
            bias=None,
            query_start_loc=query_start_loc_npu,
            cache_indices=cache_indices_npu,
            initial_state_mode=initial_state_mode_npu,
            activation=None,
            residual_connection=residual_connection,
            pad_slot_id=pad_slot_id,
        )

        return [output.cpu(), conv_states_npu.cpu()]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "6_CausalConv1dFn.json")
INPUT_CASES = _load_jsonl_cases(_JSONL_PATH)
def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    value_range = spec.get("range")
    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)
    if value_range is not None:
        if isinstance(value_range, dict):
            pass  # fall through to randn
        else:
            low, high = value_range
            if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
                return torch.randint(low, high + 1, tuple(shape), dtype=dtype)
            return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)
    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)
    return torch.randn(*shape, dtype=dtype)


def _make_causal_conv1d_fn_inputs(case):
    """Custom input generation for CausalConv1dFn with valid queryStartLoc and cacheIndices."""
    inputs = case["inputs"]
    x_spec = inputs[0]
    weight_spec = inputs[1]
    conv_states_spec = inputs[2]
    qsl_spec = inputs[3]
    ci_spec = inputs[4]
    ism_spec = inputs[5]

    x_dtype = _DTYPE_MAP[x_spec["dtype"]]
    cu_seq_len = x_spec["shape"][0]
    dim = x_spec["shape"][1]
    batch = qsl_spec["shape"][0] - 1
    num_states = conv_states_spec["shape"][0]

    x = torch.randn(cu_seq_len, dim, dtype=x_dtype).clamp_(-1, 1)
    weight = torch.randn(*weight_spec["shape"], dtype=_DTYPE_MAP[weight_spec["dtype"]]).clamp_(-1, 1)
    conv_states = torch.randn(*conv_states_spec["shape"], dtype=_DTYPE_MAP[conv_states_spec["dtype"]]).clamp_(-1, 1)

    if batch > 0 and cu_seq_len >= batch:
        min_per_seg = 1
        remaining = cu_seq_len - batch * min_per_seg
        if remaining > 0:
            extras = sorted(torch.randint(0, remaining + 1, (batch - 1,)).tolist())
        else:
            extras = [0] * (batch - 1)
        qsl = [0]
        cumsum = 0
        for i in range(batch - 1):
            cumsum += min_per_seg + (extras[i] - (extras[i - 1] if i > 0 else 0))
            qsl.append(cumsum)
        qsl.append(cu_seq_len)
    else:
        qsl = [0, cu_seq_len]
    query_start_loc = torch.tensor(qsl[:qsl_spec["shape"][0]], dtype=torch.int32)

    cache_indices = torch.randperm(min(num_states, batch * 10))[:batch].to(torch.int32)
    initial_state_mode = torch.randint(1, 3, (batch,), dtype=torch.int32)

    return [x, weight, conv_states, query_start_loc, cache_indices, initial_state_mode]


def get_input_groups():
    _NAME_MAP = {"padSlotId": "pad_slot_id", "residualConnection": "residual_connection"}
    _ATTR_DEFAULTS = {"pad_slot_id": -1, "residual_connection": 1}
    results = []
    for case in INPUT_CASES:
        base_args = _make_causal_conv1d_fn_inputs(case)
        attrs = {}
        for spec in case["inputs"]:
            if spec["type"] == "attr":
                mapped = _NAME_MAP.get(spec["name"], spec["name"])
                attrs[mapped] = spec["value"]
        pad_slot_id = attrs.get("pad_slot_id", _ATTR_DEFAULTS["pad_slot_id"])
        residual_connection = attrs.get("residual_connection", _ATTR_DEFAULTS["residual_connection"])
        results.append(base_args + [pad_slot_id, residual_connection])


    return results

def get_init_inputs():
    return []
