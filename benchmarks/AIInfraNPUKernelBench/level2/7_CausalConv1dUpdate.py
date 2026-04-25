"""Golden for L2/7 CausalConv1dUpdate — NPU kernel via omni_custom_ops.

Uses torch.ops.custom.npu_ai_infra_causal_conv1d_update_add which performs
causal conv1d update for speculative decoding on NPU.

API schema:
  npu_ai_infra_causal_conv1d_update_add(x, weight, conv_states, bias=None,
      query_start_loc=None, cache_indices=None, num_accepted_tokens=None,
      activation=None, residual_connection=1, pad_slot_id=-1) -> Tensor

Note: conv_states is modified in-place. The API returns only the output tensor,
not the updated conv_states.

Reference contract:
  forward(x, weight, conv_states, query_start_loc=None, cache_indices=None,
          num_accepted_tokens=None, residual_connection=1, pad_slot_id=-1)
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
        query_start_loc: torch.Tensor = None,
        cache_indices: torch.Tensor = None,
        num_accepted_tokens: torch.Tensor = None,
        residual_connection: int = 1,
        pad_slot_id: int = -1,
    ) -> List[torch.Tensor]:
        # Move to NPU
        x_npu = x.npu() if not x.is_npu else x
        weight_npu = weight.npu() if not weight.is_npu else weight
        conv_states_npu = conv_states.clone().npu() if not conv_states.is_npu else conv_states.clone()

        query_start_loc_npu = None
        if query_start_loc is not None:
            query_start_loc_npu = query_start_loc.npu() if not query_start_loc.is_npu else query_start_loc

        cache_indices_npu = None
        if cache_indices is not None:
            cache_indices_npu = cache_indices.npu() if not cache_indices.is_npu else cache_indices

        num_accepted_tokens_npu = None
        if num_accepted_tokens is not None:
            num_accepted_tokens_npu = num_accepted_tokens.npu() if not num_accepted_tokens.is_npu else num_accepted_tokens

        # Call NPU kernel (conv_states_npu is modified in-place)
        output = torch.ops.custom.npu_ai_infra_causal_conv1d_update_add(
            x_npu, weight_npu, conv_states_npu,
            bias=None,
            query_start_loc=query_start_loc_npu,
            cache_indices=cache_indices_npu,
            num_accepted_tokens=num_accepted_tokens_npu,
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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "7_CausalConv1dUpdate.json")
INPUT_CASES = _load_jsonl_cases(_JSONL_PATH)
def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    value_range = spec.get("range")
    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)
    if value_range is not None:
        if isinstance(value_range, dict):
            pass
        else:
            low, high = value_range
            if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
                return torch.randint(low, high + 1, tuple(shape), dtype=dtype)
            return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)
    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)
    return torch.randn(*shape, dtype=dtype)


def _make_causal_conv1d_update_inputs(case):
    """Custom input generation for CausalConv1dUpdate with valid cacheIndices and numAcceptedTokens."""
    inputs = case["inputs"]
    spec_by_name = {inp["name"]: inp for inp in inputs}

    x_spec = spec_by_name["x"]
    weight_spec = spec_by_name["weight"]
    conv_states_spec = spec_by_name["convStates"]
    ci_spec = spec_by_name["cacheIndices"]
    nat_spec = spec_by_name["numAcceptedTokens"]

    x_dtype = _DTYPE_MAP[x_spec["dtype"]]
    x_shape = x_spec["shape"]
    num_states = conv_states_spec["shape"][0]
    state_len = conv_states_spec["shape"][1]

    x = torch.randn(*x_shape, dtype=x_dtype).clamp_(-1, 1)
    weight = torch.randn(*weight_spec["shape"], dtype=_DTYPE_MAP[weight_spec["dtype"]]).clamp_(-1, 1)
    conv_states = torch.randn(*conv_states_spec["shape"], dtype=_DTYPE_MAP[conv_states_spec["dtype"]]).clamp_(-1, 1)

    batch = ci_spec["shape"][0]
    cache_indices = torch.randperm(min(num_states, batch * 10))[:batch].to(torch.int32)

    if len(x_shape) == 3:
        seq_len = x_shape[1]
        max_accepted = min(seq_len, state_len)
        num_accepted_tokens = torch.randint(1, max(2, max_accepted + 1), (batch,), dtype=torch.int32)
    else:
        num_accepted_tokens = torch.ones(batch, dtype=torch.int32)

    query_start_loc = None
    if "queryStartLoc" in spec_by_name:
        qsl_spec = spec_by_name["queryStartLoc"]
        qsl_len = qsl_spec["shape"][0]
        if len(x_shape) == 3:
            total = x_shape[0] * x_shape[1]
        else:
            total = x_shape[0]
        seg = max(1, total // (qsl_len - 1))
        qsl = [0]
        for i in range(1, qsl_len - 1):
            qsl.append(min(i * seg, total))
        qsl.append(total)
        query_start_loc = torch.tensor(qsl[:qsl_len], dtype=torch.int32)

    attrs = {}
    for inp in inputs:
        if inp["type"] == "attr":
            attrs[inp["name"]] = inp["value"]

    pad_slot_id = attrs.get("padSlotId", -1)
    residual_connection = attrs.get("residualConnection", 1)

    return [x, weight, conv_states, query_start_loc, cache_indices, num_accepted_tokens,
            residual_connection, pad_slot_id]


def get_input_groups():
    results = []
    for case in INPUT_CASES:
        results.append(_make_causal_conv1d_update_inputs(case))


    return results

def get_init_inputs():
    return []
