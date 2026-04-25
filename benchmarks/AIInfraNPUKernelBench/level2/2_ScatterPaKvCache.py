"""Golden for L2/2 ScatterPaKvCache — PTA direct call.

Requires unsetting ASCEND_CUSTOM_OPP_PATH to avoid custom op path
overriding the mainline kernel binary.

API: npu_scatter_pa_kv_cache(key, value, key_cache!, value_cache!,
                             slot_mapping, cache_mode="PA_NZ") -> ()
     inplace on key_cache and value_cache.

Reference contract:
  forward(key, key_cache, slot_mapping, value, value_cache, cache_mode="PA_NZ")
  -> [key_cache_updated, value_cache_updated]
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List

# Unset ASCEND_CUSTOM_OPP_PATH before importing torch_npu to use mainline kernel
if "ASCEND_CUSTOM_OPP_PATH" in _os.environ:
    del _os.environ["ASCEND_CUSTOM_OPP_PATH"]

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        key: torch.Tensor,
        key_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        value: torch.Tensor,
        value_cache: torch.Tensor,
        cache_mode: str = "PA_NZ",
    ) -> List[torch.Tensor]:
        key_npu = key.npu() if not key.is_npu else key
        value_npu = value.npu() if not value.is_npu else value
        key_cache_npu = key_cache.clone().npu() if not key_cache.is_npu else key_cache.clone()
        value_cache_npu = value_cache.clone().npu() if not value_cache.is_npu else value_cache.clone()
        slot_mapping_npu = slot_mapping.npu() if not slot_mapping.is_npu else slot_mapping

        torch_npu.npu_scatter_pa_kv_cache(
            key_npu, value_npu, key_cache_npu, value_cache_npu, slot_mapping_npu
        )
        torch.npu.synchronize()

        return [key_cache_npu.cpu(), value_cache_npu.cpu()]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "2_ScatterPaKvCache.json")
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


def _make_scatter_kv_inputs(case):
    """Custom input generation for ScatterPaKvCache with valid slot_mapping."""
    inputs = case["inputs"]
    args = []
    for inp in inputs:
        if inp["type"] == "attr":
            args.append(inp["value"])
            continue
        if inp["name"] == "slotMapping":
            key_cache_spec = next(i for i in inputs if i["name"] == "keyCache")
            num_blocks = key_cache_spec["shape"][0]
            block_size = key_cache_spec["shape"][2]
            total_slots = num_blocks * block_size
            n_tokens = inp["shape"][0]
            slot_mapping = torch.randperm(total_slots)[:n_tokens].to(torch.int32)
            args.append(slot_mapping)
        elif inp["name"] in ("keyCache", "valueCache"):
            dtype = _DTYPE_MAP[inp["dtype"]]
            args.append(torch.zeros(*inp["shape"], dtype=dtype))
        else:
            args.append(_make_tensor(inp))
    return args


def get_input_groups():
    results = []
    for case in INPUT_CASES:
        results.append(_make_scatter_kv_inputs(case))


    return results

def get_init_inputs():
    return []
