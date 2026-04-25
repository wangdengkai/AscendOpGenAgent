"""Golden for L1/1 AdvanceStepV2 — direct wrapper around `torch_npu.npu_advance_step_flashattn`.

Schema (runtime-confirmed via `torch.ops.npu.npu_advance_step_flashattn.default._schema`):
    npu_advance_step_flashattn(
        Tensor(a!) input_tokens, Tensor sampled_token_ids,
        Tensor(b!) input_positions, Tensor(c!) seq_lens,
        Tensor(d!) slot_mapping, Tensor block_tables,
        int num_seqs, int num_queries, int block_size, *,
        Tensor? spec_token=None, Tensor? accepted_num=None
    ) -> ()

The op is IN-PLACE on input_tokens, input_positions, seq_lens, slot_mapping.
Reference returns 4-tuple [input_tokens, input_positions, seq_lens, slot_mapping].
Golden clones all mutable inputs, calls the op, then returns the same 4-tuple.

Cases come from neighboring `1_AdvanceStepV2.json` (one JSON record per line).
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
        input_tokens: torch.Tensor,
        sampled_tokens: torch.Tensor,
        input_positions: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        spec_tokens: torch.Tensor,
        accepted_num: torch.Tensor,
        num_seqs: int,
        num_queries: int,
        block_size: int,
    ) -> List[torch.Tensor]:
        # Clone mutable inputs so the original tensors are not modified
        it = input_tokens.clone().npu()
        ip = input_positions.clone().npu()
        sl = seq_lens.clone().npu()
        sm = slot_mapping.clone().npu()
        st = sampled_tokens.npu()
        bt = block_table.npu()
        sp = spec_tokens.npu()
        an = accepted_num.npu()

        torch_npu.npu_advance_step_flashattn(
            it, st, ip, sl, sm, bt,
            num_seqs, num_queries, block_size,
            spec_token=sp, accepted_num=an,
        )
        return [it.cpu(), ip.cpu(), sl.cpu(), sm.cpu()]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "1_AdvanceStepV2.json")
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
