"""Golden for L3/3 QkvRmsNormRopeCache -- direct wrapper around `torch_npu.npu_qkv_rms_norm_rope_cache`.

Schema (runtime-confirmed via `torch.ops.npu.npu_qkv_rms_norm_rope_cache.default._schema`):
    npu_qkv_rms_norm_rope_cache(
        Tensor qkv, Tensor q_gamma, Tensor k_gamma,
        Tensor cos, Tensor sin, Tensor index,
        Tensor(a!) q_out, Tensor(b!) k_cache, Tensor(c!) v_cache,
        int[4] qkv_size, int[3] head_nums, *,
        Tensor? k_scale=None, Tensor? v_scale=None,
        Tensor? k_offset=None, Tensor? v_offset=None,
        float epsilon=1e-06,
        str cache_mode="PA_NZ",
        bool is_output_qkv=False
    ) -> (Tensor, Tensor, Tensor)

Note: qkv_size and head_nums are positional (before kw-only *).
      q_out, k_cache, v_cache are in-place write-back buffers.
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
        qkv: torch.Tensor,
        gamma_q: torch.Tensor,
        gamma_k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        index: torch.Tensor,
        q_out: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        k_offset: Optional[torch.Tensor] = None,
        v_offset: Optional[torch.Tensor] = None,
        qkv_size: list = None,
        head_nums: list = None,
        epsilon: float = 1e-6,
        cache_mode: str = "PA_NZ",
        is_output_qkv: bool = False,
    ) -> List[torch.Tensor]:
        if qkv_size is None:
            qkv_size = [1, qkv.shape[0], 1, qkv.shape[1]]
        if head_nums is None:
            head_nums = [1, 0, 0]

        _to_npu = lambda t: t.npu() if isinstance(t, torch.Tensor) else t
        # Move buffers to NPU — kernel writes in-place and may return None
        q_out_npu = _to_npu(q_out)
        k_cache_npu = _to_npu(k_cache)
        v_cache_npu = _to_npu(v_cache)
        result = torch_npu.npu_qkv_rms_norm_rope_cache(
            _to_npu(qkv), _to_npu(gamma_q), _to_npu(gamma_k),
            _to_npu(cos), _to_npu(sin), _to_npu(index),
            q_out_npu, k_cache_npu, v_cache_npu,
            qkv_size, head_nums,
            k_scale=_to_npu(k_scale),
            v_scale=_to_npu(v_scale),
            k_offset=_to_npu(k_offset),
            v_offset=_to_npu(v_offset),
            epsilon=epsilon,
            cache_mode=cache_mode,
            is_output_qkv=is_output_qkv,
        )
        # Kernel modifies buffers in-place; result tuple may contain None.
        # Use result element if it is a tensor, else fall back to the buffer.
        def _pick(res_el, buf):
            if isinstance(res_el, torch.Tensor):
                return res_el.cpu()
            return buf.cpu() if isinstance(buf, torch.Tensor) else buf
        q_r = _pick(result[0], q_out_npu)
        # Reference returns the ORIGINAL k_cache/v_cache buffers (un-scattered).
        k_r = k_cache if isinstance(k_cache, torch.Tensor) else k_cache
        v_r = v_cache if isinstance(v_cache, torch.Tensor) else v_cache
        # Reference returns [q_out, k_cache, v_cache] or 6-element list
        if is_output_qkv:
            return [q_r, k_r, v_r, q_r, k_r, v_r]
        else:
            return [q_r, k_r, v_r]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "3_QkvRmsNormRopeCache.json")
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
        "qkv", "gamma_q", "gamma_k", "cos", "sin", "index",
        "q_out", "k_cache", "v_cache",
        "k_scale", "v_scale", "k_offset", "v_offset",
        "qkv_size", "head_nums", "epsilon", "cache_mode", "is_output_qkv",
    ]
    _PARAM_DEFAULTS = {
        "k_scale": None, "v_scale": None, "k_offset": None, "v_offset": None,
        "qkv_size": None, "head_nums": None, "epsilon": 1e-6,
        "cache_mode": "PA_NZ", "is_output_qkv": False,
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
