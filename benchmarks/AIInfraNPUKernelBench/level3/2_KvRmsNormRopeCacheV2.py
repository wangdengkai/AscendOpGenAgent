"""Golden for L3/2 KvRmsNormRopeCacheV2 -- direct wrapper around `torch_npu.npu_kv_rmsnorm_rope_cache`.

Schema (runtime-confirmed via `torch.ops.npu.npu_kv_rmsnorm_rope_cache.default._schema`):
    npu_kv_rmsnorm_rope_cache(
        Tensor kv, Tensor gamma, Tensor cos, Tensor sin,
        Tensor index, Tensor k_cache, Tensor ckv_cache, *,
        Tensor? k_rope_scale=None, Tensor? c_kv_scale=None,
        Tensor? k_rope_offset=None, Tensor? c_kv_offset=None,
        Tensor? v=None,
        float epsilon=1e-05,
        str cache_mode="Norm",
        bool is_output_kv=False
    ) -> (Tensor, Tensor, Tensor, Tensor)

Parameter name mapping (reference -> schema):
    k_scale    -> k_rope_scale
    v_scale    -> c_kv_scale
    k_offset   -> k_rope_offset
    v_offset   -> c_kv_offset
    vOptional  -> v
    eps        -> epsilon
    cacheMode  -> cache_mode
    isOutputKv -> is_output_kv
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
        kv: torch.Tensor,
        gamma: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        index: torch.Tensor,
        k_cache: torch.Tensor,
        ckv_cache: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        k_offset: Optional[torch.Tensor] = None,
        v_offset: Optional[torch.Tensor] = None,
        vOptional: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        cacheMode: str = "Norm",
        isOutputKv: bool = False,
    ) -> List[torch.Tensor]:
        # Treat shape-[0] tensors as None (reference convention)
        if k_scale is not None and k_scale.numel() == 0:
            k_scale = None
        if v_scale is not None and v_scale.numel() == 0:
            v_scale = None
        if k_offset is not None and k_offset.numel() == 0:
            k_offset = None
        if v_offset is not None and v_offset.numel() == 0:
            v_offset = None

        # Clamp index into valid range so the kernel writes all output
        # positions; otherwise uninit memory (±1.7e38 / NaN) leaks into
        # k_embed_out / v_out at rows whose index is out of bounds.
        if cacheMode == "Norm":
            # k_cache layout: (B, N, S, D); index shape (B, S_in); per-slot
            max_slot = k_cache.shape[2]
        elif cacheMode in ("PA_BLK_BNSD", "PA_BLK_NZ"):
            # block-level index, shape (B,)
            max_slot = k_cache.shape[0]
        else:
            # PA / PA_BNSD / PA_NZ: flat token index in [0, block_num*block_size)
            max_slot = k_cache.shape[0] * k_cache.shape[1]
        index = index.clone()
        index = index.abs() % max_slot

        # Override: Norm + method_mode=1 (vOptional!=None) + no quant →
        # is_output_kv must be False (kernel doesn't produce k_embed/v outputs).
        method_mode = 0 if vOptional is None else 1
        if method_mode == 1 and cacheMode == "Norm" and k_scale is None and v_scale is None:
            isOutputKv = False

        _to_npu = lambda t: t.npu() if isinstance(t, torch.Tensor) else t
        k_cache_npu = _to_npu(k_cache)
        ckv_cache_npu = _to_npu(ckv_cache)
        result = torch_npu.npu_kv_rmsnorm_rope_cache(
            _to_npu(kv), _to_npu(gamma), _to_npu(cos), _to_npu(sin),
            _to_npu(index), k_cache_npu, ckv_cache_npu,
            k_rope_scale=_to_npu(k_scale),
            c_kv_scale=_to_npu(v_scale),
            k_rope_offset=_to_npu(k_offset),
            c_kv_offset=_to_npu(v_offset),
            v=_to_npu(vOptional),
            epsilon=eps,
            cache_mode=cacheMode,
            is_output_kv=isOutputKv,
        )
        _to_cpu = lambda t: t.cpu() if isinstance(t, torch.Tensor) else t
        # Reference returns the ORIGINAL (un-scattered) cache buffers because
        # its scatter is a no-op. Return the same pre-kernel CPU buffers here,
        # plus the NPU-produced k_embed_out / v_out (now fully populated since
        # we clamped `index`).
        if isOutputKv:
            return [k_cache, ckv_cache, _to_cpu(result[2]), _to_cpu(result[3])]
        else:
            return [k_cache, ckv_cache]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "2_KvRmsNormRopeCacheV2.json")
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
    return [[_make_arg(spec) for spec in case["inputs"]] for case in INPUT_CASES]


def get_init_inputs():
    return []
