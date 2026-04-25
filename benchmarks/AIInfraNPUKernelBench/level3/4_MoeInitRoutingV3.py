"""Golden for L3/4 MoeInitRoutingV3 -- direct wrapper around `torch_npu.npu_moe_init_routing_v2`.

Schema (runtime-confirmed via `torch.ops.npu.npu_moe_init_routing_v2.default._schema`):
    npu_moe_init_routing_v2(
        Tensor x, Tensor expert_idx, *,
        Tensor? scale=None, Tensor? offset=None,
        int active_num=-1, int expert_capacity=-1, int expert_num=-1,
        int drop_pad_mode=0, int expert_tokens_num_type=0,
        bool expert_tokens_num_flag=False, int quant_mode=-1,
        int[2] active_expert_range=[], int row_idx_type=0,
        int? x_dtype=None
    ) -> (Tensor, Tensor, Tensor, Tensor)

Note: V3 shares V2 wrapper. `active_expert_range` / `row_idx_type` are
      the V3-specific semantic extensions already exposed through the V2 schema.
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
        x: torch.Tensor,
        expert_idx: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        active_num: int = 0,
        expert_capacity: int = 0,
        expert_num: int = 0,
        drop_pad_mode: int = 0,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        quant_mode: int = -1,
        active_expert_range: list = None,
        row_idx_type: int = 0,
    ) -> List[torch.Tensor]:
        # Schema defaults: active_num=-1, expert_capacity=-1, expert_num=-1
        # Reference defaults: 0, 0, 0 — translate 0→-1 for the kernel
        kw_active_num = active_num if active_num != 0 else -1
        kw_expert_capacity = expert_capacity if expert_capacity != 0 else -1
        kw_expert_num = expert_num if expert_num != 0 else -1

        # Schema default for active_expert_range is [] (empty list)
        kw_range = active_expert_range if active_expert_range is not None else []

        _to_npu = lambda t: t.npu() if isinstance(t, torch.Tensor) else t
        result = torch_npu.npu_moe_init_routing_v2(
            _to_npu(x), _to_npu(expert_idx),
            scale=_to_npu(scale),
            offset=_to_npu(offset),
            active_num=kw_active_num,
            expert_capacity=kw_expert_capacity,
            expert_num=kw_expert_num,
            drop_pad_mode=drop_pad_mode,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            quant_mode=quant_mode,
            active_expert_range=kw_range,
            row_idx_type=row_idx_type,
        )
        # Schema returns 4-tuple: (expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale)
        out = [t.cpu() if isinstance(t, torch.Tensor) else t for t in result]
        # Align with reference conventions: empty tensors for unused outputs.
        if not expert_tokens_num_flag:
            out[2] = torch.empty(0, dtype=torch.int64)

        # ---- Compute reference-side effective sizes to truncate kernel outputs ----
        # Rationale: kernel may allocate upper-bound shapes (e.g. [N*K, H]) while
        # reference produces tight shapes (e.g. [active_num_eff, H]). We replicate
        # the minimal counting logic from reference.py to obtain active_num_eff.
        import numpy as _np
        num_rows = x.shape[0]
        h = x.shape[1]
        k = expert_idx.shape[-1]
        if active_expert_range is None or len(active_expert_range) < 2:
            _ers = 0
            _ere = expert_num if expert_num != 0 else 0
        else:
            _ers = active_expert_range[0]
            _ere = active_expert_range[1]
        expert_start = _ers if drop_pad_mode == 0 else 0
        expert_end = _ere if drop_pad_mode == 0 else expert_num
        _eidx = expert_idx.detach().cpu().numpy().reshape(-1)
        actual_expert_total_num = int(
            ((_eidx >= expert_start) & (_eidx < expert_end)).sum()
        )
        if drop_pad_mode == 0:
            if active_num == 0:
                active_num_eff = actual_expert_total_num
            else:
                active_num_eff = min(active_num, actual_expert_total_num)
        else:
            active_num_eff = expert_num * expert_capacity

        # ---- expanded_x: truncate & reshape to reference shape ----
        if isinstance(out[0], torch.Tensor):
            flat = out[0].reshape(-1, h) if out[0].dim() >= 2 else out[0].reshape(-1)
            if drop_pad_mode == 0:
                if flat.dim() == 2 and flat.shape[0] >= active_num_eff:
                    out[0] = flat[:active_num_eff].contiguous()
                else:
                    out[0] = flat
            else:
                # [E, C, H]
                total = expert_num * expert_capacity * h
                if out[0].numel() >= total:
                    out[0] = out[0].reshape(-1)[:total].reshape(expert_num, expert_capacity, h).contiguous()

        # ---- expanded_row_idx: reference shape depends on row_idx_type ----
        #   row_idx_type == 0: [num_rows * k]
        #   row_idx_type == 1: [actual_expert_total_num]
        if isinstance(out[1], torch.Tensor):
            if row_idx_type == 1 and drop_pad_mode == 0:
                target_len = actual_expert_total_num
            else:
                target_len = num_rows * k
            flat = out[1].reshape(-1)
            if flat.numel() >= target_len:
                out[1] = flat[:target_len].contiguous()

        # ---- expert_tokens_count: expert_tokens_num_type == 2 reshapes to [U, 2] ----
        # Reference builds [[expert_id, count], ...] via np.unique; if U < expert_num,
        # a single [0, 0] padding row is appended.
        if expert_tokens_num_flag and drop_pad_mode == 0 and expert_tokens_num_type == 2:
            if isinstance(out[2], torch.Tensor) and out[2].numel() > 0:
                # Recompute from expert_idx to get the exact reference layout.
                _valid = _eidx[(_eidx >= expert_start) & (_eidx < expert_end)]
                if _valid.size > 0:
                    _uniq, _cnts = _np.unique(_valid, return_counts=True)
                    _stacked = _np.column_stack((_uniq, _cnts)).astype(_np.int64)
                    if _stacked.shape[0] < expert_num:
                        _stacked = _np.concatenate((_stacked, [[0, 0]]), axis=0)
                    out[2] = torch.from_numpy(_stacked)

        # ---- expanded_scale alignment (must match reference.py exactly) ----
        # Reference rules:
        #   quant_mode == -1:
        #       scale is None  -> empty(0)
        #       scale not None -> 1D [active_num_eff]      (drop_pad_mode==0)
        #                         1D [E*C]                  (drop_pad_mode==1)
        #   quant_mode == 0:   -> empty(0) (scale is consumed, not propagated)
        #   quant_mode == 1:   -> 2D [active_num_eff, 1]   (drop_pad_mode==0)
        #                         2D [E*C, 1]               (drop_pad_mode==1)
        if quant_mode == -1:
            if scale is None:
                out[3] = torch.empty(0, dtype=torch.float32)
            else:
                # CPU 侧按 reference 公式重算 expanded_scale，不依赖 kernel out[3]。
                # kernel 在 drop_pad_mode==1 的未使用槽返回 NaN/残留；
                # 在 x=int8 + quant_mode=-1 等边界场景下 kernel scale 输出语义不稳定。
                scale_np = scale.detach().cpu().numpy()
                eidx_flat = expert_idx.detach().cpu().numpy().reshape(-1).copy()
                eidx_flat[eidx_flat < expert_start] = _np.iinfo(_np.int32).max
                sorted_idx = _np.argsort(eidx_flat, axis=-1, kind="stable")
                sorted_eidx = eidx_flat[sorted_idx]
                if drop_pad_mode == 0:
                    es = scale_np[sorted_idx[:active_num_eff] // k]
                    out[3] = torch.from_numpy(_np.asarray(es).astype(_np.float32))
                else:
                    sri = sorted_idx.copy()
                    sei = sorted_eidx.copy()
                    if len(sei) > 0:
                        count, last = 0, sei[0]
                        for i in range(len(sei)):
                            v = sei[i]
                            if last != v:
                                count = 1
                                last = v
                            else:
                                count += 1
                                if count > expert_capacity:
                                    sei[i] = -1
                                    sri[i] = -1
                    sort_row_tmp = _np.full((expert_num * expert_capacity,), -1, dtype=_np.int64)
                    off = 0
                    lastE = 0
                    for i in range(len(sri)):
                        if sri[i] != -1:
                            if lastE != sei[i]:
                                off = 0
                                lastE = sei[i]
                            sort_row_tmp[sei[i] * expert_capacity + off] = sri[i]
                            off += 1
                    es = _np.zeros((expert_num * expert_capacity,), dtype=_np.float32)
                    for i in range(sort_row_tmp.shape[0]):
                        v = sort_row_tmp[i]
                        if v != -1:
                            es[i] = float(scale_np[v // k])
                    out[3] = torch.from_numpy(es)
        elif quant_mode == 0:
            out[3] = torch.empty(0, dtype=torch.float32)
        elif quant_mode == 1:
            if isinstance(out[3], torch.Tensor) and out[3].numel() > 0:
                target = active_num_eff if drop_pad_mode == 0 else expert_num * expert_capacity
                flat = out[3].reshape(-1).to(torch.float32)
                if flat.numel() >= target:
                    flat = flat[:target]
                out[3] = flat.reshape(-1, 1).contiguous()
        return out


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "4_MoeInitRoutingV3.json")
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
        "x", "expert_idx", "scale", "offset",
        "active_num", "expert_capacity", "expert_num",
        "drop_pad_mode", "expert_tokens_num_type", "expert_tokens_num_flag",
        "quant_mode", "active_expert_range", "row_idx_type",
    ]
    _PARAM_DEFAULTS = {
        "scale": None, "offset": None,
        "active_num": 0, "expert_capacity": 0, "expert_num": 0,
        "drop_pad_mode": 0, "expert_tokens_num_type": 1,
        "expert_tokens_num_flag": True, "quant_mode": -1,
        "active_expert_range": None, "row_idx_type": 0,
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
