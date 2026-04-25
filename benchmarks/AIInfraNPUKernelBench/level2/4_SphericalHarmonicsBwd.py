"""Golden for L2/4 SphericalHarmonicsBwd — hand-written reference (no torch_npu wrapper).

Per OPERATOR_TORCH_NPU_MAPPING.md classification: meta_gauss_render not installed
in this environment, so golden mirrors prompt_reference.py exactly. If/when
meta_gauss_render._C is packaged, swap forward() to call the real kernel.
"""
import json as _json
import os as _os
from pathlib import Path as _Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

_DTYPE_ALIAS = {
    "bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
    "fp64": "float64",
}


class Model(nn.Module):
    """
    Spherical Harmonics Backward pass for 3DGS rendering.

    Computes gradients of direction vectors and SH coefficients
    given upstream gradients of colors.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        dirs: torch.Tensor,
        coeffs: torch.Tensor,
        v_colors: torch.Tensor,
        degree: int = 0,
    ) -> List[torch.Tensor]:
        """
        Args:
            dirs: (B, 3, N) direction vectors.
            coeffs: (B, num_sh, 3, N) SH coefficients.
            v_colors: (B, 3, N) upstream color gradients.
            degree: SH degree (0-4).

        Returns:
            List of [v_dirs, v_coeffs].
        """
        dirs = dirs.float()
        coeffs_f = coeffs.float()
        v_colors_f = v_colors.float()

        coeffs_f = coeffs_f.permute(0, 3, 1, 2).contiguous()
        dirs = dirs.permute(0, 2, 1).contiguous()
        v_colors_f = v_colors_f.permute(0, 2, 1).contiguous()

        v_coeffs = torch.zeros_like(coeffs_f)
        v_dirs = torch.zeros_like(dirs)

        c00 = 0.2820947917738781
        v_coeffs[..., 0, :3] = c00 * v_colors_f[..., :3]

        if degree == 0:
            v_coeffs = v_coeffs.permute(0, 2, 3, 1).contiguous()
            v_dirs = v_dirs.permute(0, 2, 1).contiguous()
            return [v_dirs, v_coeffs]

        inorm = torch.rsqrt((dirs ** 2).sum(-1, keepdim=True))
        dirs = F.normalize(dirs, p=2, dim=-1)
        x, y, z = dirs.unbind(-1)

        v_coeffs[..., 1, :3] = -0.48860251190292 * y[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 2, :3] = 0.48860251190292 * z[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 3, :3] = -0.48860251190292 * x[..., None] * v_colors_f[..., :3]

        v_x = (-0.48860251190292 * coeffs_f[..., 3, :3] * v_colors_f[..., :3]).sum(-1, keepdim=True)
        v_y = (-0.48860251190292 * coeffs_f[..., 1, :3] * v_colors_f[..., :3]).sum(-1, keepdim=True)
        v_z = (0.48860251190292 * coeffs_f[..., 2, :3] * v_colors_f[..., :3]).sum(-1, keepdim=True)

        if degree == 1:
            v_dir_n = torch.cat([v_x, v_y, v_z], dim=-1)
            v_d = (v_dir_n - torch.einsum('bni,bni->bn', v_dir_n, dirs)[..., None] * dirs) * inorm
            v_dirs = v_dirs + v_d
            v_coeffs = v_coeffs.permute(0, 2, 3, 1).contiguous()
            v_dirs = v_dirs.permute(0, 2, 1).contiguous()
            return [v_dirs, v_coeffs]

        z2 = z * z
        fTmp0B = -1.092548430592079 * z
        fC1 = x * x - y * y
        fS1 = 2.0 * x * y
        pSH6 = (0.9461746957575601 * z2 - 0.3153915652525201)
        pSH7 = fTmp0B * x
        pSH5 = fTmp0B * y
        pSH8 = 0.5462742152960395 * fC1
        pSH4 = 0.5462742152960395 * fS1

        v_coeffs[..., 4, :3] = pSH4[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 5, :3] = pSH5[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 6, :3] = pSH6[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 7, :3] = pSH7[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 8, :3] = pSH8[..., None] * v_colors_f[..., :3]

        fTmp0B_z = -1.092548430592079
        fC1_x = 2.0 * x
        fC1_y = -2.0 * y
        fS1_x = 2.0 * y
        fS1_y = 2.0 * x
        pSH6_z = 2.0 * 0.9461746957575601 * z
        pSH7_x = fTmp0B
        pSH7_z = fTmp0B_z * x
        pSH5_y = fTmp0B
        pSH5_z = fTmp0B_z * y
        pSH8_x = 0.5462742152960395 * fC1_x
        pSH8_y = 0.5462742152960395 * fC1_y
        pSH4_x = 0.5462742152960395 * fS1_x
        pSH4_y = 0.5462742152960395 * fS1_y

        v_x = v_x + (v_colors_f * (pSH4_x[..., None] * coeffs_f[..., 4, :3] + pSH8_x[..., None] * coeffs_f[..., 8, :3] +
                    pSH7_x[..., None] * coeffs_f[..., 7, :3])).sum(-1, keepdim=True)
        v_y = v_y + (v_colors_f * (pSH4_y[..., None] * coeffs_f[..., 4, :3] + pSH8_y[..., None] * coeffs_f[..., 8, :3] +
                    pSH5_y[..., None] * coeffs_f[..., 5, :3])).sum(-1, keepdim=True)
        v_z = v_z + (v_colors_f * (pSH6_z[..., None] * coeffs_f[..., 6, :3] + pSH7_z[..., None] * coeffs_f[..., 7, :3] +
                    pSH5_z[..., None] * coeffs_f[..., 5, :3])).sum(-1, keepdim=True)

        if degree < 3:
            v_dir_n = torch.cat([v_x, v_y, v_z], dim=-1)
            v_d = (v_dir_n - torch.einsum('bni,bni->bn', v_dir_n, dirs)[..., None] * dirs) * inorm
            v_dirs = v_dirs + v_d
            v_coeffs = v_coeffs.permute(0, 2, 3, 1).contiguous()
            v_dirs = v_dirs.permute(0, 2, 1).contiguous()
            return [v_dirs, v_coeffs]

        fTmp0C = -2.285228997322329 * z2 + 0.4570457994644658
        fTmp1B = 1.445305721320277 * z
        fC2 = x * fC1 - y * fS1
        fS2 = x * fS1 + y * fC1
        pSH12 = z * (1.865881662950577 * z2 - 1.119528997770346)
        pSH13 = fTmp0C * x
        pSH11 = fTmp0C * y
        pSH14 = fTmp1B * fC1
        pSH10 = fTmp1B * fS1
        pSH15 = -0.5900435899266435 * fC2
        pSH9 = -0.5900435899266435 * fS2

        v_coeffs[..., 9, :3] = pSH9[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 10, :3] = pSH10[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 11, :3] = pSH11[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 12, :3] = pSH12[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 13, :3] = pSH13[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 14, :3] = pSH14[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 15, :3] = pSH15[..., None] * v_colors_f[..., :3]

        fTmp0C_z = -2.285228997322329 * 2.0 * z
        fTmp1B_z = 1.445305721320277
        fC2_x = fC1 + x * fC1_x - y * fS1_x
        fC2_y = x * fC1_y - fS1 - y * fS1_y
        fS2_x = fS1 + x * fS1_x + y * fC1_x
        fS2_y = x * fS1_y + fC1 + y * fC1_y
        pSH12_z = 3.0 * 1.865881662950577 * z2 - 1.119528997770346
        pSH13_x = fTmp0C
        pSH13_z = fTmp0C_z * x
        pSH11_y = fTmp0C
        pSH11_z = fTmp0C_z * y
        pSH14_x = fTmp1B * fC1_x
        pSH14_y = fTmp1B * fC1_y
        pSH14_z = fTmp1B_z * fC1
        pSH10_x = fTmp1B * fS1_x
        pSH10_y = fTmp1B * fS1_y
        pSH10_z = fTmp1B_z * fS1
        pSH15_x = -0.5900435899266435 * fC2_x
        pSH15_y = -0.5900435899266435 * fC2_y
        pSH9_x = -0.5900435899266435 * fS2_x
        pSH9_y = -0.5900435899266435 * fS2_y

        v_x = v_x + (v_colors_f *
                    (pSH9_x[..., None] * coeffs_f[..., 9, :3] + pSH15_x[..., None] * coeffs_f[..., 15, :3] +
                        pSH10_x[..., None] * coeffs_f[..., 10, :3] + pSH14_x[..., None] * coeffs_f[..., 14, :3] +
                        pSH13_x[..., None] * coeffs_f[..., 13, :3])).sum(dim=-1, keepdim=True)
        v_y = v_y + (v_colors_f *
                    (pSH9_y[..., None] * coeffs_f[..., 9, :3] + pSH15_y[..., None] * coeffs_f[..., 15, :3] +
                        pSH10_y[..., None] * coeffs_f[..., 10, :3] + pSH14_y[..., None] * coeffs_f[..., 14, :3] +
                        pSH11_y[..., None] * coeffs_f[..., 11, :3])).sum(dim=-1, keepdim=True)
        v_z = v_z + (v_colors_f *
                    (pSH12_z[..., None] * coeffs_f[..., 12, :3] + pSH13_z[..., None] * coeffs_f[..., 13, :3] +
                    pSH11_z[..., None] * coeffs_f[..., 11, :3] + pSH14_z[..., None] * coeffs_f[..., 14, :3] +
                    pSH10_z[..., None] * coeffs_f[..., 10, :3])).sum(dim=-1, keepdim=True)

        if degree == 3:
            v_dir_n = torch.cat([v_x, v_y, v_z], dim=-1)
            v_d = (v_dir_n - torch.einsum('bni,bni->bn', v_dir_n, dirs)[..., None] * dirs) * inorm
            v_dirs = v_dirs + v_d
            v_coeffs = v_coeffs.permute(0, 2, 3, 1).contiguous()
            v_dirs = v_dirs.permute(0, 2, 1).contiguous()
            return [v_dirs, v_coeffs]

        fTmp0D = z * (-4.683325804901025 * z2 + 2.007139630671868)
        fTmp1C = 3.31161143515146 * z2 - 0.47308734787878
        fTmp2B = -1.770130769779931 * z
        fC3 = x * fC2 - y * fS2
        fS3 = x * fS2 + y * fC2
        pSH20 = (1.984313483298443 * z * pSH12 + -1.006230589874905 * pSH6)
        pSH21 = fTmp0D * x
        pSH19 = fTmp0D * y
        pSH22 = fTmp1C * fC1
        pSH18 = fTmp1C * fS1
        pSH23 = fTmp2B * fC2
        pSH17 = fTmp2B * fS2
        pSH24 = 0.6258357354491763 * fC3
        pSH16 = 0.6258357354491763 * fS3

        v_coeffs[..., 16, :3] = pSH16[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 17, :3] = pSH17[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 18, :3] = pSH18[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 19, :3] = pSH19[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 20, :3] = pSH20[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 21, :3] = pSH21[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 22, :3] = pSH22[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 23, :3] = pSH23[..., None] * v_colors_f[..., :3]
        v_coeffs[..., 24, :3] = pSH24[..., None] * v_colors_f[..., :3]

        fTmp0D_z = 3.0 * -4.683325804901025 * z2 + 2.007139630671868
        fTmp1C_z = 2.0 * 3.31161143515146 * z
        fTmp2B_z = -1.770130769779931
        fC3_x = fC2 + x * fC2_x - y * fS2_x
        fC3_y = x * fC2_y - fS2 - y * fS2_y
        fS3_x = fS2 + y * fC2_x + x * fS2_x
        fS3_y = x * fS2_y + fC2 + y * fC2_y
        pSH20_z = 1.984313483298443 * (pSH12 + z * pSH12_z) + (-1.006230589874905 * pSH6_z)
        pSH21_x = fTmp0D
        pSH21_z = fTmp0D_z * x
        pSH19_y = fTmp0D
        pSH19_z = fTmp0D_z * y
        pSH22_x = fTmp1C * fC1_x
        pSH22_y = fTmp1C * fC1_y
        pSH22_z = fTmp1C_z * fC1
        pSH18_x = fTmp1C * fS1_x
        pSH18_y = fTmp1C * fS1_y
        pSH18_z = fTmp1C_z * fS1
        pSH23_x = fTmp2B * fC2_x
        pSH23_y = fTmp2B * fC2_y
        pSH23_z = fTmp2B_z * fC2
        pSH17_x = fTmp2B * fS2_x
        pSH17_y = fTmp2B * fS2_y
        pSH17_z = fTmp2B_z * fS2
        pSH24_x = 0.6258357354491763 * fC3_x
        pSH24_y = 0.6258357354491763 * fC3_y
        pSH16_x = 0.6258357354491763 * fS3_x
        pSH16_y = 0.6258357354491763 * fS3_y

        v_x = v_x + (v_colors_f *
                (pSH16_x[..., None] * coeffs_f[..., 16, :3] + pSH24_x[..., None] * coeffs_f[..., 24, :3] +
                pSH17_x[..., None] * coeffs_f[..., 17, :3] + pSH23_x[..., None] * coeffs_f[..., 23, :3] +
                pSH18_x[..., None] * coeffs_f[..., 18, :3] + pSH22_x[..., None] * coeffs_f[..., 22, :3] +
                pSH21_x[..., None] * coeffs_f[..., 21, :3])).sum(dim=-1, keepdim=True)
        v_y = v_y + (v_colors_f *
                (pSH16_y[..., None] * coeffs_f[..., 16, :3] + pSH24_y[..., None] * coeffs_f[..., 24, :3] +
                pSH17_y[..., None] * coeffs_f[..., 17, :3] + pSH23_y[..., None] * coeffs_f[..., 23, :3] +
                pSH18_y[..., None] * coeffs_f[..., 18, :3] + pSH22_y[..., None] * coeffs_f[..., 22, :3] +
                pSH19_y[..., None] * coeffs_f[..., 19, :3])).sum(dim=-1, keepdim=True)
        v_z = v_z + (v_colors_f *
                (pSH20_z[..., None] * coeffs_f[..., 20, :3] + pSH21_z[..., None] * coeffs_f[..., 21, :3] +
                pSH19_z[..., None] * coeffs_f[..., 19, :3] + pSH22_z[..., None] * coeffs_f[..., 22, :3] +
                pSH18_z[..., None] * coeffs_f[..., 18, :3] + pSH23_z[..., None] * coeffs_f[..., 23, :3] +
                pSH17_z[..., None] * coeffs_f[..., 17, :3])).sum(dim=-1, keepdim=True)

        v_dir_n = torch.cat([v_x, v_y, v_z], dim=-1)
        v_d = (v_dir_n - torch.einsum('bni,bni->bn', v_dir_n, dirs)[..., None] * dirs) * inorm
        v_dirs = v_dirs + v_d

        v_coeffs = v_coeffs.permute(0, 2, 3, 1).contiguous()
        v_dirs = v_dirs.permute(0, 2, 1).contiguous()

        return [v_dirs, v_coeffs]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "4_SphericalHarmonicsBwd.json")
INPUT_CASES = _load_jsonl_cases(_JSONL_PATH)
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


def _make_boxes(shape, dtype):
    leading_shape = tuple(shape[:-1])
    mins = torch.rand(*leading_shape, 2, dtype=torch.float32)
    sizes = torch.rand(*leading_shape, 2, dtype=torch.float32) + 0.05
    maxs = mins + sizes
    boxes = torch.cat([mins, maxs], dim=-1)
    return boxes.to(dtype=dtype)


def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    name = spec["name"]
    value_range = spec.get("range")

    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)

    if name in {"boxes", "bboxes", "gtboxes"} and shape and shape[-1] == 4 and dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }:
        return _make_boxes(shape, dtype)

    if value_range is not None:
        low, high = value_range
        if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            high_exclusive = high + 1
            return torch.randint(low, high_exclusive, tuple(shape), dtype=dtype)
        return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)

    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)

    return torch.randn(*shape, dtype=dtype)


def _make_tensor_list(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    return [torch.randn(*shape, dtype=dtype) for shape in spec["shapes"]]


def _make_arg(spec):
    spec_type = spec["type"]
    if spec_type == "tensor":
        return _make_tensor(spec)
    if spec_type == "tensor_list":
        return _make_tensor_list(spec)
    if spec_type == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {spec_type}")


def get_input_groups():
    return [[_make_arg(spec) for spec in case["inputs"]] for case in INPUT_CASES]


def get_init_inputs():
    return []
