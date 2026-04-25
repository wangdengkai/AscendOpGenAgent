"""Golden for L2/5 FullyFusedProjectionBwd — hand-written reference (no torch_npu wrapper).

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
from typing import List, Optional, Tuple

_DTYPE_ALIAS = {
    "bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
    "fp64": "float64",
}
from torch import Tensor


class Model(nn.Module):
    """
    Fully Fused Projection Backward for 3DGS rendering.

    Computes gradients for Gaussian positions, quaternions, scales,
    rotation matrices, colors, and opacities.
    """

    def __init__(self):
        super(Model, self).__init__()

    def _inverse_vjp(self, Minv, v_Minv):
        return -Minv @ v_Minv @ Minv

    def _quat_to_rotmat(self, quats):
        quats = F.normalize(quats, p=2, dim=-1)
        w, x, y, z = torch.unbind(quats, dim=-1)
        R = torch.stack([
            1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2),
        ], dim=-1)
        return R.reshape(quats.shape[:-1] + (3, 3))

    def _quat_to_rotmat_vjp(self, quats, v_R):
        quats_n = F.normalize(quats, p=2, dim=-1)
        w, x, y, z = torch.unbind(quats_n, dim=-1)
        vR12_vR21 = v_R[..., 2, 1] - v_R[..., 1, 2]
        vR20_vR02 = v_R[..., 0, 2] - v_R[..., 2, 0]
        vR01_vR10 = v_R[..., 1, 0] - v_R[..., 0, 1]
        vR11_add_vR22 = v_R[..., 1, 1] + v_R[..., 2, 2]
        vR00_add_vR22 = v_R[..., 0, 0] + v_R[..., 2, 2]
        vR00_add_vR11 = v_R[..., 0, 0] + v_R[..., 1, 1]
        vR01_add_vR10 = v_R[..., 1, 0] + v_R[..., 0, 1]
        vR02_add_vR20 = v_R[..., 2, 0] + v_R[..., 0, 2]
        vR12_add_vR21 = v_R[..., 2, 1] + v_R[..., 1, 2]
        v_quat_n = torch.stack([
            2.0 * (x * vR12_vR21 + y * vR20_vR02 + z * vR01_vR10),
            2.0 * (-2 * x * vR11_add_vR22 + y * vR01_add_vR10 + z * vR02_add_vR20 + w * vR12_vR21),
            2.0 * (x * vR01_add_vR10 - 2.0 * y * vR00_add_vR22 + z * vR12_add_vR21 + w * vR20_vR02),
            2.0 * (x * vR02_add_vR20 + y * vR12_add_vR21 - 2.0 * z * vR00_add_vR11 + w * vR01_vR10)
        ], dim=-1)
        v_quat = (v_quat_n - torch.einsum('...ni,...ni->...n', v_quat_n, quats_n)[..., None] * quats_n) * torch.rsqrt((quats * quats).sum(-1, keepdim=True))
        return v_quat

    def _quat_scale_to_covar(self, quats, scales):
        R = self._quat_to_rotmat(quats)
        M = R * scales[..., None, :]
        covars = torch.einsum("...ij,...kj -> ...ik", M, M)
        return covars, R

    def _quat_scale_to_covar_vjp(self, quats, scales, R, v_covar):
        M = R * scales[..., None, :]
        v_M = (v_covar + v_covar.transpose(-2, -1)) @ M
        v_R = v_M * scales[..., None, :]
        v_quats = self._quat_to_rotmat_vjp(quats, v_R)
        v_scales = torch.einsum("...ij,...ij->...j", R, v_M)
        return v_quats, v_scales

    def _posW2C(self, R, t, pW):
        pC = torch.einsum("...cij,...nj->...cni", R, pW) + t[..., None, :]
        return pC

    def _posW2C_VJP(self, R, t, pW, v_pC):
        v_R = torch.einsum('...cni,...nj->...cij', v_pC, pW)
        v_t = torch.einsum('...cni->...ci', v_pC)
        v_pW = torch.einsum("...cji,...cnj->...ni", R, v_pC)
        return v_R, v_t, v_pW

    def _covarW2C(self, R, covarW):
        covars_c = torch.einsum("...cij,...njk,...clk->...cnil", R, covarW, R)
        return covars_c

    def _covarW2C_VJP(self, R, covarW, v_covarC, v_R):
        v_R = v_R + torch.einsum("...cnij,...cjk,...nlk->...cil", v_covarC, R, covarW) + torch.einsum("...cnji,...cjk,...nkl->...cil", v_covarC, R, covarW)
        v_covarW = torch.einsum("...cij,...cnik,...ckl->...njl", R, v_covarC, R)
        return v_R, v_covarW

    def _persp_proj_vjp(self, means, cov3d, Ks, width, height, v_cov2d, v_mean2d):
        batch_dims = means.shape[:-3]
        C, N = means.shape[-3:-1]
        tx, ty, tz = torch.unbind(means, dim=-1)
        tz2 = tz**2
        fx = Ks[..., 0, 0, None]
        fy = Ks[..., 1, 1, None]
        cx = Ks[..., 0, 2, None]
        cy = Ks[..., 1, 2, None]
        tan_fovx = 0.5 * width / fx
        tan_fovy = 0.5 * height / fy
        lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
        lim_x_neg = cx / fx + 0.3 * tan_fovx
        lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
        lim_y_neg = cy / fy + 0.3 * tan_fovy

        x_clipping_mask = (means[..., 0] / tz <= lim_x_pos) & (means[..., 0] / tz >= -lim_x_neg)
        y_clipping_mask = (means[..., 1] / tz <= lim_y_pos) & (means[..., 1] / tz >= -lim_y_neg)
        tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
        ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)
        O = torch.zeros(batch_dims + (C, N), device=means.device, dtype=means.dtype)

        J = torch.stack(
            [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
        ).reshape(batch_dims + (C, N, 2, 3))

        v_cov3d = J.transpose(-2, -1) @ v_cov2d @ J

        v_mean3d = torch.stack(
            [fx / tz * v_mean2d[..., 0],
             fy / tz * v_mean2d[..., 1],
             -(fx * means[..., 0] * v_mean2d[..., 0] + fy * means[..., 1] * v_mean2d[..., 1]) / tz2], dim=-1)

        tz3 = tz2 * tz
        v_J = v_cov2d @ J @ cov3d.transpose(-2, -1) + v_cov2d.transpose(-2, -1) @ J @ cov3d

        v_mean3d[..., 0] = v_mean3d[..., 0] - (fx / tz2 * v_J[..., 0, 2]) * x_clipping_mask
        v_mean3d[..., 2] = v_mean3d[..., 2] - (fx / tz3 * v_J[..., 0, 2] * tx) * (~x_clipping_mask)
        v_mean3d[..., 1] = v_mean3d[..., 1] - (fy / tz2 * v_J[..., 1, 2]) * y_clipping_mask
        v_mean3d[..., 2] = v_mean3d[..., 2] - (fy / tz3 * v_J[..., 1, 2] * ty) * (~y_clipping_mask)
        v_mean3d[..., 2] = v_mean3d[..., 2] - fx / tz2 * v_J[..., 0, 0] - fy / tz2 * v_J[..., 1, 1] + \
                2.0 * fx * tx / tz3 * v_J[..., 0, 2] + \
                2.0 * fy * ty / tz3 * v_J[..., 1, 2]

        return v_mean3d, v_cov3d, v_J

    def forward(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        conics: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        v_means2d: torch.Tensor,
        v_depths: torch.Tensor,
        v_conics: torch.Tensor,
        v_colors_culling: torch.Tensor,
        v_opacities_culling: torch.Tensor,
        filter_mask: torch.Tensor,
        width: int = 648,
        height: int = 420,
    ) -> List[torch.Tensor]:
        """
        Args:
            means: (B, 3, N)
            quats: (B, 4, N)
            scales: (B, 3, N)
            conics: (B, C, 3, N)
            viewmats: (B, C, 4, 4)
            Ks: (B, C, 3, 3)
            v_means2d: (B, C, 2, N)
            v_depths: (B, C, N)
            v_conics: (B, C, 3, N)
            v_colors_culling: (B, C, 3, N)
            v_opacities_culling: (B, C, N)
            filter_mask: (B, C, M) uint8 bitmask
            width, height: int attrs

        Returns:
            List of [v_means, v_quats, v_scales, v_R, v_colors, v_opacities]
        """
        means = means.float().permute(0, 2, 1).contiguous()
        quats = quats.float().permute(0, 2, 1).contiguous()
        scales = scales.float().permute(0, 2, 1).contiguous()
        conics = conics.float().permute(0, 1, 3, 2).contiguous()
        viewmats = viewmats.float()
        Ks = Ks.float()
        v_means2d_culling = v_means2d.float().permute(0, 1, 3, 2).contiguous()
        v_depths_culling = v_depths.float()
        v_conics_culling = v_conics.float().permute(0, 1, 3, 2).contiguous()
        v_colors_cull = v_colors_culling.float().permute(0, 1, 3, 2).contiguous()
        v_opacities_cull = v_opacities_culling.float()

        B, C, N = v_opacities_cull.shape
        filter_byte = filter_mask.byte()

        v_conics_out = torch.zeros_like(v_conics_culling)
        v_means2d_out = torch.zeros_like(v_means2d_culling)
        v_depths_out = torch.zeros_like(v_depths_culling)
        v_colors_out = torch.zeros_like(v_colors_cull)
        v_opacities_out = torch.zeros_like(v_opacities_cull)

        bit_mask = (1 << torch.arange(8, dtype=torch.uint8))
        filter_bool = (filter_byte.unsqueeze(-1).bitwise_and(bit_mask) != 0).reshape(B, C, -1)
        filter_bool = filter_bool[:, :, :N]

        for b in range(B):
            for c in range(C):
                cnt = filter_bool[b, c].sum()
                v_conics_out[b, c, filter_bool[b, c]] = v_conics_culling[b, c, :cnt]
                v_means2d_out[b, c, filter_bool[b, c]] = v_means2d_culling[b, c, :cnt]
                v_depths_out[b, c, filter_bool[b, c]] = v_depths_culling[b, c, :cnt]
                v_colors_out[b, c, filter_bool[b, c]] = v_colors_cull[b, c, :cnt]
                v_opacities_out[b, c, filter_bool[b, c]] = v_opacities_cull[b, c, :cnt]

        covar2d_inv = torch.stack([conics[..., 0], conics[..., 1], conics[..., 1], conics[..., 2]], dim=-1).reshape(conics.shape[:-1] + (2, 2))
        v_covar2d_inv = torch.stack([v_conics_out[..., 0], v_conics_out[..., 1] * 0.5, v_conics_out[..., 1] * 0.5, v_conics_out[..., 2]], dim=-1).reshape(conics.shape[:-1] + (2, 2))
        v_covar2d = self._inverse_vjp(covar2d_inv, v_covar2d_inv)

        R = viewmats[..., :3, :3]
        t = viewmats[..., :3, 3]
        covars, rotmat = self._quat_scale_to_covar(quats, scales)
        mean_c = self._posW2C(R, t, means)
        covar_c = self._covarW2C(R, covars)
        v_mean_c, v_covar_c, v_J = self._persp_proj_vjp(mean_c, covar_c, Ks, width, height, v_covar2d, v_means2d_out)

        v_mean_c[..., 2] = v_mean_c[..., 2] + v_depths_out

        v_R, v_t, v_pW = self._posW2C_VJP(R, t, means, v_mean_c)
        v_R, v_covar = self._covarW2C_VJP(R, covars, v_covar_c, v_R)

        v_quats, v_scales = self._quat_scale_to_covar_vjp(quats, scales, rotmat, v_covar)

        v_colors_final = v_colors_out.sum(dim=1).permute(0, 2, 1).contiguous()
        v_opacities_final = v_opacities_out.sum(dim=1)

        return [v_pW, v_quats, v_scales, v_R, v_colors_final, v_opacities_final]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "5_FullyFusedProjectionBwd.json")
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
