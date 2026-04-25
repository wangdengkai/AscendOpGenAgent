## 功能说明
- 3DGS渲染高斯球过滤接口，计算每个相机视角下有效高斯球的位置并对相关输入进行过滤。

```python
class Model(nn.Module):
    """
    3DGS Gaussian Filter: filters valid Gaussians per camera view.

    Computes valid Gaussian positions based on depth, determinant, and screen bounds,
    then culls and compacts the inputs accordingly.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        det: torch.Tensor,
        opacities: torch.Tensor,
        means2d: torch.Tensor,
        depths: torch.Tensor,
        radius: torch.Tensor,
        conics: torch.Tensor,
        covars2d: torch.Tensor,
        near_plane: float = 0.0,
        far_plane: float = 2.0,
        width: int = 200,
        height: int = 600,
    ) -> List[torch.Tensor]:
        """
        Args:
            means: (B, 3, N) float32
            colors: (B, 3, N) float32
            det: (B, C, N) float32
            opacities: (B, N) float32
            means2d: (B, C, 2, N) float32
            depths: (B, C, N) float32
            radius: (B, C, 2, N) float32
            conics: (B, C, 3, N) float32
            covars2d: (B, C, 3, N) float32
            near_plane, far_plane, width, height: scalar attrs

        Returns:
            List of culled tensors + filter_uint8 + cnt.
        """
        B, C, N = det.shape

        opacities_exp = opacities.unsqueeze(1).expand(B, C, N)

        means_t = means.permute(0, 2, 1).contiguous()
        means2d_t = means2d.permute(0, 1, 3, 2).contiguous()
        radius_t = radius.permute(0, 1, 3, 2).contiguous()
        radius_out = radius.permute(0, 1, 3, 2).contiguous()
        conics_t = conics.permute(0, 1, 3, 2).contiguous()
        colors_t = colors.permute(0, 2, 1).contiguous()
        covars2d_t = covars2d.permute(0, 1, 3, 2).contiguous()

        valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
        radius_t[~valid] = 0.0
        inside = (
            (means2d_t[..., 0] + radius_t[..., 0] > 0)
            & (means2d_t[..., 0] - radius_t[..., 0] < width)
            & (means2d_t[..., 1] + radius_t[..., 1] > 0)
            & (means2d_t[..., 1] - radius_t[..., 1] < height)
        )
        radius_t[~inside] = 0.0
        filter_mask = torch.logical_and(inside, valid)

        means_culling = torch.ones_like(conics_t)
        radius_culling = torch.ones_like(radius_t)
        means2d_culling = torch.ones_like(means2d_t)
        depths_culling = torch.ones_like(depths)
        opacities_culling = torch.ones_like(depths)
        conics_culling = torch.ones_like(conics_t)
        colors_culling = torch.ones_like(conics_t)
        covars2d_culling = torch.ones_like(covars2d_t)

        for b in range(B):
            for c in range(C):
                cnt = filter_mask[b, c].sum()
                radius_culling[b, c, :cnt] = radius_out[b, c, filter_mask[b, c]]
                means_culling[b, c, :cnt] = means_t[b, filter_mask[b, c]]
                means2d_culling[b, c, :cnt] = means2d_t[b, c, filter_mask[b, c]]
                depths_culling[b, c, :cnt] = depths[b, c, filter_mask[b, c]]
                conics_culling[b, c, :cnt] = conics_t[b, c, filter_mask[b, c]]
                colors_culling[b, c, :cnt] = colors_t[b, filter_mask[b, c]]
                covars2d_culling[b, c, :cnt] = covars2d_t[b, c, filter_mask[b, c]]
                opacities_culling[b, c, :cnt] = opacities_exp[b, c, filter_mask[b, c]]

        cnt_out = filter_mask.sum(-1)

        filter_bool = filter_mask.bool()
        remainder = N % 8
        if remainder != 0:
            pad_size = 8 - remainder
            filter_bool = F.pad(filter_bool, (0, pad_size), mode='constant', value=False)
        M = (N + 7) // 8
        filter_reshaped = filter_bool.reshape(B, C, M, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=filter_mask.device)
        filter_uint8 = (filter_reshaped.to(torch.uint8) * powers).sum(dim=-1, dtype=torch.uint8)

        means_culling = means_culling.permute(0, 1, 3, 2).contiguous()
        radius_culling = radius_culling.permute(0, 1, 3, 2).contiguous()
        means2d_culling = means2d_culling.permute(0, 1, 3, 2).contiguous()
        conics_culling = conics_culling.permute(0, 1, 3, 2).contiguous()
        colors_culling = colors_culling.permute(0, 1, 3, 2).contiguous()
        covars2d_culling = covars2d_culling.permute(0, 1, 3, 2).contiguous()

        return [means_culling, colors_culling, means2d_culling,
                depths_culling, radius_culling, covars2d_culling,
                conics_culling, opacities_culling, filter_uint8, cnt_out.to(torch.int32)]
```
