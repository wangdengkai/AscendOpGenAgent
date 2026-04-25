## 功能说明
- 算子功能：mhc_post基于一系列计算对mHC架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_j$进行ResMapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。

- 计算公式：
  $$
  x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}
  $$

```python
class Model(nn.Module):
    """
    Manifold Constrained Hyper Connection Post:
    x_{l+1} = (H_res)^T * x_l + h_out (x) H_post
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        """
        mHC Post forward.

        y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
            + sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)

        All bf16/fp16 inputs are cast to float32 for computation;
        result is cast back to original dtype.

        Args:
            x:      (..., n, D), bf16/fp16
            h_res:  (..., n, n), float32
            h_out:  (..., D), bf16/fp16
            h_post: (..., n), float32

        Returns:
            y: (..., n, D)
        """
        orig_dtype = x.dtype
        x_f = x.float()
        h_out_f = h_out.float()

        y = h_post.unsqueeze(-1) * h_out_f.unsqueeze(-2) + torch.sum(
            h_res.unsqueeze(-1) * x_f.unsqueeze(-2), dim=-3
        )
        return y.to(orig_dtype)
```
