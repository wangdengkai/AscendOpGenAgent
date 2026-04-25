## 功能说明

- 算子功能：mhc_post基于一系列计算对mHC架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_j$进行ResMapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。该算子实现前述过程的反向功能。

- 计算公式：
  $$
  grad\_x = H_{l}^{res} \times grad\_output\\
  grad\_h\_res = x_{l} \times {grad\_output}^{T}
  $$

  $$
  grad\_h\_out=({grad\_output} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_output} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

```python
class Model(nn.Module):
    """
    mHC Post backward: computes gradients for ManifoldConstrainedHyperConnectionPost.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Functional implementation of mHC Post backward.

        Computes four gradients via autograd of the forward:
          y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
              + sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)

        All bf16/fp16 inputs are cast to float32 for computation;
        grad_x and grad_h_out are cast back to original dtype.

        Args:
            grad_output: (..., n, D), bf16/fp16
            x:           (..., n, D), bf16/fp16
            h_res:       (..., n, n), float32
            h_out:       (..., D), bf16/fp16
            h_post:      (..., n), float32

        Returns:
            List of [grad_x, grad_h_res, grad_h_out, grad_h_post]
        """
        orig_dtype = grad_output.dtype

        grad_output_f = grad_output.float()
        x_f = x.float()
        h_out_f = h_out.float()

        # grad_x = h_res @ grad_output
        grad_x = torch.matmul(h_res, grad_output_f).to(orig_dtype)

        # grad_h_res = x @ grad_output^T
        grad_h_res = torch.matmul(x_f, grad_output_f.transpose(-1, -2))

        # grad_h_out = sum(grad_output * h_post.unsqueeze(-1), dim=-2)
        grad_h_out = torch.sum(grad_output_f * h_post.unsqueeze(-1), dim=-2).to(orig_dtype)

        # grad_h_post = sum(grad_output * h_out.unsqueeze(-2), dim=-1)
        grad_h_post = torch.sum(grad_output_f * h_out_f.unsqueeze(-2), dim=-1)

        return [grad_x, grad_h_res, grad_h_out, grad_h_post]
```
