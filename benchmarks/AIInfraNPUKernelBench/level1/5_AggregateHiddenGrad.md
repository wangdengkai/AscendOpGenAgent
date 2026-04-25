## 功能说明

- 接口功能：对hidden层的token之间进行一维分组卷积操作的反向梯度计算。

- 计算公式：
  假定卷积输入input、卷积输出的梯度grad_output和卷积输入的梯度grad_input的shape是[S, B, H]，weight的shape是[W, H]，i和j分别表示S/B轴的索引，k为卷积窗口W内的索引，那么计算将被表示为：

  $$
  grad\_output\_masked[i,j] = mask[j,i] * grad\_output[i,j]
  $$

  $$
  grad\_input[i,j] = \sum_{k=0}^{W-1} grad\_output\_masked[i+k,j] * weight[W-1-k]
  $$

  $$
  grad\_weight[k] = \sum_{j=0}^{B-1}\sum_{i=0}^{S-1} grad\_output\_masked[i+W-1-k,j] * input[i,j]
  $$

  其中，无效位置的padding为0填充；当前W仅支持3。

```python
class Model(nn.Module):
    """
    AggregateHiddenGrad: backward gradient computation for AggregateHidden.
    Computes grad_input and grad_weight from grad_output, input, weight, mask.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Backward gradient computation for AggregateHidden.

        grad_output_masked[i,j] = mask[j,i] * grad_output[i,j]
        grad_input[i,j] = sum_{k=0}^{W-1} grad_output_masked[i+k,j] * weight[W-1-k]
        grad_weight[k] = sum_j sum_i grad_output_masked[i+W-1-k,j] * input[i,j]

        Args:
            grad_output: (S, B, H), bf16/fp16/fp32
            input:       (S, B, H), bf16/fp16/fp32
            weight:      (W, H), bf16/fp16/fp32
            mask:        (B, S), bool

        Returns:
            List of [grad_input, grad_weight]
        """
        orig_dtype = input.dtype
        S, B, H = input.shape
        sliding_window = weight.shape[0]

        # Use autograd approach: build forward, then backward
        # detach to ensure leaf tensor for autograd
        x_f = input.detach().float().requires_grad_(True)

        merge_conv = torch.nn.Conv1d(H, H, sliding_window, groups=H, bias=False)
        merge_conv.weight.data = weight.unsqueeze(1).transpose(0, 2).float()
        merge_conv.weight.retain_grad()

        # Forward pass
        conv_input = torch.cat(
            [torch.zeros((B, H, sliding_window - 1), dtype=torch.float32),
             x_f.permute(1, 2, 0)],
            dim=-1
        )
        conv_output = merge_conv(conv_input)
        bsh_output = conv_output.permute(0, 2, 1)  # (B, S, H)
        if mask is not None:
            bsh_output = bsh_output.clone()
            bsh_output[~mask] = 0
        output = bsh_output.view(B, S, H).transpose(1, 0)  # (S, B, H)

        # Backward
        grad_f = grad_output.float()
        loss = torch.sum(grad_f * output)
        loss.backward()

        grad_input = x_f.grad.to(orig_dtype)
        grad_weight = merge_conv.weight.grad.transpose(0, 2).squeeze(1).to(orig_dtype)

        return [grad_input, grad_weight]
```
