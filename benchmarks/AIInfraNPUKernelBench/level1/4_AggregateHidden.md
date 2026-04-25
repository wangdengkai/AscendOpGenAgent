## 功能说明

- 算子功能：对hidden层的token之间进行一维分组卷积操作。

- 计算公式：
  假设输入input和输出output的shape是[S, B, H]，卷积权重weight的shape是[W, H]，i和j分别表示S和B轴的索引，那么输出将被表示为：

  $$
  output[i,j] = mask[j,i] * \sum_{k=0}^{W-1} input[i-k,j] * weight[W-1-k]
  $$

  其中，无效位置的padding为0填充；当前W仅支持3。

```python
class Model(nn.Module):
    """
    AggregateHidden: 1D grouped convolution on hidden tokens with mask,
    plus backward gradients for input and weight.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grad_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Computes forward convolution output and backward gradients.

        output[i,j] = mask[j,i] * sum_{k=0}^{W-1} input[i-k,j] * weight[W-1-k]

        Uses Conv1d with groups=H for the forward pass, then autograd
        for grad_input and grad_weight.

        Args:
            grad_out: (S, B, H), bf16/fp16/fp32 - gradient of output
            input:    (S, B, H), bf16/fp16/fp32
            weight:   (W, H), bf16/fp16/fp32 - W is typically 3
            mask:     (B, S), bool

        Returns:
            List of [output, grad_input, grad_weight]
        """
        orig_dtype = input.dtype
        S, B, H = input.shape
        sliding_window = weight.shape[0]

        # Cast to float32 for computation — detach to ensure leaf tensor for autograd
        x_f = input.detach().float().requires_grad_(True)

        # Build Conv1d: weight shape [W, H] -> conv weight [H, 1, W]
        merge_conv = torch.nn.Conv1d(H, H, sliding_window, groups=H, bias=False)
        merge_conv.weight.data = weight.unsqueeze(1).transpose(0, 2).float()
        merge_conv.weight.retain_grad()

        # Forward: input (S,B,H) -> (B,H,S), pad zeros, conv, mask
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
        grad_f = grad_out.float()
        loss = torch.sum(grad_f * output)
        loss.backward()

        grad_input = x_f.grad.to(orig_dtype)
        grad_weight = merge_conv.weight.grad.transpose(0, 2).squeeze(1).to(orig_dtype)

        return [output.detach().to(orig_dtype), grad_input, grad_weight]
```
