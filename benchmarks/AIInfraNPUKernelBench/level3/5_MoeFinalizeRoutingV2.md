## 功能说明

- **算子功能**：MoE计算中，最后处理合并MoE FFN的输出结果。
- **计算公式**：

  $$
  expertid=expertIdx[i,k]
  $$
  
  $$
  out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
  $$

```python
class Model(nn.Module):
    """MoE finalize routing V2: merge MoE FFN output results."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        expanded_x: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor = None,
        bias: torch.Tensor = None,
        scales: torch.Tensor = None,
        expert_idx: torch.Tensor = None,
        drop_pad_mode: int = 0,
    ) -> torch.Tensor:
        """
        MoE finalize routing computation.

        out(i,j) = x1[i,j] + x2[i,j] + sum_k(scales[i,k] * (expanded_x[expanded_row_idx[i+k*num_rows],j] + bias[expert_idx[i,k],j]))

        Args:
            expanded_x: (bsk, h) expanded expert outputs, bf16/fp16
            expanded_row_idx: (bsk,) row index mapping, int32
            x1: (num_rows, h) optional residual, bf16/fp16
            x2: (num_rows, h) optional residual, bf16/fp16
            bias: (num_experts, h) expert bias, bf16/fp16
            scales: (num_rows, K) routing scales, fp32
            expert_idx: (num_rows, K) expert indices, int32
            drop_pad_mode: int, routing mode
        Returns:
            out: (num_rows, h) merged output, bf16/fp16
        """
        bsk = expanded_row_idx.shape[0]
        h = expanded_x.shape[-1]
        expanded_x_flat = expanded_x.reshape(-1, h).float()
        K = 1
        if scales is not None:
            K = scales.shape[-1]
        num_rows = bsk // K

        out = torch.zeros((num_rows, h), dtype=torch.float32)
        if x1 is not None:
            out = out + x1.float()
        if x2 is not None:
            out = out + x2.float()

        for k in range(K):
            if drop_pad_mode == 0 or drop_pad_mode == 1:
                indices = torch.arange(num_rows) + k * num_rows
            else:
                indices = torch.arange(num_rows) * K + k

            row_idx_vals = expanded_row_idx[indices].long()

            if drop_pad_mode == 1 or drop_pad_mode == 3:
                valid = row_idx_vals != -1
            else:
                valid = row_idx_vals < expanded_x_flat.shape[0]

            if not valid.any():
                continue

            valid_row_idx = row_idx_vals[valid]
            dst_rows = expanded_x_flat[valid_row_idx]

            if bias is not None and expert_idx is not None:
                eids = expert_idx[valid, k].long()
                dst_rows = dst_rows + bias[eids].float()

            if scales is not None:
                dst_rows = dst_rows * scales[valid, k].float().unsqueeze(1)

            out[valid] = out[valid] + dst_rows

        return out
```
