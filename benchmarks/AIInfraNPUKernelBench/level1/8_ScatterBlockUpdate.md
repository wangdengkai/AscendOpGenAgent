## 功能说明

- 计算公式

$$
input[indices[k,0],indices[k,1],:] = update[k,:]
$$

```python
class Model(nn.Module):
    """
    ScatterNdUpdate: input[indices[k,0], indices[k,1], :] = update[k, :]
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        input: torch.Tensor,
        indices: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """
        ScatterNdUpdate Golden implementation.
        Pure move operator: input[indices[k,0], indices[k,1], :] = update[k, :]

        Args:
            input:   (D0, D1, D2), bf16/fp16/fp32
            indices: (K, 2), int32
            update:  (K, D2), bf16/fp16/fp32

        Returns:
            output tensor with same shape as input
        """
        output = input.clone()
        for k in range(indices.shape[0]):
            idx0 = indices[k, 0].item()
            idx1 = indices[k, 1].item()
            output[idx0, idx1, :] = update[k, :]
        return output
```
