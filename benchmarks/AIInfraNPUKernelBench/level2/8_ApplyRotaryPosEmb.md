## 功能说明

- 接口功能：推理网络为了提升性能，将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。
- 计算公式：

  $$
  query\_q1 = query[..., : query.shape[-1] // 2]
  $$
  
  $$
  query\_q2 = query[..., query.shape[-1] // 2 :]
  $$
  
  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1)
  $$
  
  $$
  key\_k1 = key[..., : key.shape[-1] // 2]
  $$
  
  $$
  key\_k2 = key[..., key.shape[-1] // 2 :]
  $$
  
  $$
  key\_rotate = torch.cat((-key\_k2, key\_k1), dim=-1)
  $$
  
  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$
  
  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

```python
class Model(nn.Module):
    """
    Fused Apply Rotary Position Embedding for query and key.

    Applies half-rotation RoPE to both query and key tensors in a single pass.
    Supports partial rotary dimension and bf16/fp16 inputs (cast to fp32 internally).
    """

    def __init__(self):
        super(Model, self).__init__()

    def _single_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if dtype == torch.bfloat16 or dtype == torch.float16:
            x = x.float()
            cos = cos.float()
            sin = sin.float()

        d_rotary = cos.shape[-1]
        if x.shape[-1] != d_rotary:
            x_front = x[..., :d_rotary]
            x_back = x[..., d_rotary:]
            split = d_rotary // 2
            x1 = x_front[..., :split]
            x2 = x_front[..., split:]
            x_new = torch.cat((-x2, x1), dim=-1)
            res_front = x_front * cos + x_new * sin
            res = torch.cat((res_front, x_back), dim=-1)
        else:
            x1, x2 = torch.chunk(x, 2, dim=-1)
            x_new = torch.cat((-x2, x1), dim=-1)
            res = x * cos + x_new * sin

        return res.to(dtype)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                layout: str = "BSND") -> List[torch.Tensor]:
        """
        Apply rotary position embedding to query and key.

        Args:
            query: Query tensor.
            key: Key tensor.
            cos: Cosine frequency tensor.
            sin: Sine frequency tensor.
            layout: "BSND" or "TND".

        Returns:
            List of [q_embed, k_embed].
        """
        q_res = self._single_rope(query, cos, sin)
        k_res = self._single_rope(key, cos, sin)
        return [q_res, k_res]
```
