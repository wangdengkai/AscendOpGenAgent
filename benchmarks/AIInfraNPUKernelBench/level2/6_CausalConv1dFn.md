## 功能说明

- **算子功能**：对序列执行因果一维卷积：沿序列维度，使用缓存数据（长度为卷积核宽减 1）对各序列头部进行 padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列尾部的数据（长度为卷积核宽减1）更新到缓存。

- **支持场景**：
    - x 支持 CuSeqLen 格式，shape 是 [cu_seq_len, dim]，其中 cu_seq_len 为 batch 内所有变长序列拼接后的总长度。
    - weight 的 shape 是[K, dim], K 固定为3。
    - 每个序列卷积前，使用长度为 K-1 的缓存数据对序列头部进行 padding，保证因果性。

- **计算公式**：
  K 是卷积核宽度， L 是原始序列长度， dim 是特征维度, batchId 是当前卷积处理的变长序列。

  - 缓存写入

  $$
  cacheIndex = cacheIndices[batchId]
  $$

  $$
  cacheState[i, dim] = convStates[cacheIndex]
  $$
  
  - 缓存拼接
  
  $$
  x'[i, dim] = 
  \begin{cases} 
  cacheState[i, dim], & 0 \leq i < K-1 \\
  x[i - (K-1), dim], & K-1 \leq i < L + K - 1
  \end{cases}
  $$
  
  - 因果1维卷积
  
  $$
  y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim]
  $$
  
  - 残差连接（可选）
  
  $$
  y[i, dim] = x[i, dim] + y[i, dim]
  $$

  - 缓存更新

  $$
  cacheState[i, dim] = x'[L + i, dim], \quad i = 0, 1, \dots, K-2
  $$

```python
class Model(nn.Module):
    """
    Causal 1D convolution for variable-length sequences (CuSeqLen format).

    Performs grouped depthwise causal convolution along the sequence dimension,
    using cached states for padding. Updates the cache after convolution.
    Optionally adds a residual connection.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor,
        pad_slot_id: int = -1,
        residual_connection: int = 1,
    ) -> List[torch.Tensor]:
        """
        Args:
            x: (cu_seq_len, dim) input tokens.
            weight: (K, dim) convolution kernel, K=3.
            conv_states: (-1, K-1, dim) cached states.
            query_start_loc: (batch+1,) int32, cumulative sequence start indices.
            cache_indices: (batch,) int32, index into conv_states per batch.
            initial_state_mode: (batch,) int32, 1=use cache, 2=zero cache.

        Returns:
            List of [output, updated_conv_states].
        """
        orig_dtype = x.dtype
        x_f = x.float()
        weight_f = weight.float()
        conv_states_f = conv_states.float()

        cu_seq_len, dim = x_f.shape
        batch_size = query_start_loc.shape[0] - 1
        kernel_width = weight_f.size(0)
        state_len = kernel_width - 1
        residual = residual_connection

        out = torch.zeros_like(x_f)

        for batch_idx in range(batch_size):
            start_idx = query_start_loc[batch_idx].item()
            end_idx = query_start_loc[batch_idx + 1].item()
            seq_len = end_idx - start_idx
            seq_x = x_f[start_idx:end_idx]

            _state = int(initial_state_mode[batch_idx])

            if _state == 1:
                cached_state = conv_states_f[cache_indices[batch_idx]]
            else:
                cached_state = torch.zeros((state_len, dim), device=x_f.device, dtype=x_f.dtype)

            padded_input = torch.cat([cached_state, seq_x], dim=0)

            result = F.conv1d(
                padded_input.transpose(0, 1).unsqueeze(0),
                weight_f.transpose(0, 1).unsqueeze(1),
                bias=None,
                stride=1,
                padding=0,
                groups=dim,
            )
            result = result.squeeze(0).transpose(0, 1)

            if _state == 2:
                result[:kernel_width - 1] = 0

            out[start_idx:end_idx] = result

            conv_states_f[cache_indices[batch_idx]] = padded_input[-state_len:]

        out = out + x_f if residual else out
        return [out.to(orig_dtype), conv_states_f.to(orig_dtype)]
```
