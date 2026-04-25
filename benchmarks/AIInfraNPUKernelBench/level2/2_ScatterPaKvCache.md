## 功能说明
接口功能：更新KvCache中指定位置的key和value。

```
key:[batch * seq_len, num_head, k_head_size]
value:[batch * seq_len, num_head, v_head_size]
keyCache:[num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]/[num_blocks, num_head, k_head_size // last_dim_k, block_size, last_dim_k]
valueCache:[num_blocks, num_head * v_head_size // last_dim_v, block_size, last_dim_v]/[num_blocks, num_head, v_head_size // last_dim_v, block_size, last_dim_v]
slotMapping:[batch * seq_len]
cacheMode:"PA_NZ"

其中k_head_size与v_head_size可以不同，也可以相同。
```

```python
class Model(nn.Module):
    """
    Scatter PA KV Cache: updates KV cache at specified slot positions.

    Scatters key and value tokens into paged attention KV cache blocks
    based on slot_mapping indices.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        key: torch.Tensor,
        key_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        value: torch.Tensor,
        value_cache: torch.Tensor,
        cache_mode: str = "PA_NZ",
    ) -> List[torch.Tensor]:
        """
        Args:
            key: (batch*seq_len, num_head, k_head_size)
            key_cache: (num_blocks, num_head*k_head_size//last_dim, block_size, last_dim)
            slot_mapping: (batch*seq_len,) int32
            value: (batch*seq_len, num_head, v_head_size)
            value_cache: (num_blocks, num_head*v_head_size//last_dim, block_size, last_dim)
            cache_mode: str, "PA_NZ"

        Returns:
            List of [updated_key_cache, updated_value_cache]
        """
        key_cache_out = key_cache.clone()
        value_cache_out = value_cache.clone()

        block_size = key_cache.shape[2]
        num_kv_slices = key_cache.shape[1]
        last_dim_k = key_cache.shape[3]
        num_vv_slices = value_cache.shape[1]
        last_dim_v = value_cache.shape[3]

        for i in range(slot_mapping.shape[0]):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size

            token_key = key[i].reshape(-1)
            for k in range(num_kv_slices):
                key_cache_out[block_index][k][block_offset][:] = token_key[k * last_dim_k: k * last_dim_k + last_dim_k]

            token_value = value[i].reshape(-1)
            for v in range(num_vv_slices):
                value_cache_out[block_index][v][block_offset][:] = token_value[v * last_dim_v: v * last_dim_v + last_dim_v]

        return [key_cache_out, value_cache_out]
```
