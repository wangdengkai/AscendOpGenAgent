## 功能说明

- 接口功能：将各SP域PA算子的输出的中间结果lse，localOut两个局部变量结果更新成全局结果。
- 计算公式：

$$
lse_{max} = \text{max}lse_i
$$

$$
lse = \sum_i \text{exp}(lse_i - lse_{max})
$$

$$
lse_m = lse_{max} + \text{log}(lse)
$$

$$
O = \sum_i O_i \cdot \text{exp}(lse_i - lse_m)
$$

```python
class Model(nn.Module):
    """
    AttentionUpdate: merges local SP-domain PA outputs (lse, localOut)
    into global results using log-sum-exp reduction.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        lse_list: List[torch.Tensor],
        local_out_list: List[torch.Tensor],
        update_type: int = 0,
    ) -> List[torch.Tensor]:
        """
        Merge local attention outputs into global result.

        lse_max = max(lse_i)
        lse = sum(exp(lse_i - lse_max))
        lse_m = lse_max + log(lse)
        O = sum(O_i * exp(lse_i - lse_m))

        Args:
            lse_list:       list of K tensors, each (N,), float32
            local_out_list: list of K tensors, each (N, H), fp16/bf16/fp32
            update_type:    0 or 1

        Returns:
            List of [all_out, lse_out]
        """
        out_dtype = local_out_list[0].dtype

        lse_cpu = [t.detach().float() for t in lse_list]
        out_cpu = [t.detach().float() for t in local_out_list]

        all_lse = torch.stack(lse_cpu, dim=0)
        all_out = torch.stack(out_cpu, dim=0)

        sp = all_out.shape[0]

        # Handle both 1D (N,) and 2D (N, H) local_out shapes
        need_unsqueeze = (all_out.dim() == 2)
        if need_unsqueeze:
            all_out = all_out.unsqueeze(-1)  # (sp, N) -> (sp, N, 1)

        hd = all_out.shape[-1]
        total_length = all_out.shape[1]

        # Log-sum-exp reduction
        lse_max, _ = torch.max(all_lse, dim=0)
        lse_max_expand = lse_max.unsqueeze(0)
        lse_sub = all_lse - lse_max_expand
        lse_sub_exp = torch.exp(lse_sub)
        lse_sub_exp_sum = torch.sum(lse_sub_exp, dim=0)
        lse_out = lse_max + torch.log(lse_sub_exp_sum)

        # Weighted sum of outputs
        lse_out_expand = lse_out.unsqueeze(0)
        lse_out_expand = all_lse - lse_out_expand
        lse_out_expand = lse_out_expand.unsqueeze(2)
        lse_out_expand = lse_out_expand.expand(sp, total_length, hd)
        lse_out_expand = torch.exp(lse_out_expand)

        prod_per_sp = all_out * lse_out_expand
        all_out = torch.sum(prod_per_sp, dim=0)

        if need_unsqueeze:
            all_out = all_out.squeeze(-1)  # (N, 1) -> (N,)

        all_out = all_out.to(out_dtype)

        return [all_out, lse_out]
```
