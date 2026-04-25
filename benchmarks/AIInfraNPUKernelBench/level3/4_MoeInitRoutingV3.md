## 功能说明

- 算子功能：MoE的routing计算，，支持不量化和动态量化模式。

- 计算公式：  

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

  3.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.计算quant结果：
    - 动态quant：
        - 若不输入scale：
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
        - 若输入scale:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
  
  5.对quantResult取前NUM\_ROWS个sortedRowIdx的对应位置的值，得出expandedXOut：

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
    $$

  6.expandedRowIdxOut的有效元素数量availableIdxNum计算方式为，expertIdx中activeExpertRangeOptional范围内的元素的个数
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

```python
class Model(nn.Module):
    """MoE init routing V3: routing + optional quantization."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        expert_idx: torch.Tensor,
        scale: torch.Tensor = None,
        offset: torch.Tensor = None,
        active_num: int = 0,
        expert_capacity: int = 0,
        expert_num: int = 0,
        drop_pad_mode: int = 0,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        quant_mode: int = -1,
        active_expert_range: list = None,
        row_idx_type: int = 0,
    ) -> List[torch.Tensor]:
        """
        MoE routing: sort expert_idx, compute expanded_row_idx, optional quant.

        Args:
            x: (num_rows, h) input hidden states
            expert_idx: (num_rows, k) expert assignments
            scale: optional scale tensor
            offset: optional offset tensor
            active_num: max active tokens
            expert_capacity: capacity per expert (drop_pad_mode=1)
            expert_num: total number of experts
            drop_pad_mode: 0=no drop, 1=drop+pad
            expert_tokens_num_type: histogram type
            expert_tokens_num_flag: whether to output histogram
            quant_mode: -1=none, 0=static, 1=dynamic
            active_expert_range: (start, end) range
            row_idx_type: 0 or 1
        Returns:
            List of [expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale]
        """
        if active_expert_range is None:
            active_expert_range = [0, expert_num]

        def to_np(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                if t.dtype in (torch.bfloat16, torch.float16):
                    return t.float().numpy()
                return t.numpy()
            return t

        x_np = to_np(x)
        expert_idx_np = to_np(expert_idx)
        scale_np = to_np(scale)
        offset_np = to_np(offset)

        expert_start = active_expert_range[0] if drop_pad_mode == 0 else 0
        expert_end = active_expert_range[1] if drop_pad_mode == 0 else expert_num
        num_rows = x_np.shape[0]
        h = x_np.shape[1]
        k = expert_idx_np.shape[-1]
        expert_idx_in = expert_idx_np.copy().reshape(-1)
        actual_expert_total_num = int(np.sum((expert_idx_in >= expert_start) & (expert_idx_in < expert_end)))

        expert_idx_in[(expert_idx_in < expert_start)] = np.int32(np.iinfo(np.int32).max)
        sorted_expert_indices = np.argsort(expert_idx_in, axis=-1, kind="stable")
        sorted_expert_idx = expert_idx_in[sorted_expert_indices]

        if row_idx_type == 1:
            expanded_row_idx = sorted_expert_indices[:actual_expert_total_num]
        else:
            expanded_row_idx = np.ones(num_rows * k, dtype=np.int32) * -1
            tmp_indices = np.arange(actual_expert_total_num)
            expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

        if not expert_tokens_num_flag:
            expert_tokens_count = None
        else:
            if drop_pad_mode == 0:
                if expert_tokens_num_type == 1:
                    expert_tokens_count = np.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
                    expert_tokens_count = np.concatenate([expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                elif expert_tokens_num_type == 0:
                    expert_tokens_count = np.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
                    expert_tokens_count = np.concatenate([expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                    expert_tokens_count = np.cumsum(expert_tokens_count)
                elif expert_tokens_num_type == 2:
                    expert_id, counts = np.unique(sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                    expert_tokens_count = np.column_stack((expert_id, counts))
                    if expert_tokens_count.shape[0] < expert_num:
                        expert_tokens_count = np.concatenate((expert_tokens_count, [[0, 0]]), axis=0)
            else:
                expert_tokens_count = np.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate([expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
            expert_tokens_count = expert_tokens_count.astype(np.int64)

        if drop_pad_mode == 0:
            if active_num == 0:
                active_num = actual_expert_total_num
            else:
                active_num = min(active_num, actual_expert_total_num)
            expanded_scale = None
            expanded_x = x_np[sorted_expert_indices[:active_num] // k, :]
            if scale_np is not None and quant_mode == -1:
                expanded_scale = scale_np[sorted_expert_indices[:active_num] // k]
        else:
            # drop_pad_mode == 1
            def adapter_capacity(sorted_row_idx, sorted_expert_idx_l, capacity):
                count = 0
                last = sorted_expert_idx_l[0]
                for i, val in enumerate(sorted_expert_idx_l):
                    if last != val:
                        count = 1
                        last = val
                    else:
                        count += 1
                        if count > capacity:
                            sorted_expert_idx_l[i] = -1
                            sorted_row_idx[i] = -1

            adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)
            sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
            offset_tmp = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_expert_indices):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset_tmp = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset_tmp] = sorted_expert_indices[i]
                    offset_tmp += 1

            expanded_row_idx = np.full(sorted_expert_indices.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            expanded_x_mask = np.full((expert_num * expert_capacity, h), 1, dtype=int)
            expanded_x = np.full((expert_num * expert_capacity, h), 0, dtype=x_np.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = x_np[val // k]
                    expanded_x_mask[i] = np.full((h,), 0, dtype=int)

        if quant_mode == -1:
            if scale_np is not None and drop_pad_mode == 1:
                expanded_scale = np.full((expert_num * expert_capacity,), 0, dtype=scale_np.dtype)
                for i, val in enumerate(sort_row_tmp):
                    if val != -1:
                        expanded_scale[i] = scale_np[val // k]
            if scale_np is None:
                expanded_scale = None
        elif quant_mode == 0:
            expanded_scale = None
            expanded_x_fp16 = expanded_x.astype(np.float16)
            scale_val = scale_np.astype(np.float16)
            offset_val = offset_np.astype(np.float16)
            scale_rst = expanded_x_fp16 * scale_val[0]
            add_offset = scale_rst + offset_val[0]
            round_data = np.rint(add_offset)
            round_data = np.clip(round_data, -128, 127)
            expanded_x = round_data.astype(np.int8)
        elif quant_mode == 1:
            x_final = expanded_x.astype(np.float32)
            if scale_np is None:
                x_abs = np.abs(x_final)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)
            else:
                if scale_np.shape[0] == 1:
                    x_final = x_final * scale_np
                else:
                    if drop_pad_mode == 0:
                        x_final = x_final * scale_np[sorted_expert_idx[:active_num] - expert_start]
                    else:
                        for i, val in enumerate(sort_row_tmp):
                            if val != -1:
                                x_final[i] = x_final[i] * scale_np[i // expert_capacity]
                x_abs = np.abs(x_final)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)

        if drop_pad_mode == 1:
            expanded_x = np.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

        results = [torch.from_numpy(expanded_x), torch.from_numpy(expanded_row_idx.astype(np.int32))]
        if expert_tokens_count is not None:
            results.append(torch.from_numpy(expert_tokens_count))
        else:
            results.append(torch.empty(0, dtype=torch.int64))
        if expanded_scale is not None:
            results.append(torch.from_numpy(np.asarray(expanded_scale).astype(np.float32)))
        else:
            results.append(torch.empty(0, dtype=torch.float32))
        return results
```
