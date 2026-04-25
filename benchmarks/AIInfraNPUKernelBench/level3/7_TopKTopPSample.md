## 功能说明

- 算子功能：根据输入词频logits、topK/topP采样参数、随机采样权重分布q，进行topK-topP-sample采样计算，输出每个batch的最大词频logitsSelectIdx，以及topK-topP采样后的词频分布logitsTopKPSelect。

  算子包含三个可单独使能，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：TopK采样、TopP采样、指数采样（本文档中Sample所指）。它们可以构成八种计算场景。如下表所示：

  | 计算场景 | TopK采样 | TopP采样 | 指数分布采样 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax采样|×|×|×|对输入logits按每个batch，取SoftMax后取最大结果|
  |topK采样|√|×|×|对输入logits按每个batch，取前topK[batch]个最大结果|
  |topP采样|×|√|×|对输入logits按每个batch从大到小排序，取累加值大于等于topP[batch]值的前n个结果进行采样|
  |Sample采样|×|×|√|对输入logits按每个batch，进行Softmax后与q进行除法取最大结果|
  |topK-topP采样|√|√|×|对输入logits按每个batch，先进行topK采样，再进行topP采样后取最大结果|
  |topK-Sample采样|√|×|√|对输入logits按每个batch，先进行topK采样，再进行Sample采样后取最大结果|
  |topP-Sample采样|×|√|√|对输入logits按每个batch，先进行topP采样，再进行Sample采样后取最大结果|
  |topK-topP-Sample采样|√|√|√|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行Sample采样后取最大结果|
  
- 计算公式：

  输入logits为大小为[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
logits中的每一行logits[batch][:]根据相应的topK[batch]、topP[batch]、q[batch, :]，执行不同的计算场景。<br>
下述公式中使用b和v来分别表示batch和voc_size方向上的索引。

  TopK采样

  1. 按分段长度v采用分段topk归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
  2. topK[batch]对应当前batch采样的k值，有效范围为1≤topK[batch]≤min(voc_size[batch], 1024)，如果top[k]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会则跳过当前batch的排序，将输入logits[batch]直接传入下一模块。

  * 对当前batch分割为若干子段，滚动计算topKValue[b]：

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  其中：

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v表示预设的滚动topK时固定的分段长度：

  $$
  v=8*1024
  $$

  * 生成需要过滤的mask

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask[b] = sortedValue[b]<Min(topKValue[b])
  $$

  * 将小于阈值的部分通过mask置为-Inf

  $$
  sortedValue[b][v]=
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true} \\
  sortedValue[b][v] & \text{topKMask[b][v]=false} &
  \end{cases}
  $$

  * 通过softmax将经过topK过滤后的logits按最后一轴转换为概率分布

  $$
  probsValue[b]=sortedValue[b].softmax (dim=-1)
  $$

  * 按最后一轴计算累积概率（从最小的概率开始累加）

  $$
  probsSum[b]=probsValue[b].cumsum (dim=-1)
  $$

  TopP采样

  * 如果前序topK采样已有排序输出结果，则根据topK采样输出计算累积词频，并根据topP截断采样：

    $$
    topPMask[b] = probsSum[b][*] < topP[b]
    $$

  * 如果topK采样被跳过，则先对输入logits[b]进行softmax处理：

  $$
  logitsValue[b] = logits[b].softmax(dim=-1)
  $$

  * 尝试使用topKGuess，对logits进行滚动排序，获取计算topP的mask：

  $$
  topPValue[b] = {Max(topKGuess)}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topPValue[b]\left \{s-1 \right \}  \cup \left \{ logitsValue[b][v] \ge topKMin[b][s-1] \right \} \right \}
  $$

  * 如果在访问到logitsValue[b]的第1e4个元素之前，下条件得到满足，则视为topKGuess成功，

  $$
  \sum^{topKGuess}(topPValue[b]) \ge topP[b]\\
  topPMask[b][Index(topPValue[b])] = false
  $$

  * 如果topKGuess失败，则对当前序logitsValue[b]进行全排序和cumsum，按topP[b]截断采样：

  $$
  sortedLogits[b] = sort(logitsValue[b], descendant) \\
  probsSum[b]=sortedLogits[b].cumsum (dim=-1) \\
  topPMask[b] = (probsSum[b] - sortedLogits[b])>topP[b]
  $$

  * 将需要过滤的位置设置为-Inf，得到sortedValue[b][v]：

    $$
    sortedValue[b][v] = \begin{cases} -Inf& \text{topPMask[b][v]=true}\\sortedValue[b][v]& \text{topPMask[b][v]=false}\end{cases}
    $$

    取过滤后sortedValue[b][v]每行中前topK个元素，查找这些元素在输入中的原始索引，整合为`logitsIdx`:

    $$
    logitsIdx[b][v] = Index(sortedValue[b][v] \in logits)
    $$

  指数采样（Sample）

  * 如果`isNeedLogits=true`，则根据`logitsIdx`，选取采样后结果输出到`logitsTopKPSelect`：

  $$
  logitsTopKPSelect[b][logitsIdx[b][v]]=sortedValue[b][v]
  $$

  * 对`logitsSort`进行指数分布采样：

    $$
    probs = softmax(logitsSort)
    $$

    $$
    probsOpt = \frac{probs}{q + eps}
    $$

  * 从`probsOpt`中取出每个batch的最大元素，从`logitsIdx`中gather相应元素的输入索引，作为输出`logitsSelectIdx`：

    $$
    logitsSelectIdx[b] = logitsIdx[b][argmax(probsOpt[b][:])]
    $$

  其中0≤b<sortedValue.size(0)，0≤v<sortedValue.size(1)。

```python
class Model(nn.Module):
    """TopK-TopP sampling: topK + topP + exponential sampling."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        q: torch.Tensor = None,
        eps: float = 1e-8,
        is_need_logits: bool = False,
        top_k_guess: int = 32,
    ) -> List[torch.Tensor]:
        """
        TopK-TopP-Sample: topK filtering -> topP filtering -> exponential sampling.

        Args:
            logits: (batch, voc_size) input logits
            top_k: (batch,) k values per batch
            top_p: (batch,) p values per batch
            q: (batch, voc_size) random weights for sampling
            eps: epsilon for numerical stability
            is_need_logits: whether to output filtered logits
            top_k_guess: guess K for topP without topK
        Returns:
            List of [logits_select_idx, logits_top_kp_select]
        """
        FLT_NEG_INF = float('-inf')
        ALL_P_MAX = 1.0
        k_max = 1024
        input_is_logits = True

        def only_softmax(x, dim=-1):
            if dim < 0:
                dim = x.dim() + dim
            max_vals = torch.max(x, dim=dim, keepdim=True)[0]
            shifted = x - max_vals
            exp_vals = torch.exp(shifted)
            return exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)

        batch_size, vocab_size = logits.shape
        rs_index = torch.zeros(batch_size, dtype=torch.long)

        if is_need_logits:
            rs_value = torch.ones((batch_size, vocab_size), dtype=torch.float32) * FLT_NEG_INF
        else:
            rs_value = torch.empty(0)

        for i in range(batch_size):
            original_logits = logits[i].float()
            k_val = top_k[i].item()
            top_ks_max = min(k_max, vocab_size)
            use_top_k = (k_val >= 1 and k_val <= top_ks_max)
            p = top_p[i].item()
            use_top_p = p < ALL_P_MAX

            topk_logits, topk_indices = torch.sort(original_logits, dim=-1, descending=True, stable=True)

            if use_top_k:
                k_val = min(k_val, vocab_size)
                topk_logits = topk_logits[:k_val]
                topk_indices = topk_indices[:k_val]

            topk_probs = only_softmax(topk_logits, dim=-1)

            if use_top_p:
                sorted_probs, sorted_probs_indices = torch.sort(topk_probs, dim=-1, descending=True, stable=True)
                if p > 0:
                    probs_sum = sorted_probs.cumsum(dim=-1)
                    top_p_mask = (probs_sum - sorted_probs) > p
                else:
                    top_p_mask = torch.tensor([True] * sorted_probs.numel())
                    top_p_mask[0] = False
                top_p_sel = ~top_p_mask
                selected_probs_indices = sorted_probs_indices[top_p_sel]
                selected_indices = topk_indices[selected_probs_indices]
                selected_logits = topk_logits[selected_probs_indices]
                false_count = (top_p_sel > 0).sum().item()
            else:
                selected_indices = topk_indices
                selected_logits = topk_probs
                false_count = topk_probs.numel()

            selected_probs = only_softmax(selected_logits, dim=-1)

            if q is not None and q[i].numel() > 0:
                q_i = q[i, :false_count]
                q_sample = selected_probs / (q_i.abs() + eps)
                probs_index = q_sample.argmax(dim=0).view(-1)
            else:
                probs_index = selected_probs.argmax(dim=0).view(-1)

            golden_index = selected_indices[probs_index].squeeze(0)
            rs_index[i] = golden_index

            if is_need_logits:
                rs_value[i, selected_indices] = original_logits[selected_indices]

        return [rs_index, rs_value]
```
