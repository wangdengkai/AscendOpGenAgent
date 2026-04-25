## 功能说明

- 算子功能：根据输入词频logits、topK/topP/minP采样参数、随机采样权重分布q，进行topK-topP-minP-sample采样计算。当输入isNeedSampleResult为false时，输出每个batch的最大词频logitsSelectIdx，以及topK-topP-minP采样后的词频分布logitsTopKPSelect；当输入isNeedSampleResult为true时，输出topK-topP-minP采样后的中间计算结果logitsIdx和logitsSortMasked，其中logitsSortMasked为词频logits经过topK-topP-minP采样计算后的中间结果，logitsIdx为logitsSortMasked在logits中对应的索引。

  算子包含四个可单独使能，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：TopK采样、TopP采样、MinP采样、指数采样（本文档中Sample所指）。目前支持以下计算场景。如下表所示：

  | 计算场景 | TopK采样 | TopP采样 | minP采样 | 指数分布采样 | 输出中间计算结果 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax采样|×|×|×|×|×|对输入logits按每个batch，取SoftMax后取最大结果|
  |topK采样|√|×|×|×|×|对输入logits按每个batch，取前topK[batch]个最大结果|
  |topP采样|×|√|×|×|×|对输入logits按每个batch从大到小排序，取累加值大于等于topP[batch]值的前n个结果进行采样|
  |Sample采样|×|×|×|√|×|对输入logits按每个batch，进行Softmax后与q进行除法取最大结果|
  |topK-topP采样|√|√|×|×|×|对输入logits按每个batch，先进行topK采样，再进行topP采样后取最大结果|
  |topK-Sample采样|√|×|×|√|×|对输入logits按每个batch，先进行topK采样，再进行Sample采样后取最大结果|
  |topP-Sample采样|×|√|×|√|×|对输入logits按每个batch，先进行topP采样，再进行Sample采样后取最大结果|
  |topK-topP-Sample采样|√|√|×|√|×|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行Sample采样后取最大结果|
  |topK-topP-minP采样-中间结果|√|√|√|×|√|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行minP采样，输出中间计算结果|
  |topK-minP采样-中间结果|√|×|√|×|√|对输入logits按每个batch，先进行topK采样，再进行minP采样，输出中间计算结果|
  |topK-topP采样-中间结果|√|√|×|×|√|对输入logits按每个batch，先进行topK采样，再进行minP采样，输出中间计算结果|
  |topK采样-中间结果|√|×|×|×|√|对输入logits按每个batch，进行topK采样，输出中间计算结果|
  
- 计算公式：

  输入logits为大小为[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
logits中的每一行logits[batch][:]根据相应的topK[batch]、topP[batch]、q[batch, :]，执行不同的计算场景。<br>
下述公式中使用b和v来分别表示batch和voc_size方向上的索引。

  TopK采样

  1. 按分段长度v采用分段topk归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
  2. topK[batch]对应当前batch采样的k值，有效范围为1≤topK[batch]≤min(voc_size[batch], ks_max)，如果top[k]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会则跳过当前batch的排序，将输入logits[batch]直接传入下一模块。

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
  v = 8 * \text{ks\_max}
  $$
  ks_max有效取值范围[1,1024]，默认为1024，并且需要向上对齐到8的整数倍。
  * 生成需要过滤的mask

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask = sortedValue \geq topKValue
  $$

  * 将小于阈值的部分通过mask置为defLogit:

  $$
  sortedValue[b][v]=
  \begin{cases}
  defLogit & \text{topKMask[b][v] = false} \\
  sortedValue[b][v] & \text{topKMask[b][v] = true} &
  \end{cases}
  $$

  * 其中defLogit取决于入参约束属性input_is_logits，该属性控制输入Logits和输出logits_top_kp_select的归一化：
  $$
    \text{defLogit} = 
    \begin{cases} 
    -inf, & \text{inputIsLogits} = \text{true} \\
    0, & \text{inputIsLogits} = \text{false}
    \end{cases}
  $$

  TopP采样
  * 根据入参约束属性inputIsLogits，如果该属性为True，则对排序后结果进行归一化：
    $$
    \text{logit\_sortProb} = 
    \begin{cases}
    \text{softmax}(\text{logits\_sort}), & \text{inputIsLogits} = \text{True} \\
    \text{logits\_sort}, & \text{inputIsLogits} = \text{False}
    \end{cases}
    $$

  * 根据输入`top_p[b]`的数值，本模块的处理策略如下：

    | 参数类型 | ≤0 | 有效域 | 无效域 |
    | :-------:| :------:|:-------:|:-------:|
    |`top_p[b]`|保留1个最大词频token|0<top_p<1,执行topP采样|top_p≥1,跳过topP采样|

  * 如果执行常规topP采样，且如果前序topK环节已有排序输出结果，则根据topK采样输出计算累积词频，并根据top_p截断采样：
    $$
    topPMask[b] =
    \begin{cases}
    0, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] > p[b] \\
    1, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] \leq p[b]
    \end{cases}
    $$
  * 如果执行常规topP采样，但前序topK环节被跳过，则计算top-p的mask:
    $$
    topPMask[b] =
    \begin{cases}
    topKMask[b][0:GuessK], & \sum_{\text{GuessK}}^{} probValue[b][*] \ge p[b] \\
    probSum[b][v] \le 1 - p[b], & \text{others}
    \end{cases}
    $$
  * 将需要过滤的位置设置为默认无效值defLogit，得到logits_sort，记为sortedValue[b][v]:
  $$
  sortedValue[b][v] =
  \begin{cases}
  defLogit & \quad \text{topPMask}[b][v] = \text{false} \\
  logit\_sortProb[b][v] & \quad \text{topPMask}[b][v] = \text{true}
  \end{cases}
  $$
  * 取过滤后sortedValue[b][v]每行中前topK个元素，查找这些元素在输入中的原始索引，整合为logits_idx:
  $$
  logitsIdx[b][v] = Index(sortedValue[b][v] \in Logits)
  $$
  * 使用截断后的sortedValue作为logitsSortMasked：
  $$
  logitsSortMasked[b,:] = sortedValue[b]
  $$
  minP采样
  * 如果min_ps[b]∈(0, 1)，则执行min_p采样：
    $$
    \text{logitsMax}[b] = \text{Max}(\text{logitsSortMasked}[b])
    $$
    $$
    \text{minPThd} = \text{logitsMax}[b] * \text{minPs}[b]
    $$
    $$
    \text{minPMask}[b] = 
    \begin{cases} 
    0, & \text{logitsSortMasked}[b] < \text{minPThd} \\
    1, & \text{logitsSortMasked}[b] \geq \text{minPThd}
    \end{cases}
    $$
    $$
    \text{logitsSortMasked}[b,:] = 
    \begin{cases} 
    \text{defLogit}, & \text{minPMask}[b] = 0 \\
    \text{logitsSortMasked}[b,:], & \text{minPMask}[b] = 1
    \end{cases}
    $$
  * 其他情况：
    $$
    \text{logitsSortMasked}[b, :] = 
    \begin{cases}
        \text{logitsSortMasked}[b, :], & \text{if } minPs[b] \leq 0 \\
        \max(\text{logitsSortMasked}[b, :]), & \text{if } minPs[b] \geq 1
    \end{cases}
    $$
    min_ps[b]≥1时，每个batch仅取1个最大token，其余位置填充defLogit。

  可选输出
  * 如果​入参属性IsNeedLogits=True，则使用topK-topP-minP联合采样后的logitsIndexMasked，进行`logits_top_kp_select`输出。
    $$
    \text{logitsIndex}[b][v] = \text{Index}(\text{logitsSortMasked}[b][v] \in \text{Logits})
    $$
    $$
    \text{logitsIndexMasked}[b,:] = \text{logitsIndex}[b,:] * \text{topKMask}[b] * \text{topPMask}[b] * \text{minPMask}[b]
    $$
    其中，topK、topP、minP采样环节如果被跳过，则相应mask为全1。
  * 接下来使用logitsIndexMasked对输入Logits进行Select，过滤输入Logits中的高频token作为`logits_top_kp_select`输出：
    $$
    \text{logitsTopKpSelect}[b][v] = 
    \begin{cases} 
    \text{logits}[b][v], & \text{if } logitsIndexMasked[b,v] = \text{True} \\
    \text{defLogit}, & \text{if } logitsIndexMasked[b,v] = \text{False}
    \end{cases}
    $$

  后继处理
  * 此阶段输入为前序对前序topK-topP-minP采样的联合结果logitsSortMasked。
  * 此处输入须要确保logitsSortMasked∈(0,1)，根据输入Logits的实际情况，配置入参约束属性inputIsLogits，即：
    $$
    \text{inputIsLogits} = 
    \begin{cases}
    True, & \text{Logits} \notin [0,1] \\
    False, & \text{Logits} \in [0,1]
    \end{cases}
    $$
    使得
    $$
    \text{probs}[b] = \text{logitsSortMasked}[b, :]
    $$
    接下来有三种模式：None，QSample，输出中间结果，通过入参约束属性isNeedSampleResult和是否输入q加以控制。
  * None:
  * isNeedSampleResult为false，且不输入q时为该模式。该模式下直接对每个batch通过Argmax取最大元素和索引，并通过gatherOut输出。
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b]\left[\text{ArgMax}(\text{probs}[b][:])\right]
    $$
  * QSample：
  * isNeedSampleResult为false，且输入q时为该模式。该模式先对probs进行指数分布采样：
    $$
    qCnt = \text{Sum}(\text{MinPMask} == 1)
    $$
    $$
    \text{probsOpt}[b] = \frac{\text{probs}[b]}{q[b, :qCnt] + \text{eps}}
    $$
  * 再进行Argmax-GatherOut输出结果：
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{ArgMax}(\text{probsOpt}[b][:])]
    $$
  * 输出中间结果:
  * isNeedSampleResult为true时，为该模式。此时会输出经过采样后的logitsSortMasked及其在输入中的原始索引logitsIdx：

    $$
    \text{logitsSortMasked}[b, v] = 
    \begin{cases}
        \text{logitsSortMasked}[b, v], & \text{if } \text{minPMask}[b, v] = 1 \\
        0, & \text{if } \text{minPMask}[b, v] = 0
    \end{cases}
    $$

    $$
    logitsIdx[b][v] = Index(logitsSortMasked[b][v])
    $$

```python
class Model(nn.Module):
    """TopK-TopP-MinP sampling V2 with extended features."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        q: torch.Tensor = None,
        min_ps: torch.Tensor = None,
        eps: float = 1e-8,
        is_need_logits: bool = False,
        top_k_guess: int = 32,
        ks_max: int = 1024,
        input_is_logits: bool = True,
        is_need_sample_result: bool = False,
    ) -> List[torch.Tensor]:
        """
        TopK-TopP-MinP-Sample V2.

        Extended sampling with minP support and intermediate result output.

        Args:
            logits: (batch, voc_size) input logits
            top_k: (batch,) k values
            top_p: (batch,) p values
            min_ps: (batch,) optional minP thresholds
            eps: epsilon
            is_need_logits: output filtered logits
            top_k_guess: guess K
            ks_max: max K
            input_is_logits: whether input needs softmax
            is_need_sample_result: output intermediate results
        Returns:
            List of [logits_select_idx, logits_top_kp_select] or
            [logits_idx, logits_sort_masked, logits_select_idx, logits_top_kp_select]
        """
        FLT_NEG_INF = float('-inf')
        ALL_P_MAX = 1.0

        k_max_aligned = (ks_max * 4 + 32 - 1) // 32 * 32 // 4
        k_max = k_max_aligned if k_max_aligned < 1024 else 1024

        def only_softmax(x, dim=-1):
            if dim < 0:
                dim = x.dim() + dim
            max_vals = torch.max(x, dim=dim, keepdim=True)[0]
            shifted = x - max_vals
            exp_vals = torch.exp(shifted)
            return exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)

        batch_size, vocab_size = logits.shape
        rs_index = torch.zeros(batch_size, dtype=torch.long)
        logits_idx = torch.zeros((batch_size, vocab_size), dtype=torch.long)
        logits_sort_masked = torch.zeros((batch_size, vocab_size), dtype=torch.float32)

        if is_need_logits:
            if input_is_logits:
                rs_value = torch.ones((batch_size, vocab_size), dtype=torch.float32) * FLT_NEG_INF
            else:
                rs_value = torch.zeros((batch_size, vocab_size), dtype=torch.float32)
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

            if input_is_logits:
                topk_probs = only_softmax(topk_logits, dim=-1)
            else:
                topk_probs = topk_logits

            if use_top_p:
                sorted_probs, sorted_probs_indices = torch.sort(topk_probs, dim=-1, descending=True, stable=True)
                if p > 0:
                    probs_sum = sorted_probs.cumsum(dim=-1)
                    top_p_mask = (probs_sum - sorted_probs) > p
                else:
                    top_p_mask = torch.tensor([True] * sorted_probs.numel())
                    top_p_mask[0] = False
                top_p_mask = torch.tensor(top_p_mask) if not isinstance(top_p_mask, torch.Tensor) else top_p_mask
                top_p_sel = ~top_p_mask
                selected_probs_indices = sorted_probs_indices[top_p_sel]
                selected_indices = topk_indices[selected_probs_indices]
                selected_logits = sorted_probs[top_p_sel]
                false_count = (top_p_sel > 0).sum().item()
            else:
                selected_indices = topk_indices
                selected_logits = topk_probs
                false_count = topk_probs.numel()
                top_p_sel = torch.tensor([True] * false_count)

            if p <= 0 and input_is_logits:
                selected_logits[0] = 1

            # MinP
            if min_ps is not None:
                min_p = min_ps[i].item()
            else:
                min_p = -1

            if not use_top_k and not use_top_p and min_p < 1:
                selected_indices = torch.arange(len(original_logits))
                if input_is_logits:
                    selected_logits = only_softmax(original_logits, dim=-1)
                else:
                    selected_logits = original_logits

            if min_p <= 0:
                min_p_sel = [True] * false_count
            elif min_p < 1:
                min_p_thd = torch.max(selected_logits) * min_p
                sel_prob_mask = selected_logits >= min_p_thd
                min_p_sel = [a and b for a, b in zip(top_p_sel.tolist(), sel_prob_mask.tolist())]
            else:
                min_p_sel = [False] * false_count
                min_p_sel[0] = True

            min_p_sel = torch.tensor(min_p_sel)
            selected_logits = selected_logits[min_p_sel]
            selected_indices = selected_indices[min_p_sel]
            false_count = selected_logits.numel()

            selected_probs = selected_logits

            # Post sample
            post_sample = "qSample" if q is not None else "None"
            if is_need_sample_result:
                post_sample = "multiNomial"

            if post_sample == "multiNomial":
                logits_sort_masked[i, :len(selected_logits)] = selected_probs
                logits_idx[i, :len(selected_indices)] = selected_indices
            elif post_sample == "qSample":
                q_i = q[i, :false_count]
                q_sample = selected_probs / (q_i.abs() + eps)
                probs_index = q_sample.argmax(dim=0).view(-1)
                golden_index = selected_indices[probs_index].squeeze(0)
                rs_index[i] = golden_index
            else:
                probs_index = selected_probs.argmax(dim=0).view(-1)
                golden_index = selected_indices[probs_index].squeeze(0)
                rs_index[i] = golden_index

            if is_need_logits:
                rs_value[i, selected_indices] = original_logits[selected_indices]

        # NPU kernel always returns 2-tuple (rs_index, rs_value); drop
        # intermediate logits_idx / logits_sort_masked to align with kernel.
        return [rs_index, rs_value]
```
