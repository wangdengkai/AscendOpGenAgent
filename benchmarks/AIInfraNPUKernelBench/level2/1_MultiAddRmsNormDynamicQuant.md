## 功能说明

- 算子功能：RmsNorm算子是大模型常用的归一化操作。DynamicQuant算子则是为输入张量进行对称动态量化的算子。MultiAddRmsNormDynamicQuant算子将RmsNorm前的n个Add算子和RmsNorm归一化输出给到的DynamicQuant算子融合起来，减少搬入搬出操作（n支持0到4）。
- 计算公式：

  $$
  x_{1}=x_{1a}+x_{1bOptional}+x_{1cOptional}+x_{1dOptional}+x_{1eOptional}  (x_{1}为长度为1到5的listOfTensor)
  $$

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - 若smoothScale1Optional和smoothScale2Optional均不输入，则y2和scale2输出无实际意义。计算过程如下所示：

  $$
   scale1=row\_max(abs(y))/127
  $$

  $$
   y1=round(y/scale1)
  $$

  - 若仅输入smoothScale1Optional，则y2和scale2输出无实际意义。计算过程如下所示：
  $$
    input = y\cdot smoothScale1Optional
  $$
  $$
   scale1=row\_max(abs(input))/127
  $$

  $$
   y1=round(input/scale1)
  $$

  - 若smoothScale1Optional和smoothScale2Optional均输入，则算子的五个输出均为有效输出。计算过程如下所示：
  $$
    input1 = y\cdot smoothScale1Optional
  $$
  $$
    input2 = y\cdot smoothScale2Optional
  $$
  $$
   scale1=row\_max(abs(input1))/127
  $$
  $$
   scale2=row\_max(abs(input2))/127
  $$
  $$
   y1=round(input1/scale1)
  $$
  $$
   y2=round(input2/scale2)
  $$

  其中row\_max代表每行求最大值。

```python
class Model(nn.Module):
    """
    Multi-Add RmsNorm with Dynamic Quantization.

    Fuses n Add operations before RmsNorm and a DynamicQuant after RmsNorm.
    Supports optional smooth scales for single or dual quantization paths.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: torch.Tensor,
        smooth_scale1: torch.Tensor = None,
        smooth_scale2: torch.Tensor = None,
        epsilon: float = 1e-5,
    ) -> List[torch.Tensor]:
        """
        Args:
            x1: Input tensor (batch, hidden) or (batch, seq, hidden).
            x2: Input tensor, same shape as x1.
            gamma: RmsNorm weight (hidden,).
            smooth_scale1: Optional smooth scale for path 1 (hidden,).
            smooth_scale2: Optional smooth scale for path 2 (hidden,).
            epsilon: RmsNorm epsilon.

        Returns:
            List of [x_sum, y_norm, y1_quant, scale1, y2_quant, scale2].
        """
        orig_dtype = x1.dtype
        x = x1.float() + x2.float()
        x_sum = x.to(orig_dtype)

        # RmsNorm
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + epsilon)
        y = (x / rms) * gamma.float()
        y_norm = y.to(orig_dtype)

        # Dynamic quantization
        if smooth_scale1 is not None and smooth_scale2 is not None:
            input1 = y * smooth_scale1.float()
            input2 = y * smooth_scale2.float()
            scale1 = torch.amax(torch.abs(input1), dim=-1) / 127.0
            scale2 = torch.amax(torch.abs(input2), dim=-1) / 127.0
            y1 = torch.round(input1 / scale1.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
            y2 = torch.round(input2 / scale2.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
        elif smooth_scale1 is not None:
            input1 = y * smooth_scale1.float()
            scale1 = torch.amax(torch.abs(input1), dim=-1) / 127.0
            y1 = torch.round(input1 / scale1.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
            y2 = y1.clone()
            scale2 = scale1.clone()
        else:
            scale1 = torch.amax(torch.abs(y), dim=-1) / 127.0
            y1 = torch.round(y / scale1.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
            y2 = y1.clone()
            scale2 = scale1.clone()

        return [x_sum, y_norm, y1, scale1.float(), y2, scale2.float()]
```
