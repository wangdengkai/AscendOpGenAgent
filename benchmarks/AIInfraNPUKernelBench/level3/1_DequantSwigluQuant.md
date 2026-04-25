## 功能说明

- 算子功能：在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。
- swiglu_mode为0时的计算公式：  

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  swigluOut_i = Swiglu(dequantOut_i)=Swish(A_i)*B_i
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  其中，A<sub>i</sub>表示dequantOut<sub>i</sub>的前半部分，B<sub>i</sub>表示dequantOut<sub>i</sub>的后半部分。

- swiglu_mode为1时的计算公式：  

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clamp\_limit)
  $$
  
  $$
  x\_linear = x\_linear.clamp(min=-clamp\_limit, max=clamp\_limit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(glu\_alpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + glu\_bias)
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  其中，x\_glu表示dequantOut<sub>i</sub>的偶数索引部分，x\_linear表示dequantOut<sub>i</sub>的奇数索引部分。

```python
class Model(nn.Module):
    """Dequant + Swiglu + Quant fused operator."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight_scale: torch.Tensor = None,
        activate_scale: torch.Tensor = None,
        bias: torch.Tensor = None,
        quant_scale: torch.Tensor = None,
        quant_offset: torch.Tensor = None,
        group_num: torch.Tensor = None,
        activate_left: bool = True,
        quant_mode: str = "dynamic",
        swiglu_mode: int = 0,
        clamp_limit: float = 5.0,
        glu_alpha: float = 1.0,
        glu_bias: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        DequantSwigluQuant: dequant -> swiglu -> quant.

        swiglu_mode=0: out = Quant(Swish(A) * B) where A,B = split(Dequant(x))
        swiglu_mode=1: clipped swiglu with glu_alpha and glu_bias
        """
        if group_num is None:
            group_num = torch.tensor([x.shape[0]], dtype=torch.int64)

        offset = 0
        res_y = torch.zeros([x.shape[0], x.shape[1] // 2], dtype=torch.float32)
        res_scale = torch.zeros([x.shape[0]], dtype=torch.float32)

        for g_idx in range(group_num.shape[0]):
            groupIdx = group_num[g_idx].item()
            x_tensor = x[offset:offset + groupIdx].to(torch.float32)

            # Dequant
            res = x_tensor
            if weight_scale is not None:
                res = res * weight_scale[g_idx].to(torch.float32)
            if activate_scale is not None:
                res = res * activate_scale[offset:offset + groupIdx].unsqueeze(-1).to(torch.float32)
            if bias is not None:
                res = res + bias[g_idx].to(torch.float32)

            # Swiglu
            if swiglu_mode == 0:
                out = torch.chunk(res, 2, dim=-1)
                if activate_left:
                    self_tensor = out[0]
                    other = out[1]
                else:
                    self_tensor = out[1]
                    other = out[0]
                output = torch.nn.functional.silu(self_tensor) * other
            else:
                x_glu = res[..., ::2]
                x_linear = res[..., 1::2]
                x_glu = x_glu.clamp(max=clamp_limit)
                x_linear = x_linear.clamp(min=-clamp_limit, max=clamp_limit)
                out_glu = x_glu * torch.sigmoid(glu_alpha * x_glu)
                output = out_glu * (x_linear + glu_bias)

            # Quant
            if quant_scale is not None:
                output = output * quant_scale[g_idx].to(torch.float32)
            if quant_offset is not None:
                output = output + quant_offset[g_idx].to(torch.float32)

            if quant_mode == "dynamic":
                abs_val = torch.abs(output)
                max_values = torch.amax(abs_val, dim=-1)
                scale_out = max_values / 127.0
                max_values = 127.0 / max_values
                output = output * max_values.unsqueeze(1)

            output = torch.clamp(output, -128, 127)
            output = torch.round(output)
            res_y[offset:offset + groupIdx] = output
            res_scale[offset:offset + groupIdx] = scale_out
            offset = offset + groupIdx

        return [res_y.to(torch.int8), res_scale]
```
