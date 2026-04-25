# L1/2 ClippedSwiglu

> 摘自 `/home/Code/benchMD/ops-nn/activation/clipped_swiglu/README.md` 与 `docs/aclnnClippedSwiglu.md`。
> Golden 接口：`torch_npu.npu_clipped_swiglu`（CANN 8.5.0 已打包，eager direct）。

## 功能说明

带截断的 Swish 门控线性单元激活函数（变体 SwiGLU）。相较标准 SwiGLU，新增 `group_index`、`alpha`、`limit`、`bias`、`interleaved` 五个参数，用于支持 GPT-OSS 模型变体与 MoE 分组场景。

## 计算流程

输入 `x` 维度 `[a,b,c,d,e,f,g,...]`：

1. **合轴**：基于 `dim` 把 x 合并成 `[pre, cut, after]`，再把 `cut` 与 `after` 合并为 `[pre, cut]`。
2. **分组过滤**（可选）：当传入 `group_index` 时，`sum = Σ group_index`，`x = x[:sum, :]`。
3. **切分**：
   - `interleaved=True`（奇偶切分）：`A = x[:, ::2]`，`B = x[:, 1::2]`
   - `interleaved=False`（前后切分）：`h = x.shape[1] // 2`，`A = x[:, :h]`，`B = x[:, h:]`
4. **变体 SwiGLU**：
   - `A = clamp(A, max=limit)`
   - `B = clamp(B, -limit, limit)`
   - `y_glu = A * sigmoid(alpha * A)`
   - `y = y_glu * (B + bias)`
5. **重塑**：恢复到合轴前的维度数量，`dim` 轴大小减半，其余维度保持。

## 参数表

| 参数 | I/O | 描述 | dtype | format |
|---|---|---|---|---|
| `x` | 输入 | 主输入，`dim` 对应轴必须为偶数 | FLOAT / FLOAT16 / BFLOAT16 | ND |
| `group_index` | 可选输入 | 分组过滤索引，1 维 | INT64 | - |
| `dim` | 可选属性 | 合轴/切分维度，范围 `[-x.dim(), x.dim()-1]`，默认 `-1` | INT64 | - |
| `alpha` | 可选属性 | sigmoid 缩放因子，默认 `1.702` | FLOAT | - |
| `limit` | 可选属性 | clamp 门限，默认 `7.0` | FLOAT | - |
| `bias` | 可选属性 | 输出加性偏置，默认 `1.0` | FLOAT | - |
| `interleaved` | 可选属性 | true=奇偶切分，false=前后切分，默认 `True` | BOOL | - |
| `y` | 输出 | `dim` 轴减半，其余与 `x` 一致 | 同 `x` | ND |

## torch_npu schema（运行时确认）

```
npu_clipped_swiglu(
    Tensor x, *,
    Tensor? group_index=None,
    int dim=-1, float alpha=1.702, float limit=7., float bias=1.,
    bool interleaved=True
) -> Tensor
```

## 产品支持

| 产品 | 支持 |
|---|---|
| Atlas A3 训练/推理 | √ |
| Atlas A2 训练/推理 | √ |
| 其他 | × |

## 约束

无。

## 参考调用

```cpp
// aclnn 层：见源仓 examples/test_aclnn_clipped_swiglu.cpp
aclnnClippedSwigluGetWorkspaceSize(...)
aclnnClippedSwiglu(...)
```

```python
class Model(nn.Module):
    """
    ClippedSwiglu: Clipped Swish-Gated Linear Unit activation.
    Supports interleaved/non-interleaved split, group_index filtering,
    and parameterized alpha/limit/bias.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        group_index: Optional[torch.Tensor],
        dim: int,
        alpha: float,
        limit: float,
        bias: float,
        interleaved: bool,
    ) -> torch.Tensor:
        """
        Clipped SwiGLU computation.

        1. Merge dims around `dim` into [pre, cut*after] = [pre, cut]
        2. Filter by group_index sum if provided
        3. Split into A, B (interleaved or halved)
        4. A = clamp(A, max=limit); B = clamp(B, -limit, limit)
        5. y = A * sigmoid(alpha * A) * (B + bias)
        6. Reshape output to match input dims with dim halved

        Args:
            x:            input tensor, bf16/fp16/fp32
            group_index:  (G,) int64 or None
            dim:          int, dimension to split
            alpha:        float
            limit:        float
            bias:         float
            interleaved:  bool

        Returns:
            output tensor with dim halved at `dim`
        """
        input_x = x
        orig_dtype = input_x.dtype

        # Merge dims around `dim` into [pre, cut]
        before_shape = input_x.shape[:dim]
        before_total = 1
        for s in before_shape:
            before_total *= s
        after_shape = input_x.shape[dim:]
        after_total = 1
        for s in after_shape:
            after_total *= s
        x_2d = input_x.reshape(before_total, after_total)

        if orig_dtype != torch.float32:
            x_2d = x_2d.to(torch.float32)

        # Group filtering
        if group_index is not None:
            group_sum = min(int(torch.sum(group_index).item()), x_2d.shape[0])
        else:
            group_sum = x_2d.shape[0]
        x_tensor = x_2d[:group_sum]

        # Split
        if interleaved:
            x_glu = x_tensor[..., ::2]
            x_linear = x_tensor[..., 1::2]
        else:
            out = torch.chunk(x_tensor, 2, dim=-1)
            x_glu = out[0]
            x_linear = out[1]

        # Clamp and compute
        x_glu = x_glu.clamp(min=None, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
        sigmoid_part = torch.sigmoid(alpha * x_glu)
        result = x_glu * sigmoid_part * (x_linear + bias)
        result = result.to(orig_dtype)

        # Build output with zeros for filtered rows
        y = torch.zeros((x_2d.shape[0], x_2d.shape[1] // 2), dtype=orig_dtype)
        y[:group_sum] = result

        # Reshape to original dims with dim halved
        shape = list(input_x.shape)
        shape[dim] = shape[dim] // 2
        return y.reshape(shape)
```
