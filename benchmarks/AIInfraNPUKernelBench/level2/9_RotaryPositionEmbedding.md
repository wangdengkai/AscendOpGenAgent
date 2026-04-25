## 功能说明

- 接口功能：执行单路旋转位置编码计算。
- 计算公式：
    （1）half模式（mode等于0）：

    $$
    x1 = x[..., : x.shape[-1] // 2]
    $$

    $$
    x2 = x[..., x.shape[-1] // 2 :]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1)
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$

    （2）interleave模式（mode等于1）：

    $$
    x1 = x[..., ::2].view(-1, 1)
    $$

    $$
    x2 = x[..., 1::2].view(-1, 1)
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$

    （3）quarter模式（mode等于2）：

    $$
    x1 = x[..., : x.shape[-1] // 4]
    $$

    $$
    x2 = x[..., x.shape[-1] // 4 : x.shape[-1] // 2]
    $$

    $$
    x3 = x[..., x.shape[-1] // 2 : x.shape[-1] // 4 * 3]
    $$

    $$
    x4 = x[..., x.shape[-1] // 4 * 3 :]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1, -x4, x3), dim=-1)
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$

    （4）interleave-half模式（mode等于3），该模式会先将奇数位的输入抽取到前半部分，将偶数位的输入抽取到后半部分，再进行half处理：

    $$
    x1 = x[..., ::2]
    $$

    $$
    x2 = x[..., 1::2]
    $$

    $$
    x\_part1 = torch.cat((x1, x2), dim=-1)
    $$

    $$
    x\_part2 = torch.cat((-x2, x1), dim=-1)
    $$

    $$
    y = x\_part1 * cos + x\_part2 * sin
    $$

```python
class Model(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    Applies rotary position encoding to the input tensor using cosine and sine
    frequency tensors. Supports three rotation modes:
      - mode 0 (half): rotate_half strategy
      - mode 1 (interleave / every_two): rotate_every_two strategy
      - mode 3 (interleave-half): interleave-half rotation strategy

    Also supports partial RoPE where the embedding dimension D of cos/sin is
    smaller than the last dimension Dx of x; in that case only x[..., :D] is
    rotated and x[..., D:] is passed through unchanged.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, mode: int = 0) -> torch.Tensor:
        """
        Applies Rotary Position Embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [..., Dx].
            cos (torch.Tensor): Cosine frequency tensor of shape [..., D].
            sin (torch.Tensor): Sine frequency tensor, same shape as cos.
            mode (int): Rotation mode. 0=half, 1=interleave, 3=interleave-half.

        Returns:
            torch.Tensor: Output tensor with RoPE applied, same shape as x.
        """
        dtype = x.dtype

        if x.dtype in [torch.bfloat16, torch.float16]:
            x = x.to(torch.float32)
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

        Dx = x.shape[-1]
        D = cos.shape[-1]

        if Dx > D:
            x_rope = x[..., :D]
            x_pass = x[..., D:]
        else:
            x_rope = x
            x_pass = None

        if mode == 0:
            x_chunk = torch.chunk(x_rope, 2, -1)
            rotated = torch.cat((x_chunk[1] * (-1), x_chunk[0]), dim=-1)
            y_rope = cos * x_rope + rotated * sin
        elif mode == 1:
            x_even = x_rope[..., ::2]
            x_odd = x_rope[..., 1::2]
            stacked = torch.stack((-x_odd, x_even), dim=-1)
            rotated = stacked.reshape(x_rope.shape)
            y_rope = cos * x_rope + rotated * sin
        else:  # mode == 3
            x_even = x_rope[..., ::2]
            x_odd = x_rope[..., 1::2]
            input_part1 = torch.cat((x_even, x_odd), dim=-1)
            input_part2 = torch.cat((x_odd * (-1), x_even), dim=-1)
            y_rope = input_part2 * sin + input_part1 * cos

        if x_pass is not None:
            y = torch.cat([y_rope, x_pass], dim=-1)
        else:
            y = y_rope

        return y.to(dtype)
```
