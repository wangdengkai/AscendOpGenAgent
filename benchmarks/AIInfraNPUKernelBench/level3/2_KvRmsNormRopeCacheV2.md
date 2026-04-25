## 功能说明

- 接口功能：对输入张量（kv）的尾轴，拆分出左半边用于rms_norm计算，右半边用于RoPE计算，再将计算结果分别scatter到两块cache中。

- 计算公式：
  
  (1) interleaveRope:

  $$
  x=kv[...,Dv:]
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y=x\_part1*cos+x\_part2*sin
  $$

  (2) rmsNorm:

  $$
  x=kv[...,:Dv]
  $$

  $$
  square\_x=x*x
  $$

  $$
  mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
  $$

  $$
  rms=torch.sqrt(mean\_square\_x+epsilon)
  $$

  $$
  y=(x/rms)*gamma
  $$

```python
class Model(nn.Module):
    """KV RmsNorm + interleave RoPE + Cache scatter V2."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        kv: torch.Tensor,
        gamma: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        index: torch.Tensor,
        k_cache: torch.Tensor,
        ckv_cache: torch.Tensor,
        k_scale: torch.Tensor = None,
        v_scale: torch.Tensor = None,
        k_offset: torch.Tensor = None,
        v_offset: torch.Tensor = None,
        vOptional: torch.Tensor = None,
        eps: float = 1e-5,
        cacheMode: str = "Norm",
        isOutputKv: bool = False,
    ) -> List[torch.Tensor]:
        """
        KV RmsNorm + interleave RoPE + cache scatter.

        (1) interleaveRope on kv[..., Dv:]:
            x1 = x[...,::2], x2 = x[...,1::2]
            x_part1 = cat(x1, x2), x_part2 = cat(-x2, x1)
            y = x_part1 * cos + x_part2 * sin

        (2) rmsNorm on kv[..., :Dv]:
            y = (x / sqrt(mean(x^2) + eps)) * gamma

        All bf16/fp16 inputs cast to fp32 for computation.

        Returns:
            List of [k_cache, ckv_cache] (+ optional kv outputs)
        """
        # Handle optional tensors with shape [0]
        if k_scale is not None and k_scale.numel() == 0:
            k_scale = None
        if v_scale is not None and v_scale.numel() == 0:
            v_scale = None
        if k_offset is not None and k_offset.numel() == 0:
            k_offset = None
        if v_offset is not None and v_offset.numel() == 0:
            v_offset = None

        kv_dtype = kv.dtype
        kv_f = kv.float()
        gamma_f = gamma.float()
        cos_f = cos.float()
        sin_f = sin.float()

        kv_shape = kv_f.shape
        Bkv = kv_shape[0]
        Nkv = kv_shape[1]
        Skv = kv_shape[2]
        Dkv = kv_shape[3]

        if vOptional is None:
            method_mode = 0
        else:
            method_mode = 1
            vOptional_f = vOptional.float()

        if method_mode == 0:
            v_dim = gamma_f.shape[-1]
            k_dim = cos_f.shape[-1]
        else:
            v_dim = vOptional.shape[3]
            k_dim = Dkv

        # Transpose to (B, S, N, D)
        kv_bsnd = kv_f.permute(0, 2, 1, 3)
        cos_bsnd = cos_f.permute(0, 2, 1, 3)
        sin_bsnd = sin_f.permute(0, 2, 1, 3)

        if method_mode == 0:
            rms_in = kv_bsnd[..., :v_dim]
            rope_in_raw = kv_bsnd[..., v_dim:]

            # RmsNorm
            v = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + eps)
            v = v * gamma_f
            v_out = v.permute(0, 2, 1, 3)
            if v_scale is not None:
                v = v * v_scale.float()
            if v_offset is not None:
                v = v + v_offset.float()
            if v_scale is not None:
                v = torch.round(v).clamp(-128, 127)

            # Interleave RoPE
            k = rope_in_raw.reshape(Bkv, Skv, Nkv, k_dim // 2, 2).transpose(-1, -2).reshape(Bkv, Skv, Nkv, k_dim)
            k1 = k[..., :k.shape[-1] // 2]
            k2 = k[..., k.shape[-1] // 2:]
            rotate_half_k = torch.cat((-k2, k1), dim=-1)
            k_embed = (k * cos_bsnd) + (rotate_half_k * sin_bsnd)
            k_embed_out = k_embed.permute(0, 2, 1, 3)
            if k_scale is not None:
                k_embed = k_embed * k_scale.float()
            if k_offset is not None:
                k_embed = k_embed + k_offset.float()
            if k_scale is not None:
                k_embed = torch.round(k_embed).clamp(-128, 127)
        else:
            rms_in = kv_bsnd
            v_in = vOptional_f.permute(0, 2, 1, 3)

            # RmsNorm
            v = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + eps)
            v = v * gamma_f

            # RoPE
            rope_dim = cos_bsnd.shape[-1]
            rope_in = v[..., :rope_dim]
            k = rope_in.reshape(Bkv, Skv, Nkv, rope_dim // 2, 2).transpose(-1, -2).reshape(Bkv, Skv, Nkv, rope_dim)
            k1 = k[..., :k.shape[-1] // 2]
            k2 = k[..., k.shape[-1] // 2:]
            rotate_half_k = torch.cat((-k2, k1), dim=-1)
            k_embed = (k * cos_bsnd) + (rotate_half_k * sin_bsnd)
            kv_out = torch.cat([k_embed, v[..., rope_dim:]], dim=-1)
            k_embed_out = kv_out.permute(0, 2, 1, 3).to(kv_dtype)
            if k_scale is not None:
                kv_out = kv_out * k_scale.float()
            if k_offset is not None:
                kv_out = kv_out + k_offset.float()
            if k_scale is not None:
                kv_out = torch.round(kv_out).clamp(-128, 127)
            k_embed = kv_out

            v_out = v_in.permute(0, 2, 1, 3).to(kv_dtype)
            if v_scale is not None:
                v_in = v_in * v_scale.float()
            if v_offset is not None:
                v_in = v_in + v_offset.float()
            if v_scale is not None:
                v_in = torch.round(v_in).clamp(-128, 127)
            v = v_in

        k_embed_out = k_embed_out.to(kv_dtype)
        v_out = v_out.to(kv_dtype)

        # Scatter into caches (simplified: just return the computed outputs)
        if isOutputKv:
            return [k_cache, ckv_cache, k_embed_out, v_out]
        else:
            return [k_cache, ckv_cache]
```
