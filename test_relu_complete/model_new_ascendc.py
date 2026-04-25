import torch
import torch.nn as nn
import os

# 导入编译后的 AscendC extension
try:
    import _relu_ext as relu_ext
except ImportError:
    print("Warning: AscendC extension not found, using PyTorch fallback")
    relu_ext = None

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if relu_ext is not None:
            # 使用 AscendC kernel
            output = torch.empty_like(x)
            relu_ext.relu_forward(x.data_ptr(), output.data_ptr(), x.numel(), 
                                 torch.cuda.current_stream().cuda_stream)
            return output
        else:
            # Fallback to PyTorch
            return torch.nn.functional.relu(x)

def get_inputs():
    return [torch.randn(128, 128, dtype=torch.float16)]

def get_init_inputs():
    return []
