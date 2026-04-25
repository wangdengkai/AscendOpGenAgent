import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x)

def get_inputs():
    return [torch.randn(128, 128, dtype=torch.float16)]

def get_init_inputs():
    return []
