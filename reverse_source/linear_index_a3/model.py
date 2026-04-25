import torch
import torch.nn as nn


def _case_basic_int32():
    indices = [
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
    ]
    stride = torch.tensor([10, 1], dtype=torch.int32)
    value_size = torch.tensor([10, 10], dtype=torch.int32)
    return [indices, stride, value_size]


def _case_negative_int64():
    indices = [
        torch.tensor([-5, -1, 0, 8, 12, 19], dtype=torch.int64),
        torch.tensor([3, -4, 5, -8, 1, 0], dtype=torch.int64),
        torch.tensor([7, 6, -5, 4, -3, 2], dtype=torch.int64),
    ]
    stride = torch.tensor([12, 4, 1], dtype=torch.int32)
    value_size = torch.tensor([5, 3, 4], dtype=torch.int32)
    return [indices, stride, value_size]


def _case_multicore_int32():
    base = torch.arange(35, dtype=torch.int32).reshape(5, 7)
    indices = [
        base - 11,
        base * 2 - 7,
        base * 3 + 5,
    ]
    stride = torch.tensor([100, 10, 1], dtype=torch.int32)
    value_size = torch.tensor([7, 5, 3], dtype=torch.int32)
    return [indices, stride, value_size]


def _case_empty_slot_int64():
    valid0 = torch.tensor(
        [[-9, -1, 0, 2], [5, 7, 8, -6], [11, 13, 15, -3], [4, -8, 6, 10]],
        dtype=torch.int64,
    )
    valid1 = torch.tensor(
        [[3, 0, -4, 5], [7, -2, 9, 1], [0, 4, -7, 8], [2, 6, -5, 11]],
        dtype=torch.int64,
    )
    indices = [
        torch.empty((0,), dtype=torch.int64),
        valid0,
        valid1,
    ]
    stride = torch.tensor([100, 10, 1], dtype=torch.int32)
    value_size = torch.tensor([9, 6, 5], dtype=torch.int32)
    return [indices, stride, value_size]


def get_input_groups():
    return [
        _case_basic_int32(),
        _case_negative_int64(),
        _case_multicore_int32(),
        _case_empty_slot_int64(),
    ]


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, indices, stride: torch.Tensor, value_size: torch.Tensor):
        valid_tensors = [tensor for tensor in indices if tensor.numel() > 0]
        if not valid_tensors:
            raise ValueError("at least one indices tensor must be non-empty")

        output = torch.zeros_like(valid_tensors[0], dtype=torch.int32)
        for index_tensor, stride_value, value_value in zip(indices, stride, value_size):
            if index_tensor.numel() == 0:
                continue
            remainder = torch.remainder(index_tensor, value_value.to(index_tensor.dtype))
            output = output + (remainder * stride_value.to(index_tensor.dtype)).to(torch.int32)
        return output
