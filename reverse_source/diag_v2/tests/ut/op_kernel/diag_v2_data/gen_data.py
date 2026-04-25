#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import torch


def gen_golden_data(params, data_dir):
    input_x = np.random.uniform(-100, 100, params[0].shape).astype(params[0].np_dtype)
    inputX = torch.as_tensor(input_x)
    outputY = torch.diag(inputX)
    golden = outputY.numpy().astype(params[1].np_dtype)

    input_x.tofile(str(data_dir / params[0].data_path))
    golden.tofile(str(data_dir / params[1].golden_path))


def gen_golden_data_simple(x, y, dtype):
    input_x = np.random.uniform(-1, 100, [int(x), int(y)]).astype(dtype)
    inputX = torch.as_tensor(input_x)
    outputY = torch.diag(inputX)
    golden = outputY.numpy().astype(dtype)

    input_x.tofile("./input_x.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3])