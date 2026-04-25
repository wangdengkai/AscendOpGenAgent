/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../op_host/diag_v2_tiling.h"

using namespace std;
using namespace ge;

class DiagV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DiagV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DiagV2Tiling TearDown" << std::endl;
  }
};

TEST_F(DiagV2Tiling, ascend910B1_test_tiling__001)
{
    optiling::DiagV2CompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "DiagV2",
        {
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("diagonal", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
         &compileInfo);
    uint64_t expectTilingKey = 2102;
    string expectTilingData = "8 8 0 8 1 16 8 2102 128 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}                