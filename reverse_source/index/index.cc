/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file index.cc
 * \brief
 */
#include "runtime2_util.h"
#include "index.h"


using namespace ge;
namespace {
constexpr int64_t INPUT_IDX_X = 0;
constexpr int64_t INPUT_IDX_INDEXED_SIZES = 1;
constexpr int64_t INPUT_IDX_INDICES = 3;
constexpr int64_t TILING_MODE_FIRST_AXIS = 0;
constexpr int64_t TILING_MODE_NON_FIRST_AXIS = 1;
constexpr int64_t TILING_MODE_NON_FIRST_AXIS_2D = 2;
constexpr int64_t BYTES_PER_BLOCK = 32;
constexpr int64_t DIM_MAX = 10;
constexpr int64_t DATA_LIMIT_MULTI_INDEX = 500000;
const std::string OP_NAME = "Index";
constexpr int64_t TWO_DIMS = 2;
}  // namespace

namespace optiling {
static void PrintInfo(const IndexTilingData *params) {
  for (size_t idx = 0; idx < DIM_MAX; idx++) {
    OP_LOGD(OP_NAME.c_str(), "op [TilingData] : input_batch_num=%ld.", params->batch_num_list[idx]);
  }
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : core_num_var=%ld.", params->core_num_var);
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : output_size=%ld.", params->output_size);
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : output_num_per_batch=%ld.", params->output_num_per_batch);
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : output_batch_num=%ld.", params->output_batch_num);
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : input_size=%ld.", params->input_size);
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : input_shape_dim_num=%ld.", params->input_shape_dim_num);
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : index_size=%ld.", params->index_size);
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : masks_num=%ld.", params->masks_num);
  OP_LOGD(OP_NAME.c_str(), "op [TilingData] : tiling_mode=%ld.", params->tiling_mode);
}

static ge::graphStatus IsAiCore(IndexTilingData *params, const IndexCompileInfo *compile_info) {
  if (compile_info->indices_num < params->masks_num) {
    if (params->tiling_mode == TILING_MODE_NON_FIRST_AXIS_2D) {
      // This is the UB size judgment in the non-first axis case
      if (compile_info->available_size <
          (compile_info->task_align) * (params->batch_num_list[1] + params->index_size)) {
        OP_LOGW(OP_NAME.c_str(), "there is not enough space in the UB, execute the AICPU engine.");
        return ge::GRAPH_FAILED;
      }
    } else if (params->tiling_mode != TILING_MODE_NON_FIRST_AXIS) {
      OP_LOGW(OP_NAME.c_str(), "tail size is too small for non_first_axis case, execute the AICPU engine.");
      return ge::GRAPH_FAILED;
    }
  }
  
  // full_axis case only supports output size < 50W
  if (compile_info->after_v200 &&
      params->output_size > DATA_LIMIT_MULTI_INDEX && 
      compile_info->indices_num == params->input_shape_dim_num) {
    OP_LOGW(OP_NAME.c_str(), "in multiple index mode, compute size is too much, execute the AICPU engine.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Index(gert::TilingContext *context) {
  OP_LOGD(OP_NAME.c_str(), "Tiling4Index rt2.0 is running.");
  auto compile_info = reinterpret_cast<const IndexCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  auto params = context->GetTilingData<IndexTilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);
  
  // get input size & input shape's dim num
  auto const input_shape = context->GetInputShape(INPUT_IDX_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_shape);
  auto const input_shape_val = input_shape->GetStorageShape();
  params->input_size = input_shape_val.GetShapeSize();
  params->input_shape_dim_num = input_shape_val.GetDimNum();
  
  // get output size
  auto const output_size = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_size);
  auto const output_size_val = output_size->GetStorageShape();
  params->output_size = output_size_val.GetShapeSize();
  
  // get masks' num
  auto const indexed_sizes = context->GetInputShape(INPUT_IDX_INDEXED_SIZES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, indexed_sizes);
  auto const indexed_sizes_shape = indexed_sizes->GetStorageShape();
  auto const indexed_sizes_num = indexed_sizes_shape.GetDim(0);
  params->masks_num = indexed_sizes_num;
  
  // get index's size
  auto const index = context->GetInputShape(INPUT_IDX_INDICES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, index);
  auto const index_shape = index->GetStorageShape();
  auto const index_size = index_shape.GetShapeSize();
  params->index_size = index_size;

  auto num_head_zeros_mask = params->masks_num - compile_info->indices_num;
  auto first_dim = 1;
  auto last_dim = 1;
  // combine axes not indexed
  for (int32_t idx = 0; idx < num_head_zeros_mask; idx++) {
    first_dim = first_dim * input_shape_val.GetDim(idx);
  }
  for (int32_t idx = params->masks_num; idx < params->input_shape_dim_num; idx++) {
    last_dim = last_dim * input_shape_val.GetDim(idx);
  }

  for (int32_t idx = 0; idx < DIM_MAX; idx++) {
    params->batch_num_list[idx] = 0;
  }

  params->batch_num_list[0] = first_dim;
  for (int32_t idx = 0; idx < compile_info->indices_num; idx++) {
    params->batch_num_list[idx + 1] = input_shape_val.GetDim(idx + num_head_zeros_mask);
  }
  params->batch_num_list[compile_info->indices_num + 1] = last_dim;
  
  if (compile_info->indices_num == params->masks_num) {
    params->tiling_mode = TILING_MODE_FIRST_AXIS;
    params->output_batch_num = params->index_size;
  } else {
    if (params->input_shape_dim_num == TWO_DIMS) {
      params->tiling_mode = TILING_MODE_NON_FIRST_AXIS_2D;
    } else if (last_dim >= compile_info->align_num){
      params->tiling_mode = TILING_MODE_NON_FIRST_AXIS;
    }
    params->output_batch_num = first_dim;
  }
  params->output_num_per_batch = last_dim;

  auto isAiCore = IsAiCore(params, compile_info);
  if (isAiCore == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }
  params->input_shape_dim_num = TWO_DIMS + compile_info->indices_num;

  params->core_num_var = compile_info->core_num;
  context->SetBlockDim(compile_info->core_num);
  PrintInfo(params);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Index(gert::TilingParseContext* context) {
  auto compile_info = GetCompileInfoPtr<IndexCompileInfo>(context);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
  OP_TILING_CHECK(compile_info == nullptr || parsed_object_cinfo == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(OP_NAME, "compile_info or json_str nullptr!"),
                  return ge::GRAPH_FAILED);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(OP_NAME, "get vars failed."), return ge::GRAPH_FAILED);
  optiling::ReadCompileItem(vars, "task_align", compile_info->task_align);
  optiling::ReadCompileItem(vars, "core_num", compile_info->core_num);
  optiling::ReadCompileItem(vars, "available_size", compile_info->available_size);
  optiling::ReadCompileItem(vars, "indices_num", compile_info->indices_num);
  optiling::ReadCompileItem(vars, "align_num", compile_info->align_num);

  uint32_t core_num_temp = 0;
  OP_TILING_CHECK(!GetTilingCoreNum(context, core_num_temp),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get tiling core num failed."),
                  return ge::GRAPH_FAILED);

  compile_info->core_num = static_cast<int32_t>(core_num_temp);

  OP_LOGD(OP_NAME.c_str(), "op task_align=%d.", compile_info->task_align);
  OP_LOGD(OP_NAME.c_str(), "op core_num=%d.", compile_info->core_num);
  OP_LOGD(OP_NAME.c_str(), "op available_size=%d.", compile_info->available_size);
  OP_LOGD(OP_NAME.c_str(), "op indices_num=%d.", compile_info->indices_num);
  OP_LOGD(OP_NAME.c_str(), "op align_num=%d.", compile_info->align_num);
  OP_LOGD(OP_NAME.c_str(), "op after_v200=%d.", compile_info->after_v200);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Index).Tiling(Tiling4Index)
              .TilingParse<IndexCompileInfo>(TilingPrepare4Index);
}  // namespace optiling
