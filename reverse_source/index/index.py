# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
index
"""
from functools import reduce as functools_reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from tbe.common.platform.platform_info import get_soc_spec
from impl.util.platform_adapter import tbe_context
from impl.util.util_soc_common import after_v200


class Constant:
    """
    The class for constant
    """
    INT64_ALIGN = 4
    LARGE_TASK_ALIGN = 32
    TASK_ALIGN = 128
    MAX_INT32 = 2 ** 31 - 1
    BYTE_PER_BLOCK = 32
    STRIDE_OFFSET = 2
    IDX_OFFSET = 3
    TILING_DATA_NUM_ALIGN = 20
    DATA_LIMIT_FIRST_AXIS = 50000
    DATA_LIMIT_MULTI_INDEX = 500000
    RESERVED_UB_SIZE = 1024
    DTYPE_BYTES_DICT = {"float16": 2, "float32": 4, "int64": 8, "int32": 4, "uint8": 1, "int8": 1, "bool": 1}
    MAX_DIM = 10
    TAIL_SIZE_LIMIT = 32
    TILING_MODE_FIRST_AXIS = 0
    TILING_MODE_NON_FIRST_AXIS = 1
    TILING_MODE_NON_FIRST_AXIS_2D = 2


def check_indices_continuos(masks):
    masks_num = masks.get('shape')[0]
    for i in range(1, masks_num):
        if (masks.get('const_value')[i] - masks.get('const_value')[i - 1]) < 0:
            return False
    return True


def check_indices_same_shape(indices):
    for i in range(1, len(indices)):
        if indices[i].get('shape') != indices[i - 1].get('shape'):
            return False
    return True


def check_aicore(x, indices, masks):
    # if 'const_value' not in masks or len(indices) == 0, do nothing
    if not masks.get('const_value', None) or not indices:
        return True

    # indices must be continuous
    if not check_indices_continuos(masks):
        return False
    
    # indices must have same shape
    if not check_indices_same_shape(indices):
        return False

    # The input of the aicore does not support float64.
    if x.get('dtype') in ('float64', 'double'):
        return False
    
    return True


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
def check_supported(x, indexed_sizes, indexed_strides, indices, y, kernel_name="index"):
    """
    check the op support situation.
    """
    x_shape = x.get("shape")
    y_shape = y.get("shape")
    masks_shape = indexed_sizes.get("shape")
    masks_num = masks_shape[0]
    x_type = x.get("dtype").lower()
    x_bytes = Constant.DTYPE_BYTES_DICT.get(x_type, 4)
    available_size_check = (get_soc_spec("UB_SIZE") - Constant.TASK_ALIGN * 32) // x_bytes
    output_num = functools_reduce(lambda x, y: x * y, y_shape)
    tail_size = 1
    if masks_num < len(x_shape):
        tail_size = functools_reduce(lambda x, y: x * y, x_shape[masks_num:])

    if int(-1) in x_shape or int(-2) in x_shape:
        return "Unknown"
    if len(indices) < masks_num and len(x_shape) == 2:
        if available_size_check < Constant.TASK_ALIGN * (x_shape[1] + y_shape[1]):
            return False
    else:
        if not check_aicore(x, indices, indexed_sizes):
            return False
        if len(indices) < masks_num and tail_size < Constant.TAIL_SIZE_LIMIT:
            return False
        if len(indices) == len(x_shape) and output_num > Constant.DATA_LIMIT_MULTI_INDEX:
            return False
    return True


# 'pylint: disable=useless-object-inheritance,too-many-instance-attributes
@register_operator("index")
class Index(object):
    """
       Function: Index op
    """

    def __init__(self, x, indices, kernel_name):
        """
        Init Index base parameters

        Parameters
        ----------
        x : src data
        indices : indices info
        kernel_name : str
            kernel name, default value is "index"

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.used_aicore_num = tik.Dprofile().get_aicore_num()
        self.kernel_name = kernel_name
        self.x_shape = x.get("shape")
        self.indices_shape = indices[0].get("shape")
        self.after_v200 = after_v200()

        self.input_type = x.get("dtype").lower()
        if self.input_type == "bool":
            self.input_type = "int8"
        elif self.input_type == "bfloat16":
            self.input_type = "float16"
        self.input_bytes = Constant.DTYPE_BYTES_DICT.get(self.input_type, 4)
        self.align_num = Constant.BYTE_PER_BLOCK // self.input_bytes
        self.indices_type = indices[0].get("dtype").lower()
        self.indices_bytes = Constant.DTYPE_BYTES_DICT.get(self.indices_type, 8)
        self.indices_per_block = Constant.BYTE_PER_BLOCK // self.indices_bytes

        # Parameters check
        self.indices_num = len(indices)

        # create global memory objects for inputs & outputs & tiling_gm
        self.x_data_gm = self.tik_instance.Tensor(self.input_type, [Constant.MAX_INT32], name="x_data_gm",
                                                  scope=tik.scope_gm)
        self.indexed_sizes_data_gm = self.tik_instance.Tensor("int64", [8],
                                                              name="indexed_sizes_data_gm", scope=tik.scope_gm)
        self.indexed_strides_data_gm = self.tik_instance.Tensor("int64", [8],
                                                                name="indexed_strides_data_gm", scope=tik.scope_gm)
        self.input_tensors = [self.x_data_gm, self.indexed_sizes_data_gm, self.indexed_strides_data_gm]
        for idx in range(self.indices_num):
            tensor_name = "_".join(["indices_data_gm", str(idx)])
            gm_tensor = self.tik_instance.Tensor(self.indices_type, [Constant.MAX_INT32], name=tensor_name,
                                                 scope=tik.scope_gm)
            self.input_tensors.append(gm_tensor)

        self.y_data_gm = self.tik_instance.Tensor(self.input_type, [Constant.MAX_INT32], name="y_data_gm",
                                                  scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int64", [Constant.TILING_DATA_NUM_ALIGN], name="tiling_gm",
                                                  scope=tik.scope_gm)

        # init scalar objects for tiling_data
        self.input_batch_num_0 = self.tik_instance.Scalar()
        self.input_batch_num_1 = self.tik_instance.Scalar()
        self.input_batch_num_2 = self.tik_instance.Scalar()
        self.input_batch_num_3 = self.tik_instance.Scalar()
        self.input_batch_num_4 = self.tik_instance.Scalar()
        self.input_batch_num_5 = self.tik_instance.Scalar()
        self.input_batch_num_6 = self.tik_instance.Scalar()
        self.input_batch_num_7 = self.tik_instance.Scalar()
        self.input_batch_num_8 = self.tik_instance.Scalar()
        self.input_batch_num_9 = self.tik_instance.Scalar()
        self.core_num_var = self.tik_instance.Scalar()
        self.output_size = self.tik_instance.Scalar()
        self.output_num_per_batch = self.tik_instance.Scalar()
        self.output_batch_num = self.tik_instance.Scalar()
        self.input_size = self.tik_instance.Scalar()
        self.input_shape_dim_num = self.tik_instance.Scalar()
        self.index_size = self.tik_instance.Scalar()
        self.masks_num = self.tik_instance.Scalar()
        self.tiling_mode = self.tik_instance.Scalar()

        self.input_batch_num_list = [self.input_batch_num_0, self.input_batch_num_1, self.input_batch_num_2,
                                     self.input_batch_num_3, self.input_batch_num_4, self.input_batch_num_5,
                                     self.input_batch_num_6, self.input_batch_num_7, self.input_batch_num_8,
                                     self.input_batch_num_9]
        # gloable obj
        self.batch_align = self.tik_instance.Scalar()
        self.task_num = self.tik_instance.Scalar()
        # get tiling data from tiling_gm
        self.get_tiling_args()
        # compute data_align
        self.output_num_per_batch_align = self.ceil_div(self.output_num_per_batch, self.align_num) * self.align_num
        self.output_size_align = self.ceil_div(self.output_size, self.align_num) * self.align_num

        self.available_size = (get_soc_spec("UB_SIZE") - Constant.TASK_ALIGN * 32) // self.input_bytes
        self.max_elem_per_ub = ((get_soc_spec("UB_SIZE") - Constant.RESERVED_UB_SIZE -
                                 Constant.TASK_ALIGN * self.indices_num * self.indices_bytes) //
                                self.input_bytes // self.align_num * self.align_num)

    @staticmethod
    def ceil_div(val_x, val_y):
        """
        ceiling division
        """
        return (val_x + val_y - 1) // val_y
    
    @staticmethod
    def get_stride_list(self):
        """
        calculating strides
        """
        stride_list = []
        for indices_idx in range(self.indices_num):
            scalar_name = "_".join(["scalar_", str(indices_idx)])
            stride_val = self.tik_instance.Scalar(name=scalar_name, init_value=1)
            for count in range(indices_idx + Constant.STRIDE_OFFSET, Constant.MAX_DIM):
                with self.tik_instance.if_scope(self.input_batch_num_list[count] != 0):
                    stride_val.set_as(stride_val * self.input_batch_num_list[count])
            stride_list.append(stride_val)
        return stride_list

    def get_tiling_args(self):
        """get_tiling_args"""
        # apply to load tiling_data(int64)
        tiling_ub = self.tik_instance.Tensor("int64", [Constant.TILING_DATA_NUM_ALIGN], name='tiling_ub',
                                             scope=tik.scope_ubuf)
        burst_len = self.tik_instance.Scalar(init_value=Constant.TILING_DATA_NUM_ALIGN / Constant.INT64_ALIGN)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, burst_len, 0, 0)
        self.input_batch_num_list[0].set_as(tiling_ub[0])
        self.input_batch_num_list[1].set_as(tiling_ub[1])
        self.input_batch_num_list[2].set_as(tiling_ub[2])
        self.input_batch_num_list[3].set_as(tiling_ub[3])
        self.input_batch_num_list[4].set_as(tiling_ub[4])
        self.input_batch_num_list[5].set_as(tiling_ub[5])
        self.input_batch_num_list[6].set_as(tiling_ub[6])
        self.input_batch_num_list[7].set_as(tiling_ub[7])
        self.input_batch_num_list[8].set_as(tiling_ub[8])
        self.input_batch_num_list[9].set_as(tiling_ub[9])

        self.core_num_var.set_as(tiling_ub[10])
        self.output_size.set_as(tiling_ub[11])
        self.output_num_per_batch.set_as(tiling_ub[12])
        self.output_batch_num.set_as(tiling_ub[13])
        self.input_size.set_as(tiling_ub[14])
        self.input_shape_dim_num.set_as(tiling_ub[15])
        self.index_size.set_as(tiling_ub[16])
        self.masks_num.set_as(tiling_ub[17])
        self.tiling_mode.set_as(tiling_ub[18])

    def task_schedule(self):
        """task_schedule"""
        # allocate tasks according to different modes
        batch_num_per_aicore = self.tik_instance.Scalar()
        batch_tail = self.tik_instance.Scalar()
        # infer task allocation
        with self.tik_instance.if_scope(tik.all(self.tiling_mode == Constant.TILING_MODE_FIRST_AXIS,
                                                self.indices_num == 1,
                                                self.output_num_per_batch >= self.align_num,
                                                self.input_batch_num_1 <= Constant.DATA_LIMIT_FIRST_AXIS)):
            self.task_num.set_as(self.ceil_div(self.output_batch_num, Constant.LARGE_TASK_ALIGN))
        with self.tik_instance.else_scope(): 
            self.task_num.set_as(self.ceil_div(self.output_batch_num, Constant.TASK_ALIGN))
        batch_num_per_aicore.set_as(self.task_num // self.core_num_var)
        batch_tail.set_as(self.task_num % self.core_num_var)

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as i:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_FIRST_AXIS):
                with self.tik_instance.if_scope(tik.all(self.indices_num == 1, 
                                                self.input_batch_num_1 <= Constant.DATA_LIMIT_FIRST_AXIS)):
                    with self.tik_instance.if_scope(self.output_num_per_batch >= self.align_num):
                        with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                            self.index_compute_int_indices_large_num_core(i + j * self.core_num_var)
                        with self.tik_instance.if_scope(i < batch_tail):
                            self.index_compute_int_indices_large_num_core(batch_num_per_aicore * self.core_num_var + i)
                    with self.tik_instance.else_scope():                      
                        with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                            self.index_compute_int_indices_core(i + j * self.core_num_var)
                        with self.tik_instance.if_scope(i < batch_tail):
                            self.index_compute_int_indices_core(batch_num_per_aicore * self.core_num_var + i)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                        self.index_compute_first_axis_core(i + j * self.core_num_var)
                    with self.tik_instance.if_scope(i < batch_tail):
                        self.index_compute_first_axis_core(batch_num_per_aicore * self.core_num_var + i)
            with self.tik_instance.elif_scope(self.tiling_mode == Constant.TILING_MODE_NON_FIRST_AXIS_2D):
                with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                    self.index_compute_non_first_axis_2d_core(i + j * self.core_num_var)
                with self.tik_instance.if_scope(i < batch_tail):
                    self.index_compute_non_first_axis_2d_core(batch_num_per_aicore * self.core_num_var + i)
            with self.tik_instance.elif_scope(self.tiling_mode == Constant.TILING_MODE_NON_FIRST_AXIS):
                with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                    self.index_compute_non_first_axis_core(i + j * self.core_num_var)
                with self.tik_instance.if_scope(i < batch_tail):
                    self.index_compute_non_first_axis_core(batch_num_per_aicore * self.core_num_var + i)

        # add compile info for op_tiling
        tbe_context.get_context().add_compile_info(
            "vars", {
                "task_align": Constant.TASK_ALIGN,
                "core_num": self.used_aicore_num,
                "available_size": self.available_size,
                "indices_num": self.indices_num,
                "align_num": self.align_num,
                "after_v200": self.after_v200
            })

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.input_tensors,
            outputs=[self.y_data_gm],
            flowtable=[self.tiling_gm])

        return self.tik_instance

    def move_data_by_bytes(self, dst, src, size, dst_offset=0, src_offset=0):
        # use tik.data_move_pad to fix global memory out of bounds
        if dst.dtype != "int64":
            self.tik_instance.data_move_pad(dst[dst_offset], src[src_offset], 1, size, 0, 0)
        else:
            # tik.data_move_pad not support int64, reinterpret_cast_to int32
            dst_int32 = dst.reinterpret_cast_to("int32")
            src_int32 = src.reinterpret_cast_to("int32")
            self.tik_instance.data_move_pad(dst_int32[dst_offset * 2], src_int32[src_offset * 2], 1, size, 0, 0)
    
    def index_compute_first_axis_core(self, task_idx):
        """index_compute_first_axis_core"""
        task_offset = self.tik_instance.Scalar(init_value=task_idx * Constant.TASK_ALIGN)

        indices_val = self.tik_instance.Scalar()
        output_offset = self.tik_instance.Scalar()
        output_num_per_batch_align = self.tik_instance.Scalar(init_value=self.output_num_per_batch //
                                                              self.align_num * self.align_num)
        output_num_tail = self.tik_instance.Scalar(init_value=self.output_num_per_batch -
                                                   output_num_per_batch_align)
        offset = self.tik_instance.Scalar(init_value=0)
        # valid indices num
        compute_counts = self.tik_instance.Scalar(init_value=Constant.TASK_ALIGN)
        with self.tik_instance.if_scope(self.output_batch_num < (task_offset + Constant.TASK_ALIGN)):
            compute_counts.set_as(self.output_batch_num % Constant.TASK_ALIGN)

        indices_ub_list = []
        for indices_idx in range(self.indices_num):
            tensor_name = "_".join(["indices_data_ub", str(indices_idx)])
            # apply to load indices_data
            indices_ub = self.tik_instance.Tensor(self.indices_type, [compute_counts], name=tensor_name,
                                                  scope=tik.scope_ubuf)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.move_data_by_bytes(
                    indices_ub, self.input_tensors[Constant.IDX_OFFSET + indices_idx],
                    compute_counts * self.indices_bytes, src_offset=task_offset)
            else:
                self.tik_instance.data_move(
                    indices_ub, self.input_tensors[Constant.IDX_OFFSET + indices_idx][task_offset], 0, 1,
                    self.ceil_div(compute_counts, self.indices_per_block), 0, 0)
            indices_ub_list.append(indices_ub)
        
        stride_list = self.get_stride_list(self)
            
        # apply to record res
        res_ub = self.tik_instance.Tensor(self.input_type, [self.max_elem_per_ub], name="res_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor(self.input_type, [self.align_num], name="tmp_ub", scope=tik.scope_ubuf)
        
        iters_per_batch = self.tik_instance.Scalar(init_value=output_num_per_batch_align // self.max_elem_per_ub)
        tail_per_batch = self.tik_instance.Scalar(init_value=output_num_per_batch_align % self.max_elem_per_ub)
        x_gm_offset = self.tik_instance.Scalar()
        y_gm_offset = self.tik_instance.Scalar()
        burst_len = self.tik_instance.Scalar()
            
        with self.tik_instance.for_range(0, compute_counts) as batch_idx:
            output_offset.set_as(task_offset + batch_idx)
            offset.set_as(0)
            for indices_idx in range(self.indices_num):
                indices_val.set_as(indices_ub_list[indices_idx][batch_idx])
                # fix negative index
                indices_val.set_as(
                    (indices_val + self.input_batch_num_list[indices_idx + 1]) %
                    self.input_batch_num_list[indices_idx + 1])
                offset.set_as(offset + stride_list[indices_idx] * indices_val)
            # store compute_counts * output_num_per_batch in ub and move to gm at once
            with self.tik_instance.if_scope(self.output_num_per_batch < self.align_num):
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.move_data_by_bytes(
                        tmp_ub, self.x_data_gm, self.output_num_per_batch * self.input_bytes, src_offset=offset)
                else:
                    self.tik_instance.data_move(tmp_ub, self.x_data_gm[offset], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.output_num_per_batch) as inner_offset:
                    res_ub[batch_idx * self.output_num_per_batch + inner_offset].set_as(tmp_ub[inner_offset])
            # move one batch to ub and move to gm        
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, iters_per_batch) as iters_idx:
                    x_gm_offset.set_as(offset + iters_idx * self.max_elem_per_ub)
                    y_gm_offset.set_as(output_offset * self.output_num_per_batch + iters_idx * self.max_elem_per_ub)
                    burst_len.set_as(self.max_elem_per_ub // self.align_num)
                    self.tik_instance.data_move(res_ub, self.x_data_gm[x_gm_offset], 0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(self.y_data_gm[y_gm_offset], res_ub, 0, 1, burst_len, 0, 0)
                with self.tik_instance.if_scope(tail_per_batch > 0):
                    x_gm_offset.set_as(offset + iters_per_batch * self.max_elem_per_ub)
                    y_gm_offset.set_as(output_offset * self.output_num_per_batch +
                                       iters_per_batch * self.max_elem_per_ub)
                    if tbe_platform.api_check_support("tik.data_move_pad"):
                        self.move_data_by_bytes(
                            res_ub, self.x_data_gm, tail_per_batch * self.input_bytes, src_offset=x_gm_offset)
                        self.move_data_by_bytes(
                            self.y_data_gm, res_ub, tail_per_batch * self.input_bytes, dst_offset=y_gm_offset)
                    else:
                        burst_len.set_as(tail_per_batch // self.align_num)
                        self.tik_instance.data_move(res_ub, self.x_data_gm[x_gm_offset], 0, 1, burst_len, 0, 0)
                        self.tik_instance.data_move(self.y_data_gm[y_gm_offset], res_ub, 0, 1, burst_len, 0, 0)
                with self.tik_instance.if_scope(output_num_tail > 0):
                    if tbe_platform.api_check_support("tik.data_move_pad"):
                        x_gm_offset.set_as(offset + output_num_per_batch_align)
                        y_gm_offset.set_as(output_offset * self.output_num_per_batch + output_num_per_batch_align)
                        self.move_data_by_bytes(
                            res_ub, self.x_data_gm, output_num_tail * self.input_bytes, src_offset=x_gm_offset)
                        self.move_data_by_bytes(
                            self.y_data_gm, res_ub, output_num_tail * self.input_bytes, dst_offset=y_gm_offset)
                    else:
                        x_gm_offset.set_as(offset + self.output_num_per_batch - self.align_num)
                        y_gm_offset.set_as(
                            output_offset * self.output_num_per_batch + self.output_num_per_batch - self.align_num)
                        self.tik_instance.data_move(res_ub, self.x_data_gm[x_gm_offset], 0, 1, 1, 0, 0)
                        self.tik_instance.data_move(self.y_data_gm[y_gm_offset], res_ub, 0, 1, 1, 0, 0)
        with self.tik_instance.if_scope(self.output_num_per_batch < self.align_num):
            y_gm_offset.set_as(task_offset * self.output_num_per_batch)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.move_data_by_bytes(
                    self.y_data_gm, res_ub, compute_counts * self.output_num_per_batch * self.input_bytes,
                    dst_offset=y_gm_offset)
            else:
                burst_len.set_as(self.ceil_div(compute_counts * self.output_num_per_batch, self.align_num))
                self.tik_instance.data_move(self.y_data_gm[y_gm_offset], res_ub, 0, 1, burst_len, 0, 0)

    def index_compute_non_first_axis_core(self, task_idx):
        """index_compute_non_first_axis_core"""
        task_offset = self.tik_instance.Scalar(init_value=task_idx * Constant.TASK_ALIGN)
        indices_val = self.tik_instance.Scalar()
        output_offset = self.tik_instance.Scalar()
        
        output_num_per_batch_align = self.tik_instance.Scalar(init_value=self.output_num_per_batch //
                                                              self.align_num * self.align_num)
        output_num_tail = self.tik_instance.Scalar(init_value=self.output_num_per_batch % self.align_num)
        # valid batch nums
        compute_counts = self.tik_instance.Scalar(init_value=Constant.TASK_ALIGN)
        with self.tik_instance.if_scope(self.input_batch_num_list[0] < (task_offset + Constant.TASK_ALIGN)):
            compute_counts.set_as(self.input_batch_num_list[0] % Constant.TASK_ALIGN)

        stride_x = self.input_size // self.input_batch_num_list[0]            
        # calculate strides
        stride_list = self.get_stride_list(self)
        res_ub = self.tik_instance.Tensor(self.input_type, [self.max_elem_per_ub], name="res_ub", scope=tik.scope_ubuf)
        
        iters_per_batch = self.tik_instance.Scalar(init_value=output_num_per_batch_align // self.max_elem_per_ub)
        tail_per_batch = self.tik_instance.Scalar(init_value=output_num_per_batch_align % self.max_elem_per_ub)
        offset_x_start = self.tik_instance.Scalar()
        offset_x = self.tik_instance.Scalar()
        offset_y = self.tik_instance.Scalar()
        iters_per_index_batch = self.tik_instance.Scalar(init_value=self.index_size // Constant.TASK_ALIGN)
        tail_per_index_batch = self.tik_instance.Scalar(init_value=self.index_size % Constant.TASK_ALIGN)
        burst_len = self.tik_instance.Scalar(init_value=0)
        with self.tik_instance.for_range(0, iters_per_index_batch) as index_idx:
            indices_ub_list = []
            # move indices to ub
            for indices_idx in range(self.indices_num):
                tensor_name = "_".join(["indices_data_ub", str(indices_idx)])
                # apply to load indices_data
                indices_ub = self.tik_instance.Tensor(self.indices_type, [Constant.TASK_ALIGN], name=tensor_name,
                                                    scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    indices_ub, 
                    self.input_tensors[Constant.IDX_OFFSET + indices_idx][index_idx * Constant.TASK_ALIGN],
                    0, 1,
                    Constant.TASK_ALIGN // self.indices_per_block,
                    0, 0)
                indices_ub_list.append(indices_ub)
            with self.tik_instance.for_range(0, compute_counts) as batch_idx:
                output_offset.set_as(task_offset + batch_idx)
                with self.tik_instance.for_range(0, Constant.TASK_ALIGN) as i:
                    offset_x_start.set_as(output_offset * stride_x)
                    for j in range(self.indices_num):
                        indices_val.set_as(indices_ub_list[j][i])
                        # fix negative index
                        indices_val.set_as(
                            (indices_val + self.input_batch_num_list[j + 1]) % self.input_batch_num_list[j + 1])
                        offset_x_start.set_as(offset_x_start + indices_val * stride_list[j])
                    with self.tik_instance.for_range(0, iters_per_batch) as iters_idx:
                        offset_x.set_as(offset_x_start + iters_idx * self.max_elem_per_ub)
                        offset_y.set_as(output_offset * self.index_size * self.output_num_per_batch +
                                        (index_idx * Constant.TASK_ALIGN + i) * self.output_num_per_batch +
                                        iters_idx * self.max_elem_per_ub)
                        burst_len.set_as(self.max_elem_per_ub // self.align_num)
                        self.tik_instance.data_move(res_ub, self.x_data_gm[offset_x], 0, 1, burst_len, 0, 0)
                        self.tik_instance.data_move(self.y_data_gm[offset_y], res_ub, 0, 1, burst_len, 0, 0)
                    with self.tik_instance.if_scope(tail_per_batch > 0):
                        offset_x.set_as(offset_x_start + iters_per_batch * self.max_elem_per_ub)
                        offset_y.set_as(output_offset * self.index_size * self.output_num_per_batch + 
                                        (index_idx * Constant.TASK_ALIGN + i) * self.output_num_per_batch +
                                        iters_per_batch * self.max_elem_per_ub)
                        burst_len.set_as(tail_per_batch // self.align_num)
                        self.tik_instance.data_move(res_ub, self.x_data_gm[offset_x], 0, 1, burst_len, 0, 0)
                        self.tik_instance.data_move(self.y_data_gm[offset_y], res_ub, 0, 1, burst_len, 0, 0)
                    with self.tik_instance.if_scope(output_num_tail > 0):
                        offset_x.set_as(offset_x_start + self.output_num_per_batch - self.align_num)
                        offset_y.set_as(output_offset * self.index_size * self.output_num_per_batch +
                                        (index_idx * Constant.TASK_ALIGN + i) * self.output_num_per_batch +
                                        self.output_num_per_batch - self.align_num)
                        self.tik_instance.data_move(res_ub, self.x_data_gm[offset_x], 0, 1, 1, 0, 0)
                        self.tik_instance.data_move(self.y_data_gm[offset_y], res_ub, 0, 1, 1, 0, 0)
        
        with self.tik_instance.if_scope(tail_per_index_batch > 0):
            indices_ub_list = []    
            # move indices to ub
            for indices_idx in range(self.indices_num):
                tensor_name = "_".join(["indices_data_ub", str(indices_idx)])
                # apply to load indices_data
                indices_ub = self.tik_instance.Tensor(self.indices_type, [Constant.TASK_ALIGN], name=tensor_name,
                                                    scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    indices_ub, 
                    self.input_tensors[Constant.IDX_OFFSET + indices_idx][iters_per_index_batch * Constant.TASK_ALIGN],
                    0, 1,
                    self.ceil_div(tail_per_index_batch, self.indices_per_block),
                    0, 0)
                indices_ub_list.append(indices_ub)
            with self.tik_instance.for_range(0, compute_counts) as batch_idx:
                output_offset.set_as(task_offset + batch_idx)
                with self.tik_instance.for_range(0, tail_per_index_batch) as i:
                    offset_x_start.set_as(output_offset * stride_x)
                    for j in range(self.indices_num):
                        indices_val.set_as(indices_ub_list[j][i])
                        # fix negative index
                        indices_val.set_as(
                            (indices_val + self.input_batch_num_list[j + 1]) % self.input_batch_num_list[j + 1])
                        offset_x_start.set_as(offset_x_start + indices_val * stride_list[j])
                    with self.tik_instance.for_range(0, iters_per_batch) as iters_idx:
                        offset_x.set_as(offset_x_start + iters_idx * self.max_elem_per_ub)
                        offset_y.set_as(output_offset * self.index_size * self.output_num_per_batch + 
                                        (iters_per_index_batch * Constant.TASK_ALIGN + i) * self.output_num_per_batch +
                                        iters_idx * self.max_elem_per_ub)
                        burst_len.set_as(self.max_elem_per_ub // self.align_num)
                        self.tik_instance.data_move(res_ub, self.x_data_gm[offset_x], 0, 1, burst_len, 0, 0)
                        self.tik_instance.data_move(self.y_data_gm[offset_y], res_ub, 0, 1, burst_len, 0, 0)
                    with self.tik_instance.if_scope(tail_per_batch > 0):
                        offset_x.set_as(offset_x_start + iters_per_batch * self.max_elem_per_ub)
                        offset_y.set_as(output_offset * self.index_size * self.output_num_per_batch + 
                                        (iters_per_index_batch * Constant.TASK_ALIGN + i) * self.output_num_per_batch +
                                        iters_per_batch * self.max_elem_per_ub)
                        burst_len.set_as(tail_per_batch // self.align_num)
                        self.tik_instance.data_move(res_ub, self.x_data_gm[offset_x], 0, 1, burst_len, 0, 0)
                        self.tik_instance.data_move(self.y_data_gm[offset_y], res_ub, 0, 1, burst_len, 0, 0)
                    with self.tik_instance.if_scope(output_num_tail > 0):
                        offset_x.set_as(offset_x_start + self.output_num_per_batch - self.align_num)
                        offset_y.set_as(output_offset * self.index_size * self.output_num_per_batch +
                                        (iters_per_index_batch * Constant.TASK_ALIGN + i) * self.output_num_per_batch +
                                        self.output_num_per_batch - self.align_num)
                        self.tik_instance.data_move(res_ub, self.x_data_gm[offset_x], 0, 1, 1, 0, 0)
                        self.tik_instance.data_move(self.y_data_gm[offset_y], res_ub, 0, 1, 1, 0, 0)

    def index_compute_non_first_axis_2d_core(self, task_idx):
        """index_compute_int_indices_core"""
        data_offset = task_idx * Constant.TASK_ALIGN
        x_val = self.tik_instance.Scalar(self.input_type)
        indices_val = self.tik_instance.Scalar("int64")
        input_num_per_batch = self.tik_instance.Scalar("int64", init_value=self.input_batch_num_1)
        # apply to load indices_data
        indices_ub = self.tik_instance.Tensor(self.indices_type, [self.index_size], name="indices_ub",
                                              scope=tik.scope_ubuf)

        self.tik_instance.data_move(indices_ub, self.input_tensors[Constant.IDX_OFFSET][0],
                                    0, 1,
                                    self.ceil_div(self.index_size, self.indices_per_block),
                                    0, 0)

        # apply to record res
        res_ub = self.tik_instance.Tensor(self.input_type, [Constant.TASK_ALIGN * self.index_size], 
                                          name="res_ub", scope=tik.scope_ubuf)
        # apply to load x_data
        x_ub = self.tik_instance.Tensor(self.input_type, [Constant.TASK_ALIGN, input_num_per_batch], 
                                        name="x_ub", scope=tik.scope_ubuf)
        
        # valid indices num
        compute_counts = self.tik_instance.Scalar('int64', init_value=Constant.TASK_ALIGN)
        with self.tik_instance.if_scope(self.output_batch_num < (data_offset + Constant.TASK_ALIGN)):
            compute_counts.set_as(self.output_batch_num % Constant.TASK_ALIGN)

        self.tik_instance.data_move(
            x_ub, self.x_data_gm[data_offset * input_num_per_batch], 0, 1,
            self.ceil_div(compute_counts * input_num_per_batch, self.align_num), 0, 0)

        with self.tik_instance.for_range(0, self.index_size) as indices_idx:
            # record index val
            indices_val.set_as(indices_ub[indices_idx])
            # fix negative index
            indices_val.set_as((indices_val + input_num_per_batch) % input_num_per_batch)

            with self.tik_instance.for_range(0, compute_counts) as compute_idx:
                x_val.set_as(x_ub[indices_val + compute_idx * input_num_per_batch])
                res_ub[indices_idx + compute_idx * self.index_size].set_as(x_val)
    
        # valid indices num
        move_counts = self.tik_instance.Scalar('int64', init_value=compute_counts * self.index_size)
        self.tik_instance.data_move(
                self.y_data_gm[data_offset * self.index_size], res_ub, 0, 1,
                self.ceil_div(move_counts, self.align_num), 0, 0)
    
    def index_compute_int_indices_core(self, task_idx):
        """index_compute_int_indices_core"""
        # infer data apply batch limit by ub_size
        self.batch_align.set_as(
            (self.available_size - Constant.TASK_ALIGN * self.output_num_per_batch) \
            // self.output_num_per_batch)
        data_offset = self.tik_instance.Scalar(init_value=task_idx * Constant.TASK_ALIGN)
        x_rounds = self.tik_instance.Scalar(init_value=self.input_batch_num_1 // self.batch_align)
        x_tail = self.tik_instance.Scalar(init_value=self.input_batch_num_1 % self.batch_align)

        x_val = self.tik_instance.Scalar(self.input_type)
        indices_val = self.tik_instance.Scalar()

        # valid indices num
        compute_counts = self.tik_instance.Scalar(init_value=Constant.TASK_ALIGN)
        with self.tik_instance.if_scope(self.output_batch_num < (data_offset + Constant.TASK_ALIGN)):
            compute_counts.set_as(self.output_batch_num % Constant.TASK_ALIGN)

        # apply to load indices_data
        indices_ub = self.tik_instance.Tensor(self.indices_type, [compute_counts], name="indices_ub",
                                              scope=tik.scope_ubuf)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.move_data_by_bytes(
                indices_ub, self.input_tensors[Constant.IDX_OFFSET], compute_counts * self.indices_bytes,
                src_offset=data_offset)
        else:
            self.tik_instance.data_move(
                indices_ub, self.input_tensors[Constant.IDX_OFFSET][data_offset], 0, 1,
                self.ceil_div(compute_counts, self.indices_per_block), 0, 0)

        # apply to record res
        res_ub = self.tik_instance.Tensor(self.input_type, [compute_counts, self.output_num_per_batch],
                                          name="res_ub", scope=tik.scope_ubuf)
        # apply to load x_data
        x_ub = self.tik_instance.Tensor(self.input_type, [self.batch_align, self.output_num_per_batch], name="x_ub",
                                        scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, x_rounds) as x_part_idx:
            self.tik_instance.data_move(
                x_ub, self.x_data_gm[self.batch_align * x_part_idx * self.output_num_per_batch], 0, 1,
                (self.batch_align * self.output_num_per_batch + self.align_num - 1) // self.align_num, 0, 0)
            with self.tik_instance.for_range(0, compute_counts) as compute_idx:
                indices_val.set_as(indices_ub[compute_idx])
                # fix negative index
                indices_val.set_as((indices_val + self.input_batch_num_1) % self.input_batch_num_1)
                indices_val.set_as(indices_val - x_part_idx * self.batch_align)
                # check if the indices_val in the current x_data
                with self.tik_instance.if_scope(tik.all(indices_val >= 0, indices_val < self.batch_align)):
                    # compute the offset in x and y
                    offset_1 = indices_val * self.output_num_per_batch
                    offset_2 = compute_idx * self.output_num_per_batch
                    with self.tik_instance.for_range(0, self.output_num_per_batch) as last_dim_idx:
                        x_val.set_as(x_ub[offset_1 + last_dim_idx])
                        res_ub[offset_2 + last_dim_idx].set_as(x_val)
        with self.tik_instance.if_scope(x_tail > 0):
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.move_data_by_bytes(
                    x_ub, self.x_data_gm, x_tail * self.output_num_per_batch * self.input_bytes,
                    src_offset=self.batch_align * x_rounds * self.output_num_per_batch)
            else:
                self.tik_instance.data_move(
                    x_ub, self.x_data_gm[self.batch_align * x_rounds * self.output_num_per_batch], 0, 1,
                    (x_tail * self.output_num_per_batch + self.align_num - 1) // self.align_num, 0, 0)
            with self.tik_instance.for_range(0, compute_counts) as compute_idx:
                indices_val.set_as(indices_ub[compute_idx])
                # fix negative index
                indices_val.set_as((indices_val + self.input_batch_num_1) % self.input_batch_num_1)
                indices_val.set_as(indices_val - x_rounds * self.batch_align)
                # check if the indices_val in the current x_data
                with self.tik_instance.if_scope(tik.all(indices_val >= 0, indices_val < self.batch_align)):
                    # compute the offset in x and y
                    offset_1 = indices_val * self.output_num_per_batch
                    offset_2 = compute_idx * self.output_num_per_batch
                    with self.tik_instance.for_range(0, self.output_num_per_batch) as last_dim_idx:
                        x_val.set_as(x_ub[offset_1 + last_dim_idx])
                        res_ub[offset_2 + last_dim_idx].set_as(x_val)

        # Move data outward according to data volume
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.move_data_by_bytes(
                self.y_data_gm, res_ub, compute_counts * self.output_num_per_batch * self.input_bytes,
                dst_offset=data_offset * self.output_num_per_batch)
        else:
            self.tik_instance.data_move(
                self.y_data_gm[data_offset * self.output_num_per_batch], res_ub, 0, 1,
                (compute_counts * self.output_num_per_batch + self.align_num - 1) // self.align_num, 0, 0)


    def index_compute_int_indices_large_num_core(self, task_idx):
        """index_compute_int_indices_large_num_core"""
        # infer data apply batch limit by ub_size
        self.batch_align.set_as(self.available_size // 2 // self.align_num * self.align_num)
        data_offset = self.tik_instance.Scalar(init_value=task_idx * Constant.LARGE_TASK_ALIGN)
        indices_val = self.tik_instance.Scalar()
        # whole_block
        x_rounds = self.tik_instance.Scalar(init_value=self.output_num_per_batch // self.batch_align)
        whole_block = self.tik_instance.Scalar(init_value=x_rounds * self.batch_align)
        # tail_block
        tail_block = self.tik_instance.Scalar(init_value=self.output_num_per_batch % self.batch_align)

        # valid indices num
        compute_counts = self.tik_instance.Scalar(init_value=Constant.LARGE_TASK_ALIGN)
        with self.tik_instance.if_scope(self.output_batch_num < (data_offset + Constant.LARGE_TASK_ALIGN)):
            compute_counts.set_as(self.output_batch_num % Constant.LARGE_TASK_ALIGN)

        # apply to load indices_data
        indices_ub = self.tik_instance.Tensor(self.indices_type, [compute_counts + 1], 
                                              name="indices_ub", scope=tik.scope_ubuf)
        
        # apply to record res
        res_ub = self.tik_instance.Tensor(self.input_type, [self.batch_align], name="res_ub", scope=tik.scope_ubuf)
        tail_ub = self.tik_instance.Tensor(self.input_type, [self.align_num], name="tail_ub", scope=tik.scope_ubuf)
        output_offset = self.tik_instance.Scalar()

        with self.tik_instance.if_scope(Constant.LARGE_TASK_ALIGN * (1 + task_idx) < self.output_batch_num):
            self.tik_instance.data_move(indices_ub, self.input_tensors[Constant.IDX_OFFSET][data_offset], 0, 1,
                                        compute_counts // self.indices_per_block + 1, 0, 0)
        with self.tik_instance.else_scope():
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.move_data_by_bytes(
                    indices_ub, self.input_tensors[Constant.IDX_OFFSET],
                    compute_counts * self.indices_bytes, src_offset=data_offset)
            else:
                self.tik_instance.data_move(
                    indices_ub, self.input_tensors[Constant.IDX_OFFSET][data_offset], 0, 1,
                    (compute_counts + self.indices_per_block - 1) // self.indices_per_block, 0, 0)

        with self.tik_instance.for_range(0, Constant.LARGE_TASK_ALIGN) as current_idx:
            output_offset.set_as(current_idx + data_offset)
            with self.tik_instance.if_scope(output_offset < self.output_batch_num):
                indices_val.set_as(indices_ub[current_idx])
                # fix negative index
                indices_val.set_as((indices_val + self.input_batch_num_1) % self.input_batch_num_1)
                with self.tik_instance.for_range(0, x_rounds) as x_part_idx:
                    self.tik_instance.data_move(
                        res_ub, 
                        self.x_data_gm[indices_val * self.output_num_per_batch + x_part_idx * self.batch_align],
                        0, 1, (self.batch_align + self.align_num - 1) // self.align_num, 0, 0)
                    self.tik_instance.data_move(
                        self.y_data_gm[output_offset * self.output_num_per_batch + x_part_idx * self.batch_align],
                        res_ub, 0, 1, (self.batch_align + self.align_num - 1) // self.align_num, 0, 0)
                with self.tik_instance.if_scope(tail_block > 0):
                    if tbe_platform.api_check_support("tik.data_move_pad"):
                        self.move_data_by_bytes(
                            res_ub, self.x_data_gm, tail_block * self.input_bytes,
                            src_offset=indices_val * self.output_num_per_batch + whole_block)
                    else:
                        self.tik_instance.data_move(
                            res_ub, self.x_data_gm[indices_val * self.output_num_per_batch + whole_block],
                            0, 1, (tail_block + self.align_num - 1) // self.align_num, 0, 0)

                    with self.tik_instance.if_scope(
                            tik.all(current_idx == Constant.LARGE_TASK_ALIGN - 1, tail_block % self.align_num > 0,
                                    Constant.LARGE_TASK_ALIGN * (1 + task_idx) < self.output_batch_num)):
                        indices_val.set_as(indices_ub[Constant.LARGE_TASK_ALIGN])
                        # fix negative index
                        indices_val.set_as((indices_val + self.input_batch_num_1) % self.input_batch_num_1)

                        if tbe_platform.api_check_support("tik.data_move_pad"):
                            self.move_data_by_bytes(
                                tail_ub, self.x_data_gm,
                                tail_block * self.input_bytes, src_offset=indices_val * self.output_num_per_batch)
                        else:
                            self.tik_instance.data_move(
                                tail_ub, self.x_data_gm[indices_val * self.output_num_per_batch], 0, 1, 1, 0, 0)
                        with self.tik_instance.for_range(0, self.align_num) as tail_idx:
                            res_ub[tail_block + tail_idx].set_as(tail_ub[tail_idx])

                    if tbe_platform.api_check_support("tik.data_move_pad"):
                        self.move_data_by_bytes(
                            self.y_data_gm, res_ub, tail_block * self.input_bytes,
                            dst_offset=output_offset * self.output_num_per_batch + whole_block)
                    else:
                        self.tik_instance.data_move(
                            self.y_data_gm[output_offset * self.output_num_per_batch + whole_block],
                            res_ub, 0, 1, (tail_block + self.align_num - 1) // self.align_num, 0, 0)


# 'pylint: disable=too-many-arguments, too-many-instance-attributes，unused-argument, too-many-locals, too-many-lines
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def index(x, indexed_sizes, indexed_strides, indices, y, kernel_name="index"):
    """
    index

    Parameters
    ----------
    x : dict
        shape and dtype of input x
    indexed_sizes : dict
        shape and dtype of input indexed_sizes, type only support int64
    indexed_strides : dict
        shape and dtype of input indexed_strides, type only support int64
    indices : dict
        shape and dtype of indices, type only support int64
    y : dict
        shape and dtype of output y, should be same shape and type as input x
    kernel_name : str
        kernel name, default value is "index"

    Returns
    -------
    None
    """
    op_obj = Index(x, indices, kernel_name)
    return op_obj.task_schedule()
