#!/usr/bin/env python3
"""
lingxi_profiling_runner.py — ACL-native msprof simulator profiling for Lingxi operators.

Lingxi operators (built via build_ascendc.py) use pybind11 .so files that depend on
libtorch_npu.so. Under msprof simulator, this dependency cannot be resolved because
msprof replaces ACL runtime libraries via LD_LIBRARY_PATH injection.

This script provides an alternative profiling path:
1. Build a standalone .so from libkernels.a (no torch_npu dependency)
2. Generate an ACL-native Python script using ctypes (no torch_npu import)
3. Run msprof simulator with the generated script

Usage:
    python3 lingxi_profiling_runner.py \
        --kernel-dir /path/to/task/kernel \
        --output-dir /path/to/output/sim_profiling \
        [--soc-version Ascend910B3] \
        [--timeout 120]

    # Or as a Python module:
    from lingxi_profiling_runner import run_cv_simulator_profiling
    result = run_cv_simulator_profiling(kernel_dir="/path/to/task/kernel")
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def parse_pybind11_cpp(pybind_path: str) -> dict:
    """
    Parse pybind11.cpp to extract the kernel launch function signature.

    Returns:
        {
            "kernel_fn": "int8_matmul_scale_do",
            "arg_names": ["blockDim", "stream", "a", "b", "scale", "c", "workspace", "tiling"],
            "arg_types": ["uint32_t", "void*", "uint8_t*", ...],
            "n_device_args": 6,   # number of device pointer args (excluding blockDim, stream)
            "blockDim_value": 2,  # from usedCoreNum in source
            "tiling_struct": "Int8MatmulScaleTiling",
            "tiling_header": "int8_matmul_scale_tiling.h",
        }
    """
    src = Path(pybind_path).read_text(encoding="utf-8")

    # Find extern "C" declaration
    # Handles multi-line declarations
    extern_pattern = re.compile(
        r'extern\s+"C"\s+void\s+(\w+)\s*\((.*?)\)\s*;',
        re.DOTALL,
    )
    m = extern_pattern.search(src)
    if not m:
        raise ValueError(f"No extern \"C\" void function found in {pybind_path}")

    kernel_fn = m.group(1)
    raw_args = m.group(2).strip()

    # Parse argument list
    arg_names = []
    arg_types = []
    for arg_str in raw_args.split(","):
        arg_str = arg_str.strip()
        if not arg_str:
            continue
        # Split type and name: "uint32_t blockDim" or "void *stream" or "uint8_t *a"
        parts = arg_str.replace("*", "* ").split()
        if len(parts) >= 2:
            arg_type = " ".join(parts[:-1]).replace(" *", "*").replace("* ", "*")
            arg_name = parts[-1].lstrip("*")
            arg_names.append(arg_name)
            arg_types.append(arg_type)

    n_device_args = len(arg_names) - 2  # minus blockDim and stream

    # Extract usedCoreNum (blockDim value)
    # Handles: usedCoreNum = 32 (literal), usedCoreNum = mNum (variable)
    blockDim_value = None
    blockDim_expr = None  # Python expression for generated script

    # Try literal: usedCoreNum = 32
    core_literal = re.search(r'usedCoreNum\s*=\s*(\d+)\s*;', src)
    if core_literal:
        blockDim_value = int(core_literal.group(1))
    else:
        # Try variable: usedCoreNum = mNum;
        core_var = re.search(r'usedCoreNum\s*=\s*([a-zA-Z_]\w*)\s*;', src)
        if core_var:
            var_name = core_var.group(1)
            # Trace variable definition: int32_t mNum = M / DEFAULT_BASE_M;
            var_def = re.search(
                rf'(?:int32_t|uint32_t|int)\s+{var_name}\s*=\s*(.+?)\s*;', src
            )
            if var_def:
                expr = var_def.group(1).strip()
                # Convert C expression to Python: M / DEFAULT_BASE_M → M // DEFAULT_BASE_M
                blockDim_expr = expr.replace("/", "//")

    if blockDim_value is None and blockDim_expr is None:
        blockDim_value = 2  # safe default

    # Extract tiling struct name and header
    tiling_struct = None
    tiling_header = None
    struct_match = re.search(r'reinterpret_cast<(\w+)\s*\*>', src)
    if struct_match:
        tiling_struct = struct_match.group(1)
    header_match = re.search(r'#include\s+"(\w+_tiling\.h)"', src)
    if header_match:
        tiling_header = header_match.group(1)

    # Extract computed tiling field assignments from pybind11.cpp
    # e.g., tp->nTiles = N / DEFAULT_BASE_N;  or  int32_t nTiles = N / DEFAULT_BASE_N;
    computed_tiling_fields = {}
    for cm in re.finditer(
        r'(?:tp->|(?:int32_t|uint32_t|int)\s+)(\w+)\s*=\s*(\w+)\s*/\s*(\w+)\s*;', src
    ):
        field, numerator, denominator = cm.group(1), cm.group(2), cm.group(3)
        computed_tiling_fields[field] = (numerator, denominator)

    return {
        "kernel_fn": kernel_fn,
        "arg_names": arg_names,
        "arg_types": arg_types,
        "n_device_args": n_device_args,
        "blockDim_value": blockDim_value,
        "blockDim_expr": blockDim_expr,
        "tiling_struct": tiling_struct,
        "tiling_header": tiling_header,
        "computed_tiling_fields": computed_tiling_fields,
    }


def parse_tiling_header(header_path: str) -> dict:
    """
    Parse tiling header to extract struct fields and constexpr defaults.

    Returns:
        {
            "struct_name": "Int8MatmulScaleTiling",
            "fields": [("M", "int32_t"), ("N", "int32_t"), ...],
            "defaults": {"DEFAULT_BASE_M": 128, "DEFAULT_BASE_N": 256, ...},
            "pack_format": "<6i",   # struct.pack format
            "pack_size": 24,        # bytes
        }
    """
    src = Path(header_path).read_text(encoding="utf-8")

    # Extract constexpr values
    defaults = {}
    for m in re.finditer(r'constexpr\s+\w+\s+(\w+)\s*=\s*(\d+)', src):
        defaults[m.group(1)] = int(m.group(2))

    # Extract struct fields
    struct_name = None
    struct_match = re.search(r'struct\s+(\w+)\s*\{(.*?)\}', src, re.DOTALL)
    if not struct_match:
        return {
            "struct_name": None,
            "fields": [],
            "defaults": defaults,
            "pack_format": "",
            "pack_size": 0,
        }

    struct_name = struct_match.group(1)
    body = struct_match.group(2)

    fields = []
    type_to_fmt = {
        "int32_t": "i",
        "uint32_t": "I",
        "int64_t": "q",
        "uint64_t": "Q",
        "float": "f",
        "double": "d",
        "int16_t": "h",
        "uint16_t": "H",
        "int8_t": "b",
        "uint8_t": "B",
    }

    for line in body.split(";"):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        # "int32_t M" or "int32_t M, N"
        parts = line.split()
        if len(parts) < 2:
            continue
        field_type = parts[0]
        for name_part in parts[1:]:
            name = name_part.strip().rstrip(",")
            if name and name[0].isalpha():
                fields.append((name, field_type))

    # Build pack format
    fmt_chars = []
    for _, ftype in fields:
        fmt_chars.append(type_to_fmt.get(ftype, "i"))  # default to int32

    pack_format = "<" + "".join(fmt_chars)
    pack_size = struct.calcsize(pack_format)

    return {
        "struct_name": struct_name,
        "fields": fields,
        "defaults": defaults,
        "pack_format": pack_format,
        "pack_size": pack_size,
    }


def parse_model_inputs(kernel_dir: str) -> dict:
    """
    Parse pybind11.cpp to extract input tensor info: names, dtypes, shapes.

    Shape inference strategy:
    1. Extract input tensor names from run_xxx() params
    2. Extract dtypes from TORCH_CHECK scalar_type assertions
    3. Extract ndim from TORCH_CHECK dim() assertions
    4. Extract dim_vars: e.g. m = a.sizes()[0] → a has shape[0] = M
    5. Build shape tuples from dim_vars

    Returns:
        {
            "inputs": [
                {"name": "a", "dtype": "int8", "numpy_dtype": "np.int8",
                 "ndim": 2, "shape_expr": "(M, K)"},
                ...
            ],
            "output_dtype_bytes": 2,
            "dim_vars": {"m": ("a", 0), "n": ("b", 1), "k": ("a", 1)},
        }
    """
    kernel_dir = Path(kernel_dir)

    pybind_path = kernel_dir / "pybind11.cpp"
    if not pybind_path.exists():
        return {"inputs": [], "dim_vars": {}, "output_dtype_bytes": 2}

    pybind_src = pybind_path.read_text(encoding="utf-8")

    # Step 1: Extract input tensor names
    inputs = []
    run_fn = re.search(
        r'(?:at::Tensor|void)\s+run_\w+\((.*?)\)',
        pybind_src, re.DOTALL,
    )
    if run_fn:
        param_str = run_fn.group(1)
        for pm in re.finditer(r'const\s+at::Tensor\s*&\s*(\w+)', param_str):
            inputs.append({"name": pm.group(1)})

    # Step 2: Extract dtypes from TORCH_CHECK
    dtype_map = {
        "kChar": ("int8", "np.int8"),
        "kByte": ("uint8", "np.uint8"),
        "kShort": ("int16", "np.int16"),
        "kInt": ("int32", "np.int32"),
        "kLong": ("int64", "np.int64"),
        "kHalf": ("float16", "np.float16"),
        "kFloat": ("float32", "np.float32"),
        "kDouble": ("float64", "np.float64"),
        "kBFloat16": ("bfloat16", "np.uint16"),  # numpy has no bf16; uint16 has same 2-byte size, sufficient for profiling
    }

    for inp in inputs:
        check = re.search(
            rf'{inp["name"]}\.scalar_type\(\)\s*==\s*at::(\w+)',
            pybind_src,
        )
        if check:
            key = check.group(1)
            if key in dtype_map:
                inp["dtype"], inp["numpy_dtype"] = dtype_map[key]
        if "dtype" not in inp:
            inp["dtype"] = "float16"
            inp["numpy_dtype"] = "np.float16"

    # Step 3: Extract ndim from TORCH_CHECK(x.dim() == N)
    for inp in inputs:
        dim_check = re.search(
            rf'{inp["name"]}\.dim\(\)\s*==\s*(\d+)',
            pybind_src,
        )
        if dim_check:
            inp["ndim"] = int(dim_check.group(1))

    # Step 4: Extract dim_vars: variable = tensor.sizes()[index]
    dim_vars = {}
    for dm in re.finditer(
        r'(?:uint32_t|int32_t|int)\s+(\w+)\s*=\s*(\w+)\.sizes\(\)\[(\d+)\]',
        pybind_src,
    ):
        var_name = dm.group(1).lower()
        tensor_name = dm.group(2)
        dim_index = int(dm.group(3))
        dim_vars[var_name] = (tensor_name, dim_index)

    # Step 5: Build shape info per input from dim_vars
    # For each input, collect all (dim_index, var_name) mappings
    for inp in inputs:
        name = inp["name"]
        ndim = inp.get("ndim", 2)
        shape_parts = [None] * ndim

        for var_name, (tensor_name, dim_index) in dim_vars.items():
            if tensor_name == name and dim_index < ndim:
                shape_parts[dim_index] = var_name.upper()

        # Also check equality constraints: a.sizes()[1] == b.sizes()[0]
        for eq_match in re.finditer(
            rf'(\w+)\.sizes\(\)\[(\d+)\]\s*==\s*(\w+)\.sizes\(\)\[(\d+)\]',
            pybind_src,
        ):
            t1, d1, t2, d2 = eq_match.group(1), int(eq_match.group(2)), eq_match.group(3), int(eq_match.group(4))
            if t1 == name and d1 < ndim and shape_parts[d1] is None:
                # Find what var maps to t2[d2]
                for vn, (tn, di) in dim_vars.items():
                    if tn == t2 and di == d2:
                        shape_parts[d1] = vn.upper()
                        break
            if t2 == name and d2 < ndim and shape_parts[d2] is None:
                for vn, (tn, di) in dim_vars.items():
                    if tn == t1 and di == d1:
                        shape_parts[d2] = vn.upper()
                        break

        # Fill unknowns with the first available shape key
        for i in range(ndim):
            if shape_parts[i] is None:
                shape_parts[i] = "N"  # default

        inp["shape_parts"] = shape_parts
        inp["shape_expr"] = "(" + ", ".join(shape_parts) + ")"

    # Step 6: Extract output tensor dtype
    output_match = re.search(
        r'at::Tensor\s+\w+\s*=\s*at::empty\(\{.*?\}.*?dtype\(at::(\w+)\)',
        pybind_src,
    )
    output_dtype_bytes = 2  # default fp16
    if output_match:
        dtype_key = output_match.group(1)
        size_map = {"kHalf": 2, "kFloat": 4, "kChar": 1, "kByte": 1, "kInt": 4}
        output_dtype_bytes = size_map.get(dtype_key, 2)

    return {
        "inputs": inputs,
        "dim_vars": dim_vars,
        "output_dtype_bytes": output_dtype_bytes,
    }


# --------------------------------------------------------------------------- #
# ACL Script Generation
# --------------------------------------------------------------------------- #

def generate_acl_profiling_script(
    kernel_dir: str,
    pybind_info: dict,
    tiling_info: dict,
    model_info: dict,
    shapes: dict = None,
) -> str:
    """
    Generate an ACL-native Python profiling script using ctypes.

    Args:
        kernel_dir: Path to kernel/ directory
        pybind_info: Output from parse_pybind11_cpp()
        tiling_info: Output from parse_tiling_header()
        model_info: Output from parse_model_inputs()
        shapes: Optional override shapes, e.g. {"M": 1024, "N": 1024, "K": 1024}

    Returns:
        Python script content as string
    """
    kernel_fn = pybind_info["kernel_fn"]
    n_device_args = pybind_info["n_device_args"]
    blockDim_value = pybind_info.get("blockDim_value")
    blockDim_expr = pybind_info.get("blockDim_expr")
    computed_tiling = pybind_info.get("computed_tiling_fields", {})
    arg_names = pybind_info["arg_names"][2:]  # skip blockDim, stream

    # Default shapes (can be overridden)
    if shapes is None:
        shapes = {"M": 1024, "N": 1024, "K": 1024}

    # Build tiling data
    tiling_fields = tiling_info["fields"]
    tiling_defaults = tiling_info["defaults"]
    pack_format = tiling_info["pack_format"]

    # Build a resolved values dict for computed field lookups
    resolved = dict(shapes)
    for dk, dv in tiling_defaults.items():
        resolved[dk] = dv

    # Map field names to values
    tiling_values = []
    tiling_value_strs = []
    for field_name, field_type in tiling_fields:
        val = None
        # Priority 1: direct shape match
        if field_name in shapes:
            val = shapes[field_name]
        # Priority 2: constexpr default match (e.g., DEFAULT_BASE_M for baseM)
        elif f"DEFAULT_{field_name.upper()}" in tiling_defaults:
            val = tiling_defaults[f"DEFAULT_{field_name.upper()}"]
        elif f"DEFAULT_BASE_{field_name.replace('base', '').upper()}" in tiling_defaults:
            val = tiling_defaults[f"DEFAULT_BASE_{field_name.replace('base', '').upper()}"]
        elif field_name in tiling_defaults:
            val = tiling_defaults[field_name]
        # Priority 3: computed field from pybind11.cpp (e.g., nTiles = N / DEFAULT_BASE_N)
        elif field_name in computed_tiling:
            numerator, denominator = computed_tiling[field_name]
            num_val = resolved.get(numerator)
            den_val = resolved.get(denominator)
            if num_val is not None and den_val is not None and den_val != 0:
                val = num_val // den_val
        # Priority 4: fuzzy match against defaults
        if val is None:
            for dk, dv in tiling_defaults.items():
                if field_name.upper() in dk.upper():
                    val = dv
                    break
        if val is None:
            val = 128  # safe default
        tiling_values.append(val)
        tiling_value_strs.append(str(val))
        resolved[field_name] = val  # make available for subsequent computed fields

    # Detect output arg name from signature (not hardcoded)
    output_arg_name = None
    input_names_set = {inp["name"] for inp in model_info.get("inputs", [])}
    for an in arg_names:
        if an in ("c", "y", "output", "out", "result", "dst"):
            output_arg_name = an
            break
    if output_arg_name is None:
        # Heuristic: first arg that is not workspace/tiling/input
        for an in arg_names:
            if an not in ("workspace", "ws", "w", "tiling") and an not in input_names_set:
                output_arg_name = an
                break
    if output_arg_name is None:
        output_arg_name = "output"

    # Build input_ptr_names and cleanup_ptrs using detected output arg
    input_ptr_names = []
    cleanup_ptrs = []
    for arg_name in arg_names:
        if arg_name == "tiling":
            input_ptr_names.append("tiling_dev")
            cleanup_ptrs.append("tiling_dev")
        elif arg_name in ("workspace", "ws", "w"):
            input_ptr_names.append("ws_dev")
            cleanup_ptrs.append("ws_dev")
        else:
            input_ptr_names.append(f"{arg_name}_dev")
            cleanup_ptrs.append(f"{arg_name}_dev")

    # Resolve blockDim for the generated script
    blockDim_value = pybind_info.get("blockDim_value")
    blockDim_expr = pybind_info.get("blockDim_expr")
    if blockDim_value is not None:
        blockdim_line = f"    blockDim = {blockDim_value}"
    elif blockDim_expr:
        bd_expr = blockDim_expr
        for sk, sv in shapes.items():
            bd_expr = bd_expr.replace(sk, str(sv))
        for dk, dv in tiling_defaults.items():
            bd_expr = bd_expr.replace(dk, str(dv))
        try:
            blockDim_value = eval(bd_expr)
            blockdim_line = f"    blockDim = {blockDim_value}  # computed: {blockDim_expr}"
        except Exception:
            blockdim_line = "    blockDim = 2  # WARNING: could not evaluate blockDim expr"
    else:
        blockdim_line = "    blockDim = 2  # default fallback"

    # Auto shape reduction for msprof simulator (large shapes cause timeout)
    MAX_SIMULATOR_ELEMENTS = 512 * 1024  # 512K elements
    total_elements = 1
    for v in shapes.values():
        total_elements *= v
    sim_shapes = dict(shapes)
    if total_elements > MAX_SIMULATOR_ELEMENTS:
        scale = (MAX_SIMULATOR_ELEMENTS / total_elements) ** (1.0 / len(shapes))
        for k in sim_shapes:
            align = 64
            for dk, dv in tiling_defaults.items():
                if dk.startswith("DEFAULT_BASE_") and dk.replace("DEFAULT_BASE_", "") in k.upper():
                    align = dv
                    break
            sim_shapes[k] = max((int(sim_shapes[k] * scale) // align) * align, align)
        # Recompute tiling values for reduced shapes
        for i, (field_name, _) in enumerate(tiling_fields):
            if field_name in sim_shapes:
                tiling_values[i] = sim_shapes[field_name]
                tiling_value_strs[i] = str(sim_shapes[field_name])
            elif field_name in computed_tiling:
                num_name, den_name = computed_tiling[field_name]
                nv = sim_shapes.get(num_name, resolved.get(num_name))
                dv = sim_shapes.get(den_name, resolved.get(den_name))
                if nv and dv and dv != 0:
                    tiling_values[i] = nv // dv
                    tiling_value_strs[i] = str(nv // dv)
        # Recompute blockDim for reduced shapes
        if blockDim_expr:
            bd_expr2 = blockDim_expr
            for sk, sv in sim_shapes.items():
                bd_expr2 = bd_expr2.replace(sk, str(sv))
            for dk, dv in tiling_defaults.items():
                bd_expr2 = bd_expr2.replace(dk, str(dv))
            try:
                bd_val = eval(bd_expr2)
                blockdim_line = f"    blockDim = {bd_val}  # reduced shape, computed: {blockDim_expr}"
            except Exception:
                pass

    # Absolute path for standalone .so (msprof changes cwd to /)
    abs_so_path = os.path.join(str(Path(kernel_dir).resolve()), "build", "libkernel_standalone.so")

    # Workspace size heuristic
    ws_depth = tiling_defaults.get("WORKSPACE_DEPTH", 4)
    baseM_val = tiling_defaults.get("DEFAULT_BASE_M", 128)
    baseN_val = tiling_defaults.get("DEFAULT_BASE_N", 128)

    # Build input tensor creation lines
    inputs = model_info.get("inputs", [])
    input_tensor_lines = []
    for inp in inputs:
        name = inp["name"]
        dtype = inp.get("dtype", "float16")
        np_dtype = inp.get("numpy_dtype", "np.float16")
        shape_parts = inp.get("shape_parts", list(sim_shapes.keys())[:2])
        shape_vals = ", ".join(shape_parts)

        if dtype == "int8":
            input_tensor_lines.append(f'    {name}_host = np.random.randint(-128, 127, ({shape_vals}), dtype=np.int8)')
        elif dtype in ("bfloat16",):
            # numpy has no bf16; use uint16 random bits (same 2-byte size, sufficient for profiling)
            input_tensor_lines.append(f'    {name}_host = np.random.randint(0, 65535, ({shape_vals}), dtype=np.uint16)')
        elif dtype == "float32":
            input_tensor_lines.append(f'    {name}_host = np.random.randn({shape_vals}).astype(np.float32)')
        elif dtype == "uint8":
            input_tensor_lines.append(f'    {name}_host = np.random.randint(0, 255, ({shape_vals}), dtype=np.uint8)')
        elif dtype == "int32":
            input_tensor_lines.append(f'    {name}_host = np.random.randint(-1000, 1000, ({shape_vals}), dtype=np.int32)')
        else:
            input_tensor_lines.append(f'    {name}_host = np.random.randn({shape_vals}).astype({np_dtype})')

    # Output size: use first two shape keys as M, N
    output_dtype_bytes = model_info.get("output_dtype_bytes", 2)
    shape_keys = list(sim_shapes.keys())
    out_m = sim_shapes.get("M", sim_shapes.get(shape_keys[0], 1024) if shape_keys else 1024)
    out_n = sim_shapes.get("N", sim_shapes.get(shape_keys[1], 1024) if len(shape_keys) > 1 else 1024)

    # Allocate lines
    alloc_lines = []
    for inp in inputs:
        alloc_lines.append(f'    {inp["name"]}_dev = acl_malloc({inp["name"]}_host.nbytes)')
    alloc_lines.append(f'    {output_arg_name}_dev = acl_malloc(output_size)')
    alloc_lines.append(f'    ws_dev = acl_malloc(ws_size)')
    alloc_lines.append(f'    tiling_dev = acl_malloc(len(tiling_data))')

    # Copy lines
    copy_lines = []
    for inp in inputs:
        copy_lines.append(f'    acl_memcpy_h2d({inp["name"]}_dev, {inp["name"]}_host)')
    copy_lines.append('    acl_memcpy_h2d(tiling_dev, np.frombuffer(tiling_data, dtype=np.uint8))')

    # Assemble the complete script
    script = f'''"""ACL-native script for msprof simulator profiling (auto-generated by lingxi_profiling_runner.py)."""
import ctypes
import numpy as np
import os
import struct

def main():
    # --- ACL Init ---
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0")
    acl = ctypes.CDLL(os.path.join(ascend_home, "lib64", "libascendcl.so"))
    ret = acl.aclInit(None)
    ret = acl.aclrtSetDevice(0)
    stream = ctypes.c_void_p()
    ret = acl.aclrtCreateStream(ctypes.byref(stream))

    # --- Load standalone kernel .so (absolute path — msprof changes cwd to /) ---
    kernel_so = ctypes.CDLL("{abs_so_path}")
    launch_fn = kernel_so.{kernel_fn}
    launch_fn.restype = None
    launch_fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * {n_device_args}

    # --- Shapes and Tiling ---
    {"; ".join(f"{k} = {v}" for k, v in sim_shapes.items())}
{blockdim_line}

    # Tiling data (packed struct: {", ".join(f"{fn}={fv}" for (fn, _), fv in zip(tiling_fields, tiling_value_strs))})
    tiling_data = struct.pack('{pack_format}', {", ".join(tiling_value_strs)})

    # Workspace size
    ws_size = {baseM_val} * {baseN_val} * {ws_depth} * 4 * blockDim

    # --- Create input tensors ---
{chr(10).join(input_tensor_lines)}

    output_size = {out_m} * {out_n} * {output_dtype_bytes}  # output tensor bytes

    # --- ACL Memory Helpers ---
    def acl_malloc(size):
        ptr = ctypes.c_void_p()
        ret = acl.aclrtMalloc(ctypes.byref(ptr), ctypes.c_size_t(size), 0)
        return ptr if ret == 0 else None

    def acl_memcpy_h2d(dev_ptr, host_arr):
        acl.aclrtMemcpy(dev_ptr, ctypes.c_size_t(host_arr.nbytes),
                        host_arr.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_size_t(host_arr.nbytes), 1)

    # --- Allocate device memory ---
{chr(10).join(alloc_lines)}

    # --- Copy data to device ---
{chr(10).join(copy_lines)}
    acl.aclrtSynchronizeStream(stream)

    # --- Launch kernel ---
    launch_fn(blockDim, stream, {", ".join(input_ptr_names)})
    acl.aclrtSynchronizeStream(stream)
    print("Done")

    # --- Cleanup ---
    for ptr in [{", ".join(cleanup_ptrs)}]:
        if ptr: acl.aclrtFree(ptr)
    acl.aclrtDestroyStream(stream)
    acl.aclrtResetDevice(0)
    acl.aclFinalize()

if __name__ == "__main__":
    main()
'''
    return script


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #

def _detect_soc_version() -> str:
    """Auto-detect SoC version via npu-smi."""
    try:
        out = subprocess.run(
            "npu-smi info", shell=True, capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.splitlines():
            if not line.strip().startswith("|"):
                continue
            m = re.search(r'\b(\d{3}[A-Za-z]\w*)\b', line)
            if m:
                return f"Ascend{m.group(1)}"
    except Exception:
        pass
    return None


def _find_simulator_output(output_dir: str) -> str:
    """Find the most recent simulator trace directory."""
    if not os.path.isdir(output_dir):
        return None

    entries = sorted(
        [e for e in os.listdir(output_dir) if e.startswith("OPPROF_")],
        reverse=True,
    )
    for entry in entries:
        opprof_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(opprof_dir):
            continue
        for root, dirs, files in os.walk(opprof_dir):
            if "simulator" in dirs:
                sim_dir = os.path.join(root, "simulator")
                for d in os.listdir(sim_dir):
                    if d.startswith("core"):
                        trace = os.path.join(sim_dir, d, "trace.json")
                        if os.path.isfile(trace):
                            return sim_dir
    return None


def run_cv_simulator_profiling(
    kernel_dir: str,
    output_dir: str = None,
    soc_version: str = None,
    aic_metrics: str = "PipeUtilization",
    timeout: int = 120,
    shapes: dict = None,
) -> dict:
    """
    Run msprof simulator profiling for a Lingxi operator using ACL-native approach.

    This is the alternative to profiling_runner.py for operators whose pybind11 .so
    depends on libtorch_npu.so.

    Args:
        kernel_dir: Path to the kernel/ directory (must contain pybind11.cpp,
                    build/lib/libkernels.a, and a *_tiling.h file)
        output_dir: Where to store profiling output (default: kernel_dir/../sim_profiling)
        soc_version: SoC version for simulator (default: auto-detect)
        aic_metrics: Metrics to collect (default: PipeUtilization)
        timeout: msprof timeout in seconds (default: 120)
        shapes: Optional shape overrides, e.g. {"M": 1024, "N": 1024, "K": 1024}

    Returns:
        {"success": bool, "simulator_dir": str|None, "error": str|None, ...}
    """
    kernel_dir = str(Path(kernel_dir).resolve())
    task_dir = str(Path(kernel_dir).parent)

    if output_dir is None:
        output_dir = os.path.join(task_dir, "sim_profiling")

    result = {
        "success": False,
        "simulator_dir": None,
        "error": None,
        "mode": "cv_acl_native",
    }

    # --- Step 1: Build standalone .so ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_script = os.path.join(script_dir, "build_standalone_kernel.sh")

    if not os.path.isfile(build_script):
        result["error"] = f"build_standalone_kernel.sh not found at {build_script}"
        return result

    standalone_so = os.path.join(kernel_dir, "build", "libkernel_standalone.so")
    if not os.path.isfile(standalone_so):
        print(f"[lingxi_profiling_runner] Building standalone .so ...")
        build_result = subprocess.run(
            ["bash", build_script, kernel_dir],
            capture_output=True, text=True, timeout=60,
        )
        if build_result.returncode != 0:
            result["error"] = f"build_standalone_kernel.sh failed: {build_result.stderr[-500:]}"
            return result
        print(f"[lingxi_profiling_runner] Standalone .so built: {standalone_so}")
    else:
        print(f"[lingxi_profiling_runner] Standalone .so exists: {standalone_so}")

    # --- Step 2: Parse operator info ---
    pybind_path = os.path.join(kernel_dir, "pybind11.cpp")
    if not os.path.isfile(pybind_path):
        result["error"] = f"pybind11.cpp not found at {pybind_path}"
        return result

    try:
        pybind_info = parse_pybind11_cpp(pybind_path)
    except ValueError as e:
        result["error"] = str(e)
        return result

    # Find tiling header
    tiling_header = pybind_info.get("tiling_header")
    tiling_path = None
    if tiling_header:
        tiling_path = os.path.join(kernel_dir, tiling_header)
    if not tiling_path or not os.path.isfile(tiling_path):
        # Search for any *_tiling.h
        import glob
        candidates = glob.glob(os.path.join(kernel_dir, "*_tiling.h"))
        if candidates:
            tiling_path = candidates[0]

    if not tiling_path or not os.path.isfile(tiling_path):
        result["error"] = f"Tiling header not found in {kernel_dir}"
        return result

    tiling_info = parse_tiling_header(tiling_path)
    model_info = parse_model_inputs(kernel_dir)

    # --- Step 3: Determine shapes ---
    if shapes is None:
        shapes = {"M": 1024, "N": 1024, "K": 1024}  # default

    # --- Step 4: Generate ACL profiling script ---
    acl_script_content = generate_acl_profiling_script(
        kernel_dir=kernel_dir,
        pybind_info=pybind_info,
        tiling_info=tiling_info,
        model_info=model_info,
        shapes=shapes,
    )

    acl_script_path = os.path.join(task_dir, "run_acl_profiling.py")
    with open(acl_script_path, "w") as f:
        f.write(acl_script_content)
    print(f"[lingxi_profiling_runner] Generated ACL script: {acl_script_path}")

    # --- Step 5: Run msprof simulator ---
    if not soc_version:
        soc_version = _detect_soc_version()

    os.makedirs(output_dir, exist_ok=True)

    # Use absolute python3 path (msprof may alter PATH)
    python3_path = sys.executable or "/usr/bin/python3"

    msprof_cmd = (
        f'msprof op simulator'
        f' --application="{python3_path} {acl_script_path}"'
        f' --output="{output_dir}"'
        f' --aic-metrics={aic_metrics}'
        f' --timeout={timeout}'
    )
    if soc_version:
        msprof_cmd += f' --soc-version={soc_version}'

    print(f"[lingxi_profiling_runner] Running: {msprof_cmd}")

    try:
        msprof_result = subprocess.run(
            msprof_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # extra buffer
            cwd=task_dir,
        )

        if msprof_result.returncode != 0:
            result["error"] = (
                f"msprof failed (rc={msprof_result.returncode}): "
                f"{msprof_result.stderr[-500:]}"
            )
            result["stdout"] = msprof_result.stdout[-500:]
            return result

    except subprocess.TimeoutExpired:
        result["error"] = f"msprof timed out after {timeout + 60}s"
        return result
    except FileNotFoundError:
        result["error"] = "msprof not found in PATH"
        return result

    # --- Step 6: Find simulator output ---
    simulator_dir = _find_simulator_output(output_dir)
    if not simulator_dir:
        result["error"] = f"msprof completed but no simulator trace found in {output_dir}"
        result["stdout"] = msprof_result.stdout[-500:]
        return result

    result["success"] = True
    result["simulator_dir"] = simulator_dir
    result["command"] = msprof_cmd
    print(f"[lingxi_profiling_runner] Success! Simulator dir: {simulator_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="ACL-native msprof simulator profiling for Lingxi operators"
    )
    parser.add_argument("--kernel-dir", required=True,
                        help="Path to the kernel/ directory")
    parser.add_argument("--output-dir", default=None,
                        help="Profiling output directory (default: task_dir/sim_profiling)")
    parser.add_argument("--soc-version", default=None,
                        help="SoC version (e.g., Ascend910B3)")
    parser.add_argument("--aic-metrics", default="PipeUtilization")
    parser.add_argument("--timeout", type=int, default=120,
                        help="msprof timeout in seconds (default: 120)")
    parser.add_argument("--shapes", default=None,
                        help='JSON string of shapes, e.g. \'{"M":1024,"N":1024,"K":1024}\'')
    parser.add_argument("--output", default=None,
                        help="Output result JSON to file")
    args = parser.parse_args()

    shapes = None
    if args.shapes:
        shapes = json.loads(args.shapes)

    result = run_cv_simulator_profiling(
        kernel_dir=args.kernel_dir,
        output_dir=args.output_dir,
        soc_version=args.soc_version,
        aic_metrics=args.aic_metrics,
        timeout=args.timeout,
        shapes=shapes,
    )

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)
    print(output_json)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
