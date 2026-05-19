#!/usr/bin/env python3
"""
enable_ttk.py - Migrate custom operator files to CANN TBE directory for TTK testing.

Supports two source modes:
  1. --vendor-dir: Directly copy from a variant's vendors/{pkg_name} directory
     (e.g., from ops-evo evolved output). No need to install the custom package first.
  2. Legacy mode: Copy from ${ASCEND_HOME_PATH}/opp/vendors/{pkg_name} (installed package).

Copies kernel implementations, dynamic compilation scripts, binary files,
and merges JSON config nodes into the CANN built-in TBE directory.
"""

import argparse
import json
import os
import shutil
import sys


def get_ascend_home():
    path = os.environ.get("ASCEND_HOME_PATH")
    if not path:
        print("ERROR: ASCEND_HOME_PATH is not set", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(path):
        print(f"ERROR: ASCEND_HOME_PATH={path} does not exist", file=sys.stderr)
        sys.exit(1)
    return path


def snake_to_camel(name):
    """Convert snake_case to CamelCase. e.g. dynamic_quant_update_scatter_v2 -> DynamicQuantUpdateScatterV2."""
    parts = name.split("_")
    result = []
    i = 0
    while i < len(parts):
        part = parts[i]
        # Handle trailing version like "v2", "v3" -> merge with previous as "V2", "V3"
        if i > 0 and len(part) >= 2 and part[0] == "v" and part[1:].isdigit():
            result.append(part[0].upper() + part[1:])
        else:
            result.append(part.capitalize())
        i += 1
    return "".join(result)


def merge_json_node(tgt_json_path, src_json_path, op_name):
    """Merge a single op_name key from src JSON into tgt JSON.

    Tries both snake_case (op_name) and CamelCase variants as keys,
    since CANN JSON configs may use either convention.
    """
    if not os.path.isfile(src_json_path):
        print(f"  WARN: source JSON not found: {src_json_path}")
        return False
    if not os.path.isfile(tgt_json_path):
        print(f"  WARN: target JSON not found: {tgt_json_path}")
        return False

    with open(tgt_json_path, "r") as f:
        tgt_data = json.load(f)
    with open(src_json_path, "r") as f:
        src_data = json.load(f)

    # Try snake_case first, then CamelCase
    camel_name = snake_to_camel(op_name)
    actual_key = None
    if op_name in src_data:
        actual_key = op_name
    elif camel_name in src_data:
        actual_key = camel_name
    else:
        print(f"  WARN: key '{op_name}' (nor '{camel_name}') not found in {src_json_path}")
        return False

    tgt_data[actual_key] = src_data[actual_key]

    with open(tgt_json_path, "w") as f:
        json.dump(tgt_data, f, indent=4)

    print(f"  Merged key '{actual_key}' into {tgt_json_path}")
    return True


def detect_pkg_name(vendor_dir):
    """Auto-detect package name from vendor directory structure.

    The vendor directory typically contains {pkg_name}_impl/ under
    op_impl/ai_core/tbe/. We detect it from there.
    """
    tbe_path = os.path.join(vendor_dir, "op_impl", "ai_core", "tbe")
    if not os.path.isdir(tbe_path):
        return None
    for entry in os.listdir(tbe_path):
        if entry.endswith("_impl") and os.path.isdir(os.path.join(tbe_path, entry)):
            return entry  # e.g. "custom_nn_impl"
    return None


def detect_soc_versions(vendor_dir):
    """Detect available SOC versions from vendor directory config."""
    config_path = os.path.join(vendor_dir, "op_impl", "ai_core", "tbe", "config")
    if not os.path.isdir(config_path):
        return []
    return [d for d in os.listdir(config_path)
            if os.path.isdir(os.path.join(config_path, d))]


def migrate_op(op_name, cann_tbe_path, custom_tbe_path, pkg_name, repo_name, soc,
               cust_config_name, build_mode="release"):
    """Migrate a single operator's files from custom vendor dir to CANN TBE dir."""
    print(f"\n=== Migrating operator: {op_name} ===")
    errors = []

    # 1. Kernel implementation files (ascendc directory)
    src_kernel = os.path.join(custom_tbe_path, f"{pkg_name}_impl", "ascendc", op_name)
    dst_kernel = os.path.join(cann_tbe_path, "impl", repo_name, "ascendc")
    if os.path.isdir(src_kernel):
        dst_kernel_op = os.path.join(dst_kernel, op_name)
        if os.path.exists(dst_kernel_op):
            shutil.rmtree(dst_kernel_op)
        os.makedirs(dst_kernel, exist_ok=True)
        shutil.copytree(src_kernel, dst_kernel_op)
        print(f"  Copied kernel impl: {src_kernel} -> {dst_kernel_op}")
    else:
        errors.append(f"Kernel impl dir not found: {src_kernel}")

    # 2. Dynamic compilation script
    src_dynamic = os.path.join(custom_tbe_path, f"{pkg_name}_impl", "dynamic", f"{op_name}.py")
    dst_dynamic_dir = os.path.join(cann_tbe_path, "impl", repo_name, "dynamic")
    if os.path.isfile(src_dynamic):
        os.makedirs(dst_dynamic_dir, exist_ok=True)
        dst_dynamic = os.path.join(dst_dynamic_dir, f"{op_name}.py")
        shutil.copy2(src_dynamic, dst_dynamic)
        os.chmod(dst_dynamic, 0o755)
        print(f"  Copied dynamic script: {src_dynamic} -> {dst_dynamic}")
    else:
        errors.append(f"Dynamic script not found: {src_dynamic}")

    # 3. Operator registration JSON (only for binary mode)
    if build_mode == "release":
        src_op_json = os.path.join(custom_tbe_path, "kernel", "config", soc, f"{op_name}.json")
        dst_op_json_dir = os.path.join(cann_tbe_path, "kernel", "config", soc, repo_name)
        if os.path.isfile(src_op_json):
            os.makedirs(dst_op_json_dir, exist_ok=True)
            shutil.copy2(src_op_json, os.path.join(dst_op_json_dir, f"{op_name}.json"))
            print(f"  Copied op JSON: {src_op_json} -> {dst_op_json_dir}/")
        else:
            errors.append(f"Op registration JSON not found: {src_op_json}")

        # 4. Operator binary files
        src_bin = os.path.join(custom_tbe_path, "kernel", soc, op_name)
        dst_bin_dir = os.path.join(cann_tbe_path, "kernel", soc, repo_name)
        if os.path.isdir(src_bin):
            dst_bin = os.path.join(dst_bin_dir, op_name)
            if os.path.exists(dst_bin):
                shutil.rmtree(dst_bin)
            os.makedirs(dst_bin_dir, exist_ok=True)
            shutil.copytree(src_bin, dst_bin)
            print(f"  Copied binary: {src_bin} -> {dst_bin}")
        else:
            errors.append(f"Binary dir not found: {src_bin}")

        # 5. Merge binary_info_config.json
        src_binary_info = os.path.join(custom_tbe_path, "kernel", "config", soc, "binary_info_config.json")
        tgt_binary_info = os.path.join(cann_tbe_path, "kernel", "config", soc, repo_name, "binary_info_config.json")
        if not merge_json_node(tgt_binary_info, src_binary_info, op_name):
            errors.append(f"Failed to merge binary_info_config.json for {op_name}")

    # 6. Merge ops-info.json
    src_ops_info = os.path.join(custom_tbe_path, "config", soc, f"aic-{soc}-ops-info.json")
    tgt_ops_info = os.path.join(cann_tbe_path, "config", soc, cust_config_name)
    if not merge_json_node(tgt_ops_info, src_ops_info, op_name):
        errors.append(f"Failed to merge ops-info.json for {op_name}")

    if errors:
        print(f"\n  Warnings for {op_name}:")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"  Migration complete for {op_name}")

    return len(errors) == 0


def install_vendor_to_cann(vendor_dir, ascend_home, pkg_name):
    """Install vendor directory to CANN opp/vendors/ (direct copy).

    This is the new approach: directly copy the variant's vendors/{pkg_name}
    directory to ${ASCEND_HOME_PATH}/opp/vendors/{pkg_name}.
    """
    cann_vendor_dst = os.path.join(ascend_home, "opp", "vendors", pkg_name)

    if os.path.exists(cann_vendor_dst):
        print(f"  Removing existing vendor: {cann_vendor_dst}")
        shutil.rmtree(cann_vendor_dst)

    os.makedirs(os.path.dirname(cann_vendor_dst), exist_ok=True)
    shutil.copytree(vendor_dir, cann_vendor_dst)
    print(f"  Installed vendor: {vendor_dir} -> {cann_vendor_dst}")
    return cann_vendor_dst


def main():
    parser = argparse.ArgumentParser(
        description="Migrate custom operator files to CANN TBE for TTK testing")
    parser.add_argument("--op-names", required=True,
                        help="Comma-separated operator names")
    parser.add_argument("--repo-name", required=True,
                        help="Target repo name (e.g. ops_transformer, ops_nn)")
    parser.add_argument("--soc", required=True,
                        help="SoC model (e.g. ascend910b, ascend910_93)")
    parser.add_argument("--vendor-dir", default=None,
                        help="Path to variant's vendor directory "
                             "(e.g. .../round_5/parallel_3/evolved/vendors/custom_nn). "
                             "When provided, the directory is first copied to "
                             "${ASCEND_HOME_PATH}/opp/vendors/{pkg_name}, "
                             "then files are migrated to CANN built-in TBE. "
                             "This eliminates the need to pre-install the custom package.")
    parser.add_argument("--pkg-name", default=None,
                        help="Custom package name (e.g. custom_nn). "
                             "Auto-detected from --vendor-dir if not specified.")
    parser.add_argument("--cust-config-name", default=None,
                        help="Custom config JSON name (auto-derived if not set)")
    parser.add_argument("--build-mode", default="release",
                        choices=["release", "dynamic"],
                        help="Build mode: release (binary) or dynamic (compile-on-fly)")

    args = parser.parse_args()

    ascend_home = get_ascend_home()
    cann_tbe_path = os.path.join(ascend_home, "opp", "built-in", "op_impl", "ai_core", "tbe")

    if not os.path.isdir(cann_tbe_path):
        print(f"ERROR: CANN TBE path not found: {cann_tbe_path}", file=sys.stderr)
        sys.exit(1)

    # --- Determine pkg_name ---
    pkg_name = args.pkg_name
    if args.vendor_dir:
        # --vendor-dir mode: directly copy variant vendors to CANN
        vendor_dir = os.path.abspath(args.vendor_dir)
        if not os.path.isdir(vendor_dir):
            print(f"ERROR: vendor-dir not found: {vendor_dir}", file=sys.stderr)
            sys.exit(1)

        # Auto-detect pkg_name from directory name if not specified
        if not pkg_name:
            pkg_name = os.path.basename(vendor_dir)
            print(f"Auto-detected pkg_name from vendor-dir: {pkg_name}")

        # Step 1: Copy vendor dir to CANN opp/vendors/
        print(f"\n=== Installing vendor package to CANN ===")
        install_vendor_to_cann(vendor_dir, ascend_home, pkg_name)
    else:
        # Legacy mode: vendor must already be installed in CANN
        if not pkg_name:
            print("ERROR: --pkg-name is required when --vendor-dir is not specified",
                  file=sys.stderr)
            sys.exit(1)

    custom_tbe_path = os.path.join(
        ascend_home, "opp", "vendors", pkg_name, "op_impl", "ai_core", "tbe")

    if not os.path.isdir(custom_tbe_path):
        print(f"ERROR: Custom TBE path not found: {custom_tbe_path}", file=sys.stderr)
        sys.exit(1)

    # Auto-derive cust_config_name
    cust_config_name = args.cust_config_name
    if not cust_config_name:
        # ops_transformer -> transformer, ops_nn -> nn
        repo_suffix = args.repo_name.replace("ops_", "")
        cust_config_name = f"aic-{args.soc}-ops-info-{repo_suffix}.json"
        print(f"Auto-derived cust_config_name: {cust_config_name}")

    op_names = [n.strip() for n in args.op_names.split(",") if n.strip()]
    print(f"\nASCEND_HOME_PATH: {ascend_home}")
    print(f"CANN TBE: {cann_tbe_path}")
    print(f"Custom TBE: {custom_tbe_path}")
    print(f"Package: {pkg_name}")
    print(f"Operators: {op_names}")
    print(f"Build mode: {args.build_mode}")

    # Step 2: Migrate each operator's files to CANN built-in TBE
    all_ok = True
    for op_name in op_names:
        ok = migrate_op(op_name, cann_tbe_path, custom_tbe_path,
                        pkg_name, args.repo_name, args.soc,
                        cust_config_name, args.build_mode)
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\nAll {len(op_names)} operator(s) migrated successfully.")
    else:
        print(f"\nMigration completed with warnings. Check output above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
