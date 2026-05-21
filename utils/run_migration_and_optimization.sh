#!/bin/bash
# 批量调度 CANN 算子迁移与优化，支持多 NPU 并行
#
# 功能：
# 1. 使用 origin-ascendc-to-tilang-design-ascendc-implementation 进行算子迁移
# 2. 使用 kope_ascendc 对迁移后的算子进行性能优化
#
# 支持两种模式：
# 1. 单 NPU 模式（--npu）：串行执行，向后兼容
# 2. 多 NPU 并行模式（--npu-list）：NPU 间并行，NPU 内串行
#
# 用法:
#   # 单 NPU 模式
#   bash utils/run_migration_and_optimization.sh --source-dir /home/w30031530/cann/ops-nn/activation/elu --output /path/to/output --npu 0
#
#   # 多 NPU 并行模式
#   bash utils/run_migration_and_optimization.sh --source-dir /home/w30031530/cann/ops-nn --output /path/to/output --npu-list "0,1,2,3"

set -euo pipefail

# ── 默认值 ──
SOURCE_DIR=""
OUTPUT_DIR=""
NPU_ID=0
NPU_LIST=""
TIMEOUT_SEC=7200  # 2小时默认超时
OP_FILTER=""      # 可选：只处理特定算子，如 "relu,elu,gelu"
SKIP_OPTIMIZATION=false  # 是否跳过优化阶段，只做迁移

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --source-dir)       SOURCE_DIR="$2"; shift 2 ;;
        --output)           OUTPUT_DIR="$2"; shift 2 ;;
        --npu)              NPU_ID="$2"; shift 2 ;;
        --npu-list)         NPU_LIST="$2"; shift 2 ;;
        --timeout)          TIMEOUT_SEC="$2"; shift 2 ;;
        --op-filter)        OP_FILTER="$2"; shift 2 ;;
        --skip-optimization) SKIP_OPTIMIZATION=true; shift ;;
        -h|--help)
            echo "用法: bash utils/run_migration_and_optimization.sh --source-dir <path> --output <path> [--npu <id> | --npu-list <list>] [options]"
            echo ""
            echo "参数:"
            echo "  --source-dir       CANN 算子源码目录 (必填)，可以是单个算子目录或包含多个算子的父目录"
            echo "  --output           输出目录 (必填)"
            echo "  --npu              单 NPU 设备 ID，如 0 (默认 0，与 --npu-list 互斥)"
            echo "  --npu-list         多 NPU 列表，逗号分隔，如 0,1,2,3 (与 --npu 互斥，优先级更高)"
            echo "  --timeout          单个算子总超时时间（迁移+优化），单位秒 (默认 7200 = 2小时)"
            echo "  --op-filter        可选：只处理特定算子，逗号分隔，如 \"relu,elu,gelu\""
            echo "  --skip-optimization 可选：只执行迁移，跳过性能优化阶段"
            echo ""
            echo "示例:"
            echo "  # 单 NPU 模式 - 处理单个算子"
            echo "  bash utils/run_migration_and_optimization.sh --source-dir /home/w30031530/cann/ops-nn/activation/elu --output /path/to/output --npu 0"
            echo ""
            echo "  # 单 NPU 模式 - 处理目录下所有算子"
            echo "  bash utils/run_migration_and_optimization.sh --source-dir /home/w30031530/cann/ops-nn/activation --output /path/to/output --npu 0"
            echo ""
            echo "  # 多 NPU 并行模式"
            echo "  bash utils/run_migration_and_optimization.sh --source-dir /home/w30031530/cann/ops-nn/activation --output /path/to/output --npu-list \"0,1,2,3\""
            echo ""
            echo "  # 只迁移不优化"
            echo "  bash utils/run_migration_and_optimization.sh --source-dir /home/w30031530/cann/ops-nn/activation/elu --output /path/to/output --npu 0 --skip-optimization"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
if [[ -z "$SOURCE_DIR" ]]; then
    echo "错误: 必须指定 --source-dir"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 必须指定 --output"
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "错误: 目录不存在: ${SOURCE_DIR}"
    exit 1
fi

# ── 确定执行模式 ──
USE_PARALLEL=false
if [[ -n "$NPU_LIST" ]]; then
    USE_PARALLEL=true
    # 解析 NPU 列表
    IFS=',' read -ra NPU_ARRAY <<< "$NPU_LIST"
    NPU_COUNT=${#NPU_ARRAY[@]}
    if [[ $NPU_COUNT -eq 0 ]]; then
        echo "错误: NPU 列表为空"
        exit 1
    fi
else
    # 单 NPU 模式
    NPU_ARRAY=("$NPU_ID")
    NPU_COUNT=1
fi

# ── 扫描算子目录 ──
OP_DIRS=()

# 判断 source_dir 是单个算子还是包含多个算子的父目录
if [[ -f "${SOURCE_DIR}/op_kernel/"*.cpp ]] || [[ -f "${SOURCE_DIR}/op_host/"*_def.cpp ]]; then
    # 单个算子目录
    OP_NAME=$(basename "$SOURCE_DIR")
    
    # 检查是否在过滤列表中
    if [[ -n "$OP_FILTER" ]]; then
        if [[ ",$OP_FILTER," == *",$OP_NAME,"* ]]; then
            OP_DIRS+=("$SOURCE_DIR")
            echo "找到算子: $OP_NAME (在过滤列表中)"
        else
            echo "跳过算子: $OP_NAME (不在过滤列表中)"
        fi
    else
        OP_DIRS+=("$SOURCE_DIR")
        echo "找到算子: $OP_NAME"
    fi
else
    # 父目录，扫描子目录
    echo "扫描目录: $SOURCE_DIR"
    for subdir in "$SOURCE_DIR"/*/; do
        if [[ -d "$subdir" ]]; then
            op_name=$(basename "$subdir")
            
            # 检查是否是有效的算子目录
            if [[ -d "${subdir}op_kernel" ]] && [[ -d "${subdir}op_host" ]]; then
                # 检查是否在过滤列表中
                if [[ -n "$OP_FILTER" ]]; then
                    if [[ ",$OP_FILTER," == *",$op_name,"* ]]; then
                        OP_DIRS+=("$subdir")
                        echo "找到算子: $op_name (在过滤列表中)"
                    else
                        echo "跳过算子: $op_name (不在过滤列表中)"
                    fi
                else
                    OP_DIRS+=("$subdir")
                    echo "找到算子: $op_name"
                fi
            fi
        fi
    done
fi

if [[ ${#OP_DIRS[@]} -eq 0 ]]; then
    echo "错误: 未找到任何有效的算子目录"
    exit 1
fi

TOTAL=${#OP_DIRS[@]}
echo ""
echo "================================================================"
echo "发现 ${TOTAL} 个算子待处理"
echo "================================================================"

# ── 创建输出目录 ──
mkdir -p "$OUTPUT_DIR"

# ── 创建文件锁 ──
touch "${OUTPUT_DIR}/.lock"
touch "${OUTPUT_DIR}/.trace_lock"

# ── 结果记录 ──
REPORT_FILE="${OUTPUT_DIR}/batch_report.md"
echo "# 批量迁移与优化报告" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- 源目录: ${SOURCE_DIR}" >> "$REPORT_FILE"
echo "- 输出目录: ${OUTPUT_DIR}" >> "$REPORT_FILE"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "- NPU 列表: ${NPU_LIST}" >> "$REPORT_FILE"
    echo "- 执行模式: 多 NPU 并行（NPU 间并行，NPU 内串行）" >> "$REPORT_FILE"
else
    echo "- NPU: ${NPU_ID}" >> "$REPORT_FILE"
    echo "- 执行模式: 单 NPU 串行" >> "$REPORT_FILE"
fi
echo "- 超时设置: ${TIMEOUT_SEC}s" >> "$REPORT_FILE"
echo "- 跳过优化: ${SKIP_OPTIMIZATION}" >> "$REPORT_FILE"
echo "- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 算子名称 | 迁移状态 | 优化状态 | 总耗时(s) | 备注 |" >> "$REPORT_FILE"
echo "|---------|---------|---------|----------|------|" >> "$REPORT_FILE"

SUCCESS=0
FAIL=0
PARTIAL=0  # 迁移成功但优化失败

# ── 定义处理单个算子的函数 ──
process_operator() {
    local op_dir="$1"
    local npu="$2"
    local output_base="$3"
    
    local op_name=$(basename "$op_dir")
    local op_output="${output_base}/${op_name}"
    
    mkdir -p "$op_output"
    
    local start_time=$(date +%s)
    local migration_status="❌ 失败"
    local optimization_status="⏭️ 跳过"
    local remark=""
    
    echo ""
    echo "================================================================"
    echo "[NPU $npu] 开始处理算子: ${op_name}"
    echo "源目录: ${op_dir}"
    echo "输出目录: ${op_output}"
    echo "================================================================"
    
    # ========== Phase 1: 算子迁移 ==========
    echo "[NPU $npu] Phase 1: 执行算子迁移..."
    
    local migration_prompt="使用 origin-ascendc-to-tilang-design-ascendc-implementation skill 迁移算子，source_dir=${op_dir}，output_dir=${op_output}，npu=${npu}"
    
    if timeout "$TIMEOUT_SEC" claude -p "$migration_prompt" \
        --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)' \
        >> "${op_output}/migration.log" 2>&1; then
        
        migration_status="✅ 成功"
        echo "[NPU $npu] ✅ 算子 ${op_name} 迁移完成"
        
        # 检查迁移输出是否完整
        if [[ -f "${op_output}/model_new_ascendc.py" ]] && [[ -d "${op_output}/kernel" ]]; then
            echo "[NPU $npu] ✓ 迁移输出完整"
        else
            migration_status="⚠️ 部分成功"
            remark="迁移输出不完整"
            echo "[NPU $npu] ⚠️ 警告: 迁移输出可能不完整"
        fi
    else
        migration_status="❌ 失败"
        remark="迁移超时或失败"
        echo "[NPU $npu] ❌ 算子 ${op_name} 迁移失败"
        
        # 迁移失败，直接返回
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        
        # 加锁写入报告
        {
            flock -x 200
            echo "| ${op_name} | ${migration_status} | ⏭️ 跳过 | ${elapsed} | ${remark} |" >> "$REPORT_FILE"
        } 200>"${OUTPUT_DIR}/.lock"
        
        return 1
    fi
    
    # ========== Phase 2: 性能优化（可选）==========
    if [[ "$SKIP_OPTIMIZATION" == false ]]; then
        echo "[NPU $npu] Phase 2: 执行性能优化..."
        
        # 准备 kope_ascendc 所需的项目结构
        local kope_project="${op_output}/kope_project"
        mkdir -p "${kope_project}/output/${op_name}"
        
        # 复制迁移后的代码到 kope 项目结构
        if [[ -d "${op_output}/kernel" ]]; then
            cp -r "${op_output}/kernel"/* "${kope_project}/output/${op_name}/" 2>/dev/null || true
        fi
        
        if [[ -f "${op_output}/model_new_ascendc.py" ]]; then
            cp "${op_output}/model_new_ascendc.py" "${kope_project}/output/${op_name}/${op_name}_ascendc.cpp" 2>/dev/null || true
        fi
        
        if [[ -f "${op_output}/model.py" ]]; then
            cp "${op_output}/model.py" "${kope_project}/output/${op_name}/${op_name}_reference.py" 2>/dev/null || true
        fi
        
        local optimization_prompt="使用 kope_ascendc agent 优化算子性能，工作目录=${kope_project}，算子名称=${op_name}，npu=${npu}"
        
        # 为优化阶段分配剩余时间
        local remaining_time=$((TIMEOUT_SEC / 2))
        
        if timeout "$remaining_time" claude -p "$optimization_prompt" \
            --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)' \
            >> "${op_output}/optimization.log" 2>&1; then
            
            optimization_status="✅ 成功"
            echo "[NPU $npu] ✅ 算子 ${op_name} 优化完成"
            
            # 检查优化输出
            if [[ -f "${kope_project}/output/${op_name}/${op_name}_optimized_*.cpp" ]]; then
                echo "[NPU $npu] ✓ 找到优化后的代码"
                # 复制优化结果回主输出目录
                cp "${kope_project}/output/${op_name}/${op_name}_optimized_"*.cpp "${op_output}/" 2>/dev/null || true
            fi
        else
            optimization_status="❌ 失败"
            if [[ -z "$remark" ]]; then
                remark="优化超时或失败"
            else
                remark="${remark}; 优化失败"
            fi
            echo "[NPU $npu] ❌ 算子 ${op_name} 优化失败"
        fi
    fi
    
    # ========== 汇总结果 ==========
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    
    echo "[NPU $npu] 算子 ${op_name} 处理完成，总耗时: ${elapsed}s"
    echo "  迁移状态: ${migration_status}"
    echo "  优化状态: ${optimization_status}"
    
    # 加锁写入报告
    {
        flock -x 200
        echo "| ${op_name} | ${migration_status} | ${optimization_status} | ${elapsed} | ${remark} |" >> "$REPORT_FILE"
    } 200>"${OUTPUT_DIR}/.lock"
    
    # 返回状态码
    if [[ "$migration_status" == "✅ 成功" ]]; then
        if [[ "$optimization_status" == "✅ 成功" ]] || [[ "$SKIP_OPTIMIZATION" == true ]]; then
            return 0  # 完全成功
        else
            return 2  # 部分成功（迁移成功但优化失败）
        fi
    else
        return 1  # 失败
    fi
}

# ── 执行模式选择 ──
if [[ "$USE_PARALLEL" == true ]]; then
    # ========== 多 NPU 并行模式 ==========
    echo ""
    echo "================================================================"
    echo "多 NPU 并行模式: ${NPU_COUNT} 个 NPU，${TOTAL} 个算子"
    echo "NPU 列表: ${NPU_LIST}"
    echo "超时设置: ${TIMEOUT_SEC}s"
    echo "================================================================"
    echo ""
    
    # 任务分配：轮询分配算子到各 NPU 队列
    declare -A npu_tasks
    npu_index=0
    for op_dir in "${OP_DIRS[@]}"; do
        npu=${NPU_ARRAY[$((npu_index % NPU_COUNT))]}
        npu_tasks[$npu]+="${op_dir} "
        npu_index=$((npu_index + 1))
    done
    
    # 为每个 NPU 启动 worker 进程
    worker_pids=()
    for npu in "${NPU_ARRAY[@]}"; do
        # 检查该 NPU 是否有任务
        if [[ -n "${npu_tasks[$npu]:-}" ]]; then
            (
                # ========== Worker 进程开始 ==========
                echo "[Worker NPU $npu] 启动，处理 ${npu_tasks[$npu]} 中的算子"
                
                for op_dir in ${npu_tasks[$npu]}; do
                    process_operator "$op_dir" "$npu" "$OUTPUT_DIR"
                    result=$?
                    
                    # 根据返回值统计
                    if [[ $result -eq 0 ]]; then
                        SUCCESS=$((SUCCESS + 1))
                    elif [[ $result -eq 2 ]]; then
                        PARTIAL=$((PARTIAL + 1))
                    else
                        FAIL=$((FAIL + 1))
                    fi
                done
                
                echo "[Worker NPU $npu] 完成所有任务"
                # ========== Worker 进程结束 ==========
            ) &
            worker_pids+=($!)
        fi
    done
    
    # 等待所有 worker 完成
    echo ""
    echo "等待所有 NPU worker 完成..."
    for pid in "${worker_pids[@]}"; do
        wait $pid || true
    done
    
else
    # ========== 单 NPU 串行模式 ==========
    echo ""
    echo "================================================================"
    echo "单 NPU 串行模式: NPU ${NPU_ID}，${TOTAL} 个算子"
    echo "超时设置: ${TIMEOUT_SEC}s"
    echo "================================================================"
    echo ""
    
    CURRENT=0
    for op_dir in "${OP_DIRS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "================================================================"
        echo "[${CURRENT}/${TOTAL}] 处理算子: $(basename $op_dir)"
        echo "================================================================"
        
        process_operator "$op_dir" "$NPU_ID" "$OUTPUT_DIR"
        result=$?
        
        # 根据返回值统计
        if [[ $result -eq 0 ]]; then
            SUCCESS=$((SUCCESS + 1))
        elif [[ $result -eq 2 ]]; then
            PARTIAL=$((PARTIAL + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    done
fi

# ── 写入汇总 ──
echo "" >> "$REPORT_FILE"
echo "## 汇总" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- 总数: ${TOTAL}" >> "$REPORT_FILE"
echo "- 完全成功: ${SUCCESS}" >> "$REPORT_FILE"
echo "- 部分成功（仅迁移）: ${PARTIAL}" >> "$REPORT_FILE"
echo "- 失败: ${FAIL}" >> "$REPORT_FILE"
echo "- 结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"

if [[ "$USE_PARALLEL" == true ]]; then
    echo "- 执行模式: 多 NPU 并行" >> "$REPORT_FILE"
fi

echo ""
echo "================================================================"
echo "批量执行完成"
echo "  总数: ${TOTAL}"
echo "  完全成功: ${SUCCESS}"
echo "  部分成功（仅迁移）: ${PARTIAL}"
echo "  失败: ${FAIL}"
echo "报告: ${REPORT_FILE}"
echo "================================================================"
