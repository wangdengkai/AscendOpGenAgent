#!/bin/bash
# 批量调度 ascendc-coder，支持多 NPU 并行
#
# 改进点：
# - 每个算子独立日志文件（不再按 NPU 合并）
# - 产物校验 + 重试机制
# - 优雅超时（SIGTERM → 等待 → SIGKILL）
# - 时间戳子目录自动生成
# - 单 NPU 模式也有持久化日志
#
# 用法:
#   # 单 NPU 模式
#   bash utils/run_benchmark_ascendc_aiinfra.sh --benchmark-dir /path/to/KernelBench --level 1 --range 1-30 --npu 0 --output /path/to/output
#
#   # 多 NPU 并行模式
#   bash utils/run_benchmark_ascendc_aiinfra.sh --benchmark-dir /path/to/KernelBench --level 1 --range 1-30 --npu-list "0,1,2,3,4,5" --output /path/to/output
#
#   # 输出目录会自动追加时间戳子目录，如 /path/to/output/20260421_143000/

set -euo pipefail

# ── 默认值 ──
BENCHMARK_DIR=""
LEVEL=""
RANGE=""
IDS=""
NPU_ID=0
NPU_LIST=""
OUTPUT_DIR=""
TIMEOUT_SEC=7200
MAX_RETRIES=1  # 失败后最多重试次数（总尝试 = 1 + MAX_RETRIES）

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --level)         LEVEL="$2"; shift 2 ;;
        --range)         RANGE="$2"; shift 2 ;;
        --ids)           IDS="$2"; shift 2 ;;
        --npu)           NPU_ID="$2"; shift 2 ;;
        --npu-list)      NPU_LIST="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --timeout)       TIMEOUT_SEC="$2"; shift 2 ;;
        --max-retries)   MAX_RETRIES="$2"; shift 2 ;;
        -h|--help)
            cat <<'HELP'
用法: bash utils/run_benchmark_ascendc_aiinfra.sh [选项]

必填参数:
  --benchmark-dir <path>   KernelBench 根目录路径
  --level <N>              Level 编号，如 1, 2, 3
  --output <path>          输出根目录（自动追加时间戳子目录）

算子选择 (二选一):
  --range <start-end>      算子范围，如 1-30
  --ids <id_list>          指定算子编号，逗号分隔，如 3,7,15

NPU 选择 (二选一):
  --npu <id>               单 NPU 设备 ID (默认 0)
  --npu-list <list>        多 NPU 列表，逗号分隔，如 0,1,2,3,4,5

可选参数:
  --timeout <seconds>      单次算子超时 (默认 7200 = 2小时)
  --max-retries <N>        失败重试次数 (默认 1，总尝试 = 1+N)
HELP
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
[[ -z "$BENCHMARK_DIR" ]] && { echo "错误: 必须指定 --benchmark-dir"; exit 1; }
[[ -z "$LEVEL" ]]         && { echo "错误: 必须指定 --level"; exit 1; }
[[ -z "$RANGE" && -z "$IDS" ]] && { echo "错误: 必须指定 --range 或 --ids"; exit 1; }
[[ -z "$OUTPUT_DIR" ]]    && { echo "错误: 必须指定 --output"; exit 1; }

LEVEL_DIR="${BENCHMARK_DIR}/level${LEVEL}"
[[ ! -d "$LEVEL_DIR" ]] && { echo "错误: 目录不存在: ${LEVEL_DIR}"; exit 1; }

# ── 时间戳输出目录 ──
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

# ── PID 追踪与清理 ──
CLAUDE_PIDS_FILE="${RUN_DIR}/.claude_pids"
touch "$CLAUDE_PIDS_FILE"

cleanup() {
    local exit_code=$?
    [[ "${_CLEANUP_DONE:-}" == "1" ]] && return
    _CLEANUP_DONE=1

    echo ""
    echo "$(date '+%H:%M:%S') [CLEANUP] 正在清理所有 claude 进程..."

    if [[ -f "$CLAUDE_PIDS_FILE" ]]; then
        while IFS= read -r pid; do
            [[ -z "$pid" ]] && continue
            kill -0 "$pid" 2>/dev/null || continue
            # 杀整个进程组
            kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        done < "$CLAUDE_PIDS_FILE"

        sleep 3

        # SIGKILL 残留进程
        while IFS= read -r pid; do
            [[ -z "$pid" ]] && continue
            if kill -0 "$pid" 2>/dev/null; then
                echo "$(date '+%H:%M:%S') [CLEANUP] SIGKILL pid=${pid}"
                kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            fi
        done < "$CLAUDE_PIDS_FILE"
    fi

    # 杀掉 worker 子进程
    jobs -p 2>/dev/null | xargs -r kill -TERM 2>/dev/null || true

    echo "$(date '+%H:%M:%S') [CLEANUP] 清理完成"
    exit "$exit_code"
}

trap cleanup INT TERM
trap 'cleanup' EXIT

# ── 确定执行模式 ──
USE_PARALLEL=false
if [[ -n "$NPU_LIST" ]]; then
    USE_PARALLEL=true
    IFS=',' read -ra NPU_ARRAY <<< "$NPU_LIST"
    NPU_COUNT=${#NPU_ARRAY[@]}
    [[ $NPU_COUNT -eq 0 ]] && { echo "错误: NPU 列表为空"; exit 1; }
else
    NPU_ARRAY=("$NPU_ID")
    NPU_COUNT=1
fi

# ── 构建算子 ID 列表 ──
OP_IDS=()
if [[ -n "$RANGE" ]]; then
    START=$(echo "$RANGE" | cut -d'-' -f1)
    END=$(echo "$RANGE" | cut -d'-' -f2)
    for i in $(seq "$START" "$END"); do
        OP_IDS+=("$i")
    done
elif [[ -n "$IDS" ]]; then
    IFS=',' read -ra OP_IDS <<< "$IDS"
fi

# ── 扫描算子文件 ──
declare -A OP_FILES
for id in "${OP_IDS[@]}"; do
    matched=$(find "$LEVEL_DIR" -maxdepth 1 -name "${id}_*.py" -type f 2>/dev/null | head -1)
    if [[ -n "$matched" ]]; then
        OP_FILES[$id]="$matched"
    else
        echo "警告: 未找到算子 ${id} 的文件，跳过"
    fi
done

[[ ${#OP_FILES[@]} -eq 0 ]] && { echo "错误: 未找到任何算子文件"; exit 1; }

TOTAL=${#OP_FILES[@]}

# ── 文件锁 ──
touch "${RUN_DIR}/.lock"

# ── 工具函数 ──

# 产物校验
check_deliverables() {
    local task_dir="$1"
    local required=("model_new_ascendc.py" "model_new_tilelang.py")
    local missing=()
    for f in "${required[@]}"; do
        local fpath="${task_dir}/${f}"
        if [[ ! -f "$fpath" ]]; then
            missing+=("${f}(不存在)")
        elif [[ ! -s "$fpath" ]]; then
            missing+=("${f}(空文件)")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "MISSING:${missing[*]}"
        return 1
    fi
    echo "OK"
    return 0
}

# 执行单个算子（含重试）
run_single_op() {
    local id="$1"
    local npu="$2"
    local file="${OP_FILES[$id]}"
    local filename
    filename=$(basename "$file")
    local op_name="${filename%.*}"
    local target_dir="${RUN_DIR}/${op_name}"
    local op_log="${LOG_DIR}/${op_name}.log"

    mkdir -p "$target_dir"

    local max_attempts=$((1 + MAX_RETRIES))
    local attempt=1
    local status="❌ 失败"
    local total_elapsed=0

    while [[ $attempt -le $max_attempts ]]; do
        local attempt_label="[NPU ${npu}] 算子 ${id} (${filename}) 尝试 ${attempt}/${max_attempts}"
        echo "$(date '+%H:%M:%S') ${attempt_label} 开始" >> "$op_log"

        local start_time
        start_time=$(date +%s)

        local PROMPT="使用当前agent生成ascendC算子，npu=${npu}，算子描述文件为 ${file}，输出到 ${target_dir}/。若 ${file%.py}.md 存在，请先阅读该文件以理解算子定义、约束与 CPU 参考实现（文末的 Model 类）。"

        # 用 setsid 启动 claude，使其成为独立进程组组长，方便清理
        setsid claude --verbose \
            --permission-mode acceptEdits \
            -p "$PROMPT" \
            --output-format stream-json \
            --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)' \
            >> "$op_log" 2>&1 &
        local claude_pid=$!

        # 记录 PID 用于清理
        echo "$claude_pid" >> "$CLAUDE_PIDS_FILE"

        # 等待完成或超时
        local timed_out=false
        local exit_code=0
        local waited=0
        while kill -0 "$claude_pid" 2>/dev/null; do
            if [[ $waited -ge $TIMEOUT_SEC ]]; then
                timed_out=true
                # 杀整个进程组
                kill -TERM -- "-${claude_pid}" 2>/dev/null || kill -TERM "$claude_pid" 2>/dev/null || true
                sleep 5
                kill -KILL -- "-${claude_pid}" 2>/dev/null || kill -KILL "$claude_pid" 2>/dev/null || true
                break
            fi
            sleep 5
            waited=$((waited + 5))
        done

        wait "$claude_pid" 2>/dev/null || exit_code=$?

        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        total_elapsed=$((total_elapsed + elapsed))

        echo "$(date '+%H:%M:%S') ${attempt_label} exit_code=${exit_code} elapsed=${elapsed}s" >> "$op_log"

        # 产物校验
        local check_result
        check_result=$(check_deliverables "$target_dir")

        if [[ "$check_result" == "OK" ]]; then
            status="✅ 成功"
            echo "$(date '+%H:%M:%S') ${attempt_label} 产物校验通过" >> "$op_log"
            break
        else
            echo "$(date '+%H:%M:%S') ${attempt_label} 产物校验失败: ${check_result}" >> "$op_log"
            if [[ "$timed_out" == true ]]; then
                echo "$(date '+%H:%M:%S') ${attempt_label} 超时，不再重试" >> "$op_log"
                status="⏰ 超时"
                break
            fi
            if [[ $attempt -lt $max_attempts ]]; then
                echo "$(date '+%H:%M:%S') ${attempt_label} 将重试..." >> "$op_log"
            fi
        fi

        attempt=$((attempt + 1))
    done

    # 写入报告（加锁）
    {
        flock -x 200
        echo "| ${id} | ${filename} | ${status} | ${total_elapsed} | ${attempt}/${max_attempts} |" >> "${RUN_DIR}/batch_report.md"
    } 200>"${RUN_DIR}/.lock"

    # 输出到终端（仅一行状态）
    echo "[NPU ${npu}] ${status} 算子 ${id}: ${filename} (${total_elapsed}s, ${attempt}次)" >&2
}

# ── 初始化报告 ──
cat > "${RUN_DIR}/batch_report.md" <<HEADER
# 批量执行报告

- benchmark: ${BENCHMARK_DIR}
- level: ${LEVEL}
$(if [[ "$USE_PARALLEL" == true ]]; then
    echo "- npu-list: ${NPU_LIST}"
    echo "- 执行模式: 多 NPU 并行（NPU 间并行，NPU 内串行）"
else
    echo "- npu: ${NPU_ID}"
    echo "- 执行模式: 单 NPU 串行"
fi)
- 超时设置: ${TIMEOUT_SEC}s
- 最大重试: ${MAX_RETRIES}
- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')
- 输出目录: ${RUN_DIR}

| 算子ID | 文件 | 状态 | 耗时(s) | 尝试次数 |
|--------|------|------|---------|----------|
HEADER

echo ""
echo "================================================================"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "多 NPU 并行模式: ${NPU_COUNT} 个 NPU，${TOTAL} 个算子"
    echo "NPU 列表: ${NPU_LIST}"
else
    echo "单 NPU 串行模式: NPU ${NPU_ID}，${TOTAL} 个算子"
fi
echo "超时: ${TIMEOUT_SEC}s | 重试: ${MAX_RETRIES} | 输出: ${RUN_DIR}"
echo "================================================================"
echo ""

# ── 执行 ──
if [[ "$USE_PARALLEL" == true ]]; then
    # 轮询分配算子到各 NPU 队列
    declare -A npu_tasks
    npu_index=0
    for id in "${OP_IDS[@]}"; do
        if [[ -v OP_FILES[$id] ]]; then
            npu=${NPU_ARRAY[$((npu_index % NPU_COUNT))]}
            npu_tasks[$npu]+="${id} "
            npu_index=$((npu_index + 1))
        fi
    done

    # 为每个 NPU 启动 worker
    for npu in "${NPU_ARRAY[@]}"; do
        if [[ -n "${npu_tasks[$npu]:-}" ]]; then
            (
                for id in ${npu_tasks[$npu]}; do
                    run_single_op "$id" "$npu"
                done
            ) &
        fi
    done
    wait
else
    CURRENT=0
    for id in $(echo "${!OP_FILES[@]}" | tr ' ' '\n' | sort -n); do
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL}] 算子 ${id}..."
        run_single_op "$id" "$NPU_ID"
    done
fi

# ── 写入汇总 ──
SUCCESS=$(grep -c "✅ 成功" "${RUN_DIR}/batch_report.md" 2>/dev/null || echo 0)
FAIL=$(grep -c "❌ 失败" "${RUN_DIR}/batch_report.md" 2>/dev/null || echo 0)
TIMEOUT_COUNT=$(grep -c "⏰ 超时" "${RUN_DIR}/batch_report.md" 2>/dev/null || echo 0)

cat >> "${RUN_DIR}/batch_report.md" <<SUMMARY

## 汇总

- 总数: ${TOTAL}
- 成功: ${SUCCESS}
- 失败: ${FAIL}
- 超时: ${TIMEOUT_COUNT}
- 结束时间: $(date '+%Y-%m-%d %H:%M:%S')
- 日志目录: ${LOG_DIR}
SUMMARY

# ── 调用汇总脚本 ──
echo ""
echo "================================================================"
echo "正在生成详细报告..."
echo "================================================================"
python3 generate_report_dynamic.py -i "$RUN_DIR" -o "${RUN_DIR}/final_report.md" 2>/dev/null || \
    echo "警告: generate_report_dynamic.py 执行失败，跳过详细报告"

# ── 最终输出 ──
echo ""
echo "================================================================"
echo "批量执行完成: 成功 ${SUCCESS}/${TOTAL}, 失败 ${FAIL}/${TOTAL}, 超时 ${TIMEOUT_COUNT}/${TOTAL}"
echo "报告: ${RUN_DIR}/batch_report.md"
echo "日志: ${LOG_DIR}/"
echo "================================================================"
