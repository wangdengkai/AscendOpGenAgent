# ops-partial 子agent Prompt 模板

供 ops-evo 步骤 4.3.2 中启动 ops-partial 子agent 时使用。
主 agent 读取此模板，按 ops-evo.md 中的"Prompt 变量填充规则"填充 `{变量}` 后作为 prompt 传给子 agent。

---

```
[TASK]
Modify, build, and evaluate AscendC kernel code for {op_name} operator in ops repository.

Operator: {op_description_from_def_cpp}
Source: {output_dir}/shared/original/
Output: {output_dir}/round_{r}/parallel_{p}/modified_files/

[MANDATORY OPTIMIZATION DIRECTION]
你被分配的优化方向是:
  方向描述: {node_description}
  策略组合: {strategy_combination}
  模式: {mode}

[WARNING] 你必须严格按此方向实现优化。
- 禁止偏离到其他方向（如仅调整 tiling 阈值、仅改 BUFFER_NUM 等简单参数修改不算"实现策略"）
- 你的修改必须体现上述方向描述中的核心优化思路
- 如果此方向在技术上不可行，在 implementation_note.txt 中说明原因，而不是静默切换到其他方向
- 禁止与兄弟变体做同一件事：若你的策略组合与下方任一兄弟的 sig 重叠超过 60%，必须在 implementation_note.txt 写明差异化实现点。

同轮其他变体的方向（禁止重复）:
{other_variants_summary}

[Optimization Approach]
If Mode=open_exploration:
  DO NOT read strategy files. Reason from first principles.
  Analyze bottleneck and profiling data, design novel optimization approach.

If Mode=profiling_driven:
  DO NOT read strategy files. The node description contains specific profiling bottleneck data — use it as your optimization target.
  a. Read parent kernel code and understand current implementation
  b. Parse the profiling bottleneck from node description (e.g., "MTE2 stall 62%", "scalar compute blocking 40%")
  c. Design a TARGETED optimization specifically addressing the diagnosed bottleneck
  d. Modify kernel/tiling files with targeted patches (don't rewrite from scratch)
  e. In implementation_note: state what profiling bottleneck was targeted and how

If Mode=strategy_guided:
  The [MANDATORY OPTIMIZATION DIRECTION] already contains the strategy direction and key points.
  DO NOT read strategy files from evolution/meta_prompts/strategies/ — implement based on the direction description above.
  If the direction is unclear, read at most 1 strategy file for the primary strategy ID only.

[CONTEXT]
Node: {node_id} | Parent: {parent_solution_ref} | Best: {best_solution_ref}

{inspirations_text}

Kernel: {kernel_summary}
Score: global best {best_score}x | this variant from {parent_score}x ({parent_solution_ref or "baseline"})

[Profiling Context]
Baseline:  bn={baseline_bottleneck_type} | recommended={baseline_suggested_strategies} | anti={baseline_anti_strategies}
Parent:    bn={bottleneck} | pipeline={profiling_one_liner} | recommended={recommended_strategies}
{若有 profiling_evidence:}
Bubble Analysis: D-class={d_class_pct}% C-class={c_class_pct}%
Primary stall: {primary_bottleneck}
DMA efficiency: MTE2 short={mte2_short_pct}% MTE3 short={mte3_short_pct}%
Pattern: {pattern_type} | Overlap: {overlap_status}
{/若}

[Alignment — 实现代码前必检]
本变体 opt_type={optimization_type}；strategy_combination={strategy_combination}
- 若 strategy_combination 含 Baseline anti 中任一策略 → 必须移除，或在 implementation_note.txt 第一行写明例外理由
- 若 opt_type 与 Baseline bn 相悖（示例：bn=compute_bound 但 opt_type=bandwidth）→ 在 implementation_note.txt 写明"故意背离"的理由
- 若 Baseline 行任一字段为 `N/A`，说明根级 baseline_evidence 未挂载，跳过对齐检查

Bottleneck-targeted guidance:
  memory_bound → increase tile size, enable double buffer, merge DMA transfers
  scalar_bound → reduce GetValue calls, vectorize reductions, hoist loop-invariants
  compute_bound → algorithm optimization, eliminate redundant compute
  mte2_stall → redesign data layout, enlarge transfer blocks, batch small DMAs
  tiling_imbalance → rebalance core workload, adjust block_dim partitioning
  no_overlap → restructure CopyIn/Compute/CopyOut stage boundaries for pipeline overlap

Hardware: {hw_params_one_liner}

Steps:
1. Read original kernel: shared/original/ (op_kernel/*.cpp, *.h, op_host/*_tiling.h, *_tiling.cpp)
2. If parent_solution_ref non-empty: read parent variant from {output_dir}/{parent_solution_ref}/modified_files/
3. Apply optimization per [MANDATORY OPTIMIZATION DIRECTION]
4. operator-coding-red-line check (read .claude/skills/operator-coding-red-line/SKILL.md only, skip ascendc-code-review)
5. Save modified files + code_changes.json + implementation_note.txt
6. Apply modified files to worktree and build
7. Evaluate with eval-lock (baseline vs evolved on bound device)

Modification Constraints (STRICTLY ENFORCED):
- op_kernel/*.cpp, *.h: Free to modify (do not change call signatures)
- op_host/*_tiling.h, *_tiling.cpp: Allowed to modify (do not change field names)
- op_host/*_def.cpp: Restricted (compile options/optimization attributes ONLY; NEVER change inputs/outputs/type constraints)

[BUILD & EVAL CONTEXT]
Worktree Path: {worktree_base}/round_{r}_p{p}
Worktree Repo Root: {worktree_base}/round_{r}_p{p}
Worktree Op Path: {worktree_base}/round_{r}_p{p}/{OP_PATH_RELATIVE}
Z-Search Root: {z_search_root}
Install Path: $(pwd)/output/{op_name}_ops-evo_{timestamp}/round_{r}/parallel_{p}/evolved
Baseline Install Path: $(pwd)/output/{op_name}_ops-evo_{timestamp}/baseline
Device ID: {session_device_id}
Eval Lock Path: {eval_lock_path}
Baseline Cache: $(pwd)/output/{op_name}_ops-evo_{timestamp}/baseline_evaluation.json
Call Spec: $(pwd)/output/{op_name}_ops-evo_{timestamp}/shared/call_spec.json
Op Name: {op_name}
Build Op Name: {build_op_name}
SOC: {soc}
Task Type: {task_type}
Eval Lock Timeout: 300

[HISTORY]
{open_questions_rendered}

Output all modified files to: {output_dir}/round_{r}/parallel_{p}/modified_files/
```
