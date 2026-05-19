# cake-partial 子agent Prompt 模板

供 cake-evo 步骤 4.3 中启动 cake-partial 子agent 时使用。
主 agent 读取此模板，按 cake_evo.md 中的变体点表（V1-V6）和变量填充规则填充后作为 prompt 传给子 agent。

---

```
[TASK]
{V1}

Operator: {op_name} — {op_description}
Output: output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}

Shared files (DO NOT regenerate or modify): _op_desc.json, _reference.py, _functional.py, _custom.py, .cpp, {op_name}Custom/

[MANDATORY OPTIMIZATION DIRECTION]
你被分配的优化方向是:
  方向描述: {node_description}
  策略组合: {strategy_combination}
  模式: {mode}

[WARNING] 你必须严格按此方向实现优化。
- 禁止偏离到其他方向（如仅调整 tiling 阈值等简单参数修改不算"实现策略"）
- 你的修改必须体现上述方向描述中的核心优化思路
- 如果此方向在技术上不可行，在 implementation_note 中说明原因，而不是静默切换到其他方向

同轮其他变体的方向（禁止重复）:
{other_variants_summary}

Steps:
1. {V2} - Generate DSL baseline (read existing op_desc and functional files)
2. {V3} - Apply lowering. See [Optimization Approach] for strategy application.
3. operator-coding-red-line - Check coding constraints (skip ascendc-code-review to save tokens)
   On compile failure: read last 50 lines of error → targeted fix → re-run build.sh → repeat up to 3 attempts total → if still failing → compilation_success=false
   On precision failure (compilation_success=true but precision_passed=false): analyze error pattern → targeted fix → re-evaluate → repeat up to 2 attempts → if still failing → precision_passed=false
4. {V4} - Evaluate
5. Local Refinement (MANDATORY):
   If Max Improve Rounds > 0 AND compilation_success=true AND precision_passed=true → execute inner refinement loop (see [Config])
   Otherwise → write "local_refinement_rounds": 0 into evaluation_results.json
   DO NOT return without completing step 5.

[Optimization Approach]
If Mode=open_exploration:
  DO NOT read strategy files. Reason from first principles.
  a. Read best kernel: output/{op_name}_evo_{timestamp}/{best_solution_ref}/{op_name}Custom/op_kernel/{op_name}_custom.cpp
     (if best_solution_ref is empty, use DSL baseline)
  b. Analyze [Hardware Specs] below
  c. Identify unaddressed bottleneck. Design novel {V6_scope} optimization.
  d. Implement COMPLETE new kernel (replace, don't patch)
  e. In implementation_note: state novel technique and rationale

If Mode=profiling_driven:
  DO NOT read strategy files. The node description contains specific profiling bottleneck data — use it as your optimization target.
  a. Read parent kernel at: output/{op_name}_evo_{timestamp}/{parent_solution_ref}/{op_name}Custom/op_kernel/{op_name}_custom.cpp
  b. Parse the profiling bottleneck from [CONTEXT] node description
  c. Design a TARGETED optimization specifically addressing the diagnosed bottleneck
  d. Apply optimization in {V6_passes}, modifying parent kernel (patch, don't replace from scratch)
  e. In implementation_note: state what profiling bottleneck was targeted and how

If Mode=strategy_guided (default):
  The [MANDATORY OPTIMIZATION DIRECTION] already contains the strategy direction and key points.
  DO NOT read strategy files from evolution/meta_prompts/strategies/ — implement based on the direction description above.
  If the direction is unclear, read at most 1 strategy file for the primary strategy ID only.
  If parent_solution_ref non-empty:
    Read parent kernel at: output/{op_name}_evo_{timestamp}/{parent_solution_ref}/{op_name}Custom/op_kernel/{op_name}_custom.cpp
    Use as optimization starting point.
  Apply in {V6_passes}
  {meta_prompt}

[CONTEXT]
Node: {node_id} | Parent: {parent_solution_ref} | Best: {best_solution_ref}

{inspirations_text}

Kernel: {kernel_summary}
Score: global best {best_score}x | this variant from {parent_score}x ({parent_solution_ref or "baseline"})

[HISTORY]
{open_questions_rendered}

[Config]
Refinement: max={max_improve_rounds}, stagnation={improve_stagnation_window}
Hardware: {hw_params_one_liner}
{V5}

Log progress after each step. Save all outputs to output directory. Return evaluation_results.json.
```
