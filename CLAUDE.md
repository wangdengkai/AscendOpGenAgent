# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LINGXI is an AI-powered system that generates high-performance AscendC operators for Ascend NPU hardware. It converts PyTorch Model descriptions through a TileLang design expression and AscendC translation pipeline: **PyTorch Model → TileLang Design → AscendC Kernel**.

## Commands

### Agent Usage (Primary Interface)

```bash
mkdir genop && cp -r .claude genop/ && cd genop
# Launch Claude Code, press Tab to select agent:
#   lingxi     - Standard operator generation (TileLang→AscendC pipeline)
#   lingxi-evo - Evolutionary optimization with parallel variants
#   ops-evo  - Evolutionary optimization for ops-nn/cv/math/transformer operators
```

## Architecture

### Multi-Stage Generation Pipeline

The lingxi agent uses a TileLang → AscendC pipeline with 8 phases:

1. **Phase 0: Parameter confirmation** — Parse npu, op_file, output_dir
2. **Phase 1: Environment setup** — Copy model.py and test cases to output directory
3. **Phase 2: Test case simplification** — Reduce test cases to ≤10 (case-simplifier skill)
4. **Phase 3: TileLang design expression** — Generate block-level and tile-level designs (tilelang-designer skill, iterative with degeneration detection)
5. **Phase 4: AscendC translation & verification** — Translate TileLang → AscendC kernel (ascendc-translator skill, iterative with degeneration detection)
6. **Phase 5: Performance analysis** — Benchmark reference vs AscendC implementation (performance-analyzer skill)
7. **Phase 6: Full test case validation** — Run full test suite with evaluate_ascendc.sh
8. **Phase 7: Trace recording** — Generate structured execution trace (trace-recorder skill)

### Agents (`.claude/agents/`)

- **lingxi** (`lingxi.md`): Main orchestrator using TileLang→AscendC pipeline. Skills are pre-loaded into context (not called as tools). The agent uses Write/Edit/Bash directly following skill instructions. Max 3 retries on failure. Outputs Chinese-language progress updates.
- **lingxi-evo** (`lingxi-evo.md`): Evolutionary optimizer. Executes shared Phase 0-3 once, then spawns parallel lingxi-partial subagents for AscendC translation variants. Uses world model decision tree for evidence-driven strategy selection.
- **lingxi-partial** (`lingxi-partial.md`): Parallel subagent for lingxi-evo. Executes AscendC translation, verification, and Local Refinement for individual variants.
- **ops-evo** (`ops-evo.md`): Evolutionary optimizer for operators in ops-nn/cv/math/transformer repositories. Works directly in the repository, uses serial builds with `build.sh`, and parallel profiling. Compares baseline vs evolved performance.
- **ops-partial** (`ops-partial.md`): Code modification subagent for ops-evo. Only modifies kernel/tiling code — does not build or evaluate. Returns modified files for the main agent to build.

### Evolution System (`evolution/`)

- **`meta_prompts/`** — Strategy index, strategy files, prompt templates, and meta-prompts for evolution.
- **`knowledge_base/`** — Curated knowledge for evolution decision-making (hardware heuristics, optimization patterns, API pitfalls, algorithm insights).
- **`world_model/`** — World model operations, schema, and scripts (wm_ops.py for refine/select/diagnose).
- Scoring: 0=compile_fail, 100=runtime_error, 200=precision_fail, 300=pass, 350=optimal (speedup ≥ 2.0x).

### Output Directory Structure

```
output/{op_name}/
├── model.py                     # Operator description (PyTorch Model)
├── <op_name>.json               # Test cases (simplified)
├── <op_name>.json.bak           # Original test cases backup
├── design/                      # TileLang design
│   ├── block_level/             # Block-level design
│   └── tile_level/              # Tile-level design
├── kernel/                      # AscendC kernel files
├── model_new_tilelang.py        # TileLang optimized implementation
├── model_new_ascendc.py         # AscendC optimized implementation
├── evaluation_results.json      # Correctness and performance metrics
└── trace.md                     # Execution trace
```

Evolution output adds round/parallel structure under `output/{op_name}_evo_{timestamp}/`.

## Key Conventions

- All agent outputs (progress, explanations) are in **Chinese**.
- Skills are **pre-loaded** into agent context, not invoked via the Skill tool. Agents use Write/Edit/Bash directly.
- When fixing evaluation precision mismatches, max 2 retry attempts, only modifying files under `output/`.
- Environment variable `ASCEND_HOME_PATH` must be set for CANN compilation.

### Long-Running Loop State & Hooks

Evolution runs maintain a runtime state cursor decoupled from the decision tree:

- **`<evo_dir>/state.json`** — execution cursor (stage, current_round, partial progress, drift status). **Written exclusively by hook + wm_ops**, never by agent prompt. Agent only calls `state_ops.py init` once at session start (step 3.5).
- **`<evo_dir>/world_model.json`** — decision-tree evidence (nodes, scores, profiling). Unchanged.
- See `evolution/world_model/state_schema.md` for the full state machine + infer rules.

**v0.2 design (no circular reasoning)**: every Stop hook entry runs `state_ops.py infer` to re-derive `stage` / `current_round` / `partial_status` from filesystem evidence (round_N/parallel_K/evaluation_results.json existence + wm.session.actual_rounds_completed). LLM-written stage values are overridden if they conflict with disk state.

Three hooks (registered in `.claude/settings.json`, auto-active on clone) guard the loop:

| Hook | Trigger | Purpose |
|---|---|---|
| `.claude/hooks/loop-bash-safety.sh` | PreToolUse(Bash) | Block `cp/mv/rm` with empty path variables or `/`/`$HOME` targets (B1/B2 rules) |
| `.claude/hooks/loop-stop.sh` | Stop | (1) Run `state_ops.py infer` to refresh state from filesystem; (2) Block agent exit when state invariants violated (R2-R10: incomplete partials, drift, must_run, session mismatch, anti-skip) |
| `.claude/hooks/loop-subagent-stop.sh` | SubagentStop | Audit partial subagents' transcripts (S1) — verify they actually invoked evaluation/build scripts via `evolution/world_model/transcript_audit.py` |

**Drift circuit breaker** — when search stalls (`stagnation_count >= 2`), `wm_ops.py refine` auto-writes `state.drift_status=replan_required`. Next round's GATE diverts to `drift_replan` flow:
- `wm_ops.py select` auto-forces `⌈n/2⌉` open_exploration slots
- Agent injects fresh-source prompt to partials
- `wm_ops.py select` auto-clears `drift_status` back to `normal` after consumption

**Emergency bypass** — set `LINGXI_LOOP_HOOK_DISABLE=1` to downgrade all blocks to warn-only. Document use only.

**Resume after crash** — agent reads `state.json` first thing and continues from `stage`. See agent prompts (`lingxi-evo.md` / `ops-evo.md` "重入与状态游标" sections) for the resume protocol. State is always trustworthy because hook/wm_ops are the sole writers.

**Anti-skip rules (v0.3)** — block agent from claiming "done" without doing the work:
- **R6**: blocks stop if `actual_rounds_completed < requested_rounds`
- **R7**: blocks stop if fewer partials produced than `expected_parallel_num`
- **R8**: blocks stop if partials complete but refine never ran
- **R9**: `wm_ops refine` auto-marks `msprof` into `must_run` if profiling artifacts missing (then R4 blocks next round)
- **R10**: stderr warning when ≥50% partials fail precision

All anti-skip rules read filesystem evidence (`evaluation_results.json`, `profiling/`, `round_N/parallel_K/` dirs) written by scripts — agent cannot lie its way around them.

**Regression tests** — `bash tests/hooks/run_all.sh` runs all hook tests (8 suites: bash safety, state ops, state infer, stop hook, wm-state sync, drift breaker, anti-skip, subagent stop).

### Knowledge Base

Two-level knowledge architecture for AscendC development:

- **L1 (Curated)**: `evolution/knowledge_base/` — 19 files of curated knowledge for evolution decision-making (hardware heuristics, optimization patterns, API pitfalls, algorithm insights, proven solutions). Progressive disclosure: read `guide.md` first, drill into details on-demand.
- **L2 (Full Docs)**: `.claude/skills/ascendc-dev-knowledge/references/` — 3742 official AscendC documentation files (~35MB) for precise API lookup during code writing. Includes 5 search guide files and INDEX.md per section.
- **Search priority**: L1 guide.md → L2 grep → WebSearch

## Dependencies

- Python 3.10+
- CANN >= 8.3.RC1
- torch-npu >= 2.6.0.RC1
- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) (source compile)
- `numpy`, `attrs`, `pyyaml`, `decorator`, `scipy`, `psutil`, `protobuf`
