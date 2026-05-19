#!/usr/bin/env bash
# loop-stop.sh — Stop hook for Z-Search loop.
#
# Activated by .claude/settings.json. Triggered when the agent attempts to
# stop. Receives Claude Code's hook JSON on stdin (we mostly don't need it).
#
# Behavior:
#   1. Locate state.json by walking up from cwd.
#   2. If state.json absent or stage is terminal (done/aborted) → exit 0 (allow stop).
#   3. Otherwise run rule checks (R2-R5) and block (exit 2) if violations found.
#
# Rules (see evolution/world_model/state_schema.md):
#   R2.1 stage=round_generate but partials are running/pending
#   R2.x stage=round_refine/round_react/round_checkpoint but evaluation_results.json missing
#   R3   drift_status=replan_required and stage in {round_select, round_generate}
#   R4   must_run_before_next_round non-empty and stage=round_select
#   R5   state.evo_dir disagrees with hook-detected evo_dir
#
# Escape hatch: LINGXI_LOOP_HOOK_DISABLE=1 downgrades to warn-only.

set -uo pipefail

HOOK_NAME="loop-stop"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/state.sh
source "${SCRIPT_DIR}/lib/state.sh"

# Read stdin (we don't strictly need it but slurp to avoid SIGPIPE)
read_stdin_json || true

evo_dir="$(find_evo_dir)"
if [[ -z "${evo_dir}" ]]; then
    # No active evolution loop — never block stops
    hook_allow
fi

# Re-infer state from filesystem before validating rules. This ensures
# R2/R3/R4 check against hook-derived state, not LLM-written state.
# Failure is non-fatal — fall through to rule check with whatever state exists.
if [[ -f "${STATE_OPS_PY}" ]]; then
    python3 "${STATE_OPS_PY}" infer --evo-dir "${evo_dir}" --quiet 2>/dev/null || \
        hook_log "state_ops infer failed (non-fatal, proceeding with existing state)"
fi

stage="$(read_state_field "${evo_dir}" "stage")"

# Terminal stages: always allow stop
case "${stage}" in
    done|aborted|"")
        hook_allow
        ;;
esac

# Stop_hook_active guard: per Claude Code docs, hook input has a
# stop_hook_active=true field on the second invocation in a row.
# Don't recurse into ourselves.
stop_hook_active="$(json_get "stop_hook_active")"
if [[ "${stop_hook_active}" == "True" || "${stop_hook_active}" == "true" ]]; then
    hook_allow
fi

# Run rule checks. check_stop_rules echoes one reason per line.
violations="$(check_stop_rules "${evo_dir}")"

if [[ -n "${violations}" ]]; then
    # Assemble a multi-line block reason
    reason="$(echo "${violations}" | head -10)"
    hook_block "Stop blocked due to state.json invariant violations:
${reason}

Hint: run 'python3 evolution/world_model/state_ops.py read --evo-dir ${evo_dir}' to inspect current state, then resolve the issue (complete missing partials, run wm_ops.py refine, etc.) before retrying.
To bypass in emergency only: export LINGXI_LOOP_HOOK_DISABLE=1"
fi

hook_allow
