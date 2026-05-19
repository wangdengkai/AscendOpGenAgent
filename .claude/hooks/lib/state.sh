#!/usr/bin/env bash
# state.sh — Stop hook rule checks against state.json.
#
# Sourced by .claude/hooks/loop-stop.sh.
# Exposes:
#   check_stop_rules <evo_dir>
# which echoes a list of violation reasons (one per line) and returns 0 if no
# violations, or 1 if any.

if [[ -n "${LOOP_HOOK_STATE_SOURCED:-}" ]]; then
    return 0
fi
LOOP_HOOK_STATE_SOURCED=1

# Detect concrete violations of state.json invariants.
# Echoes reasons to stdout (caller assembles them); returns 1 if any.
check_stop_rules() {
    local evo_dir="$1"
    local state_file="${evo_dir}/state.json"

    if [[ ! -f "${state_file}" ]]; then
        # No state.json → no active evolution loop → noop
        return 0
    fi

    # Use the Python validator for the heavy lifting; it owns the schema.
    local issues_output
    issues_output="$(python3 "${STATE_OPS_PY}" validate \
        --evo-dir "${evo_dir}" \
        --check-stage-artifacts 2>&1 1>/dev/null)"
    local validate_rc=$?

    if [[ ${validate_rc} -ne 0 ]]; then
        # state_ops.py validate prints issues to stderr, one per line prefixed with "  - "
        echo "${issues_output}" | sed -n 's/^  - /R2: /p'
    fi

    # Additional rules that need stage-aware logic beyond schema validation

    local stage
    stage="$(read_state_field "${evo_dir}" "stage")"

    local drift
    drift="$(read_state_field "${evo_dir}" "drift_status")"

    local must_run_json
    must_run_json="$(read_state_field "${evo_dir}" "must_run_before_next_round")"

    local partial_json
    partial_json="$(read_state_field "${evo_dir}" "partial_status")"

    local state_evo_dir
    state_evo_dir="$(read_state_field "${evo_dir}" "evo_dir")"

    # R3: drift_status=replan_required but agent is trying to stop in mid-round_select
    if [[ "${drift}" == "replan_required" ]]; then
        case "${stage}" in
            round_select|round_generate)
                echo "R3: drift_status=replan_required but stage=${stage}; agent must run drift_replan before stopping"
                ;;
        esac
    fi

    # R4: must_run_before_next_round non-empty and agent entered round_select
    if [[ "${stage}" == "round_select" ]] && [[ -n "${must_run_json}" && "${must_run_json}" != "[]" ]]; then
        echo "R4: must_run_before_next_round=${must_run_json} but stage already=round_select"
    fi

    # R5: session drift (state.evo_dir disagrees with hook-detected evo_dir)
    if [[ -n "${state_evo_dir}" ]]; then
        local state_abs hook_abs
        state_abs="$(python3 -c "import os,sys;print(os.path.abspath('${state_evo_dir}'))" 2>/dev/null)"
        hook_abs="$(python3 -c "import os,sys;print(os.path.abspath('${evo_dir}'))" 2>/dev/null)"
        if [[ -n "${state_abs}" && -n "${hook_abs}" && "${state_abs}" != "${hook_abs}" ]]; then
            echo "R5: state.evo_dir=${state_abs} disagrees with hook-detected evo_dir=${hook_abs}"
        fi
    fi

    # R2.1: stage=round_generate and at least one partial is running/pending
    if [[ "${stage}" == "round_generate" ]]; then
        if echo "${partial_json}" | grep -qE '"(running|pending)"'; then
            echo "R2.1: stage=round_generate but partial_status has running/pending: ${partial_json}"
        fi
    fi

    # -----------------------------------------------------------------
    # Anti-skip rules — block agent from claiming "done" without
    # actually running the required rounds / partials / refine / msprof.
    # All rules rely on filesystem-derived state (post-infer), not LLM-written
    # fields, so they cannot be bypassed by agent lying.
    # -----------------------------------------------------------------

    # Read world_model.json's session for requested_rounds / actual_rounds_completed
    local wm_path="${evo_dir}/world_model.json"
    local requested_rounds=""
    local actual_rounds=""
    if [[ -f "${wm_path}" ]]; then
        requested_rounds="$(python3 -c "
import json, sys
try:
    w = json.load(open('${wm_path}'))
    print(w.get('session',{}).get('requested_rounds',''))
except Exception:
    pass
" 2>/dev/null)"
        actual_rounds="$(python3 -c "
import json, sys
try:
    w = json.load(open('${wm_path}'))
    print(w.get('session',{}).get('actual_rounds_completed',''))
except Exception:
    pass
" 2>/dev/null)"
    fi

    local expected_parallel
    expected_parallel="$(read_state_field "${evo_dir}" "expected_parallel_num")"

    # R6: agent tries to stop before completing requested_rounds.
    # Only enforce when stage is in a round-active phase. Pre-round stages
    # (init/shared_prep/baseline_build/wm_init) and terminal/finalize stages
    # are exempt — pre-round means rounds haven't even started, terminal means
    # the user accepted early exit.
    if [[ -n "${requested_rounds}" && -n "${actual_rounds}" ]]; then
        if [[ "${actual_rounds}" =~ ^[0-9]+$ && "${requested_rounds}" =~ ^[0-9]+$ ]]; then
            if (( actual_rounds < requested_rounds )); then
                case "${stage}" in
                    init|shared_prep|baseline_build|wm_init|done|aborted|finalize|report)
                        # Pre-round (agent hasn't started yet) or terminal — allow.
                        ;;
                    *)
                        echo "R6: actual_rounds_completed=${actual_rounds} < requested_rounds=${requested_rounds}; finish remaining rounds or cancel explicitly"
                        ;;
                esac
            fi
        fi
    fi

    # R7: number of parallel partials actually produced < expected_parallel_num.
    # Only enforced once we've started a round (stage past wm_init/shared_prep)
    # and expected_parallel_num was set at init time (0 = skip check).
    if [[ -n "${expected_parallel}" && "${expected_parallel}" =~ ^[0-9]+$ && "${expected_parallel}" -gt 0 ]]; then
        case "${stage}" in
            round_generate|round_refine|round_react|round_checkpoint)
                local actual_parallel
                actual_parallel="$(python3 -c "
import json
try:
    s = json.load(open('${evo_dir}/state.json'))
    print(len(s.get('partial_status', {})))
except Exception:
    print(0)
" 2>/dev/null)"
                if [[ "${actual_parallel}" =~ ^[0-9]+$ && "${actual_parallel}" -lt "${expected_parallel}" ]]; then
                    echo "R7: only ${actual_parallel} partial(s) produced, expected ${expected_parallel}; start the missing ones"
                fi
                ;;
        esac
    fi

    # R8: all partials done but refine never ran (wm.actual_rounds < max_round seen on disk).
    # This catches "agent collected partial results but skipped wm_ops refine and tried to stop".
    if [[ "${stage}" == "round_generate" && -n "${partial_json}" && "${partial_json}" != "{}" ]]; then
        # No running/pending partials (otherwise R2.1 fires first)
        if ! echo "${partial_json}" | grep -qE '"(running|pending)"'; then
            # Compare actual_rounds vs current_round
            local current_round
            current_round="$(read_state_field "${evo_dir}" "current_round")"
            if [[ "${current_round}" =~ ^[0-9]+$ && "${actual_rounds}" =~ ^[0-9]+$ ]]; then
                if (( actual_rounds < current_round )); then
                    echo "R8: round ${current_round} partials all completed but wm.actual_rounds_completed=${actual_rounds}; run wm_ops.py refine before stopping"
                fi
            fi
        fi
    fi

    return 0
}

export -f check_stop_rules
