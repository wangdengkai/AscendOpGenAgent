#!/usr/bin/env bash
# Stop hook tests (v0.2). Fixtures construct filesystem evidence so the
# hook's infer-on-entry derives state from disk, mirroring real runtime.
# Verifies R2.1, R3, R4 and graceful no-state-json case.

set -uo pipefail

STOP_HOOK="${PROJECT_ROOT}/.claude/hooks/loop-stop.sh"
STATE_OPS="${PROJECT_ROOT}/evolution/world_model/state_ops.py"

# Helper: run hook in a temp dir, echoing stop_hook_active=false
_run_hook_in() {
    local d="$1"
    (cd "${d}" && echo '{"stop_hook_active":false}' | "${STOP_HOOK}" 2>&1)
}

# Helper: build a minimal evo_dir with wm.json (no rounds yet → stage=wm_init)
_build_minimal_evo() {
    local d="$1"
    python3 "${STATE_OPS}" init --evo-dir "${d}" \
        --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
    cat > "${d}/world_model.json" <<EOF
{"session":{"requested_rounds":3,"actual_rounds_completed":0,"session_id":"sid","evo_dir":"${d}","op_name":"test"},"best_score":1.0,"stagnation_count":0,"stagnation_count_vs_base":0,"decision_tree":{"nodes":{}}}
EOF
}

# Helper: simulate round_N/parallel_K with optional eval result
_make_partial() {
    local round_dir="$1"   # e.g. evo/round_1
    local idx="$2"
    local has_eval="$3"    # "yes" or "no"
    mkdir -p "${round_dir}/parallel_${idx}"
    if [[ "${has_eval}" == "yes" ]]; then
        echo '{"compilation_success":true,"precision_passed":true,"speedup":0.7}' \
            > "${round_dir}/parallel_${idx}/evaluation_results.json"
    fi
}

# ----------------------------------------------------------------------
# Test 1: no state.json → allow
# ----------------------------------------------------------------------
TMPDIR1="$(mktemp -d)"
out="$(_run_hook_in "${TMPDIR1}")"
rc=$?
rm -rf "${TMPDIR1}"
[[ ${rc} -eq 0 ]] || { echo "FAIL: no state.json should allow exit=${rc}"; exit 1; }

# ----------------------------------------------------------------------
# Test 2: fresh state, no rounds → wm_init stage, hook allows
# ----------------------------------------------------------------------
TMPDIR2="$(mktemp -d)"
_build_minimal_evo "${TMPDIR2}"
out="$(_run_hook_in "${TMPDIR2}")"
rc=$?
rm -rf "${TMPDIR2}"
[[ ${rc} -eq 0 ]] || { echo "FAIL: wm_init stage should allow exit=${rc} out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# Test 3: round_1/parallel_0 has eval, parallel_1 doesn't
#         → infer should detect round_generate + R2.1 BLOCK
# ----------------------------------------------------------------------
TMPDIR3="$(mktemp -d)"
_build_minimal_evo "${TMPDIR3}"
_make_partial "${TMPDIR3}/round_1" 0 yes
_make_partial "${TMPDIR3}/round_1" 1 no
out="$(_run_hook_in "${TMPDIR3}")"
rc=$?
# Even if user manually wrote stage=done, hook infer overrides it
rm -rf "${TMPDIR3}"
[[ ${rc} -eq 2 ]] || { echo "FAIL: R2.1 should BLOCK exit=${rc} out=${out}"; exit 1; }
[[ "${out}" == *"R2.1"* ]] || { echo "FAIL: missing R2.1 marker out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# Test 4: stop_hook_active=true → allow (no recursion)
# ----------------------------------------------------------------------
TMPDIR4="$(mktemp -d)"
_build_minimal_evo "${TMPDIR4}"
_make_partial "${TMPDIR4}/round_1" 0 no
out="$(cd "${TMPDIR4}" && echo '{"stop_hook_active":true}' | "${STOP_HOOK}" 2>&1)"
rc=$?
rm -rf "${TMPDIR4}"
[[ ${rc} -eq 0 ]] || { echo "FAIL: stop_hook_active=true should allow exit=${rc} out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# Test 5: round_2 partials done + drift_status=replan_required
#         → infer stage=round_checkpoint (NOT round_select)
#         → R3 should NOT fire (stage is past round_select)
#         BUT we want R3 to fire when stage IS round_select. So construct
#         round_2/ with no parallels (→ stage=round_select)
# ----------------------------------------------------------------------
TMPDIR5="$(mktemp -d)"
_build_minimal_evo "${TMPDIR5}"
# Round 1 fully complete and refined
mkdir -p "${TMPDIR5}/round_1"
_make_partial "${TMPDIR5}/round_1" 0 yes
# Bump actual_rounds_completed to 1
python3 -c "
import json
wm = json.load(open('${TMPDIR5}/world_model.json'))
wm['session']['actual_rounds_completed'] = 1
json.dump(wm, open('${TMPDIR5}/world_model.json','w'))
"
# Round 2 dir but no parallels yet → infer = round_select
mkdir -p "${TMPDIR5}/round_2"
python3 "${STATE_OPS}" set-drift --evo-dir "${TMPDIR5}" --status replan_required >/dev/null
out="$(_run_hook_in "${TMPDIR5}")"
rc=$?
rm -rf "${TMPDIR5}"
[[ ${rc} -eq 2 ]] || { echo "FAIL: R3 should BLOCK exit=${rc} out=${out}"; exit 1; }
[[ "${out}" == *"R3"* ]] || { echo "FAIL: missing R3 marker out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# Test 6: must_run + stage=round_select → BLOCK (R4)
# ----------------------------------------------------------------------
TMPDIR6="$(mktemp -d)"
_build_minimal_evo "${TMPDIR6}"
mkdir -p "${TMPDIR6}/round_1"
# No parallels → infer stage=round_select
python3 "${STATE_OPS}" mark-must-run --evo-dir "${TMPDIR6}" --step msprof >/dev/null
out="$(_run_hook_in "${TMPDIR6}")"
rc=$?
rm -rf "${TMPDIR6}"
[[ ${rc} -eq 2 ]] || { echo "FAIL: R4 should BLOCK exit=${rc} out=${out}"; exit 1; }
[[ "${out}" == *"R4"* ]] || { echo "FAIL: missing R4 marker out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# Test 7: max_round == requested_rounds + all partials done → stage=finalize/done
#         (treated as terminal, hook allows)
# ----------------------------------------------------------------------
TMPDIR7="$(mktemp -d)"
_build_minimal_evo "${TMPDIR7}"
python3 -c "
import json
wm = json.load(open('${TMPDIR7}/world_model.json'))
wm['session']['actual_rounds_completed'] = 3
wm['session']['requested_rounds'] = 3
json.dump(wm, open('${TMPDIR7}/world_model.json','w'))
"
mkdir -p "${TMPDIR7}/round_3"
_make_partial "${TMPDIR7}/round_3" 0 yes
# Manually set stage=done (just to verify it's NOT overridden incorrectly)
python3 "${STATE_OPS}" write-stage --evo-dir "${TMPDIR7}" --stage done >/dev/null
out="$(_run_hook_in "${TMPDIR7}")"
rc=$?
rm -rf "${TMPDIR7}"
# After infer, stage=finalize (max==requested). loop-stop allows when stage in terminal set
# Currently terminal_set includes "done" and "aborted" only — finalize is non-terminal but no rules violate it
[[ ${rc} -eq 0 ]] || { echo "FAIL: finalize stage should allow exit=${rc} out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# Test 8: HOOK_DISABLE downgrades violations to warn
# ----------------------------------------------------------------------
TMPDIR8="$(mktemp -d)"
_build_minimal_evo "${TMPDIR8}"
_make_partial "${TMPDIR8}/round_1" 0 yes
_make_partial "${TMPDIR8}/round_1" 1 no
out="$(cd "${TMPDIR8}" && LINGXI_LOOP_HOOK_DISABLE=1 bash -c "echo '{\"stop_hook_active\":false}' | '${STOP_HOOK}'" 2>&1)"
rc=$?
rm -rf "${TMPDIR8}"
[[ ${rc} -eq 0 ]] || { echo "FAIL: HOOK_DISABLE should allow exit=${rc} out=${out}"; exit 1; }
[[ "${out}" == *"WARN"* ]] || { echo "FAIL: HOOK_DISABLE missing WARN out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# Test 9 (v0.2 specific): LLM wrote stage=done but filesystem says round_generate
#                          → infer overrides to round_generate, R2.1 BLOCKs
# ----------------------------------------------------------------------
TMPDIR9="$(mktemp -d)"
_build_minimal_evo "${TMPDIR9}"
_make_partial "${TMPDIR9}/round_1" 0 yes
_make_partial "${TMPDIR9}/round_1" 1 no
# Adversarial: write stage=done in state.json
python3 "${STATE_OPS}" write-stage --evo-dir "${TMPDIR9}" --stage done >/dev/null
out="$(_run_hook_in "${TMPDIR9}")"
rc=$?
# Verify infer overrode the stage
final_stage="$(python3 -c "import json; print(json.load(open('${TMPDIR9}/state.json'))['stage'])")"
rm -rf "${TMPDIR9}"
[[ "${final_stage}" == "round_generate" ]] || { echo "FAIL: infer didn't override LLM stage, got ${final_stage}"; exit 1; }
[[ ${rc} -eq 2 ]] || { echo "FAIL: hook should block adversarial done, exit=${rc} out=${out}"; exit 1; }

exit 0
