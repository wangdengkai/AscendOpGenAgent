#!/usr/bin/env bash
# wm_ops.py ↔ state.json integration test (v0.2 infer mode).
#
# Verifies that wm_ops subcommands trigger _maybe_infer_state which derives
# stage from filesystem evidence. Replaces v0.1's single-field update tests.

set -uo pipefail

WM_OPS="${PROJECT_ROOT}/evolution/world_model/wm_ops.py"
STATE_OPS="${PROJECT_ROOT}/evolution/world_model/state_ops.py"

# Helper: build minimal wm node fixture
_inject_open_nodes() {
    local wm_path="$1"
    python3 -c "
import json
wm = json.load(open('${wm_path}'))
wm['decision_tree'] = {'nodes': {
    'n1': {'id':'n1','mode':'strategy_guided','strategy_combination':['P1'],
           'description':'t','optimization_type':'bandwidth',
           'difficulty':2,'depth':1,'parent_id':'root',
           'status':'open','score':None,'solution_ref':None,'children':[],
           'failure_type':None,'failure_reason':None,'retry_count':0},
}}
wm['best_score'] = 1.0
wm['stagnation_count'] = 0
wm['stagnation_count_vs_base'] = 0
wm['world_model_active'] = True
json.dump(wm, open('${wm_path}','w'), indent=2)
"
}

# Test 1: wm_ops session sets state to wm_init via infer (wm.json exists, no rounds)
TMPDIR1="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR1}" --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
python3 "${WM_OPS}" session --wm-path "${TMPDIR1}/world_model.json" \
    --session-id sid --evo-dir "${TMPDIR1}" --op-name testop --requested-rounds 3 >/dev/null 2>&1
stage="$(python3 -c "import json; print(json.load(open('${TMPDIR1}/state.json'))['stage'])")"
rm -rf "${TMPDIR1}"
[[ "${stage}" == "wm_init" ]] || { echo "FAIL: stage after session=${stage} (expected wm_init)"; exit 1; }

# Test 2: wm_ops select WITH round_1 dir → infer detects round_select
TMPDIR2="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR2}" --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
python3 "${WM_OPS}" session --wm-path "${TMPDIR2}/world_model.json" \
    --session-id sid --evo-dir "${TMPDIR2}" --op-name testop --requested-rounds 3 >/dev/null 2>&1
_inject_open_nodes "${TMPDIR2}/world_model.json"
# Create round_1/ to simulate "agent created the round dir"
mkdir -p "${TMPDIR2}/round_1"
python3 "${WM_OPS}" select --path "${TMPDIR2}/world_model.json" --n 1 >/dev/null 2>&1
stage="$(python3 -c "import json; print(json.load(open('${TMPDIR2}/state.json'))['stage'])")"
rm -rf "${TMPDIR2}"
[[ "${stage}" == "round_select" ]] || { echo "FAIL: stage after select with round_1/=${stage} (expected round_select)"; exit 1; }

# Test 3: wm_ops works without state.json (backward compat)
TMPDIR3="$(mktemp -d)"
# No state_ops init this time
python3 "${WM_OPS}" session --wm-path "${TMPDIR3}/world_model.json" \
    --session-id sid --evo-dir "${TMPDIR3}" --op-name testop --requested-rounds 3 >/dev/null 2>&1
rc=$?
[[ ${rc} -eq 0 ]] || { echo "FAIL: wm_ops should work without state.json, rc=${rc}"; rm -rf "${TMPDIR3}"; exit 1; }
[[ -f "${TMPDIR3}/world_model.json" ]] || { echo "FAIL: wm_ops didn't create world_model.json"; rm -rf "${TMPDIR3}"; exit 1; }
[[ ! -f "${TMPDIR3}/state.json" ]] || { echo "FAIL: wm_ops should not create state.json"; rm -rf "${TMPDIR3}"; exit 1; }
rm -rf "${TMPDIR3}"

# Test 4 (NEW v0.2): Construct full round_generate filesystem state, run select,
# verify infer detects round_generate (partial running) — NOT round_select
TMPDIR4="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR4}" --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
python3 "${WM_OPS}" session --wm-path "${TMPDIR4}/world_model.json" \
    --session-id sid --evo-dir "${TMPDIR4}" --op-name testop --requested-rounds 3 >/dev/null 2>&1
_inject_open_nodes "${TMPDIR4}/world_model.json"
mkdir -p "${TMPDIR4}/round_1/parallel_0" "${TMPDIR4}/round_1/parallel_1"
echo '{"speedup":0.8}' > "${TMPDIR4}/round_1/parallel_0/evaluation_results.json"
# parallel_1 NO eval → infer should say partial_1 is running
python3 "${WM_OPS}" select --path "${TMPDIR4}/world_model.json" --n 1 >/dev/null 2>&1
state="$(python3 -c "
import json
s = json.load(open('${TMPDIR4}/state.json'))
print(s['stage'], s['partial_status'].get('0','-'), s['partial_status'].get('1','-'))
")"
rm -rf "${TMPDIR4}"
[[ "${state}" == "round_generate completed running" ]] || { echo "FAIL: expected 'round_generate completed running', got '${state}'"; exit 1; }

exit 0
