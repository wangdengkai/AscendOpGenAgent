#!/usr/bin/env bash
# Drift circuit breaker tests (Phase E).
# Verifies that:
#   1. refine writes state.drift_status=replan_required when stagnation_count >= 2
#   2. select detects drift and forces force_open_exploration_min
#   3. drift_status auto-clears when stagnation recovers

set -uo pipefail

WM_OPS="${PROJECT_ROOT}/evolution/world_model/wm_ops.py"
STATE_OPS="${PROJECT_ROOT}/evolution/world_model/state_ops.py"

_init_state() {
    local d="$1"
    python3 "${STATE_OPS}" init --evo-dir "${d}" \
        --agent lingxi-evo --session-id sid --max-rounds 5 >/dev/null
}

_build_wm_with_nodes() {
    local d="$1"
    local sc="${2:-0}"
    local scvb="${3:-0}"
    python3 -c "
import json
node = lambda nid, mode, dt='bandwidth': {
  'id': nid, 'mode': mode,
  'strategy_combination': ['P1'] if mode=='strategy_guided' else [],
  'description': 'test '+nid, 'optimization_type': dt,
  'difficulty': 2, 'depth': 1, 'parent_id': 'root',
  'status': 'open', 'score': None, 'solution_ref': None, 'children': [],
  'failure_type': None, 'failure_reason': None, 'retry_count': 0,
}
wm = {
  'best_score': 1.0,
  'stagnation_count': ${sc}, 'stagnation_count_vs_base': ${scvb},
  'world_model_active': True,
  'decision_tree': {'nodes': {
    'n1': node('n1','strategy_guided','bandwidth'),
    'n2': node('n2','strategy_guided','tiling'),
    'n3': node('n3','strategy_guided','algorithm'),
    'n4': node('n4','strategy_guided','bandwidth'),
    'x0': node('x0','open_exploration'),
    'x1': node('x1','open_exploration'),
  }}
}
json.dump(wm, open('${d}/world_model.json','w'), indent=2)
"
}

# Test 1: _decide_drift_status helper logic
out="$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}/evolution/world_model')
from wm_ops import _decide_drift_status, DRIFT_THRESHOLD
assert DRIFT_THRESHOLD == 2, f'expected threshold 2, got {DRIFT_THRESHOLD}'
status, _ = _decide_drift_status({'stagnation_count': 1, 'stagnation_count_vs_base': 1})
assert status == 'normal', f'sc=1,scvb=1 should be normal, got {status}'
status, _ = _decide_drift_status({'stagnation_count': 2, 'stagnation_count_vs_base': 0})
assert status == 'replan_required', f'sc=2 should trigger drift, got {status}'
status, _ = _decide_drift_status({'stagnation_count': 0, 'stagnation_count_vs_base': 2})
assert status == 'replan_required', f'scvb=2 should trigger drift, got {status}'
print('OK')
" 2>&1)"
[[ "${out}" == "OK" ]] || { echo "FAIL test1: decide_drift_status: ${out}"; exit 1; }

# Test 2: _maybe_update_drift_status writes state.drift_status when sc>=2
TMPDIR2="$(mktemp -d)"
_init_state "${TMPDIR2}"
python3 -c "
import sys, json
sys.path.insert(0, '${PROJECT_ROOT}/evolution/world_model')
from wm_ops import _maybe_update_drift_status
wm = {'stagnation_count': 2}
json.dump(wm, open('${TMPDIR2}/world_model.json','w'))
_maybe_update_drift_status('${TMPDIR2}/world_model.json', wm)
"
drift="$(python3 -c "import json; print(json.load(open('${TMPDIR2}/state.json'))['drift_status'])")"
rm -rf "${TMPDIR2}"
[[ "${drift}" == "replan_required" ]] || { echo "FAIL test2: drift=${drift} (expected replan_required)"; exit 1; }

# Test 3: cmd_refine end-to-end triggers drift when stagnation_count crosses threshold
TMPDIR3="$(mktemp -d)"
_init_state "${TMPDIR3}"
# Build a wm with: best_score=1.0, one passed node with score=0.5 (stagnation will increment)
python3 -c "
import json
nodes = {
  'n1': {
    'id':'n1','mode':'strategy_guided','strategy_combination':['P1'],
    'description':'test','optimization_type':'bandwidth',
    'difficulty':2,'depth':1,'parent_id':'root',
    'status':'in_progress','score':None,'solution_ref':None,'children':[],
    'failure_type':None,'failure_reason':None,'retry_count':0,
  }
}
wm = {
  'best_score': 1.0, 'stagnation_count': 1, 'stagnation_count_vs_base': 0,
  'world_model_active': True, 'decision_tree': {'nodes': nodes},
  'session': {'session_id':'sid','requested_rounds':5,'actual_rounds_completed':0,
              'evo_dir':'${TMPDIR3}','op_name':'test'},
}
json.dump(wm, open('${TMPDIR3}/world_model.json','w'), indent=2)
"
# Fake round results: n1 passes but with speedup < best_score → stagnation_count incremented
mkdir -p "${TMPDIR3}/round_2/parallel_0"
python3 -c "
import json
json.dump({
  'compilation_success': True, 'precision_passed': True, 'speedup': 0.8,
  'evolved': {'pipeline': None},
}, open('${TMPDIR3}/round_2/parallel_0/evaluation_results.json','w'))
"
python3 "${WM_OPS}" refine \
    --wm-path "${TMPDIR3}/world_model.json" \
    --round 2 \
    --results-dir "${TMPDIR3}/round_2" \
    --parallel-map '{"0":"n1"}' \
    --task-type vector >/dev/null 2>&1
drift="$(python3 -c "import json; print(json.load(open('${TMPDIR3}/state.json'))['drift_status'])")"
sc="$(python3 -c "import json; print(json.load(open('${TMPDIR3}/world_model.json'))['stagnation_count'])")"
rm -rf "${TMPDIR3}"
[[ "${sc}" -ge 2 ]] || { echo "FAIL test3: stagnation_count=${sc} (expected >=2)"; exit 1; }
[[ "${drift}" == "replan_required" ]] || { echo "FAIL test3: drift=${drift} after refine"; exit 1; }

# Test 4: cmd_select with drift=replan_required raises oe_slots
TMPDIR4="$(mktemp -d)"
_init_state "${TMPDIR4}"
_build_wm_with_nodes "${TMPDIR4}"
python3 "${STATE_OPS}" set-drift --evo-dir "${TMPDIR4}" --status replan_required >/dev/null

out="$(python3 "${WM_OPS}" select --path "${TMPDIR4}/world_model.json" --n 4 2>/dev/null)"
oe_count="$(echo "${out}" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(sum(1 for n in r if n['mode']=='open_exploration'))
")"
rm -rf "${TMPDIR4}"
[[ "${oe_count}" -ge 2 ]] || { echo "FAIL test4: drift select oe_count=${oe_count} (expected >=2)"; exit 1; }

# Test 5: normal drift_status → oe_slots = ceil(n/4) (baseline)
TMPDIR5="$(mktemp -d)"
_init_state "${TMPDIR5}"
_build_wm_with_nodes "${TMPDIR5}"
# drift_status stays "normal" (default after init)

out="$(python3 "${WM_OPS}" select --path "${TMPDIR5}/world_model.json" --n 4 2>/dev/null)"
oe_count="$(echo "${out}" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(sum(1 for n in r if n['mode']=='open_exploration'))
")"
rm -rf "${TMPDIR5}"
[[ "${oe_count}" -eq 1 ]] || { echo "FAIL test5: normal select oe_count=${oe_count} (expected 1)"; exit 1; }

# Test 6: drift auto-recovery — when sc drops back to 0, drift cleared to normal
TMPDIR6="$(mktemp -d)"
_init_state "${TMPDIR6}"
# Pre-set drift to replan_required
python3 "${STATE_OPS}" set-drift --evo-dir "${TMPDIR6}" --status replan_required >/dev/null
# Now call helper with sc=0
python3 -c "
import sys, json
sys.path.insert(0, '${PROJECT_ROOT}/evolution/world_model')
from wm_ops import _maybe_update_drift_status
wm = {'stagnation_count': 0, 'stagnation_count_vs_base': 0, 'best_score': 1.5}
json.dump(wm, open('${TMPDIR6}/world_model.json','w'))
_maybe_update_drift_status('${TMPDIR6}/world_model.json', wm)
"
drift="$(python3 -c "import json; print(json.load(open('${TMPDIR6}/state.json'))['drift_status'])")"
rm -rf "${TMPDIR6}"
[[ "${drift}" == "normal" ]] || { echo "FAIL test6: drift auto-recovery failed, drift=${drift}"; exit 1; }

# Test 7: backward compat — wm without state.json, drift helpers must noop silently
TMPDIR7="$(mktemp -d)"
# No state_ops init this time
python3 -c "
import sys, json
sys.path.insert(0, '${PROJECT_ROOT}/evolution/world_model')
from wm_ops import _maybe_update_drift_status
wm = {'stagnation_count': 5}
json.dump(wm, open('${TMPDIR7}/world_model.json','w'))
_maybe_update_drift_status('${TMPDIR7}/world_model.json', wm)
print('returned without error')
" 2>&1 | grep -q "returned without error" \
    || { echo "FAIL test7: drift helper crashed when state.json missing"; rm -rf "${TMPDIR7}"; exit 1; }
[[ ! -f "${TMPDIR7}/state.json" ]] || { echo "FAIL test7: helper should not create state.json"; rm -rf "${TMPDIR7}"; exit 1; }
rm -rf "${TMPDIR7}"

exit 0
