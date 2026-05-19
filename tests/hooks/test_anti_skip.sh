#!/usr/bin/env bash
# Anti-skip rule tests (v0.3 R6 / R7 / R8 / R9 / R10).
#
# Each test constructs a filesystem state that simulates an agent trying to
# "skip a required step", then verifies the framework catches it (either via
# Stop hook block or via wm_ops auto-mark).

set -uo pipefail

STOP_HOOK="${PROJECT_ROOT}/.claude/hooks/loop-stop.sh"
STATE_OPS="${PROJECT_ROOT}/evolution/world_model/state_ops.py"
WM_OPS="${PROJECT_ROOT}/evolution/world_model/wm_ops.py"

_init_with_wm() {
    local d="$1"
    local max_rounds="$2"
    local parallel_num="$3"
    local requested="$4"
    local actual="$5"
    python3 "${STATE_OPS}" init \
        --evo-dir "${d}" --agent lingxi-evo --session-id sid \
        --max-rounds "${max_rounds}" --parallel-num "${parallel_num}" >/dev/null
    python3 -c "
import json
wm = {
  'session': {
    'requested_rounds': ${requested},
    'actual_rounds_completed': ${actual},
    'session_id': 'sid', 'op_name': 'test', 'evo_dir': '${d}',
  },
  'best_score': 1.0,
  'stagnation_count': 0, 'stagnation_count_vs_base': 0,
  'world_model_active': True,
  'decision_tree': {'nodes': {}},
}
json.dump(wm, open('${d}/world_model.json','w'), indent=2)
"
}

_run_hook() {
    local d="$1"
    (cd "${d}" && echo '{"stop_hook_active":false}' | "${STOP_HOOK}" 2>&1)
}

# ----------------------------------------------------------------------
# T1 (R6): max_rounds=3, requested=3, actual=1 → BLOCK
# ----------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
_init_with_wm "${TMPDIR}" 3 2 3 1
mkdir -p "${TMPDIR}/round_1/parallel_0" "${TMPDIR}/round_1/parallel_1"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
echo '{}' > "${TMPDIR}/round_1/parallel_1/evaluation_results.json"
out="$(_run_hook "${TMPDIR}")"
rc=$?
rm -rf "${TMPDIR}"
[[ ${rc} -eq 2 ]] || { echo "FAIL T1: R6 should BLOCK exit=${rc}"; exit 1; }
[[ "${out}" == *"R6"* ]] || { echo "FAIL T1: missing R6 marker out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# T2 (R6): actual == requested → no R6
# ----------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
_init_with_wm "${TMPDIR}" 1 2 1 1
mkdir -p "${TMPDIR}/round_1/parallel_0" "${TMPDIR}/round_1/parallel_1"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
echo '{}' > "${TMPDIR}/round_1/parallel_1/evaluation_results.json"
out="$(_run_hook "${TMPDIR}")"
rm -rf "${TMPDIR}"
[[ "${out}" != *"R6"* ]] || { echo "FAIL T2: R6 should NOT fire when actual==requested out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# T3 (R7): expected_parallel_num=3, only 1 partial dir → BLOCK
# ----------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
_init_with_wm "${TMPDIR}" 1 3 1 0
mkdir -p "${TMPDIR}/round_1/parallel_0"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
out="$(_run_hook "${TMPDIR}")"
rc=$?
rm -rf "${TMPDIR}"
[[ ${rc} -eq 2 ]] || { echo "FAIL T3: R7 should BLOCK exit=${rc}"; exit 1; }
[[ "${out}" == *"R7"* ]] || { echo "FAIL T3: missing R7 marker out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# T4 (R7 disabled): expected_parallel_num=0 → no R7 (backward compat)
# ----------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
_init_with_wm "${TMPDIR}" 1 0 1 0
mkdir -p "${TMPDIR}/round_1/parallel_0"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
out="$(_run_hook "${TMPDIR}")"
rm -rf "${TMPDIR}"
[[ "${out}" != *"R7"* ]] || { echo "FAIL T4: R7 should be disabled when expected_parallel_num=0 out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# T5 (R8): partials all completed but actual_rounds_completed=0 (refine skipped) → BLOCK
# ----------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
_init_with_wm "${TMPDIR}" 2 2 2 0
mkdir -p "${TMPDIR}/round_1/parallel_0" "${TMPDIR}/round_1/parallel_1"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
echo '{}' > "${TMPDIR}/round_1/parallel_1/evaluation_results.json"
# infer → stage=round_generate (actual<max_round), partials all completed → R8 fires
out="$(_run_hook "${TMPDIR}")"
rc=$?
rm -rf "${TMPDIR}"
[[ ${rc} -eq 2 ]] || { echo "FAIL T5: R8 should BLOCK exit=${rc}"; exit 1; }
[[ "${out}" == *"R8"* ]] || { echo "FAIL T5: missing R8 marker out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# T6 (R9): wm_ops refine detects missing profiling, marks must_run += msprof
# ----------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
_init_with_wm "${TMPDIR}" 3 2 3 0
mkdir -p "${TMPDIR}/round_1/parallel_0" "${TMPDIR}/round_1/parallel_1"
echo '{"compilation_success":true,"precision_passed":true,"speedup":0.8,"evolved":{"pipeline":null}}' \
    > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
echo '{"compilation_success":true,"precision_passed":true,"speedup":0.7,"evolved":{"pipeline":null}}' \
    > "${TMPDIR}/round_1/parallel_1/evaluation_results.json"
# Inject minimal wm nodes
python3 -c "
import json
wm = json.load(open('${TMPDIR}/world_model.json'))
node = lambda nid: {'id':nid,'mode':'strategy_guided','strategy_combination':[],
  'description':'t','optimization_type':'bandwidth',
  'difficulty':2,'depth':1,'parent_id':'root',
  'status':'in_progress','score':None,'solution_ref':None,'children':[],
  'failure_type':None,'failure_reason':None,'retry_count':0}
wm['decision_tree'] = {'nodes': {'n1': node('n1'), 'n2': node('n2')}}
json.dump(wm, open('${TMPDIR}/world_model.json','w'), indent=2)
"
# Note: no profiling/ dir created → R9 should fire
python3 "${WM_OPS}" refine \
    --wm-path "${TMPDIR}/world_model.json" \
    --round 1 \
    --results-dir "${TMPDIR}/round_1" \
    --parallel-map '{"0":"n1","1":"n2"}' \
    --task-type vector >/dev/null 2>&1
must_run="$(python3 -c "import json; print(json.load(open('${TMPDIR}/state.json'))['must_run_before_next_round'])")"
rm -rf "${TMPDIR}"
[[ "${must_run}" == *"msprof"* ]] || { echo "FAIL T6: R9 should add msprof to must_run, got ${must_run}"; exit 1; }

# ----------------------------------------------------------------------
# T7 (R9 negative): profiling dir present with op_summary CSV → no must_run mark
# ----------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
_init_with_wm "${TMPDIR}" 3 1 3 0
mkdir -p "${TMPDIR}/round_1/parallel_0/profiling/ModelNew/PROF/mindstudio_profiler_output"
echo '{"compilation_success":true,"precision_passed":true,"speedup":0.8,"evolved":{"pipeline":null}}' \
    > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
# Create the canonical profiling evidence
echo "Op Name,Task Duration" > "${TMPDIR}/round_1/parallel_0/profiling/ModelNew/PROF/mindstudio_profiler_output/op_summary_20260518.csv"
python3 -c "
import json
wm = json.load(open('${TMPDIR}/world_model.json'))
wm['decision_tree'] = {'nodes': {'n1': {'id':'n1','mode':'strategy_guided','strategy_combination':[],
  'description':'t','optimization_type':'bandwidth',
  'difficulty':2,'depth':1,'parent_id':'root',
  'status':'in_progress','score':None,'solution_ref':None,'children':[],
  'failure_type':None,'failure_reason':None,'retry_count':0}}}
json.dump(wm, open('${TMPDIR}/world_model.json','w'), indent=2)
"
python3 "${WM_OPS}" refine \
    --wm-path "${TMPDIR}/world_model.json" \
    --round 1 \
    --results-dir "${TMPDIR}/round_1" \
    --parallel-map '{"0":"n1"}' \
    --task-type vector >/dev/null 2>&1
must_run="$(python3 -c "import json; print(json.load(open('${TMPDIR}/state.json'))['must_run_before_next_round'])")"
rm -rf "${TMPDIR}"
[[ "${must_run}" != *"msprof"* ]] || { echo "FAIL T7: R9 should NOT add msprof when profiling present, got ${must_run}"; exit 1; }

# ----------------------------------------------------------------------
# T8 (R10): ≥50% precision failures → stderr warning (not blocking)
# ----------------------------------------------------------------------
TMPDIR="$(mktemp -d)"
_init_with_wm "${TMPDIR}" 1 4 1 0
for p in 0 1 2 3; do
    mkdir -p "${TMPDIR}/round_1/parallel_${p}"
    # 3 of 4 fail precision
    if [[ "${p}" == "0" ]]; then
        echo '{"compilation_success":true,"precision_passed":true,"speedup":0.5}' \
            > "${TMPDIR}/round_1/parallel_${p}/evaluation_results.json"
    else
        echo '{"compilation_success":true,"precision_passed":false,"speedup":0}' \
            > "${TMPDIR}/round_1/parallel_${p}/evaluation_results.json"
    fi
done
python3 -c "
import json
wm = json.load(open('${TMPDIR}/world_model.json'))
nodes = {}
for i in range(4):
    nid = f'n{i+1}'
    nodes[nid] = {'id':nid,'mode':'strategy_guided','strategy_combination':[],
      'description':'t','optimization_type':'bandwidth',
      'difficulty':2,'depth':1,'parent_id':'root',
      'status':'in_progress','score':None,'solution_ref':None,'children':[],
      'failure_type':None,'failure_reason':None,'retry_count':0}
wm['decision_tree'] = {'nodes': nodes}
json.dump(wm, open('${TMPDIR}/world_model.json','w'), indent=2)
"
refine_out="$(python3 "${WM_OPS}" refine \
    --wm-path "${TMPDIR}/world_model.json" \
    --round 1 \
    --results-dir "${TMPDIR}/round_1" \
    --parallel-map '{"0":"n1","1":"n2","2":"n3","3":"n4"}' \
    --task-type vector 2>&1)"
rm -rf "${TMPDIR}"
[[ "${refine_out}" == *"R10 WARN"* ]] || { echo "FAIL T8: R10 warn should fire on ≥50% precision fails out=${refine_out}"; exit 1; }
[[ "${refine_out}" == *"3/4 partials failed precision"* ]] || { echo "FAIL T8: R10 wrong count out=${refine_out}"; exit 1; }

exit 0
