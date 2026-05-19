#!/usr/bin/env bash
# Half-synthetic fixture: build an EVO_DIR pre-loaded with state suggesting
# "round 1 just finished with stagnation_count=2 → drift_status=replan_required",
# so that when the agent starts round 2, the 4.1 GATE detects drift and runs
# the drift_replan flow.
#
# Usage:
#   bash tests/manual/setup_drift_fixture.sh
#   # prints the constructed EVO_DIR path
#
# The fixture reuses shared/ from output/Add_evo_20260516_163947 (already exists
# on disk from the earlier E2E test). It does NOT run any NPU build — only
# rebuilds state.json + world_model.json + session_anchor + minimal round_1/.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"

SOURCE_EVO="output/Add_evo_20260516_163947"
[[ -d "${SOURCE_EVO}/shared" ]] || { echo "FATAL: source shared/ missing at ${SOURCE_EVO}/shared"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OP_NAME="DriftTest"
EVO="output/${OP_NAME}_evo_${TS}"
mkdir -p "${EVO}/round_1/parallel_0" "${EVO}/round_1/parallel_1"

# 1. Reuse shared/ from earlier Add run (minimal, contains model.py + design/)
cp -r "${SOURCE_EVO}/shared" "${EVO}/shared"

# 2. Build a world_model.json that looks like "round 1 just finished, both
#    variants stalled" — stagnation_count=2 (cross threshold) + a mix of
#    passed nodes + open siblings of both strategy_guided and open_exploration.
python3 -c "
import json, os
nodes = {
  # Two passed (low-score) — they got selected in round 1 but didn't beat baseline
  'n1': {
    'id':'n1','mode':'strategy_guided','strategy_combination':['P1'],
    'description':'Double buffer','optimization_type':'bandwidth',
    'difficulty':2,'depth':1,'parent_id':'root',
    'status':'passed','score':0.85,'solution_ref':'round_1/parallel_0',
    'children':['n1_cont'],'failure_type':None,'failure_reason':None,
    'retry_count':0,'profiling_insight':None,'profiling_evidence':None,
  },
  'x0': {
    'id':'x0','mode':'open_exploration','strategy_combination':[],
    'description':'Free exploration #1','optimization_type':'algorithm',
    'difficulty':3,'depth':1,'parent_id':'root',
    'status':'passed','score':0.78,'solution_ref':'round_1/parallel_1',
    'children':['x0_x1'],'failure_type':None,'failure_reason':None,
    'retry_count':0,'profiling_insight':None,'profiling_evidence':None,
  },
  # Siblings still open
  'n2': {
    'id':'n2','mode':'strategy_guided','strategy_combination':['P2'],
    'description':'Adaptive tiling','optimization_type':'tiling',
    'difficulty':3,'depth':1,'parent_id':'root',
    'status':'open','score':None,'solution_ref':None,'children':[],
    'failure_type':None,'failure_reason':None,'retry_count':0,
  },
  'n3': {
    'id':'n3','mode':'strategy_guided','strategy_combination':['P10'],
    'description':'Vectorized copy','optimization_type':'bandwidth',
    'difficulty':2,'depth':1,'parent_id':'root',
    'status':'open','score':None,'solution_ref':None,'children':[],
    'failure_type':None,'failure_reason':None,'retry_count':0,
  },
  # Children of passed nodes
  'n1_cont': {
    'id':'n1_cont','mode':'strategy_guided','strategy_combination':['P1','P7'],
    'description':'P1 + alignment','optimization_type':'bandwidth',
    'difficulty':3,'depth':2,'parent_id':'n1',
    'status':'open','score':None,'solution_ref':None,'children':[],
    'failure_type':None,'failure_reason':None,'retry_count':0,
  },
  # Open-exploration siblings (multiple so drift can saturate)
  'x1': {
    'id':'x1','mode':'open_exploration','strategy_combination':[],
    'description':'Free exploration #2','optimization_type':'algorithm',
    'difficulty':3,'depth':1,'parent_id':'root',
    'status':'open','score':None,'solution_ref':None,'children':[],
    'failure_type':None,'failure_reason':None,'retry_count':0,
  },
  'x2': {
    'id':'x2','mode':'open_exploration','strategy_combination':[],
    'description':'Free exploration #3','optimization_type':'algorithm',
    'difficulty':3,'depth':1,'parent_id':'root',
    'status':'open','score':None,'solution_ref':None,'children':[],
    'failure_type':None,'failure_reason':None,'retry_count':0,
  },
}
wm = {
  'kernel_summary': 'Drift fixture: element-wise add, FP16',
  'session': {
    'session_id': '${OP_NAME}_evo_${TS}',
    'start_time': '2026-05-17T12:00:00+0800',
    'requested_rounds': 3,
    'actual_rounds_completed': 1,
    'evo_dir': os.path.abspath('${EVO}'),
    'op_name': '${OP_NAME}',
  },
  'baseline_performance': {'speedup': 1.0, 'time_ms': None},
  'decision_tree': {'nodes': nodes},
  'open_questions': [],
  # Stagnation: 2 consecutive rounds below 1.02x threshold → triggers drift
  'stagnation_count': 2,
  'stagnation_count_vs_base': 2,
  'best_score': 1.0,
  'world_model_active': True,
  'solution_db_path': None,
  'hw_params': None,
  'discovered_strategies': [],
  'baseline_evidence': None,
}
with open('${EVO}/world_model.json','w') as f:
  json.dump(wm, f, ensure_ascii=False, indent=2)
print('  wrote world_model.json with stagnation_count=2')
"

# 3. Build state.json simulating "round 1 just finished, drift_status not yet
#    triggered". The agent will, on next round entry, *naturally* re-detect
#    drift via the GATE step. Alternatively, pre-set drift_status=replan_required
#    to skip the natural-trigger path and go straight to the drift_replan flow.
python3 evolution/world_model/state_ops.py init \
    --evo-dir "${EVO}" \
    --agent lingxi-evo \
    --session-id "${OP_NAME}_evo_${TS}" \
    --max-rounds 3 \
    --force >/dev/null

python3 evolution/world_model/state_ops.py write-stage \
    --evo-dir "${EVO}" --stage round_checkpoint --round 1 >/dev/null

python3 evolution/world_model/state_ops.py set-drift \
    --evo-dir "${EVO}" --status replan_required >/dev/null

# 4. Write fake evaluation_results.json for round 1 (for completeness / hook R2)
python3 -c "
import json
for p, score in [(0, 0.85), (1, 0.78)]:
  d = {
    'compilation_success': True, 'precision_passed': True,
    'speedup': score, 'evolved': {'pipeline': None},
  }
  json.dump(d, open(f'${EVO}/round_1/parallel_{p}/evaluation_results.json','w'), indent=2)
"

# 5. Session anchor
python3 evolution/world_model/session_anchor.py write \
    --op-name "${OP_NAME}" \
    --evo-dir "$(pwd)/${EVO}" \
    --requested-rounds 3 >/dev/null

echo
echo "=== Drift fixture ready ==="
echo "EVO_DIR=${EVO}"
echo
echo "Pre-loaded state:"
python3 -c "
import json
s = json.load(open('${EVO}/state.json'))
w = json.load(open('${EVO}/world_model.json'))
print(f'  state.stage              = {s[\"stage\"]}')
print(f'  state.current_round      = {s[\"current_round\"]}')
print(f'  state.drift_status       = {s[\"drift_status\"]}')
print(f'  wm.stagnation_count      = {w[\"stagnation_count\"]}')
print(f'  wm.stagnation_count_vs_base = {w[\"stagnation_count_vs_base\"]}')
print(f'  wm.best_score            = {w[\"best_score\"]}')
print(f'  open nodes               = {sum(1 for n in w[\"decision_tree\"][\"nodes\"].values() if n[\"status\"]==\"open\")}')
"
