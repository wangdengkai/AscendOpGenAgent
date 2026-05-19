#!/usr/bin/env bash
# state_ops.py infer tests (v0.2 core).
#
# Verifies _infer_state_from_filesystem decision table:
#   - shared_prep (no wm.json)
#   - wm_init (wm.json but no round_N)
#   - round_select (round_N dir but no parallels)
#   - round_generate (partials in mixed state)
#   - round_checkpoint (all partials done, actual_rounds==max_round)
#   - finalize (max_round == requested_rounds)
# Plus: drift_status preservation, LLM-stage override.

set -uo pipefail

STATE_OPS="${PROJECT_ROOT}/evolution/world_model/state_ops.py"

_init() {
    local d="$1"
    python3 "${STATE_OPS}" init --evo-dir "${d}" \
        --agent lingxi-evo --session-id sid --max-rounds "${2:-3}" >/dev/null
}

_write_wm() {
    local d="$1"
    local requested="${2:-3}"
    local completed="${3:-0}"
    cat > "${d}/world_model.json" <<EOF
{"session":{"requested_rounds":${requested},"actual_rounds_completed":${completed}}}
EOF
}

_get_stage() {
    python3 -c "import json; print(json.load(open('$1/state.json'))['stage'])"
}
_get_round() {
    python3 -c "import json; print(json.load(open('$1/state.json'))['current_round'])"
}
_get_partial() {
    python3 -c "import json; print(json.dumps(json.load(open('$1/state.json'))['partial_status'], sort_keys=True))"
}

# T1: no wm.json → shared_prep
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "shared_prep" ]] || { echo "FAIL T1: stage=$(_get_stage "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

# T2: wm.json, no rounds → wm_init
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "wm_init" ]] || { echo "FAIL T2: stage=$(_get_stage "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

# T3: round_1/ exists, no parallels → round_select
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}"
mkdir -p "${TMPDIR}/round_1"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "round_select" ]] || { echo "FAIL T3: stage=$(_get_stage "${TMPDIR}")"; exit 1; }
[[ "$(_get_round "${TMPDIR}")" == "1" ]] || { echo "FAIL T3: round=$(_get_round "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

# T4: 0 eval'd, 1 not eval'd → round_generate, partial_status mixed
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}"
mkdir -p "${TMPDIR}/round_1/parallel_0" "${TMPDIR}/round_1/parallel_1"
echo '{"speedup":0.7}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "round_generate" ]] || { echo "FAIL T4 stage"; exit 1; }
[[ "$(_get_partial "${TMPDIR}")" == '{"0": "completed", "1": "running"}' ]] || { echo "FAIL T4 partial=$(_get_partial "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

# T5: all 2 partials done, actual_rounds==1, max_round==1, requested==3
#     → round_checkpoint
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}" 3 1
mkdir -p "${TMPDIR}/round_1/parallel_0" "${TMPDIR}/round_1/parallel_1"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
echo '{}' > "${TMPDIR}/round_1/parallel_1/evaluation_results.json"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "round_checkpoint" ]] || { echo "FAIL T5: stage=$(_get_stage "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

# T6: max_round == requested_rounds → finalize
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}" 3 3
mkdir -p "${TMPDIR}/round_3/parallel_0"
echo '{}' > "${TMPDIR}/round_3/parallel_0/evaluation_results.json"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "finalize" ]] || { echo "FAIL T6: stage=$(_get_stage "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

# T7: drift_status preserved across infer
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}"
python3 "${STATE_OPS}" set-drift --evo-dir "${TMPDIR}" --status replan_required >/dev/null
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
drift="$(python3 -c "import json; print(json.load(open('${TMPDIR}/state.json'))['drift_status'])")"
rm -rf "${TMPDIR}"
[[ "${drift}" == "replan_required" ]] || { echo "FAIL T7: drift=${drift}"; exit 1; }

# T8 (core v0.2 property): LLM wrote wrong stage, infer overrides
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}"
mkdir -p "${TMPDIR}/round_1/parallel_0"
# Adversarial: manually write stage=done
python3 "${STATE_OPS}" write-stage --evo-dir "${TMPDIR}" --stage done >/dev/null
# Now infer — should overwrite stage based on filesystem (round_1 with parallel_0 no eval → round_generate)
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
final_stage="$(_get_stage "${TMPDIR}")"
rm -rf "${TMPDIR}"
[[ "${final_stage}" == "round_generate" ]] || { echo "FAIL T8: infer didn't override LLM stage, final=${final_stage}"; exit 1; }

# T9 (v0.5): max_round == requested_rounds + no report → "finalize"
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}" 1 1   # requested=1, actual=1
mkdir -p "${TMPDIR}/round_1/parallel_0"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "finalize" ]] || { echo "FAIL T9: expected finalize, got $(_get_stage "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

# T10 (v0.5): max_round == requested_rounds + evolution-report*.html present → "done"
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}" 1 1
mkdir -p "${TMPDIR}/round_1/parallel_0"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
echo "<html>fake</html>" > "${TMPDIR}/evolution-report_testop_20260519_120000.html"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "done" ]] || { echo "FAIL T10: expected done, got $(_get_stage "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

# T11 (v0.5): random .html file (not evolution-report) should NOT trigger done
TMPDIR="$(mktemp -d)"
_init "${TMPDIR}"
_write_wm "${TMPDIR}" 1 1
mkdir -p "${TMPDIR}/round_1/parallel_0"
echo '{}' > "${TMPDIR}/round_1/parallel_0/evaluation_results.json"
echo "<html>fake</html>" > "${TMPDIR}/some-other-report.html"
echo "<html>fake</html>" > "${TMPDIR}/index.html"
python3 "${STATE_OPS}" infer --evo-dir "${TMPDIR}" --quiet
[[ "$(_get_stage "${TMPDIR}")" == "finalize" ]] || { echo "FAIL T11: random html should not promote to done, got $(_get_stage "${TMPDIR}")"; exit 1; }
rm -rf "${TMPDIR}"

exit 0
