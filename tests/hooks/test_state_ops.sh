#!/usr/bin/env bash
# state_ops.py CLI tests. Validates init / write-stage / write-partial / validate.

set -uo pipefail

STATE_OPS="${PROJECT_ROOT}/evolution/world_model/state_ops.py"

# Test 1: init creates well-formed state.json
TMPDIR1="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR1}" --agent lingxi-evo --session-id testsid --max-rounds 5 >/dev/null
[[ -f "${TMPDIR1}/state.json" ]] || { echo "FAIL: init did not create state.json"; rm -rf "${TMPDIR1}"; exit 1; }
stage="$(python3 -c "import json; print(json.load(open('${TMPDIR1}/state.json'))['stage'])")"
[[ "${stage}" == "init" ]] || { echo "FAIL: initial stage=${stage} (expected init)"; rm -rf "${TMPDIR1}"; exit 1; }
rm -rf "${TMPDIR1}"

# Test 2: init without --force on existing file should fail
TMPDIR2="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR2}" --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
out="$(python3 "${STATE_OPS}" init --evo-dir "${TMPDIR2}" --agent lingxi-evo --session-id sid --max-rounds 3 2>&1)"
rc=$?
rm -rf "${TMPDIR2}"
[[ ${rc} -ne 0 ]] || { echo "FAIL: re-init without --force should fail"; exit 1; }

# Test 3: write-stage updates field
TMPDIR3="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR3}" --agent ops-evo --session-id sid --max-rounds 3 >/dev/null
python3 "${STATE_OPS}" write-stage --evo-dir "${TMPDIR3}" --stage shared_prep >/dev/null
stage="$(python3 -c "import json; print(json.load(open('${TMPDIR3}/state.json'))['stage'])")"
[[ "${stage}" == "shared_prep" ]] || { echo "FAIL: write-stage set stage=${stage}"; rm -rf "${TMPDIR3}"; exit 1; }
rm -rf "${TMPDIR3}"

# Test 4: write-stage with invalid stage exits non-zero
TMPDIR4="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR4}" --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
out="$(python3 "${STATE_OPS}" write-stage --evo-dir "${TMPDIR4}" --stage not_a_real_stage 2>&1)"
rc=$?
rm -rf "${TMPDIR4}"
[[ ${rc} -ne 0 ]] || { echo "FAIL: invalid stage should reject"; exit 1; }

# Test 5: write-partial / reset-partial
TMPDIR5="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR5}" --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
python3 "${STATE_OPS}" write-partial --evo-dir "${TMPDIR5}" --parallel-idx 1 --status running >/dev/null
val="$(python3 -c "import json; print(json.load(open('${TMPDIR5}/state.json'))['partial_status']['1'])")"
[[ "${val}" == "running" ]] || { echo "FAIL: write-partial val=${val}"; rm -rf "${TMPDIR5}"; exit 1; }
python3 "${STATE_OPS}" reset-partial --evo-dir "${TMPDIR5}" >/dev/null
val="$(python3 -c "import json; print(json.load(open('${TMPDIR5}/state.json'))['partial_status'])")"
[[ "${val}" == "{}" ]] || { echo "FAIL: reset-partial val=${val}"; rm -rf "${TMPDIR5}"; exit 1; }
rm -rf "${TMPDIR5}"

# Test 6: validate --check-stage-artifacts detects missing eval files
TMPDIR6="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR6}" --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
echo '{}' > "${TMPDIR6}/world_model.json"
python3 "${STATE_OPS}" write-stage --evo-dir "${TMPDIR6}" --stage round_refine --round 1 >/dev/null
python3 "${STATE_OPS}" write-partial --evo-dir "${TMPDIR6}" --parallel-idx 0 --status completed >/dev/null
out="$(python3 "${STATE_OPS}" validate --evo-dir "${TMPDIR6}" --check-stage-artifacts 2>&1)"
rc=$?
rm -rf "${TMPDIR6}"
[[ ${rc} -eq 2 ]] || { echo "FAIL: validate should exit 2 on missing artifact, got ${rc}"; exit 1; }
[[ "${out}" == *"missing artifact"* ]] || { echo "FAIL: missing 'missing artifact' marker out=${out}"; exit 1; }

# Test 7: verdict bumps stall counter for stalled, resets for advanced
TMPDIR7="$(mktemp -d)"
python3 "${STATE_OPS}" init --evo-dir "${TMPDIR7}" --agent lingxi-evo --session-id sid --max-rounds 3 >/dev/null
python3 "${STATE_OPS}" set-verdict --evo-dir "${TMPDIR7}" --verdict stalled >/dev/null
n="$(python3 -c "import json; print(json.load(open('${TMPDIR7}/state.json'))['mainline_stall_count'])")"
[[ "${n}" == "1" ]] || { echo "FAIL: stalled should set count=1, got ${n}"; rm -rf "${TMPDIR7}"; exit 1; }
python3 "${STATE_OPS}" set-verdict --evo-dir "${TMPDIR7}" --verdict stalled >/dev/null
n="$(python3 -c "import json; print(json.load(open('${TMPDIR7}/state.json'))['mainline_stall_count'])")"
[[ "${n}" == "2" ]] || { echo "FAIL: 2nd stalled should set count=2, got ${n}"; rm -rf "${TMPDIR7}"; exit 1; }
python3 "${STATE_OPS}" set-verdict --evo-dir "${TMPDIR7}" --verdict advanced >/dev/null
n="$(python3 -c "import json; print(json.load(open('${TMPDIR7}/state.json'))['mainline_stall_count'])")"
[[ "${n}" == "0" ]] || { echo "FAIL: advanced should reset count, got ${n}"; rm -rf "${TMPDIR7}"; exit 1; }
rm -rf "${TMPDIR7}"

exit 0
