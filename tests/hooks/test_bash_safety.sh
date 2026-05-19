#!/usr/bin/env bash
# Bash-safety hook tests.
# Verifies B1 (empty var as path) and B2 (root protection) rules.

set -uo pipefail

SAFETY_HOOK="${PROJECT_ROOT}/.claude/hooks/loop-bash-safety.sh"

# Test 1: safe command should allow
out="$(echo '{"tool_name":"Bash","tool_input":{"command":"ls -la /tmp"}}' | "${SAFETY_HOOK}" 2>&1)"
rc=$?
[[ ${rc} -eq 0 ]] || { echo "FAIL: safe command exit=${rc} out=${out}"; exit 1; }

# Test 2: empty var as path should block (B1)
unset EMPTYVAR
out="$(echo '{"tool_name":"Bash","tool_input":{"command":"cp -r $EMPTYVAR/* /tmp/foo"}}' | "${SAFETY_HOOK}" 2>&1)"
rc=$?
[[ ${rc} -eq 2 ]] || { echo "FAIL: empty var exit=${rc} (expected 2) out=${out}"; exit 1; }
[[ "${out}" == *"B1"* ]] || { echo "FAIL: missing B1 marker out=${out}"; exit 1; }

# Test 3: rm -rf / should block (B2)
out="$(echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' | "${SAFETY_HOOK}" 2>&1)"
rc=$?
[[ ${rc} -eq 2 ]] || { echo "FAIL: rm -rf / exit=${rc} (expected 2) out=${out}"; exit 1; }
[[ "${out}" == *"B2"* ]] || { echo "FAIL: missing B2 marker out=${out}"; exit 1; }

# Test 4: rm -rf $HOME should block (B2)
out="$(echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf $HOME"}}' | "${SAFETY_HOOK}" 2>&1)"
rc=$?
[[ ${rc} -eq 2 ]] || { echo "FAIL: rm -rf \$HOME exit=${rc} (expected 2) out=${out}"; exit 1; }

# Test 5: false-positive: cp with set var must allow
out="$(WORKDIR=/tmp/x bash -c "echo '{\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"cp -r \$WORKDIR/* dst/\"}}' | '${SAFETY_HOOK}'" 2>&1)"
rc=$?
[[ ${rc} -eq 0 ]] || { echo "FAIL: cp with set var exit=${rc} (expected 0) out=${out}"; exit 1; }

# Test 6: false-positive: non cp/mv/rm must allow even with vars
out="$(echo '{"tool_name":"Bash","tool_input":{"command":"git diff HEAD"}}' | "${SAFETY_HOOK}" 2>&1)"
rc=$?
[[ ${rc} -eq 0 ]] || { echo "FAIL: git command blocked exit=${rc} out=${out}"; exit 1; }

# Test 7: HOOK_DISABLE downgrades B1 to warn
out="$(LINGXI_LOOP_HOOK_DISABLE=1 bash -c "echo '{\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"cp -r \$EMPTYVAR/* /tmp/foo\"}}' | '${SAFETY_HOOK}'" 2>&1)"
rc=$?
[[ ${rc} -eq 0 ]] || { echo "FAIL: HOOK_DISABLE not honored exit=${rc} out=${out}"; exit 1; }
[[ "${out}" == *"WARN"* ]] || { echo "FAIL: HOOK_DISABLE missing WARN out=${out}"; exit 1; }

exit 0
