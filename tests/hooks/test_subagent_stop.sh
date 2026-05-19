#!/usr/bin/env bash
# SubagentStop hook tests (v0.4).
#
# Verifies that loop-subagent-stop.sh audits evolution partial subagents'
# transcripts correctly:
#   - Allows real partials that actually ran evaluate_ascendc.sh
#   - Blocks fake/lying partials that only claimed completion
#   - Routes non-evolution subagents (general-purpose, etc.) through
#   - Honors stop_hook_active and LINGXI_LOOP_HOOK_DISABLE

set -uo pipefail

SUB_HOOK="${PROJECT_ROOT}/.claude/hooks/loop-subagent-stop.sh"
AUDIT_PY="${PROJECT_ROOT}/evolution/world_model/transcript_audit.py"

# Build a fake transcript JSONL with the given Bash commands
_build_transcript() {
    local out="$1"
    shift
    > "${out}"
    # user prompt
    echo '{"type":"user","message":{"role":"user","content":"test prompt"}}' >> "${out}"
    # one assistant line per command, each one a tool_use Bash block
    for cmd in "$@"; do
        python3 -c "
import json
event = {'type':'assistant','message':{'role':'assistant','content':[
    {'type':'tool_use','name':'Bash','input':{'command':'''${cmd}'''}}
]}}
print(json.dumps(event))
" >> "${out}"
    done
}

# Build a transcript with one Write of evaluation_results.json
_build_transcript_with_write() {
    local out="$1"
    local write_path="$2"
    shift 2
    _build_transcript "${out}" "$@"
    python3 -c "
import json
event = {'type':'assistant','message':{'role':'assistant','content':[
    {'type':'tool_use','name':'Write','input':{'file_path':'${write_path}','content':'{}'}}
]}}
print(json.dumps(event))
" >> "${out}"
}

# Build the stdin JSON the hook expects
_make_stdin() {
    local agent_type="$1"
    local transcript="$2"
    local last_msg="$3"
    local stop_active="${4:-false}"
    # Map bash bool to Python bool
    if [[ "${stop_active}" == "true" ]]; then
        local py_bool="True"
    else
        local py_bool="False"
    fi
    python3 -c "
import json
print(json.dumps({
    'agent_type': '${agent_type}',
    'agent_transcript_path': '${transcript}',
    'last_assistant_message': '''${last_msg}''',
    'cwd': '${PROJECT_ROOT}',
    'stop_hook_active': ${py_bool}
}))
"
}

# ----------------------------------------------------------------------
# T1: lingxi-partial that actually invoked evaluate_ascendc.sh → allow
# ----------------------------------------------------------------------
TR="$(mktemp --suffix=.jsonl)"
_build_transcript_with_write "${TR}" "/some/path/round_1/parallel_0/evaluation_results.json" \
    "bash .claude/skills/ascendc-translator/references/evaluate_ascendc.sh output/round_1/parallel_0"
out="$(_make_stdin lingxi-partial "${TR}" "已完成精度通过" | "${SUB_HOOK}" 2>&1)"
rc=$?
rm -f "${TR}"
[[ ${rc} -eq 0 ]] || { echo "FAIL T1: real partial should allow, exit=${rc} out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# T2: lingxi-partial that only `cat`ed evaluate script (didn't run) → block
# ----------------------------------------------------------------------
TR="$(mktemp --suffix=.jsonl)"
_build_transcript "${TR}" \
    "cat .claude/skills/ascendc-translator/references/evaluate_ascendc.sh" \
    "ls some-dir"
out="$(_make_stdin lingxi-partial "${TR}" "已完成,精度通过" | "${SUB_HOOK}" 2>&1)"
rc=$?
rm -f "${TR}"
[[ ${rc} -eq 2 ]] || { echo "FAIL T2: passive-only fake should block, exit=${rc}"; exit 1; }
[[ "${out}" == *"S1:"* ]] || { echo "FAIL T2: missing S1 marker"; exit 1; }

# ----------------------------------------------------------------------
# T3: lingxi-partial that ran evaluate (S6 removed in v0.4.1, no longer fires)
# ----------------------------------------------------------------------
TR="$(mktemp --suffix=.jsonl)"
_build_transcript "${TR}" \
    "bash .claude/skills/ascendc-translator/references/evaluate_ascendc.sh output/round_1/parallel_0"
# No write of evaluation_results.json — but script writes it; S6 dropped
# because NLP success-marker matching was unreliable (25% false positive).
out="$(_make_stdin lingxi-partial "${TR}" "精度通过完成" | "${SUB_HOOK}" 2>&1)"
rc=$?
rm -f "${TR}"
[[ ${rc} -eq 0 ]] || { echo "FAIL T3: S1 satisfied should now allow (S6 removed) exit=${rc} out=${out}"; exit 1; }

# ----------------------------------------------------------------------
# T4: general-purpose subagent → always allow (out of scope)
# ----------------------------------------------------------------------
TR="$(mktemp --suffix=.jsonl)"
_build_transcript "${TR}" "echo hello"  # no evaluate
out="$(_make_stdin general-purpose "${TR}" "完成" | "${SUB_HOOK}" 2>&1)"
rc=$?
rm -f "${TR}"
[[ ${rc} -eq 0 ]] || { echo "FAIL T4: general-purpose should be ignored, exit=${rc}"; exit 1; }

# ----------------------------------------------------------------------
# T5: stop_hook_active=true → allow regardless of audit result (anti-recursion)
# ----------------------------------------------------------------------
TR="$(mktemp --suffix=.jsonl)"
_build_transcript "${TR}" "echo nothing"  # no evaluate (would fail audit)
out="$(_make_stdin lingxi-partial "${TR}" "已完成精度通过" true | "${SUB_HOOK}" 2>&1)"
rc=$?
rm -f "${TR}"
[[ ${rc} -eq 0 ]] || { echo "FAIL T5: stop_hook_active=true should allow, exit=${rc}"; exit 1; }

# ----------------------------------------------------------------------
# T6: HOOK_DISABLE downgrades block to warn-only (exit 0)
# ----------------------------------------------------------------------
TR="$(mktemp --suffix=.jsonl)"
_build_transcript "${TR}" "echo nothing"
out="$(LINGXI_LOOP_HOOK_DISABLE=1 bash -c "_make_stdin() { python3 -c \"
import json
print(json.dumps({
  'agent_type':'lingxi-partial',
  'agent_transcript_path':'${TR}',
  'last_assistant_message':'已完成精度通过',
  'cwd':'${PROJECT_ROOT}',
  'stop_hook_active':False
}))\"
}; _make_stdin | '${SUB_HOOK}'" 2>&1)"
rc=$?
rm -f "${TR}"
[[ ${rc} -eq 0 ]] || { echo "FAIL T6: HOOK_DISABLE should allow, exit=${rc}"; exit 1; }
[[ "${out}" == *"WARN"* ]] || { echo "FAIL T6: missing WARN marker"; exit 1; }

# ----------------------------------------------------------------------
# T7: missing transcript file → fail open (allow, log warning)
# ----------------------------------------------------------------------
out="$(_make_stdin lingxi-partial "/nonexistent/path.jsonl" "完成" | "${SUB_HOOK}" 2>&1)"
rc=$?
[[ ${rc} -eq 0 ]] || { echo "FAIL T7: missing transcript should fail open, exit=${rc}"; exit 1; }

# ----------------------------------------------------------------------
# T8: agent_type empty + meta.json resolves to lingxi-partial → audit kicks in
# ----------------------------------------------------------------------
TR="$(mktemp --suffix=.jsonl)"
META="${TR%.jsonl}.meta.json"
_build_transcript "${TR}" "echo nothing"  # no evaluate
echo '{"agentType":"lingxi-partial","description":"test"}' > "${META}"
out="$(_make_stdin "" "${TR}" "已完成精度通过" | "${SUB_HOOK}" 2>&1)"
rc=$?
rm -f "${TR}" "${META}"
[[ ${rc} -eq 2 ]] || { echo "FAIL T8: empty agent_type should resolve via meta and block, exit=${rc}"; exit 1; }
[[ "${out}" == *"S1"* ]] || { echo "FAIL T8: missing S1 marker"; exit 1; }

# ----------------------------------------------------------------------
# T9: transcript_audit list-calls CLI works (used by other scripts/audit)
# ----------------------------------------------------------------------
TR="$(mktemp --suffix=.jsonl)"
_build_transcript "${TR}" "bash test.sh" "echo ok"
out="$(python3 "${AUDIT_PY}" list-calls --transcript "${TR}" --tool Bash)"
rc=$?
rm -f "${TR}"
[[ ${rc} -eq 0 ]] || { echo "FAIL T9: list-calls failed, exit=${rc}"; exit 1; }
[[ "${out}" == *"bash test.sh"* ]] || { echo "FAIL T9: missing expected Bash command"; exit 1; }

exit 0
