#!/usr/bin/env bash
# Regression test runner for hooks. Returns 0 if all pass.
#
# Usage:
#   tests/hooks/run_all.sh
#
# Designed to be runnable standalone (no pytest dep).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
export PROJECT_ROOT CLAUDE_PROJECT_DIR="${PROJECT_ROOT}"

PASS=0
FAIL=0
FAILED_TESTS=()

run_case() {
    local name="$1"
    local script="$2"
    echo -n "  [$name] "
    local out
    out="$(bash "${script}" 2>&1)"
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        echo "${out}" | sed 's/^/    /'
        ((FAIL++))
        FAILED_TESTS+=("${name}")
    fi
}

echo "=== Z-Search hook tests ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo

for case_file in "${HERE}"/test_*.sh; do
    [[ -e "${case_file}" ]] || continue
    run_case "$(basename "${case_file}" .sh)" "${case_file}"
done

echo
echo "=== Summary: ${PASS} pass, ${FAIL} fail ==="
if [[ ${FAIL} -gt 0 ]]; then
    echo "Failed: ${FAILED_TESTS[*]}"
    exit 1
fi
exit 0
