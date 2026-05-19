#!/usr/bin/env python3
"""
analyze_ttk.py - Analyze TTK test results by comparing baseline vs evolved CSV outputs.

Features:
  - Join by testcase_name: aligns baseline and evolved rows by case name, not row index
  - Regression detection: distinguishes new failures (regression) from inherited failures
  - Per-case performance comparison: shows speedup for each matched case
  - Rich failure output: includes testcase_name, dtype, shape in failure reports
"""

import argparse
import csv
import json
import os
import sys
from statistics import mean, stdev


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _find_column(fieldnames, *candidates):
    """Find a column name by case-insensitive matching.

    Priority: exact match > substring match.
    Candidates are tried in order; first match wins.
    """
    # Pass 1: exact match (case-insensitive)
    for cand in candidates:
        for col in fieldnames:
            if col.strip().lower() == cand:
                return col
    # Pass 2: substring match
    for cand in candidates:
        for col in fieldnames:
            if cand in col.strip().lower():
                return col
    return None


def read_ttk_csv(csv_path):
    """Read a TTK result CSV and return a list of row dicts keyed by testcase_name.

    Each returned entry contains:
      - testcase_name: str
      - precision_status: str | None  (lower-cased)
      - bin_perf_us: float | None
      - dtype: str   (from stc_input_dtypes, first element)
      - shape: str   (from stc_ori_inputs, raw string)
      - raw: dict    (full original row)
    """
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        name_col = _find_column(fieldnames, "testcase_name")
        precision_col = _find_column(fieldnames, "precision_status", "precision")
        perf_col = _find_column(fieldnames, "bin_perf_us", "perf_us")
        dtype_col = _find_column(fieldnames, "stc_input_dtypes")
        shape_col = _find_column(fieldnames, "stc_ori_inputs")

        if not name_col:
            print(f"WARN: 'testcase_name' column not found in {csv_path}", file=sys.stderr)
        if not precision_col:
            print(f"WARN: 'precision_status' column not found in {csv_path}", file=sys.stderr)
        if not perf_col:
            print(f"WARN: 'bin_perf_us' column not found in {csv_path}", file=sys.stderr)

        for row in reader:
            entry = {
                "testcase_name": row.get(name_col, "").strip() if name_col else "",
                "precision_status": None,
                "bin_perf_us": None,
                "dtype": "",
                "shape": "",
                "raw": row,
            }
            if precision_col:
                entry["precision_status"] = row.get(precision_col, "").strip().lower()
            if perf_col:
                raw_val = row.get(perf_col, "").strip()
                try:
                    entry["bin_perf_us"] = float(raw_val)
                except (ValueError, TypeError):
                    entry["bin_perf_us"] = None
            if dtype_col:
                entry["dtype"] = row.get(dtype_col, "").strip()
            if shape_col:
                entry["shape"] = row.get(shape_col, "").strip()
            rows.append(entry)

    return rows


def rows_to_dict(rows):
    """Convert row list to dict keyed by testcase_name.

    If testcase_name is empty or duplicated, fall back to index-based keys.
    """
    d = {}
    for i, r in enumerate(rows):
        key = r["testcase_name"] or f"__row_{i}"
        if key in d:
            key = f"{key}__dup_{i}"
        d[key] = r
    return d


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def perf_stats(values):
    """Compute performance statistics from a list of floats."""
    if not values:
        return {"avg_us": 0, "min_us": 0, "max_us": 0, "std_us": 0, "count": 0}
    return {
        "avg_us": round(mean(values), 2),
        "min_us": round(min(values), 2),
        "max_us": round(max(values), 2),
        "std_us": round(stdev(values), 2) if len(values) > 1 else 0,
        "count": len(values),
    }


def _extract_first_dtype(dtype_str):
    """Extract first dtype from a tuple string like \"('float16', 'int32', ...)\". """
    # Strip outer parens and quotes, take first element
    s = dtype_str.strip("() '\"")
    return s.split("'")[0].split(",")[0].strip("' \"")


def analyze(baseline_rows, evolved_rows):
    """Full analysis: join by testcase_name, classify precision, compare perf."""

    base_dict = rows_to_dict(baseline_rows)
    evo_dict = rows_to_dict(evolved_rows)

    all_keys = list(dict.fromkeys(list(base_dict.keys()) + list(evo_dict.keys())))

    # Per-case comparison
    per_case = []
    # Precision classification
    regressions = []       # baseline PASS -> evolved FAIL
    inherited_fails = []   # baseline FAIL -> evolved FAIL
    improvements = []      # baseline FAIL -> evolved PASS
    both_pass = 0
    # Performance values (only for cases that both PASS with valid perf)
    base_perf_vals = []
    evo_perf_vals = []

    for key in all_keys:
        base = base_dict.get(key)
        evo = evo_dict.get(key)

        base_prec = base["precision_status"] if base else None
        evo_prec = evo["precision_status"] if evo else None
        base_perf = base["bin_perf_us"] if base else None
        evo_perf = evo["bin_perf_us"] if evo else None

        base_pass = base_prec == "pass" if base_prec else False
        evo_pass = evo_prec == "pass" if evo_prec else False

        # Speedup per case (only when both have valid perf > 0)
        case_speedup = None
        if base_perf and evo_perf and base_perf > 0 and evo_perf > 0:
            case_speedup = round(base_perf / evo_perf, 3)

        dtype = ""
        shape = ""
        if evo:
            dtype = _extract_first_dtype(evo["dtype"])
            shape = evo["shape"]
        elif base:
            dtype = _extract_first_dtype(base["dtype"])
            shape = base["shape"]

        case_entry = {
            "testcase_name": key,
            "dtype": dtype,
            "shape": shape,
            "baseline_precision": base_prec or "N/A",
            "evolved_precision": evo_prec or "N/A",
            "baseline_perf_us": base_perf,
            "evolved_perf_us": evo_perf,
            "speedup": case_speedup,
        }
        per_case.append(case_entry)

        # Classify
        if base_pass and evo_pass:
            both_pass += 1
            if base_perf and base_perf > 0:
                base_perf_vals.append(base_perf)
            if evo_perf and evo_perf > 0:
                evo_perf_vals.append(evo_perf)
        elif base_pass and not evo_pass:
            regressions.append(case_entry)
        elif not base_pass and not evo_pass:
            inherited_fails.append(case_entry)
        elif not base_pass and evo_pass:
            improvements.append(case_entry)
            if base_perf and base_perf > 0:
                base_perf_vals.append(base_perf)
            if evo_perf and evo_perf > 0:
                evo_perf_vals.append(evo_perf)

    # Overall performance (only from cases where both have valid perf)
    base_perf_summary = perf_stats(base_perf_vals)
    evo_perf_summary = perf_stats(evo_perf_vals)

    overall_speedup = 0
    if evo_perf_summary["avg_us"] > 0 and base_perf_summary["avg_us"] > 0:
        overall_speedup = round(base_perf_summary["avg_us"] / evo_perf_summary["avg_us"], 3)

    total_cases = len(all_keys)
    evo_passed = both_pass + len(improvements)
    has_regression = len(regressions) > 0

    result = {
        "precision": {
            "all_passed": not has_regression and evo_passed == total_cases,
            "no_regression": not has_regression,
            "total_cases": total_cases,
            "both_pass": both_pass,
            "regressions": len(regressions),
            "inherited_fails": len(inherited_fails),
            "improvements": len(improvements),
            "regression_details": regressions,
            "inherited_fail_details": inherited_fails,
            "improvement_details": improvements,
        },
        "performance": {
            "baseline_avg_us": base_perf_summary["avg_us"],
            "evolved_avg_us": evo_perf_summary["avg_us"],
            "speedup": overall_speedup,
            "baseline_details": base_perf_summary,
            "evolved_details": evo_perf_summary,
        },
        "per_case": per_case,
    }

    return result


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_report(result):
    """Print a human-readable summary to stdout."""
    prec = result["precision"]
    perf = result["performance"]
    per_case = result["per_case"]

    print(f"\n{'=' * 70}")
    print(f"  TTK Test Results Summary")
    print(f"{'=' * 70}")

    # Precision summary
    if prec["all_passed"]:
        prec_label = "ALL PASS"
    elif prec["no_regression"]:
        prec_label = f"NO REGRESSION (inherited fails: {prec['inherited_fails']})"
    else:
        prec_label = f"REGRESSION ({prec['regressions']} new failures)"

    print(f"  Precision:      {prec_label}")
    print(f"    Total cases:    {prec['total_cases']}")
    print(f"    Both pass:      {prec['both_pass']}")
    print(f"    Regressions:    {prec['regressions']}")
    print(f"    Inherited fail: {prec['inherited_fails']}")
    print(f"    Improvements:   {prec['improvements']}")

    # Performance summary
    print(f"\n  Performance:")
    print(f"    Baseline:       {perf['baseline_avg_us']}us  (avg, n={perf['baseline_details']['count']})")
    print(f"    Evolved:        {perf['evolved_avg_us']}us  (avg, n={perf['evolved_details']['count']})")
    print(f"    Speedup:        {perf['speedup']}x")

    # Per-case table
    print(f"\n{'=' * 70}")
    print(f"  Per-Case Comparison")
    print(f"{'=' * 70}")
    # Header
    hdr = f"  {'Case':<52s} {'dtype':<9s} {'Base(us)':>9s} {'Evo(us)':>9s} {'Speed':>6s} {'Prec':>8s}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for c in per_case:
        name = c["testcase_name"]
        # Shorten long names: keep last 50 chars
        if len(name) > 50:
            name = "..." + name[-47:]

        b_perf = f"{c['baseline_perf_us']:.2f}" if c["baseline_perf_us"] is not None else "N/A"
        e_perf = f"{c['evolved_perf_us']:.2f}" if c["evolved_perf_us"] is not None else "N/A"
        spd = f"{c['speedup']:.2f}x" if c["speedup"] is not None else "N/A"

        # Precision label for this case
        bp = c["baseline_precision"]
        ep = c["evolved_precision"]
        if bp == "pass" and ep == "pass":
            prec_lbl = "OK"
        elif bp == "pass" and ep != "pass":
            prec_lbl = "REGRESS"
        elif bp != "pass" and ep != "pass":
            prec_lbl = "INHERIT"
        else:
            prec_lbl = "FIXED"

        print(f"  {name:<52s} {c['dtype']:<9s} {b_perf:>9s} {e_perf:>9s} {spd:>6s} {prec_lbl:>8s}")

    # Regression details
    if prec["regressions"] > 0:
        print(f"\n{'=' * 70}")
        print(f"  Regression Details ({prec['regressions']} cases)")
        print(f"{'=' * 70}")
        for r in prec["regression_details"]:
            print(f"  {r['testcase_name']}")
            print(f"    dtype:  {r['dtype']}")
            print(f"    shape:  {r['shape']}")
            print(f"    base:   {r['baseline_precision']}  {r['baseline_perf_us']}us")
            print(f"    evolved: {r['evolved_precision']}")

    # Improvements
    if prec["improvements"] > 0:
        print(f"\n  Improvements ({prec['improvements']} cases):")
        for r in prec["improvement_details"]:
            print(f"    {r['testcase_name']}:  {r['baseline_precision']} -> {r['evolved_precision']}")

    print(f"\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze TTK test results: baseline vs evolved (joined by testcase_name)")
    parser.add_argument("--baseline-csv", required=True,
                        help="Path to baseline TTK result CSV")
    parser.add_argument("--evolved-csv", required=True,
                        help="Path to evolved TTK result CSV")
    parser.add_argument("--output", default="ttk_comparison.json",
                        help="Output JSON path")

    args = parser.parse_args()

    print(f"Baseline CSV: {args.baseline_csv}")
    print(f"Evolved CSV:  {args.evolved_csv}")

    # Read
    baseline_rows = read_ttk_csv(args.baseline_csv)
    evolved_rows = read_ttk_csv(args.evolved_csv)

    # Analyze
    result = analyze(baseline_rows, evolved_rows)

    # Write JSON (exclude per_case raw data to keep file clean)
    output_result = {
        "precision": {
            "all_passed": result["precision"]["all_passed"],
            "no_regression": result["precision"]["no_regression"],
            "total_cases": result["precision"]["total_cases"],
            "both_pass": result["precision"]["both_pass"],
            "regressions": result["precision"]["regressions"],
            "inherited_fails": result["precision"]["inherited_fails"],
            "improvements": result["precision"]["improvements"],
            "regression_details": [
                {k: v for k, v in r.items() if k != "raw"}
                for r in result["precision"]["regression_details"]
            ],
            "inherited_fail_details": [
                {k: v for k, v in r.items() if k != "raw"}
                for r in result["precision"]["inherited_fail_details"]
            ],
            "improvement_details": [
                {k: v for k, v in r.items() if k != "raw"}
                for r in result["precision"]["improvement_details"]
            ],
        },
        "performance": result["performance"],
        "per_case": [
            {k: v for k, v in c.items() if k != "raw"}
            for c in result["per_case"]
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_result, f, indent=2, ensure_ascii=False)

    # Print report
    print_report(result)
    print(f"  Output: {args.output}")

    # Exit code: 0 if no regression, 1 if regression
    return 0 if result["precision"]["no_regression"] else 1


if __name__ == "__main__":
    sys.exit(main())
