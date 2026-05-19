import importlib.util
import json
import sys
import types
from pathlib import Path


MODULE_PATH = Path("/mnt/workspace/Z-Search/.claude/skills/ops-evaluation/scripts/forge_evaluator.py")
SPEC = importlib.util.spec_from_file_location("forge_evaluator_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_write_forge_args_json_writes_flat_string_values(tmp_path):
    output = tmp_path / "forge_args.json"
    path = MODULE.write_forge_args_json(
        output_path=str(output),
        args_map={
            "repo_root": "/tmp/repo",
            "op_name": "MyOp",
            "install_path": "/tmp/install",
            "test_path": "/tmp/tests",
            "test_script": "test_myop.py",
            "zsearch_side": "evolved",
            "zsearch_precision_passed": "true",
            "zsearch_correctness_message": "PASS",
            "zsearch_build_success": "true",
            "zsearch_repo_type": "omni-ops",
            "zsearch_soc": "ascend910b",
            "zsearch_task_type": "performance",
        },
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["op_name"] == "MyOp"
    assert all(isinstance(v, str) for v in payload.values())


def test_update_forge_args_json_updates_same_file(tmp_path):
    output = tmp_path / "forge_args.json"
    MODULE.write_forge_args_json(
        output_path=str(output),
        args_map={
            "op_name": "MyOp",
            "zsearch_precision_passed": "unknown",
            "zsearch_correctness_message": "",
        },
    )
    path = MODULE.update_forge_args_json(
        output_path=str(output),
        updates={
            "zsearch_precision_passed": "true",
            "zsearch_correctness_message": "PASS",
        },
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["op_name"] == "MyOp"
    assert payload["zsearch_precision_passed"] == "true"
    assert payload["zsearch_correctness_message"] == "PASS"


def test_resolve_zsearch_op_filter_prefers_explicit_value(tmp_path):
    filter_map = tmp_path / "zsearch_op_filter_map.json"
    filter_map.write_text(json.dumps({"MyOp": "MappedKernel*"}), encoding="utf-8")

    result = MODULE.resolve_zsearch_op_filter(
        op_name="MyOp",
        explicit_filter="ExplicitKernel*",
        filter_map_path=str(filter_map),
    )

    assert result == "ExplicitKernel*"


def test_resolve_zsearch_op_filter_reads_mapping_by_op_name(tmp_path):
    filter_map = tmp_path / "zsearch_op_filter_map.json"
    filter_map.write_text(json.dumps({"MyOp": "MappedKernel*"}), encoding="utf-8")

    result = MODULE.resolve_zsearch_op_filter(
        op_name="MyOp",
        explicit_filter="",
        filter_map_path=str(filter_map),
    )

    assert result == "MappedKernel*"


def test_resolve_zsearch_op_filter_falls_back_to_op_name(tmp_path):
    filter_map = tmp_path / "zsearch_op_filter_map.json"
    filter_map.write_text(json.dumps({"OtherOp": "OtherKernel*"}), encoding="utf-8")

    result = MODULE.resolve_zsearch_op_filter(
        op_name="MyOp",
        explicit_filter="",
        filter_map_path=str(filter_map),
    )

    assert result == "MyOp"


def test_main_writes_resolved_zsearch_op_filter_to_forge_arg_file(monkeypatch, tmp_path):
    filter_map = tmp_path / "zsearch_op_filter_map.json"
    filter_map.write_text(json.dumps({"MyOp": "MappedKernel*"}), encoding="utf-8")
    output = tmp_path / "evaluation_results.json"
    seen = {}

    def fake_forge_build(forge_bin, config_dir, config_name, arg_file=None):
        seen["payload"] = json.loads(Path(arg_file).read_text(encoding="utf-8"))
        return True

    monkeypatch.setattr(MODULE, "forge_build", fake_forge_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forge_evaluator.py",
            "--forge-config",
            "cfg",
            "--forge-config-dir",
            "/tmp/configs",
            "--op-name",
            "MyOp",
            "--install-path",
            "/tmp/install",
            "--output",
            str(output),
            "--mode",
            "build",
            "--zsearch-op-filter-map",
            str(filter_map),
        ],
    )

    MODULE.main()

    assert seen["payload"]["op_name"] == "MyOp"
    assert seen["payload"]["zsearch_op_filter"] == "MappedKernel*"


def test_run_forge_command_appends_arg_file(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, capture_output, text, timeout):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    arg_file = tmp_path / "forge_args.json"
    arg_file.write_text("{}", encoding="utf-8")
    MODULE.run_forge_command("forge", ["workflow", "cfg", "performance_test"], "/tmp/configs", str(arg_file))
    assert seen["cmd"][-2:] == ["--arg", f"@{arg_file}"]


def test_forge_accuracy_test_returns_pass_on_success(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "run_forge_command",
        lambda forge_bin, args, config_dir, arg_file=None: types.SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    success, message = MODULE.forge_accuracy_test("forge", "/tmp/configs", "cfg", "/tmp/args.json")
    assert success is True
    assert message == "PASS"


def test_forge_accuracy_test_returns_failure_summary(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "run_forge_command",
        lambda forge_bin, args, config_dir, arg_file=None: types.SimpleNamespace(
            returncode=1,
            stdout="FAILED test_case",
            stderr="AssertionError: mismatch",
        ),
    )
    success, message = MODULE.forge_accuracy_test("forge", "/tmp/configs", "cfg", "/tmp/args.json")
    assert success is False
    assert "AssertionError" in message


def test_build_evaluation_result_prefers_raw_result_and_applies_iqr(tmp_path):
    csv1 = tmp_path / "op_summary.csv"
    header = "OP Type,Task Duration(us),aiv_vec_ratio,aiv_scalar_ratio,aiv_mte2_ratio,aiv_mte3_ratio\n"
    # 2 warmup rows (120.0 noise) + 5 real rows; after warmup_count=2 we have 5 rows
    # one outlier (120.0) should be removed by IQR, median of remaining 4 = 88.05
    csv1.write_text(
        header
        + "MyOp,120.0,0.20,0.10,0.55,0.15\n"  # warmup row 1
        + "MyOp,120.0,0.20,0.10,0.55,0.15\n"  # warmup row 2
        + "MyOp,88.1,0.49,0.08,0.28,0.15\n"
        + "MyOp,87.9,0.48,0.08,0.29,0.15\n"
        + "MyOp,120.0,0.20,0.10,0.55,0.15\n"  # outlier
        + "MyOp,88.0,0.49,0.08,0.28,0.15\n"
        + "MyOp,88.4,0.50,0.08,0.27,0.15\n",
        encoding="utf-8",
    )

    raw_result = {
        "op_name": "MyOp",
        "repo_type": "nn",
        "soc": "ascend910b",
        "evolved": {
            "build_success": True,
            "precision_passed": True,
            "correctness_message": "PASS",
            "profiling_dir": "/tmp/evolved_profile",
            "raw_op_summary_files": [str(csv1)],
            "warmup_count": 2,
            "pipeline_candidates": {
                "mte2_pct": 28.0,
                "vec_pct": 49.0,
                "scalar_pct": 8.0,
                "mte3_pct": 15.0,
            },
        },
    }

    result = MODULE.build_evaluation_result(
        compilation_success=True,
        baseline_time_us=100.0,
        raw_result=raw_result,
        evolved_install_path="/tmp/evolved_install",
    )

    assert result["op_name"] == "MyOp"
    assert result["repo_type"] == "nn"
    assert result["soc"] == "ascend910b"
    assert result["baseline"]["time_us"] == 100.0
    assert result["baseline"]["precision_passed"] is True
    assert result["evolved"]["time_us"] == 88.05
    assert result["evolved"]["n_outliers_removed"] == 1
    assert result["evolved"]["bottleneck"] == "balanced"
    assert result["comparison"]["precision_passed"] is True
    assert result["comparison"]["measurement_quality"] == "good"
    assert result["speedup"] > 1.13
    assert result["compilation_success"] is True
    assert result["eval_backend"] == "forge"


def test_build_side_result_warmup_count_skips_leading_rows_and_each_row_is_own_trial(tmp_path):
    csv1 = tmp_path / "op_summary.csv"
    header = "OP Type,Task Duration(us),aiv_vec_ratio,aiv_scalar_ratio,aiv_mte2_ratio,aiv_mte3_ratio\n"
    # 3 warmup rows (large values) + 5 real rows (small values); outlier 50 should be removed
    csv1.write_text(
        header
        + "MyOp,999.0,0.10,0.10,0.60,0.20\n"  # warmup
        + "MyOp,999.0,0.10,0.10,0.60,0.20\n"  # warmup
        + "MyOp,999.0,0.10,0.10,0.60,0.20\n"  # warmup
        + "MyOp,10.2,0.50,0.10,0.20,0.20\n"
        + "MyOp,10.4,0.49,0.10,0.21,0.20\n"
        + "MyOp,50,0.20,0.10,0.50,0.20\n"    # outlier
        + "MyOp,10.5,0.48,0.10,0.22,0.20\n"
        + "MyOp,10.6,0.47,0.10,0.23,0.20\n",
        encoding="utf-8",
    )

    side = MODULE._build_side_result(
        {
            "precision_passed": True,
            "correctness_message": "PASS",
            "raw_op_summary_files": [str(csv1)],
            "warmup_count": 3,
        },
        tag="evolved",
        op_name="MyOp",
    )

    assert side["n_samples_raw"] == 5
    assert side["n_outliers_removed"] == 1
    assert side["time_us"] == 10.45
    assert side["pipeline"]["vec_ratio"] == 0.48


def test_build_side_result_skips_missing_csv_and_keeps_valid_trials(tmp_path):
    csv1 = tmp_path / "trial1.csv"
    csv1.write_text("OP Type,Task Duration(us)\nMyOp,10\n", encoding="utf-8")
    side = MODULE._build_side_result(
        {
            "precision_passed": True,
            "correctness_message": "PASS",
            "raw_op_summary_files": [str(csv1), str(tmp_path / "missing.csv")],
        },
        tag="evolved",
        op_name="MyOp",
    )

    assert side["n_samples_raw"] == 1
    assert side["time_us"] == 10.0


def test_build_evaluation_result_falls_back_to_legacy_stdout_time():
    result = MODULE.build_evaluation_result(
        compilation_success=True,
        baseline_time_us=120.0,
        raw_result=None,
        evolved_install_path="/tmp/evolved_install",
        legacy_evolved_time_us=90.0,
    )

    assert result["baseline"]["time_us"] == 120.0
    assert result["evolved"]["time_us"] == 90.0
    assert result["comparison"]["speedup"] == 120.0 / 90.0
    assert result["comparison"]["precision_passed"] is True


def test_build_evaluation_result_raw_result_blocks_legacy_timing_without_authoritative_csvs():
    result = MODULE.build_evaluation_result(
        compilation_success=True,
        baseline_time_us=120.0,
        raw_result={
            "op_name": "MyOp",
            "evolved": {
                "precision_passed": True,
                "correctness_message": "PASS",
            },
        },
        evolved_install_path="/tmp/evolved_install",
        legacy_evolved_time_us=90.0,
    )

    assert result["evolved"]["time_us"] == -1
    assert result["comparison"]["speedup"] == 0.0


def test_build_evaluation_result_raw_result_missing_evolved_section_blocks_legacy_timing():
    result = MODULE.build_evaluation_result(
        compilation_success=True,
        baseline_time_us=120.0,
        raw_result={
            "op_name": "MyOp",
        },
        evolved_install_path="/tmp/evolved_install",
        legacy_evolved_time_us=90.0,
    )

    assert result["evolved"]["time_us"] == -1
    assert result["comparison"]["speedup"] == 0.0


def test_build_evaluation_result_non_dict_raw_result_sides_fail_closed():
    result = MODULE.build_evaluation_result(
        compilation_success=True,
        baseline_time_us=120.0,
        raw_result={
            "op_name": "MyOp",
            "baseline": "oops",
            "evolved": ["oops"],
        },
        evolved_install_path="/tmp/evolved_install",
        legacy_evolved_time_us=90.0,
    )

    assert result["baseline"]["time_us"] == 120.0
    assert result["evolved"]["time_us"] == -1
    assert result["comparison"]["speedup"] == 0.0


def test_build_side_result_no_csv_files_and_no_time_fallback_returns_minus_one():
    side = MODULE._build_side_result(
        {
            "precision_passed": True,
            "correctness_message": "PASS",
            "pipeline_candidates": {
                "vec_pct": 49.0,
            },
        },
        tag="evolved",
        op_name="MyOp",
        fallback_time_us=90.0,
        allow_time_fallback=False,
    )

    assert side["time_us"] == -1
    assert side["n_samples_raw"] == 0
    assert side["cv_pct"] == 0.0


def test_build_evaluation_result_raw_result_missing_or_unmatched_csvs_do_not_silently_use_legacy_time(tmp_path):
    missing_csv = tmp_path / "missing.csv"
    unmatched_csv = tmp_path / "unmatched.csv"
    unmatched_csv.write_text("OP Type,Task Duration(us)\nOtherOp,77.0\n", encoding="utf-8")

    result = MODULE.build_evaluation_result(
        compilation_success=True,
        baseline_time_us=120.0,
        raw_result={
            "op_name": "MyOp",
            "evolved": {
                "precision_passed": True,
                "correctness_message": "PASS",
                "raw_op_summary_files": [str(missing_csv), str(unmatched_csv)],
            },
        },
        evolved_install_path="/tmp/evolved_install",
        legacy_evolved_time_us=90.0,
    )

    assert result["evolved"]["time_us"] == -1
    assert result["evolved"]["n_samples_raw"] == 0
    assert result["comparison"]["speedup"] == 0.0


def test_build_side_result_build_failure_blocks_authoritative_csv_and_samples(tmp_path):
    csv_path = tmp_path / "trial1.csv"
    csv_path.write_text("OP Type,Task Duration(us)\nMyOp,77.0\n", encoding="utf-8")

    side = MODULE._build_side_result(
        {
            "build_success": False,
            "precision_passed": False,
            "correctness_message": "build failed",
            "raw_op_summary_files": [str(csv_path)],
            "samples": [77.0],
        },
        tag="evolved",
        op_name="MyOp",
    )

    assert side["time_us"] == -1
    assert side["n_samples_raw"] == 0


def test_build_evaluation_result_marks_failed_build_as_unsuccessful():
    result = MODULE.build_evaluation_result(
        compilation_success=False,
        baseline_time_us=120.0,
        raw_result=None,
        evolved_install_path="/tmp/evolved_install",
    )

    assert result["comparison"]["compilation_success"] is False
    assert result["comparison"]["precision_passed"] is False
    assert result["evolved"]["time_us"] == -1
    assert result["speedup"] == 0.0


def test_build_failed_precision_result_marks_precision_failure():
    result = MODULE._build_failed_precision_result(
        op_name="MyOp",
        repo_type="omni-ops",
        soc="ascend910b",
        install_path="/tmp/evolved_install",
        baseline_time_us=120.0,
        correctness_message="forge accuracy_test failed",
    )
    assert result["compilation_success"] is True
    assert result["precision_passed"] is False
    assert result["baseline"]["time_us"] == 120.0
    assert result["evolved"]["time_us"] == -1
    assert result["evolved"]["correctness_message"] == "forge accuracy_test failed"
    assert result["speedup"] == 0.0


def test_parse_raw_result_path_from_stdout():
    stdout = "some log\nZSEARCH_RAW_RESULT=/tmp/run/forge_raw_result.json\nmore log\n"
    assert MODULE._parse_raw_result_path_from_stdout(stdout) == "/tmp/run/forge_raw_result.json"


def test_resolve_forge_profiling_dir_prefers_explicit_test_path(tmp_path):
    result = MODULE.resolve_forge_profiling_dir(
        forge_config_dir=str(tmp_path),
        forge_config="cfg",
        forge_test_path="/tmp/explicit_tests",
    )
    assert result == "/tmp/explicit_tests"


def test_resolve_forge_profiling_dir_normalizes_relative_explicit_test_path(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    expected = str((config_dir / "../workspace/tests/my_op").resolve())

    result = MODULE.resolve_forge_profiling_dir(
        forge_config_dir=str(config_dir),
        forge_config="cfg",
        forge_test_path="../workspace/tests/my_op",
    )

    assert result == expected


def test_resolve_forge_profiling_dir_reads_config_working_dir_and_test_path(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "cfg.json").write_text(
        json.dumps(
            {
                "working_dir": "../workspace",
                "vars": {"test_path": "tests/my_op"},
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.resolve_forge_profiling_dir(str(config_dir), "cfg")
    assert result == str((config_dir / "../workspace/tests/my_op").resolve())


def test_extract_perf_from_profiling_dir_reads_prof_csv(tmp_path):
    csv_path = tmp_path / "PROF_1" / "mindstudio_profiler_output" / "op_summary_1.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "c0,c1,c2,c3,OP Type,c5,c6,c7,c8,Task Duration(us)\n"
        "x,x,x,x,MyOp,x,x,x,x,123.4\n",
        encoding="utf-8",
    )

    result = MODULE.extract_perf_from_profiling_dir(str(tmp_path), "MyOp")
    assert result == {"MyOp": 123.4}


def test_build_side_result_authoritative_csv_accepts_alias_headers(tmp_path):
    csv_path = tmp_path / "trial1.csv"
    csv_path.write_text(
        "Op Type,Task Duration (us)\n"
        "MyOp,123.4\n",
        encoding="utf-8",
    )

    side = MODULE._build_side_result(
        {
            "precision_passed": True,
            "correctness_message": "PASS",
            "raw_op_summary_files": [str(csv_path)],
        },
        tag="evolved",
        op_name="MyOp",
        allow_time_fallback=False,
    )

    assert side["time_us"] == 123.4
    assert side["n_samples_raw"] == 1


def test_build_side_result_authoritative_csv_skips_malformed_duration_rows(tmp_path):
    csv_path = tmp_path / "trial1.csv"
    csv_path.write_text(
        "OP Type,Task Duration(us)\n"
        "MyOp,not-a-number\n",
        encoding="utf-8",
    )

    side = MODULE._build_side_result(
        {
            "precision_passed": True,
            "correctness_message": "PASS",
            "raw_op_summary_files": [str(csv_path)],
        },
        tag="evolved",
        op_name="MyOp",
        allow_time_fallback=False,
    )

    assert side["time_us"] == -1
    assert side["n_samples_raw"] == 0


def test_build_side_result_warmup_count_all_rows_skipped_returns_minus_one(tmp_path):
    csv_path = tmp_path / "op_summary.csv"
    csv_path.write_text(
        "OP Type,Task Duration(us)\nMyOp,10.0\nMyOp,12.0\n",
        encoding="utf-8",
    )
    side = MODULE._build_side_result(
        {
            "precision_passed": True,
            "correctness_message": "PASS",
            "raw_op_summary_files": [str(csv_path)],
            "warmup_count": 5,  # more than available rows
        },
        tag="baseline",
        op_name="MyOp",
    )

    assert side["time_us"] == -1
    assert side["n_samples_raw"] == 0


def test_extract_perf_from_profiling_dir_aggregates_logically_equivalent_op_names(tmp_path):
    csv_path = tmp_path / "PROF_1" / "mindstudio_profiler_output" / "op_summary_1.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "OP Type,Task Duration(us)\n"
        "MyOp,100.0\n"
        "My Op,120.0\n",
        encoding="utf-8",
    )

    result = MODULE.extract_perf_from_profiling_dir(str(tmp_path), "MyOp")
    assert result == {"MyOp": 110.0}


def test_forge_perf_test_falls_back_to_profiling_csv_when_raw_and_stdout_unavailable(monkeypatch, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "cfg.json").write_text(
        json.dumps(
            {
                "working_dir": "../workspace",
                "vars": {"test_path": "tests/my_op"},
            }
        ),
        encoding="utf-8",
    )

    csv_path = tmp_path / "workspace" / "tests" / "my_op" / "PROF_1" / "mindstudio_profiler_output" / "op_summary_1.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("OP Type,Task Duration(us)\nMyOp,123.4\n", encoding="utf-8")

    monkeypatch.setattr(
        MODULE,
        "_run_perf_workflow",
        lambda forge_bin, config_dir, config_name, arg_file: (
            "performance_test",
            types.SimpleNamespace(returncode=0, stdout="no timings here", stderr=""),
        ),
    )
    monkeypatch.setattr(MODULE, "_find_latest_raw_result", lambda install_path, started_at: None)

    raw_result, raw_result_path, legacy_results = MODULE.forge_perf_test(
        "forge",
        str(config_dir),
        "cfg",
        "MyOp",
        "/tmp/install",
    )

    assert raw_result is None
    assert raw_result_path is None
    assert legacy_results == {"MyOp": 123.4}


def test_forge_perf_test_treats_malformed_raw_result_json_as_unavailable(monkeypatch, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    raw_result_path = tmp_path / "forge_raw_result.json"
    raw_result_path.write_text("{not-valid-json", encoding="utf-8")

    monkeypatch.setattr(
        MODULE,
        "_run_perf_workflow",
        lambda forge_bin, config_dir, config_name, arg_file: (
            "performance_test",
            types.SimpleNamespace(
                returncode=0,
                stdout=f"ZSEARCH_RAW_RESULT={raw_result_path}\nMyOp: 91.5\n",
                stderr="",
            ),
        ),
    )

    raw_result, returned_path, legacy_results = MODULE.forge_perf_test(
        "forge",
        str(config_dir),
        "cfg",
        "MyOp",
        "/tmp/install",
    )

    assert raw_result is None
    assert returned_path == str(raw_result_path)
    assert legacy_results == {"MyOp": 91.5}


def test_forge_perf_test_treats_wrong_shaped_raw_result_json_as_unavailable(monkeypatch, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    raw_result_path = tmp_path / "forge_raw_result.json"
    raw_result_path.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")

    monkeypatch.setattr(
        MODULE,
        "_run_perf_workflow",
        lambda forge_bin, config_dir, config_name, arg_file: (
            "performance_test",
            types.SimpleNamespace(
                returncode=0,
                stdout=f"ZSEARCH_RAW_RESULT={raw_result_path}\nMyOp: 91.5\n",
                stderr="",
            ),
        ),
    )

    raw_result, returned_path, legacy_results = MODULE.forge_perf_test(
        "forge",
        str(config_dir),
        "cfg",
        "MyOp",
        "/tmp/install",
    )

    assert raw_result is None
    assert returned_path == str(raw_result_path)
    assert legacy_results == {"MyOp": 91.5}


def test_forge_perf_test_preserves_raw_result_precedence_over_profiling_csv(monkeypatch, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "cfg.json").write_text(
        json.dumps(
            {
                "working_dir": "../workspace",
                "vars": {"test_path": "tests/my_op"},
            }
        ),
        encoding="utf-8",
    )

    raw_result_path = tmp_path / "forge_raw_result.json"
    raw_payload = {"op_name": "MyOp", "evolved": {"samples": [55.0]}}
    raw_result_path.write_text(json.dumps(raw_payload), encoding="utf-8")

    csv_path = tmp_path / "workspace" / "tests" / "my_op" / "PROF_1" / "mindstudio_profiler_output" / "op_summary_1.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("OP Type,Task Duration(us)\nMyOp,123.4\n", encoding="utf-8")

    monkeypatch.setattr(
        MODULE,
        "_run_perf_workflow",
        lambda forge_bin, config_dir, config_name, arg_file: (
            "performance_test",
            types.SimpleNamespace(
                returncode=0,
                stdout=f"ZSEARCH_RAW_RESULT={raw_result_path}\nno timings here",
                stderr="",
            ),
        ),
    )

    def unexpected_extract(profiling_dir, op_name):
        raise AssertionError("profiling CSV fallback should not run when raw_result is available")

    monkeypatch.setattr(MODULE, "extract_perf_from_profiling_dir", unexpected_extract)

    raw_result, selected_raw_result_path, legacy_results = MODULE.forge_perf_test(
        "forge",
        str(config_dir),
        "cfg",
        "MyOp",
        "/tmp/install",
    )

    assert raw_result == raw_payload
    assert selected_raw_result_path == str(raw_result_path)
    assert legacy_results is None
