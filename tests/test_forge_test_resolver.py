import importlib.util
import json
from pathlib import Path

import pytest

# This test intentionally loads the resolver script directly from the current checkout.
RESOLVER_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".claude"
    / "skills"
    / "ops-evaluation"
    / "scripts"
    / "forge_test_resolver.py"
)


def _write(path: Path, content: str = "print('ok')\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _load_module():
    spec = importlib.util.spec_from_file_location("forge_test_resolver_under_test", RESOLVER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module_under_test():
    return _load_module()


def _resolve_script(module, repo_root: Path, fallback_dir: Path):
    return module.resolve_forge_test_script(
        repo_root=str(repo_root),
        op_relative_path="norm/ada_layer_norm",
        custom_op_name="ada_layer_norm_custom",
        generated_test_path=str(fallback_dir),
        generated_test_script="ada_layer_norm_custom.py",
    )


def _make_repo(tmp_path: Path):
    repo_root = tmp_path / "ops-nn"
    op_dir = repo_root / "norm" / "ada_layer_norm"
    fallback_dir = tmp_path / "generated"
    return repo_root, op_dir, fallback_dir


def test_selects_single_repo_st_script(tmp_path, module_under_test):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    _write(op_dir / "tests" / "st" / "test_ada_layer_norm.py")
    _write(fallback_dir / "ada_layer_norm_custom.py")

    result = _resolve_script(module_under_test, repo_root, fallback_dir)

    assert result["source"] == "repo_st"
    assert result["test_path"] == str(op_dir / "tests" / "st")
    assert result["test_script"] == "test_ada_layer_norm.py"
    assert result["selection_reason"] == "repo_st:operator-name-match,st-directory"


def test_prefers_operator_matching_repo_st_script(tmp_path, module_under_test):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    _write(op_dir / "tests" / "st" / "test_other_op.py")
    _write(op_dir / "tests" / "st" / "test_ada_layer_norm_eager.py")
    _write(fallback_dir / "ada_layer_norm_custom.py")

    result = _resolve_script(module_under_test, repo_root, fallback_dir)

    assert result["source"] == "repo_st"
    assert result["test_script"] == "test_ada_layer_norm_eager.py"
    assert result["selection_reason"] == "repo_st:operator-name-match,st-directory,eager"


def test_prefers_name_match_over_st_directory(tmp_path, module_under_test):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    _write(op_dir / "misc" / "test_ada_layer_norm.py")
    _write(op_dir / "st" / "smoke_test.py")
    _write(fallback_dir / "ada_layer_norm_custom.py")

    result = _resolve_script(module_under_test, repo_root, fallback_dir)

    assert result["source"] == "repo_st"
    assert result["test_path"] == str(op_dir / "misc")
    assert result["test_script"] == "test_ada_layer_norm.py"
    assert result["selection_reason"] == "repo_st:operator-name-match"


def test_keeps_non_test_like_files_under_st_directories(tmp_path, module_under_test):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    _write(op_dir / "st" / "helper.py")
    _write(op_dir / "misc" / "test_other.py")
    _write(fallback_dir / "ada_layer_norm_custom.py")

    result = _resolve_script(module_under_test, repo_root, fallback_dir)

    assert result["source"] == "repo_st"
    assert result["test_path"] == str(op_dir / "st")
    assert result["test_script"] == "helper.py"
    assert result["selection_reason"] == "repo_st:st-directory"


def test_ignores_non_test_python_files_and_falls_back_to_generated(tmp_path, module_under_test):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    _write(op_dir / "state_utils.py")
    _write(fallback_dir / "ada_layer_norm_custom.py")

    result = _resolve_script(module_under_test, repo_root, fallback_dir)

    assert result["source"] == "generated"
    assert result["test_path"] == str(fallback_dir)
    assert result["test_script"] == "ada_layer_norm_custom.py"
    assert result["selection_reason"] == "generated:generated-fallback"


def test_strips_custom_suffix_case_insensitively(tmp_path, module_under_test):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    _write(op_dir / "tests" / "st" / "test_foo.py")
    _write(fallback_dir / "ada_layer_norm_custom.py")

    result = module_under_test.resolve_forge_test_script(
        repo_root=str(repo_root),
        op_relative_path="norm/ada_layer_norm",
        custom_op_name="Foo_CUSTOM",
        generated_test_path=str(fallback_dir),
        generated_test_script="ada_layer_norm_custom.py",
    )

    assert result["source"] == "repo_st"
    assert result["test_path"] == str(op_dir / "tests" / "st")
    assert result["test_script"] == "test_foo.py"
    assert result["selection_reason"] == "repo_st:operator-name-match,st-directory"


def test_rejects_operator_paths_that_escape_repo_root(tmp_path, module_under_test):
    repo_root = tmp_path / "ops-nn"
    escaped_op_dir = tmp_path / "escape" / "norm" / "ada_layer_norm"
    _write(escaped_op_dir / "tests" / "st" / "test_ada_layer_norm.py")
    _write(tmp_path / "generated" / "ada_layer_norm_custom.py")

    with pytest.raises((FileNotFoundError, ValueError), match="repo root|Operator directory"):
        module_under_test.resolve_forge_test_script(
            repo_root=str(repo_root),
            op_relative_path="../escape/norm/ada_layer_norm",
            custom_op_name="ada_layer_norm_custom",
            generated_test_path=str(tmp_path / "generated"),
            generated_test_script="ada_layer_norm_custom.py",
        )


def test_falls_back_to_generated_script_when_repo_st_missing(tmp_path, module_under_test):
    repo_root, _, fallback_dir = _make_repo(tmp_path)
    (repo_root / "norm" / "ada_layer_norm").mkdir(parents=True)
    _write(fallback_dir / "ada_layer_norm_custom.py")

    result = _resolve_script(module_under_test, repo_root, fallback_dir)

    assert result["source"] == "generated"
    assert result["test_path"] == str(fallback_dir)
    assert result["test_script"] == "ada_layer_norm_custom.py"
    assert result["selection_reason"] == "generated:generated-fallback"


def test_errors_when_no_repo_st_or_generated_fallback_exists(tmp_path, module_under_test):
    repo_root, _, _ = _make_repo(tmp_path)
    (repo_root / "norm" / "ada_layer_norm").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No forge test script found"):
        _resolve_script(module_under_test, repo_root, tmp_path / "generated")


def test_main_prints_json_result(tmp_path, capsys, module_under_test):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    _write(op_dir / "tests" / "st" / "test_ada_layer_norm.py")
    _write(fallback_dir / "ada_layer_norm_custom.py")

    exit_code = module_under_test.main([
        "--repo-root",
        str(repo_root),
        "--op-relative-path",
        "norm/ada_layer_norm",
        "--custom-op-name",
        "ada_layer_norm_custom",
        "--generated-test-path",
        str(fallback_dir),
        "--generated-test-script",
        "ada_layer_norm_custom.py",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["source"] == "repo_st"
    assert payload["test_script"] == "test_ada_layer_norm.py"
    assert payload["selection_reason"] == "repo_st:operator-name-match,st-directory"


def test_resolver_finds_benchmark_script(module_under_test, tmp_path):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    benchmark_dir = op_dir / "tests" / "benchmark"
    benchmark_dir.mkdir(parents=True)
    _write(benchmark_dir / "benchmark_ada_layer_norm.py")
    result = _resolve_script(module_under_test, repo_root, fallback_dir)
    assert result["source"] == "repo_st"
    assert "benchmark" in result["test_script"]


def test_resolver_prefers_benchmark_with_name_over_st(module_under_test, tmp_path):
    repo_root, op_dir, fallback_dir = _make_repo(tmp_path)
    st_dir = op_dir / "st"
    st_dir.mkdir(parents=True)
    _write(st_dir / "test_ada_layer_norm.py")
    benchmark_dir = op_dir / "tests" / "benchmark"
    benchmark_dir.mkdir(parents=True)
    _write(benchmark_dir / "benchmark_ada_layer_norm.py")
    result = _resolve_script(module_under_test, repo_root, fallback_dir)
    assert result["source"] == "repo_st"
    assert "benchmark" in result["test_script"]
