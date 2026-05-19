import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".claude"
    / "skills"
    / "ops-evaluation"
    / "scripts"
    / "forge_run.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("forge_run_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def test_detect_soc_parses_npu_smi_output(mod):
    fake_output = """\
+---------------------------+------------------+
| npu-smi 24.1.rc1          |                  |
+===========================+==================+
| 0     910B3               | 0000:01:00.0     |
+---------------------------+------------------+
"""
    with patch("subprocess.run", return_value=MagicMock(stdout=fake_output, returncode=0)):
        assert mod.detect_soc() == "ascend910b3"


def test_detect_soc_raises_when_npu_smi_fails(mod):
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="npu-smi"):
            mod.detect_soc()


def test_detect_repo_type_from_build_sh(mod, tmp_path):
    build_sh = tmp_path / "build.sh"
    build_sh.write_text('REPOSITORY_NAME="ops-nn"\necho build\n')
    assert mod.detect_repo_type(str(tmp_path)) == "omni-ops"


def test_detect_repo_type_missing_build_sh(mod, tmp_path):
    with pytest.raises(FileNotFoundError, match="build.sh"):
        mod.detect_repo_type(str(tmp_path))


def test_resolve_op_path_finds_directory(mod, tmp_path):
    repo = tmp_path / "ops-nn"
    (repo / "norm" / "ada_layer_norm" / "op_kernel").mkdir(parents=True)
    assert mod.resolve_op_path(str(repo), "ada_layer_norm_custom") == "norm/ada_layer_norm"


def test_resolve_op_path_strips_custom_suffix(mod, tmp_path):
    repo = tmp_path / "ops-nn"
    (repo / "math" / "my_op" / "op_kernel").mkdir(parents=True)
    assert mod.resolve_op_path(str(repo), "my_op_custom") == "math/my_op"


def test_resolve_op_path_not_found(mod, tmp_path):
    repo = tmp_path / "ops-nn"
    repo.mkdir()
    with pytest.raises(FileNotFoundError, match="no_such_op"):
        mod.resolve_op_path(str(repo), "no_such_op_custom")


def test_derive_shared_dir_from_install_path(mod, tmp_path):
    shared = tmp_path / "output" / "shared"
    shared.mkdir(parents=True)
    install = tmp_path / "output" / "baseline"
    install.mkdir(parents=True)
    assert mod.derive_shared_dir(str(install)) == str(shared)


def test_derive_shared_dir_from_evolved_path(mod, tmp_path):
    shared = tmp_path / "output" / "shared"
    shared.mkdir(parents=True)
    install = tmp_path / "output" / "round_1" / "parallel_0" / "evolved"
    install.mkdir(parents=True)
    assert mod.derive_shared_dir(str(install)) == str(shared)


def test_build_evaluator_command_contains_required_args(mod):
    cmd = mod.build_evaluator_command(
        forge_config="omni_ops_performance_pytest",
        forge_config_dir="/mnt/workspace/forge/configs",
        forge_bin="forge",
        op_name="ada_layer_norm_custom",
        repo_root="/home/user/ops-nn",
        install_path="/output/baseline",
        test_path="/home/user/ops-nn/norm/ada_layer_norm/st",
        test_script="test_ada_layer_norm.py",
        zsearch_side="baseline",
        zsearch_build_success="true",
        zsearch_repo_type="omni-ops",
        zsearch_soc="ascend910b",
        zsearch_task_type="performance",
        zsearch_precision_passed="true",
        zsearch_correctness_message="PASS",
        baseline_time_us=0.0,
        output="/dev/null",
        mode="build",
    )
    assert cmd[0] == sys.executable
    assert "--forge-config" in cmd
    assert "omni_ops_performance_pytest" in cmd
    assert "--op-name" in cmd
    assert "ada_layer_norm_custom" in cmd
    assert "--mode" in cmd
    assert "build" in cmd


def test_main_builds_correct_subprocess_call(mod, tmp_path):
    repo = tmp_path / "ops-nn"
    op_dir = repo / "norm" / "ada_layer_norm"
    st_dir = op_dir / "st"
    st_dir.mkdir(parents=True)
    (st_dir / "test_ada_layer_norm.py").write_text("print('ok')\n")
    (repo / "build.sh").write_text('REPOSITORY_NAME="ops-nn"\n')
    install_path = tmp_path / "output" / "baseline"
    install_path.mkdir(parents=True)
    shared_dir = tmp_path / "output" / "shared"
    shared_dir.mkdir(parents=True)
    output_file = tmp_path / "result.json"

    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        with patch.object(mod, "detect_soc", return_value="ascend910b"):
            mod.main([
                "--op-name", "ada_layer_norm_custom",
                "--repo-root", str(repo),
                "--install-path", str(install_path),
                "--mode", "build",
                "--output", str(output_file),
                "--zsearch-side", "baseline",
                "--shared-dir", str(shared_dir),
            ])

    cmd_str = " ".join(captured_cmd)
    assert "forge_evaluator.py" in cmd_str
    assert "--op-name" in captured_cmd
    assert "ada_layer_norm_custom" in captured_cmd
    assert "--zsearch-side" in captured_cmd
    assert "baseline" in captured_cmd
