import sys
from pathlib import Path

import pytest
import typer

from ddim.spec import DdimTrainingSpec
from runctl.cli import (
    GENERAL_ARGV_OPTIONS,
    _canonical_argv,
    _launch_preflight,
    _load_plan,
    _parse_set_overrides,
    _reject_duplicate_scalar_options,
    flow_default_config,
    resolve_flow,
)
from runctl.schemas import ExecutorType, MachineProfile


def test_cli_rejects_duplicate_and_conflicting_boolean_spellings() -> None:
    with pytest.raises(typer.BadParameter, match="duplicate"):
        _reject_duplicate_scalar_options(["train", "launch", "--max-steps", "1", "--max-steps", "2"])
    with pytest.raises(typer.BadParameter, match="duplicate"):
        _reject_duplicate_scalar_options(
            ["train", "launch", "--cache-in-memory", "--no-cache-in-memory"]
        )


def test_cli_enforces_the_single_active_experiment_config(tmp_path: Path) -> None:
    alternate = tmp_path / "alternate.yml"
    alternate.write_text("training: {}\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="single active config"):
        _load_plan("not-needed", alternate, {})


def test_canonical_command_exposes_every_varying_setting() -> None:
    """Nothing experiment-varying may stay implicit in the config file.

    Shared settings appear as typed flags; the flow's own settings appear as
    one --set each.  Both halves are asserted so a flow cannot quietly drop a
    knob out of the reproducible command.
    """

    flow = resolve_flow("ddim")
    spec = DdimTrainingSpec()
    argv = _canonical_argv("machine", flow_default_config(flow), spec, flow)

    assert argv[:5] == ("runctl", "train", "launch", "--flow", "ddim")
    for option, _field in GENERAL_ARGV_OPTIONS:
        assert argv.count(option) == 1
    assert ("--cache-in-memory" in argv) != ("--no-cache-in-memory" in argv)

    rendered = {item.split("=", 1)[0] for item in argv if "=" in item}
    for option in flow.options:
        assert option.name in rendered, option.name
    assert argv.count("--set") == len(flow.options)
    assert argv[-1] == "--yes"


def test_set_overrides_reject_unknown_names_and_duplicates() -> None:
    with pytest.raises(typer.BadParameter, match="is not a setting"):
        _parse_set_overrides(["nonsense=1"], DdimTrainingSpec)
    with pytest.raises(typer.BadParameter, match="duplicate --set"):
        _parse_set_overrides(["image_size=32", "image_size=64"], DdimTrainingSpec)
    with pytest.raises(typer.BadParameter, match="NAME=VALUE"):
        _parse_set_overrides(["image_size"], DdimTrainingSpec)


def test_set_overrides_parse_yaml_typed_values() -> None:
    parsed = _parse_set_overrides(
        ["image_size=64", "ch_mult=[1, 2, 2, 2]", "ema=false", "beta_end=0.2"],
        DdimTrainingSpec,
    )
    assert parsed == {
        "image_size": 64,
        "ch_mult": (1, 2, 2, 2),
        "ema": False,
        "beta_end": 0.2,
    }


def test_launch_preflight_accepts_selected_gpu_on_multi_gpu_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = MachineProfile(
        machine_id="four-gpu-host",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": tmp_path / "data"},
        python_executable=sys.executable,
        gpu_index=2,
        expected_gpu="Selected GPU",
    )
    report = {
        "available": True,
        "count": 4,
        "names": ["GPU 0", "GPU 1", "Selected GPU", "GPU 3"],
        "selected_index": 2,
        "selected_name": "Selected GPU",
        "selection_error": None,
        "cuda_visible_devices": None,
    }
    monkeypatch.setattr("runctl.cli._probe_configured_gpu", lambda _profile: report)

    _launch_preflight(profile)


def test_launch_preflight_checks_expected_name_on_selected_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = MachineProfile(
        machine_id="wrong-selected-gpu",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": tmp_path / "data"},
        python_executable=sys.executable,
        gpu_index=1,
        expected_gpu="H100",
    )
    report = {
        "available": True,
        "count": 4,
        "names": ["H100", "RTX A6000", "H100", "H100"],
        "selected_index": 1,
        "selected_name": "RTX A6000",
        "selection_error": None,
        "cuda_visible_devices": None,
    }
    monkeypatch.setattr("runctl.cli._probe_configured_gpu", lambda _profile: report)

    with pytest.raises(RuntimeError, match="selected GPU.*does not match"):
        _launch_preflight(profile)


def test_launch_preflight_rejects_out_of_range_gpu_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = MachineProfile(
        machine_id="bad-gpu-index",
        executor=ExecutorType.FOREGROUND,
        runs_root=tmp_path / "runs",
        datasets={"sem": tmp_path / "data"},
        python_executable=sys.executable,
        gpu_index=4,
    )
    report = {
        "available": True,
        "count": 4,
        "names": ["GPU 0", "GPU 1", "GPU 2", "GPU 3"],
        "selected_index": None,
        "selected_name": None,
        "selection_error": "configured gpu_index 4 is out of range for 4 visible CUDA GPU(s)",
        "cuda_visible_devices": None,
    }
    monkeypatch.setattr("runctl.cli._probe_configured_gpu", lambda _profile: report)

    with pytest.raises(RuntimeError, match="gpu_index 4 is out of range"):
        _launch_preflight(profile)
