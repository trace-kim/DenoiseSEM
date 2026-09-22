from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml
from typer.testing import CliRunner

from edge_denoise import real_suite as suite
from edge_denoise.cli import app
from edge_denoise.train import load_checkpoint
from test_real_sem_next_phase import prepared_uint8, real_teacher


@pytest.fixture
def suite_plan(prepared_uint8, real_teacher, tmp_path):
    args = suite.build_parser().parse_args([
        "--dataset-dir", str(prepared_uint8), "--n2n-checkpoint", str(real_teacher),
        "--run-root", str(tmp_path / "runs"), "--date", "260922", "--device", "cpu",
        "--metrology-device", "cpu", "--max-steps", "2", "--batch-size", "1",
        "--accumulation-steps", "1", "--cpu-threads", "1",
    ])
    return suite.build_plan(args)


def saved_state(plan):
    return json.loads((Path(plan["suite_dir"]) / "suite.json").read_text(encoding="utf-8"))


def complete_process(plan, command, log_path, run_dir, stop, cpu_threads, *, step=2, status="complete"):
    config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
    log_path.write_text("mocked child process\n")
    torch.save({"kind": "edge_denoise", "format": 1, "config": config, "step": step,
                "dataset_fingerprint": plan["dataset_fingerprint"]}, run_dir / "ckpt_latest.pt")
    (run_dir / "training_status.json").write_text(json.dumps({"status": status, "step": step}))
    return 0


def test_suite_preflight_is_read_only_and_keeps_order_and_teacher_settings(suite_plan):
    assert not Path(suite_plan["suite_dir"]).exists()
    assert [item["name"] for item in suite_plan["pipelines"]] == list(suite.PIPELINES)
    assert suite_plan["teacher"]["registration"] == "none"  # Preserve this teacher's actual history.
    for pipeline in suite_plan["pipelines"]:
        config = pipeline["config"]
        assert config["data"]["real_matching"]["registration"] == "affine"
        assert config["data"]["real_matching"]["brightness"] == "percentile"
        assert config["model"]["attention"] is False  # Inherited real teacher, not template attention.
        assert config["training"]["max_steps"] == 2
        assert (config["training"]["init_checkpoint"] is None) == (pipeline["name"] == "grad")


def test_suite_actual_training_entry_point_completes_all_methods(suite_plan, monkeypatch):
    calls = []
    def invoke(command, log_path, run_dir, stop, cpu_threads):
        calls.append(command)
        result = CliRunner().invoke(app, command[command.index("train"):])
        log_path.write_text(result.output, encoding="utf-8")
        assert result.exit_code == 0, (result.output, result.exception)
        return result.exit_code
    monkeypatch.setattr(suite, "_launch", invoke)
    assert suite.run_suite(suite_plan) == 0
    assert len(calls) == 5
    state = saved_state(suite_plan)
    assert state["status"] == "complete"
    assert all(row["status"] == "complete" for row in state["pipelines"].values())
    comparison = yaml.safe_load((Path(suite_plan["suite_dir"]) / "comparison.yml").read_text())
    assert list(comparison["checkpoints"]) == ["n2n", *suite.PIPELINES]
    assert comparison["contour_method"] == "otsu"
    calls.clear()
    assert suite.run_suite(suite_plan, resume=True) == 0
    assert not calls


def test_failures_continue_and_resume_retries_only_unfinished(suite_plan, monkeypatch):
    calls = []
    def fail_first_two(command, log_path, run_dir, stop, cpu_threads):
        name = Path(command[command.index("--config") + 1]).stem
        calls.append((name, "--resume" in command))
        if name == "ft_noisy":
            log_path.write_text("startup failure")
            (run_dir / "config.yml").write_text("preserve failed startup")
            return 7
        if name == "ft_consist":
            complete_process(suite_plan, command, log_path, run_dir, stop, cpu_threads, step=1, status="failed")
            return 3
        return complete_process(suite_plan, command, log_path, run_dir, stop, cpu_threads)
    monkeypatch.setattr(suite, "_launch", fail_first_two)
    assert suite.run_suite(suite_plan) == 1
    assert [name for name, _ in calls] == list(suite.PIPELINES)
    state = saved_state(suite_plan)
    assert state["status"] == "incomplete"
    assert state["pipelines"]["ft_noisy"]["status"] == "failed"
    assert state["pipelines"]["grad"]["status"] == "complete"
    calls.clear()
    def finish(command, log_path, run_dir, stop, cpu_threads):
        calls.append((Path(command[command.index("--config") + 1]).stem, "--resume" in command))
        return complete_process(suite_plan, command, log_path, run_dir, stop, cpu_threads)
    monkeypatch.setattr(suite, "_launch", finish)
    assert suite.run_suite(suite_plan, resume=True) == 0
    assert calls == [("ft_noisy", False), ("ft_consist", True)]
    first_dir = Path(suite_plan["pipelines"][0]["config"]["training"]["run_dir"])
    assert (first_dir / "failed_attempts/before_attempt_002/config.yml").read_text() == "preserve failed startup"
    assert len(saved_state(suite_plan)["pipelines"]["ft_noisy"]["attempts"]) == 2


def test_zero_exit_cannot_hide_unfinished_budget(suite_plan, monkeypatch):
    calls = []
    def unfinished(command, log_path, run_dir, stop, cpu_threads):
        calls.append(command)
        return complete_process(suite_plan, command, log_path, run_dir, stop, cpu_threads, step=1)
    monkeypatch.setattr(suite, "_launch", unfinished)
    assert suite.run_suite(suite_plan) == 1
    assert len(calls) == 5
    assert all("unfinished" in row["error"] for row in saved_state(suite_plan)["pipelines"].values())


def test_resume_marks_an_uncleanly_interrupted_attempt(suite_plan, monkeypatch):
    monkeypatch.setattr(suite, "_launch", lambda *a: complete_process(suite_plan, *a))
    assert suite.run_suite(suite_plan) == 0
    manifest = saved_state(suite_plan)
    manifest["status"] = "running"
    first = manifest["pipelines"]["ft_noisy"]
    first["status"] = first["attempts"][-1]["status"] = "running"
    first["attempts"][-1].pop("wall_seconds")
    first["attempts"][-1].pop("finished_at")
    (Path(suite_plan["suite_dir"]) / "suite.json").write_text(json.dumps(manifest))
    assert suite.run_suite(suite_plan, resume=True) == 0
    first = saved_state(suite_plan)["pipelines"]["ft_noisy"]
    assert first["attempts"][0]["status"] == "interrupted"
    assert first["attempts_without_wall_time"] == 1


def test_explicit_stop_stops_suite_but_stale_stop_status_does_not(suite_plan, monkeypatch):
    calls = []
    def stopped(command, log_path, run_dir, stop, cpu_threads):
        calls.append(command)
        return complete_process(suite_plan, command, log_path, run_dir, stop, cpu_threads, step=1, status="stopped")
    monkeypatch.setattr(suite, "_launch", stopped)
    assert suite.run_suite(suite_plan) == 130
    assert len(calls) == 1
    assert saved_state(suite_plan)["status"] == "stopped"
    calls.clear()
    def failed_start(command, log_path, run_dir, stop, cpu_threads):
        calls.append(command)
        return 9  # Dies before writing a new status file.
    monkeypatch.setattr(suite, "_launch", failed_start)
    assert suite.run_suite(suite_plan, resume=True) == 1
    assert len(calls) == 5
    assert saved_state(suite_plan)["status"] == "incomplete"


def test_foreign_runs_are_never_claimed_on_a_later_retry(suite_plan, monkeypatch):
    foreign = Path(suite_plan["pipelines"][0]["config"]["training"]["run_dir"])
    foreign.mkdir(parents=True)
    (foreign / "config.yml").write_text("unrelated user's run")
    monkeypatch.setattr(suite, "_launch", lambda *a: complete_process(suite_plan, *a))
    assert suite.run_suite(suite_plan) == 1
    assert suite.run_suite(suite_plan, resume=True) == 1
    assert (foreign / "config.yml").read_text() == "unrelated user's run"
    assert not (foreign / "failed_attempts").exists()
    assert not saved_state(suite_plan)["pipelines"]["ft_noisy"]["owned"]


def test_resume_rejects_changed_plan_and_completed_checkpoint(suite_plan, monkeypatch):
    monkeypatch.setattr(suite, "_launch", lambda *a: complete_process(suite_plan, *a))
    assert suite.run_suite(suite_plan) == 0
    modified = copy.deepcopy(suite_plan)
    modified["pipelines"][0]["config"]["training"]["max_steps"] = 3
    with pytest.raises(ValueError, match="resume plan changed"):
        suite.run_suite(modified, resume=True)
    with pytest.raises(ValueError, match="already exists"):
        suite.run_suite(suite_plan)
    manifest_path = Path(suite_plan["suite_dir"]) / "suite.json"
    original = manifest_path.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["plan"]["pipelines"][0]["config"]["training"]["max_steps"] = 99
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="resume plan changed"):
        suite.run_suite(suite_plan, resume=True)
    manifest_path.write_text(original, encoding="utf-8")
    checkpoint = Path(suite_plan["pipelines"][0]["config"]["training"]["run_dir"]) / "ckpt_latest.pt"
    payload = load_checkpoint(checkpoint)
    payload["extra"] = "changed"
    torch.save(payload, checkpoint)
    def forbidden(*args):
        raise AssertionError("should not train after checkpoint integrity failure")
    monkeypatch.setattr(suite, "_launch", forbidden)
    assert suite.run_suite(suite_plan, resume=True) == 1
    assert suite.run_suite(suite_plan, resume=True) == 1


def test_process_environment_keeps_allocated_gpus(tmp_path, monkeypatch):
    captured = {}
    class Process:
        def __init__(self, command, **kwargs):
            captured.update(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self, timeout):
            return 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(suite.subprocess, "Popen", Process)
    stop = suite.StopController(tmp_path / "stop")
    assert suite._launch(["python", "fake"], tmp_path / "log", tmp_path, stop, 2) == 0
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert captured["env"]["OMP_NUM_THREADS"] == "2"


def test_suite_lock_and_budget_validation(tmp_path):
    with suite.suite_lock(tmp_path / "suite.lock"):
        with pytest.raises(ValueError, match="already running"):
            with suite.suite_lock(tmp_path / "suite.lock"):
                pass
    assert suite._step_overrides(["grad=500", "ft_noisy=100"]) == {"grad": 500, "ft_noisy": 100}
    for value in (["bad=10"], ["grad=0"], ["grad=1", "grad=2"], ["grad=abc"]):
        with pytest.raises(ValueError):
            suite._step_overrides(value)
