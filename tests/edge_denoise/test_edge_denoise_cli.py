from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from conftest import make_config, write_burst
from PIL import Image
from typer.testing import CliRunner

from edge_denoise.cli import app

runner = CliRunner()


def _write_config_yaml(tmp_path: Path, **overrides) -> Path:
    dataset = write_burst(tmp_path / "data", bar=True)
    config = make_config(dataset, tmp_path / "run", **overrides)
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    return path


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "denoise", "evaluate", "repeatability"):
        assert command in result.output


def test_train_then_evaluate_then_repeatability(tmp_path: Path) -> None:
    config_path = _write_config_yaml(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    checkpoint = tmp_path / "run" / "ckpt_latest.pt"
    assert checkpoint.is_file()
    assert (tmp_path / "run" / "provenance.json").is_file()
    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["pipeline"] == "edge_denoise"
    assert provenance["dataset"]["distinct_contents"] == 4

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config", str(config_path),
            "--checkpoint", str(checkpoint),
            "--out", str(tmp_path / "eval"),
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    results = json.loads((tmp_path / "eval" / "results.json").read_text(encoding="utf-8"))
    assert set(results["methods"]) == {"single_frame", "avg_of_n", "one_shot"}

    result = runner.invoke(
        app,
        [
            "repeatability",
            "--config", str(config_path),
            "--checkpoint", f"mine={checkpoint}",
            "--out", str(tmp_path / "rep"),
            "--seeds", "2",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "one_shot@mine" in result.output
    assert (tmp_path / "rep" / "repeatability.json").is_file()
    # Edge arms are bound to the exact checkpoint evaluated, not just a name.
    from burst_diffusion.provenance import file_sha256

    rep = json.loads((tmp_path / "rep" / "repeatability.json").read_text(encoding="utf-8"))
    assert rep["provider_arms"] == ["mine"]
    assert rep["provider_checkpoints"]["mine"]["sha256"] == file_sha256(checkpoint)
    assert rep["provider_checkpoints"]["mine"]["step"] == 2
    assert rep["provider_checkpoints"]["mine"]["kind"] == "edge_denoise"
    assert isinstance(rep["command"], str) and rep["command"]


def test_denoise_covers_the_whole_frame_by_default(tmp_path: Path) -> None:
    """A measurement larger than the training crop is denoised in FULL by
    default (blended tiles); --center-crop restores the single-tile behavior.
    Regression for the 512x512 deployment case: the command used to silently
    center-crop every input to the training resolution."""
    config_path = _write_config_yaml(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    checkpoint = tmp_path / "run" / "ckpt_latest.pt"

    frame = np.random.default_rng(5).integers(0, 256, size=(40, 48), dtype=np.uint8)
    measurement = tmp_path / "frame.png"
    Image.fromarray(frame).save(measurement)

    result = runner.invoke(
        app,
        [
            "denoise",
            "--checkpoint", str(checkpoint),
            "--input", str(measurement),
            "--out", str(tmp_path / "out_full"),
            "--stride", "8",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "denoised the full 40x48 frame" in result.output
    with Image.open(tmp_path / "out_full" / "frame_denoised.png") as image:
        assert image.size == (48, 40)  # PIL size is (W, H): the whole frame

    result = runner.invoke(
        app,
        [
            "denoise",
            "--checkpoint", str(checkpoint),
            "--input", str(measurement),
            "--out", str(tmp_path / "out_crop"),
            "--center-crop",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    with Image.open(tmp_path / "out_crop" / "frame_denoised.png") as image:
        assert image.size == (16, 16)  # the training-resolution center tile


def test_denoise_full_dataset_frame_reports_psnr(tmp_path: Path) -> None:
    """--dataset mode denoises the full stored frame and still reports PSNR
    against the clean image at matching (full) size."""
    config_path = _write_config_yaml(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "denoise",
            "--checkpoint", str(tmp_path / "run" / "ckpt_latest.pt"),
            "--dataset", str(tmp_path / "data"),
            "--source-index", "0",
            "--out", str(tmp_path / "out"),
            "--stride", "8",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "denoised the full 20x24 frame" in result.output
    assert "PSNR vs clean [denoised]" in result.output
    with Image.open(tmp_path / "out" / "src00000_rep00000_denoised.png") as image:
        assert image.size == (24, 20)


def test_train_refuses_an_occupied_run_dir_without_overwrite(tmp_path: Path) -> None:
    config_path = _write_config_yaml(tmp_path)
    assert runner.invoke(app, ["train", "--config", str(config_path)]).exit_code == 0
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "already holds a run" in result.output and "--overwrite" in result.output
    result = runner.invoke(app, ["train", "--config", str(config_path), "--overwrite"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "ckpt_latest.pt").is_file()
    result = runner.invoke(app, ["train", "--config", str(config_path), "--resume"])
    assert result.exit_code == 0, result.output


def test_train_resume_flag_requires_a_checkpoint(tmp_path: Path) -> None:
    config_path = _write_config_yaml(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config_path), "--resume"])
    assert result.exit_code != 0
    assert "resume checkpoint not found" in result.output


def test_repeatability_rejects_duplicate_arm_names(tmp_path: Path) -> None:
    config_path = _write_config_yaml(tmp_path)
    result = runner.invoke(
        app,
        [
            "repeatability",
            "--config", str(config_path),
            "--checkpoint", "a=x.pt",
            "--checkpoint", "a=y.pt",
            "--out", str(tmp_path / "rep"),
        ],
    )
    assert result.exit_code != 0
    assert "duplicate arm" in result.output
