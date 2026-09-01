from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import make_config, write_burst

from edge_denoise.config import Config
from edge_denoise.train import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_KIND,
    LATEST_CHECKPOINT_NAME,
    Trainer,
    load_checkpoint,
)

CHECKPOINT_KEYS = {
    "format",
    "kind",
    "step",
    "config",
    "model",
    "ema",
    "optimizer",
    "torch_rng",
    "cuda_rng",
    "factory",
}


@pytest.mark.parametrize("representation", ["image", "gradient", "hybrid"])
def test_training_runs_and_checkpoints_for_every_representation(
    tmp_path: Path, representation: str
) -> None:
    dataset = write_burst(tmp_path / "data")
    config = make_config(dataset, tmp_path / "run", representation=representation)
    trainer = Trainer(config)
    checkpoint = trainer.run()
    assert trainer.step == config.training.max_steps
    assert checkpoint == tmp_path / "run" / LATEST_CHECKPOINT_NAME
    payload = load_checkpoint(checkpoint)
    assert set(payload.keys()) == CHECKPOINT_KEYS
    assert payload["kind"] == CHECKPOINT_KIND
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["ema"] is not None
    assert (tmp_path / "run" / "config.yml").is_file()
    restored = Config.model_validate(payload["config"])
    assert restored == config


def test_consistency_objective_trains_and_uses_three_replicas(tmp_path: Path) -> None:
    dataset = write_burst(tmp_path / "data", replicas=4)
    config = make_config(dataset, tmp_path / "run", lambda_consistency=0.5)
    trainer = Trainer(config)
    trainer.run()
    assert trainer.factory.need_second is True


def test_resume_restores_step_and_continues_the_exact_stream(tmp_path: Path) -> None:
    dataset = write_burst(tmp_path / "data")
    full_config = make_config(dataset, tmp_path / "run_full", max_steps=4)
    torch.manual_seed(0)
    Trainer(full_config).run()
    full = load_checkpoint(tmp_path / "run_full" / LATEST_CHECKPOINT_NAME)

    half_config = make_config(dataset, tmp_path / "run_half", max_steps=2)
    torch.manual_seed(0)
    Trainer(half_config).run()
    resumed_config = make_config(dataset, tmp_path / "run_half", max_steps=4)
    with pytest.warns(UserWarning, match="different config"):
        trainer = Trainer(
            resumed_config, resume_from=tmp_path / "run_half" / LATEST_CHECKPOINT_NAME
        )
    assert trainer.step == 2
    trainer.run()
    resumed = load_checkpoint(tmp_path / "run_half" / LATEST_CHECKPOINT_NAME)
    assert resumed["step"] == 4
    for name, value in full["model"].items():
        assert torch.allclose(resumed["model"][name], value, atol=1e-6), name


def test_stop_file_halts_training_and_still_saves(tmp_path: Path) -> None:
    dataset = write_burst(tmp_path / "data")
    config = make_config(dataset, tmp_path / "run", max_steps=50)
    trainer = Trainer(config)
    trainer.stop_file.write_text("stop", encoding="utf-8")
    checkpoint = trainer.run()
    assert trainer.step == 0
    assert checkpoint.is_file()


def test_burst_checkpoints_are_rejected(tmp_path: Path) -> None:
    payload = {"format": 1, "step": 1, "config": {}, "model": {}}
    path = tmp_path / "burst.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="edge_denoise"):
        load_checkpoint(path)
