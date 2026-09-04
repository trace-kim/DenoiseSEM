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


def test_init_checkpoint_warm_starts_from_a_burst_payload(tmp_path: Path) -> None:
    """A burst_diffusion checkpoint (bare U-Net keys, no 'kind') seeds both the
    live weights AND the fresh EMA shadow with its EMA values."""
    from edge_denoise.model import build_model

    dataset = write_burst(tmp_path / "data")
    config = make_config(dataset, tmp_path / "run", representation="image", max_steps=1)
    donor = build_model(config)
    live = {name: torch.randn_like(value) for name, value in donor.unet.state_dict().items()}
    ema = {
        name: torch.full_like(parameter, 0.5)
        for name, parameter in donor.unet.named_parameters()
    }
    teacher = tmp_path / "teacher.pt"
    torch.save({"format": 1, "step": 9, "config": {}, "model": live, "ema": ema}, teacher)

    raw = config.model_dump(mode="json")
    raw["training"]["init_checkpoint"] = str(teacher)
    trainer = Trainer(Config.model_validate(raw))
    for name, parameter in trainer.model.unet.named_parameters():
        assert torch.equal(parameter.data, torch.full_like(parameter, 0.5)), name
    assert trainer.ema is not None
    for name, value in trainer.ema.shadow.items():
        assert torch.equal(value, torch.full_like(value, 0.5)), name

    hybrid_raw = make_config(dataset, tmp_path / "run_hybrid", max_steps=1).model_dump(
        mode="json"
    )
    hybrid_raw["training"]["init_checkpoint"] = str(teacher)
    with pytest.raises(ValueError, match="backbone"):
        Trainer(Config.model_validate(hybrid_raw))  # 3-channel conv_in vs 1-channel donor


def test_init_checkpoint_accepts_edge_checkpoints_and_rejects_resume(tmp_path: Path) -> None:
    dataset = write_burst(tmp_path / "data")
    first = make_config(dataset, tmp_path / "run_a", representation="image", max_steps=1)
    checkpoint = Trainer(first).run()

    raw = make_config(dataset, tmp_path / "run_b", representation="image", max_steps=1)
    raw = raw.model_dump(mode="json")
    raw["training"]["init_checkpoint"] = str(checkpoint)
    warm_config = Config.model_validate(raw)
    warm = Trainer(warm_config)
    reference = load_checkpoint(checkpoint)
    for name, parameter in warm.model.named_parameters():
        assert torch.equal(parameter.data, reference["ema"][name]), name
    with pytest.raises(ValueError, match="warm start"):
        Trainer(warm_config, resume_from=checkpoint)


def test_training_runs_with_each_gradient_target_mode(tmp_path: Path) -> None:
    from burst_diffusion.data import BurstCache

    dataset = write_burst(tmp_path / "data")
    for mode in ("clean", "noisy_mean"):
        raw = make_config(
            dataset, tmp_path / f"run_{mode}", representation="image", max_steps=2
        ).model_dump(mode="json")
        raw["objective"]["gradient_target"] = mode
        trainer = Trainer(Config.model_validate(raw))
        trainer.run()
        assert trainer.step == 2

    cache = BurstCache(
        dataset, channels=1, min_replicas=2, min_size=16, val_fraction=0.34, split_seed=7
    )
    targets_dir = tmp_path / "targets"
    targets_dir.mkdir()
    for source in list(cache.train_sources) + list(cache.val_sources):
        import numpy as np

        np.save(
            targets_dir / f"{source.source_index:05d}.npy",
            (source.clean.astype(np.float32) / 255.0),
        )
    raw = make_config(
        dataset, tmp_path / "run_file", representation="image", max_steps=2
    ).model_dump(mode="json")
    raw["objective"]["gradient_target"] = "file"
    raw["objective"]["gradient_target_dir"] = str(targets_dir)
    trainer = Trainer(Config.model_validate(raw))
    trainer.run()
    assert trainer.step == 2


def test_fresh_run_refuses_an_occupied_run_dir_unless_overwritten(tmp_path: Path) -> None:
    """A second run in the same run_dir overwrote the 2026-09-02 ft_consist
    checkpoint and provenance while TensorBoard merged both histories; a fresh
    run now refuses the directory, resume still works, and overwrite=True
    clears exactly the previous run's artifacts first."""
    from edge_denoise.train import TENSORBOARD_DIR_NAME, existing_run_artifacts

    dataset = write_burst(tmp_path / "data")
    run_dir = tmp_path / "run"
    config = make_config(dataset, run_dir, representation="image", max_steps=2)
    assert existing_run_artifacts(run_dir) == []
    Trainer(config).run()
    (run_dir / "notes.txt").write_text("keep me", encoding="utf-8")
    names = {path.name for path in existing_run_artifacts(run_dir)}
    assert names == {"ckpt_0000002.pt", LATEST_CHECKPOINT_NAME, "config.yml", TENSORBOARD_DIR_NAME}

    with pytest.raises(FileExistsError, match="already holds a run"):
        Trainer(config)

    # Resume is the sanctioned way back into an occupied directory.
    resumed = Trainer(config, resume_from=run_dir / LATEST_CHECKPOINT_NAME)
    assert resumed.step == 2

    events_before = sorted((run_dir / TENSORBOARD_DIR_NAME).iterdir())
    assert len(events_before) == 1
    Trainer(config, overwrite=True).run()
    events_after = sorted((run_dir / TENSORBOARD_DIR_NAME).iterdir())
    assert len(events_after) == 1 and events_after != events_before
    assert (run_dir / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_provenance_records_the_warm_start_checkpoint_by_content(tmp_path: Path) -> None:
    from burst_diffusion.provenance import file_sha256

    from edge_denoise.provenance import write_provenance

    dataset = write_burst(tmp_path / "data")
    teacher_config = make_config(dataset, tmp_path / "teacher", representation="image", max_steps=1)
    teacher = Trainer(teacher_config).run()

    raw = make_config(dataset, tmp_path / "student", representation="image", max_steps=1)
    raw = raw.model_dump(mode="json")
    raw["training"]["init_checkpoint"] = str(teacher)
    student_config = Config.model_validate(raw)
    final = Trainer(student_config).run()

    record = write_provenance(tmp_path / "student", student_config, checkpoint=final)
    import json

    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["init_checkpoint"]["sha256"] == file_sha256(teacher)
    assert payload["init_checkpoint"]["step"] == 1
    assert payload["init_checkpoint"]["kind"] == CHECKPOINT_KIND
    assert payload["checkpoint"]["sha256"] == file_sha256(final)
    assert payload["checkpoint"]["sha256"] != payload["init_checkpoint"]["sha256"]

    plain = write_provenance(tmp_path / "teacher", teacher_config, checkpoint=teacher)
    assert json.loads(plain.read_text(encoding="utf-8"))["init_checkpoint"] is None
