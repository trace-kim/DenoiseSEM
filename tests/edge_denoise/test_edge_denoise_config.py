from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from edge_denoise.config import Config, load_config


def _base(**overrides) -> dict:
    raw = {
        "data": {"dataset_dir": "data/x", "image_size": 64},
        "objective": {},
        "model": {},
        "training": {"run_dir": "runs/x"},
    }
    for key, value in overrides.items():
        raw[key] = {**raw.get(key, {}), **value}
    return raw


def test_defaults_validate_and_derive_channels() -> None:
    config = Config.model_validate(_base())
    assert config.objective.representation == "hybrid"
    assert config.in_channels == 3
    assert config.out_channels == 1
    assert config.min_replicas == 2  # noisy target, no consistency


def test_representation_drives_channel_counts() -> None:
    image = Config.model_validate(_base(objective={"representation": "image"}))
    assert (image.in_channels, image.out_channels) == (1, 1)
    gradient = Config.model_validate(
        _base(objective={"representation": "gradient", "lambda_image": 0.0})
    )
    assert (gradient.in_channels, gradient.out_channels) == (2, 2)


def test_min_replicas_counts_target_and_consistency_frames() -> None:
    clean = Config.model_validate(_base(objective={"target": "clean"}))
    assert clean.min_replicas == 1
    with_consistency = Config.model_validate(
        _base(objective={"lambda_consistency": 0.5})
    )
    assert with_consistency.min_replicas == 3


def test_gradient_representation_rejects_an_image_term() -> None:
    with pytest.raises(ValidationError, match="lambda_image must be 0"):
        Config.model_validate(
            _base(objective={"representation": "gradient", "lambda_image": 1.0})
        )
    with pytest.raises(ValidationError, match="lambda_gradient must be > 0"):
        Config.model_validate(
            _base(
                objective={
                    "representation": "gradient",
                    "lambda_image": 0.0,
                    "lambda_gradient": 0.0,
                }
            )
        )


def test_consistency_alone_is_rejected_as_degenerate() -> None:
    with pytest.raises(ValidationError, match="constant output"):
        Config.model_validate(
            _base(
                objective={
                    "lambda_image": 0.0,
                    "lambda_gradient": 0.0,
                    "lambda_consistency": 1.0,
                }
            )
        )


def test_unknown_keys_are_rejected_everywhere() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(_base(objective={"lambda_grad": 1.0}))  # typo


def test_gradient_target_modes_validate() -> None:
    default = Config.model_validate(_base())
    assert default.objective.gradient_target == "target"
    assert default.objective.gradient_target_dir is None
    with pytest.raises(ValidationError, match="gradient_target_dir"):
        Config.model_validate(_base(objective={"gradient_target": "file"}))
    with pytest.raises(ValidationError, match="only meaningful"):
        Config.model_validate(
            _base(objective={"gradient_target": "clean", "gradient_target_dir": "runs/t"})
        )
    with pytest.raises(ValidationError, match="lambda_gradient must be > 0"):
        Config.model_validate(
            _base(objective={"gradient_target": "clean", "lambda_gradient": 0.0})
        )


def test_noisy_mean_gradient_target_needs_a_second_replica() -> None:
    config = Config.model_validate(
        _base(objective={"target": "clean", "gradient_target": "noisy_mean"})
    )
    assert config.min_replicas == 2  # clean target alone would need only 1


def test_init_checkpoint_field_is_accepted() -> None:
    config = Config.model_validate(
        _base(training={"init_checkpoint": "runs/ft_ladder/teacher.pt"})
    )
    assert config.training.init_checkpoint == Path("runs/ft_ladder/teacher.pt")


def test_structural_unet_checks_fire_at_load() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        Config.model_validate(
            _base(data={"image_size": 24}, model={"ch_mult": [1, 2, 2, 2, 2]})
        )
    with pytest.raises(ValidationError, match="attn_resolutions"):
        Config.model_validate(_base(model={"attn_resolutions": [24]}))


def test_load_config_round_trips_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(_base()), encoding="utf-8")
    config = load_config(path)
    assert config.data.image_size == 64
    with pytest.raises(ValueError, match="mapping"):
        bad = tmp_path / "bad.yml"
        bad.write_text("- just\n- a list\n", encoding="utf-8")
        load_config(bad)


def test_noisy_mean_target_validates_and_needs_two_replicas() -> None:
    raw = {
        "data": {"dataset_dir": "x", "image_size": 16},
        "objective": {"representation": "image", "target": "noisy_mean", "lambda_gradient": 4.0},
        "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
        "training": {"run_dir": "r"},
    }
    config = Config.model_validate(raw)
    assert config.objective.target == "noisy_mean"
    assert config.min_replicas == 2
    raw["objective"]["lambda_consistency"] = 1.0
    assert Config.model_validate(raw).min_replicas == 2


def test_shipped_sem_synth15_configs_train_native_512_full_frames() -> None:
    """The production 512 configs must stay structurally valid AND buildable:
    full-frame resolution, a bottleneck small enough for the always-on middle
    attention (16x16 at six levels), and a stage-1 -> stage-2 warm start with
    matching state dicts.  Regression for the 2026-09-08 config review."""
    from edge_denoise.model import build_model

    configs_dir = Path(__file__).resolve().parents[2] / "edge_denoise" / "configs"
    teacher = load_config(configs_dir / "sem_synth15_n2n.yml")
    student = load_config(configs_dir / "sem_synth15_ft_avgfull_consist.yml")
    for config in (teacher, student):
        assert config.data.image_size == 512  # native full frames, not crops
        levels = len(config.model.ch_mult)
        assert config.data.image_size >> (levels - 1) == 16  # bottleneck = attn geometry
    assert student.training.init_checkpoint is not None
    assert student.model == teacher.model  # warm start needs identical backbones
    teacher_state = build_model(teacher).state_dict()
    student_state = build_model(student).state_dict()
    assert set(teacher_state) == set(student_state)
    assert all(teacher_state[k].shape == student_state[k].shape for k in teacher_state)
