"""Config validation: every rejection path has a reason a user can act on."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sem_segment.config import Config, load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "sem_segment" / "configs"


def test_defaults_are_a_usable_sam3_auto_setup():
    config = Config()
    assert config.segmentation.backend == "sam3_auto"
    assert config.needs_model_weights is True
    # Cubic interpolation is the default on purpose: bilinear sampling injects a
    # once-per-pixel systematic comparable to the whole precision budget.
    assert config.refine.interp_order == 3
    # Resize, not tile, is the default path for oversized frames.
    assert config.segmentation.large_frame.mode == "resize"


def test_classical_backend_needs_no_weights():
    config = Config.model_validate({"segmentation": {"backend": "classical"}})
    assert config.needs_model_weights is False


def test_unknown_keys_are_rejected_at_every_level():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Config.model_validate({"mistyped_section": {}})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Config.model_validate({"refine": {"estimater": "erf"}})


def test_text_backend_requires_a_prompt():
    with pytest.raises(ValidationError, match="requires a non-empty 'text' prompt"):
        Config.model_validate({"segmentation": {"backend": "sam3_text"}})
    Config.model_validate({"segmentation": {"backend": "sam3_text", "text": "hole"}})


def test_prompt_backend_requires_points_or_boxes():
    with pytest.raises(ValidationError, match="requires 'points' or 'boxes'"):
        Config.model_validate({"segmentation": {"backend": "sam3_prompt"}})
    Config.model_validate({"segmentation": {"backend": "sam3_prompt", "points": [[10.0, 12.0]]}})


def test_contrast_stretch_must_be_ascending_percentiles():
    with pytest.raises(ValidationError, match="ascending percentiles"):
        Config.model_validate({"segmentation": {"contrast_stretch": [99.0, 1.0]}})
    with pytest.raises(ValidationError, match="ascending percentiles"):
        Config.model_validate({"segmentation": {"contrast_stretch": [1.0, 120.0]}})


def test_boxes_must_be_ordered_xyxy():
    with pytest.raises(ValidationError, match=r"x2 > x1"):
        Config.model_validate(
            {"segmentation": {"backend": "sam3_prompt", "boxes": [[10.0, 10.0, 5.0, 20.0]]}}
        )


def test_crop_must_be_large_enough_and_non_negative():
    with pytest.raises(ValidationError, match="at least 16 px"):
        Config.model_validate({"input": {"crop": [0, 8, 0, 8]}})
    with pytest.raises(ValidationError, match="non-negative"):
        Config.model_validate({"input": {"crop": [-1, 100, 0, 100]}})


def test_detector_levels_must_be_paired_and_ordered():
    with pytest.raises(ValidationError, match="must be set together"):
        Config.model_validate({"input": {"black_level": 5.0}})
    with pytest.raises(ValidationError, match="white_level must exceed"):
        Config.model_validate({"input": {"black_level": 100.0, "white_level": 50.0}})


def test_search_window_must_hold_enough_samples_to_fit_an_edge():
    with pytest.raises(ValidationError, match="need at least 9"):
        Config.model_validate({"refine": {"search_px": 4.0, "step_px": 2.0}})


def test_min_area_must_be_consistent_with_the_search_window():
    """A feature narrower than the profile window has no fittable edge."""
    with pytest.raises(ValidationError, match="narrower than the profile window"):
        Config.model_validate({"masks": {"min_area_px": 4.0}, "refine": {"search_px": 8.0}})
    # Disabling refinement removes the coupling.
    Config.model_validate({"masks": {"min_area_px": 4.0}, "refine": {"enabled": False}})


def test_roughness_cutoff_must_exceed_the_contour_smoothing():
    """Otherwise the high-pass band was already smoothed away and LER is noise."""
    with pytest.raises(ValidationError, match="already been smoothed away"):
        Config.model_validate(
            {"metrology": {"ler_highpass_cutoff_px": 2.0}, "refine": {"sigma_normal_px": 2.0}}
        )


def test_tiling_overlap_must_be_smaller_than_the_tile():
    with pytest.raises(ValidationError, match="overlap_px must be smaller"):
        Config.model_validate(
            {"segmentation": {"large_frame": {"mode": "tile", "tile_px": 256, "overlap_px": 256}}}
        )


def test_yaml_round_trip_preserves_every_setting(tmp_path):
    original = Config.model_validate(
        {
            "input": {"pixel_size_nm": 5.82812, "crop": [2, 441, 2, 511]},
            "segmentation": {"backend": "sam3_text", "text": "particle"},
            "refine": {"estimator": "erf"},
        }
    )
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(original.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    assert load_config(path) == original


def test_load_config_rejects_a_non_mapping_root(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config root must be a mapping"):
        load_config(path)


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIG_DIR.glob("*.yml")))
def test_shipped_configs_load(name):
    config = load_config(CONFIG_DIR / name)
    assert isinstance(config, Config)


def test_smoke_config_needs_no_model_download():
    """The committed smoke recipe must run with no GPU and no HF account."""
    config = load_config(CONFIG_DIR / "smoke.yml")
    assert config.needs_model_weights is False
