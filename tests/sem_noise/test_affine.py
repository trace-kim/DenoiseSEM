from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("skimage")
from scipy import ndimage

from sem_noise.affine import compare_models
from sem_noise.config import AnalysisConfig
from sem_noise.registration import local_diagnostics


def _tiles(matrix=None, offset=(0.0, 0.0), noise=0.005):
    matrix = np.eye(2) if matrix is None else matrix
    rng = np.random.default_rng(41)
    points = np.array([(x, y) for y in np.linspace(30, 225, 5) for x in np.linspace(30, 225, 5)])
    displacement = (points - 127.5) @ (matrix - np.eye(2)).T + offset
    displacement += rng.normal(0, noise, displacement.shape)
    return [dict(frame_position=0, frame_index=7, x_px=p[0], y_px=p[1], residual_dx_px=d[0],
                 residual_dy_px=d[1], correlation=0.95, valid=True, texture_ratio=0.8, peak_ratio=2.0)
            for p, d in zip(points, displacement)]


def _rotation(angle):
    theta = np.radians(angle)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@pytest.mark.parametrize("matrix,expected", [
    (np.eye(2), "translation"), (_rotation(0.7), "rigid"),
    (_rotation(0.7) * 1.01, "similarity"),
    (_rotation(0.5) @ np.array([[1.01, 0.012], [0, 0.994]]), "affine"),
])
def test_selects_simplest_supported_model(matrix, expected):
    rows, frames, summary = compare_models(_tiles(matrix, (0.2, -0.1)), np.zeros((1, 2)),
                                           (256, 256), AnalysisConfig(affine_diagnostics=True))
    assert frames[0]["selected_model"] == expected
    assert summary["selected_model_counts"][expected] == 1
    affine = next(r for r in rows if r["model"] == "affine")
    np.testing.assert_allclose([[affine["m00"], affine["m01"]], [affine["m10"], affine["m11"]]], matrix, atol=5e-5)
    assert affine["correction_center_dx_px"] == pytest.approx(0.2, abs=0.005)
    assert affine["correction_center_dy_px"] == pytest.approx(-0.1, abs=0.005)


def test_composes_translation_with_affine_and_reports_units():
    matrix = _rotation(0.6) @ np.diag([1.005, 0.995])
    rows, _, _ = compare_models(_tiles(matrix, noise=0), np.array([[2.0, -1.0]]), (256, 256),
                                AnalysisConfig(pixel_size_nm=2))
    row = next(r for r in rows if r["model"] == "affine")
    expected = matrix @ np.array([-1.0, 2.0])
    assert row["correction_rotation_deg"] == pytest.approx(0.6)
    assert row["correction_scale_x_percent"] == pytest.approx(0.5)
    assert row["correction_scale_y_percent"] == pytest.approx(-0.5)
    assert row["correction_center_dx_px"] == pytest.approx(expected[0])
    assert row["correction_center_dy_nm"] == pytest.approx(expected[1] * 2)


def test_outlier_does_not_create_false_affine_support():
    tiles = _tiles(offset=(0.2, -0.1))
    tiles[0]["residual_dx_px"] += 2.5
    rows, frames, _ = compare_models(tiles, np.zeros((1, 2)), (256, 256), AnalysisConfig())
    assert frames[0]["selected_model"] == "translation"
    affine = next(r for r in rows if r["model"] == "affine")
    assert abs(affine["correction_rotation_deg"]) < 0.03
    assert affine["cv_rms_error_px"] > affine["cv_median_error_px"]


@pytest.mark.parametrize("failure", ["few", "parallel", "ambiguous", "collinear", "nonfinite"])
def test_unobservable_geometry_is_unavailable(failure):
    tiles = _tiles()
    if failure == "few":
        tiles = tiles[:7]
    for tile in tiles:
        if failure == "parallel":
            tile["texture_ratio"] = 0.001
        elif failure == "ambiguous":
            tile["peak_ratio"] = 1.01
        elif failure == "collinear":
            tile["y_px"] = 100
        elif failure == "nonfinite":
            tile["residual_dx_px"] = np.nan
    rows, frames, summary = compare_models(tiles, np.zeros((1, 2)), (256, 256), AnalysisConfig())
    assert not rows
    assert frames[0]["selected_model"] is None
    assert summary["unavailable_frames"] == 1
    assert summary["affine_parameter_distributions"]["correction_rotation_deg"] is None


def test_non_affine_distortion_is_not_automatically_recommended():
    tiles = _tiles()
    rng = np.random.default_rng(19)
    for tile in tiles:
        tile["residual_dx_px"], tile["residual_dy_px"] = rng.normal(0, 2, 2)
    _, frames, _ = compare_models(tiles, np.zeros((1, 2)), (256, 256), AnalysisConfig())
    assert frames[0]["selected_model"] is None


@pytest.mark.parametrize("stretch,expected", [(np.eye(2), "rigid"), (np.eye(2) * 1.004, "similarity"),
    (np.array([[1.004, 0.006], [0, 0.996]]), "affine")])
def test_image_motion_measured_from_tiles_and_all_frame_sampling(stretch, expected):
    rng = np.random.default_rng(32)
    reference = ndimage.gaussian_filter(rng.normal(size=(320, 320)), 2) * 250 + 500
    # A native -> reference correction of +0.5 degrees (clockwise in x/y).
    matrix = _rotation(0.5) @ stretch
    swap = np.array([[0, 1], [1, 0]])
    yx = swap @ matrix @ swap
    offset = np.array([159.5, 159.5]) - yx @ np.array([159.5, 159.5])
    moving = ndimage.affine_transform(reference, yx, offset, order=3, mode="reflect")
    stack = np.stack([reference, moving]) + rng.normal(0, 0.5, (2, 320, 320))
    config = AnalysisConfig(local_grid=5, local_frames=0)
    local = local_diagnostics(stack, np.zeros((2, 2)), np.ones(2, dtype=bool), stack.mean(axis=0),
                              (slice(0, 320), slice(0, 320)), config)
    assert {r["frame_position"] for r in local} == {0, 1}
    for row in local:
        row["frame_index"] = row["frame_position"]
    rows, frames, _ = compare_models(local, np.zeros((2, 2)), (320, 320), config)
    row = next(r for r in rows if r["frame_position"] == 1 and r["model"] == "affine")
    assert row["correction_rotation_deg"] == pytest.approx(0.5, abs=0.08)
    assert row["correction_scale_x_percent"] == pytest.approx(100 * (stretch[0, 0] - 1), abs=0.08)
    assert row["correction_scale_y_percent"] == pytest.approx(100 * (stretch[1, 1] - 1), abs=0.08)
    assert frames[1]["selected_model"] == expected


@pytest.mark.parametrize("setting,value", [("local_frames", -1), ("local_frames", True),
    ("affine_diagnostics", "yes"), ("affine_min_improvement_px", 0),
    ("affine_max_cv_error_px", float("nan")), ("affine_min_texture_ratio", 1),
    ("affine_min_relative_improvement", 1)])
def test_configuration_rejects_invalid_diagnostics(setting, value):
    with pytest.raises(ValueError):
        replace(AnalysisConfig(), **{setting: value})
