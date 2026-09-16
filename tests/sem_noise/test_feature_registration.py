from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("skimage")
from scipy import ndimage

from sem_noise.config import AnalysisConfig
from sem_noise.feature_registration import (
    analyze_features, difference_examples, extract_features, fit_correspondences,
    match_features, warp_to_reference,
)


def _matrix(shape=(256, 256), angle=3.0):
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    matrix = np.eye(3)
    matrix[:2, :2] = rotation @ np.array([[1.015, 0.012], [0, 0.99]])
    center = (np.array(shape[::-1]) - 1) / 2
    matrix[:2, 2] = center - matrix[:2, :2] @ center + [3.4, -2.2]
    return matrix


def _images(size=256):
    rng = np.random.default_rng(6)
    reference = ndimage.gaussian_filter(rng.normal(size=(size, size)), 1.7) * 200 + 1000
    matrix = _matrix(reference.shape)
    moving = ndimage.affine_transform(reference, matrix[:2, :2][::-1, ::-1],
                                      matrix[:2, 2][::-1], order=3, mode="reflect")
    return reference + rng.normal(0, 1, reference.shape), moving + rng.normal(0, 1, reference.shape), matrix


@pytest.mark.parametrize("size,max_side", [(256, 768), (512, 256)])
def test_native_feature_matches_recover_affine_without_translation(size, max_side):
    reference, moving, truth = _images(size)
    config = AnalysisConfig(registration_max_side=max_side)
    fit, matches = match_features(extract_features(reference, config), extract_features(moving, config),
                                  reference.shape, config, seed=3)
    assert fit["available"], fit
    assert fit["correction_rotation_deg"] == pytest.approx(3, abs=0.12)
    assert fit["correction_scale_x_percent"] == pytest.approx(1.5, abs=0.15)
    assert fit["correction_scale_y_percent"] == pytest.approx(-1, abs=0.15)
    assert fit["correction_center_dx_px"] == pytest.approx(3.4, abs=0.2)
    assert fit["correction_center_dy_px"] == pytest.approx(-2.2, abs=0.2)
    assert {m["partition"] for m in matches} == {"training", "validation"}
    fixed, valid = warp_to_reference(moving, np.asarray(fit["matrix"]))
    assert np.mean((fixed[valid] - reference[valid])**2) < np.mean((moving[valid] - reference[valid])**2) / 5


def test_ransac_rejects_false_matches_and_validates_separate_regions():
    rng = np.random.default_rng(7)
    source = rng.uniform(10, 245, (160, 2))
    matrix = _matrix()
    target = source @ matrix[:2, :2].T + matrix[:2, 2] + rng.normal(0, 0.05, source.shape)
    target[:35] = rng.uniform(10, 245, (35, 2))
    fit, matches = fit_correspondences(source, target, (256, 256), AnalysisConfig(), seed=17)
    assert fit["available"], fit
    assert fit["training_inliers"] < fit["training_matches"]
    assert sum(not r["within_threshold"] for r in matches) >= 30
    np.testing.assert_allclose(np.array(fit["matrix"])[:2, :2], matrix[:2, :2], atol=0.001)
    repeated, _ = fit_correspondences(source, target, (256, 256), AnalysisConfig(), seed=17)
    assert fit == repeated


def test_held_out_disagreement_rejects_training_consensus():
    # Spatially separate regions have incompatible motion. Training RANSAC
    # alone could look excellent; validation must reject it.
    target = np.array([(x, y) for y in np.linspace(10, 245, 12) for x in np.linspace(10, 245, 12)])
    cells = (target / 256 * 4).astype(int)
    held = (cells[:, 0] + cells[:, 1]) % 3 == 0
    source = target.copy()
    source[held] += [8, -5]
    fit, _ = fit_correspondences(source, target, (256, 256), AnalysisConfig(), seed=5)
    assert not fit["available"]
    assert "held-out" in fit["reason"]
    assert "matrix" not in fit


@pytest.mark.parametrize("kind", ["few", "collinear", "localized"])
def test_insufficient_correspondences_never_return_identity(kind):
    rng = np.random.default_rng(10)
    source = rng.uniform(0, 255, (40, 2))
    if kind == "few":
        source = source[:3]
    elif kind == "collinear":
        source[:, 1] = source[:, 0]
    else:
        source /= 10
    fit, _ = fit_correspondences(source, source + 1, (256, 256), AnalysisConfig(), seed=4)
    assert not fit["available"]
    assert "matrix" not in fit


def test_flat_images_unavailable_and_frame_sampling_keeps_indices():
    stack = np.stack([np.full((64, 64), 50.0 + i) for i in range(6)])
    included = np.array([False, True, True, True, False, True])
    summary, rows, matches, examples = analyze_features(stack, included, np.array([0, 2, 4, 6, 8, 10]),
        AnalysisConfig(local_frames=2, diff_examples=1), lambda message: None)
    assert summary["reference_frame_index"] == 2
    assert examples == [2]
    assert rows[1]["available"]  # Explicit reference identity only.
    assert rows[2]["reason"] == "insufficient detected features"
    assert rows[3]["reason"] == "not sampled"
    assert rows[4]["reason"] == "excluded from analysis"
    assert not matches


def test_diff_sign_masks_failure_and_single_resampling():
    yy, xx = np.indices((40, 50))
    reference = xx * 2.0 + yy
    moving = reference + 7
    stack = np.stack([reference, moving])
    matrix = np.eye(3)
    matrix[:2, 2] = [1.25, -0.5]
    rows = [dict(reference_position=0), dict(reference_position=0, reference_frame_index=0,
            frame_index=9, available=True, reason="test transform", matrix=matrix.tolist())]
    shifts = np.array([[0, 0], [-0.5, 1.25]])
    pairs, maps = difference_examples(stack, rows, [1], shifts, np.ones(2, dtype=bool), True)
    pair = pairs[0]
    mask = maps["pair_00_valid"]
    assert not mask.all()
    np.testing.assert_array_equal(maps["pair_00_raw_diff"][mask], 7)
    np.testing.assert_allclose(maps["pair_00_translation_diff"], maps["pair_00_affine_diff"], equal_nan=True)
    assert np.isnan(maps["pair_00_affine_diff"][~mask]).all()
    assert pair["translation_rms_dn"] == pytest.approx(5)
    rows[1].update(available=False, reason="failed")
    pairs, maps = difference_examples(stack, rows, [1], shifts, np.array([True, False]), True)
    assert pairs[0]["available_modes"] == ["raw"]
    assert "pair_00_affine_diff" not in maps


@pytest.mark.parametrize("name,value", [("feature_match_ratio", 1), ("feature_residual_px", 0),
    ("feature_min_inliers", 3), ("feature_max_keypoints", 8), ("feature_min_overlap", float("nan")),
    ("feature_min_inlier_fraction", True), ("diff_examples", 0)])
def test_feature_settings_validated(name, value):
    with pytest.raises(ValueError):
        replace(AnalysisConfig(), **{name: value})
