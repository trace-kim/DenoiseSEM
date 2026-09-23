from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from sem_noise.pair_matching import (GeometryEstimationError, check_geometry_reference,
                                     estimate_geometry, identity_transform, match_target, measure_quantile_brightness,
                                     measure_brightness, pair_transform, regions_on_input,
                                     select_brightness_regions, warp_target)
from test_registration import TRUTH, moving_frame, specimen


def truth_matrix(shape, truth):
    cy, cx = (np.asarray(shape) - 1) / 2
    return np.array([[1 + truth["a11"], truth["a12"], truth["dx_px"] - truth["a11"] * cx - truth["a12"] * cy],
                     [truth["a21"], 1 + truth["a22"], truth["dy_px"] - truth["a21"] * cx - truth["a22"] * cy]])


def test_affine_includes_translation_and_is_independent_of_brightness() -> None:
    clean = specimen(192)
    rng = np.random.default_rng(47)
    truth = dict(TRUTH, a12=-0.006, a21=0.008, gain=0.75, offset_dn=35)
    fixed = clean + rng.normal(0, 4, clean.shape)
    target = moving_frame(clean, truth) + rng.normal(0, 4, clean.shape)
    originals = fixed.copy(), target.copy()
    fixed.flags.writeable = target.flags.writeable = False
    translation, _ = estimate_geometry(fixed, target, motion="translation")
    affine, score = estimate_geometry(fixed, target, initial=translation)
    direct, _ = estimate_geometry(fixed, target)  # affine directly from identity also supported
    corners = np.array([[20, 20, 1], [170, 20, 1], [20, 170, 1], [170, 170, 1]])
    expected = corners @ truth_matrix(clean.shape, truth).T
    np.testing.assert_allclose(corners @ affine.T, expected, atol=0.12)
    np.testing.assert_allclose(corners @ direct.T, expected, atol=0.12)
    assert score > 0.99
    np.testing.assert_array_equal(fixed, originals[0])
    np.testing.assert_array_equal(target, originals[1])


def test_pair_composition_maps_input_coordinates_to_target_and_round_trips() -> None:
    a = np.array([[1.02, -0.03, 2.0], [0.01, 0.99, -3]])
    b = np.array([[0.98, 0.02, -4.0], [-0.02, 1.01, 1]])
    ab = pair_transform(a, b)
    ba = pair_transform(b, a)
    reference = np.array([20.0, 30.0, 1])
    point_a = np.r_[a @ reference, 1]
    np.testing.assert_allclose(ab @ point_a, b @ reference)
    np.testing.assert_allclose(ba @ np.r_[ab @ point_a, 1], point_a[:2])


def test_two_noisy_frames_match_region_means_without_changing_input() -> None:
    clean = specimen(192)
    rng = np.random.default_rng(41)
    fixed = clean + rng.normal(0, 20, clean.shape)
    target = (clean - 24) / 0.8 + rng.normal(0, 20, clean.shape)
    regions = select_brightness_regions(clean, np.ones(clean.shape, dtype=bool))
    before = fixed.copy(), target.copy()
    fixed.flags.writeable = target.flags.writeable = False
    result = match_target(fixed, target, identity_transform(), regions)
    brightness = result["brightness"]
    assert brightness["gain"] == pytest.approx(0.8, abs=0.004)
    assert brightness["offset_dn"] == pytest.approx(24, abs=4)
    for label in (1, 2):
        mask = (regions == label) & result["valid"]
        assert result["corrected_target"][mask].mean() == pytest.approx(fixed[mask].mean())
    reverse = measure_brightness(target, fixed, regions, result["valid"])
    assert reverse["gain"] * brightness["gain"] == pytest.approx(1)
    assert reverse["gain"] * brightness["offset_dn"] + reverse["offset_dn"] == pytest.approx(0, abs=1e-10)
    np.testing.assert_array_equal(fixed, before[0])
    np.testing.assert_array_equal(target, before[1])


def test_low_contrast_noisy_pair_does_not_collapse_gain() -> None:
    rng = np.random.default_rng(91)
    yy, xx = np.indices((256, 256))
    clean = 70 + 2 * np.sin(xx / 9) + 2 * np.cos(yy / 11)
    stack = np.stack([clean + rng.normal(0, 12, clean.shape) for _ in range(16)])
    regions = select_brightness_regions(stack.mean(axis=0), np.ones(clean.shape, dtype=bool))
    result = match_target(stack[0], stack[8], identity_transform(), regions)
    assert result["brightness"]["gain"] == pytest.approx(1, abs=0.1)
    assert abs(result["brightness"]["offset_dn"]) < 7


def test_region_labels_follow_input_and_target_is_sampled_in_correct_direction() -> None:
    yy, xx = np.indices((64, 64))
    raw = (xx + 10 * yy).astype(np.uint16)
    matrix = identity_transform()
    matrix[:, 2] = (3, -2)
    target, valid = warp_target(raw, matrix)
    np.testing.assert_allclose(target[valid], (xx + 3 + 10 * (yy - 2))[valid])
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[20:30, 20:30] = 2
    mapped = regions_on_input(labels, matrix)
    assert mapped[18:28, 23:33].sum() == 200
    bad = np.zeros(raw.shape, dtype=bool)
    bad[30, 30] = True
    _, valid = warp_target(raw, identity_transform(), invalid=bad)
    assert not valid[28:33, 28:33].any()
    assert not valid[:2].any() and not valid[:, -2:].any()


def test_brightness_uses_shared_unclipped_region_pixels_and_does_not_clip_output() -> None:
    target = specimen(64)
    fixed = 1.5 * target + 30
    regions = select_brightness_regions(target, np.ones(target.shape, dtype=bool))
    bad = np.zeros(target.shape, dtype=bool)
    bad[20:30, 20:30] = True
    target[bad] = 65535
    result = match_target(fixed, target, identity_transform(), regions, target_invalid=bad)
    assert result["brightness"]["gain"] == pytest.approx(1.5)
    assert result["brightness"]["offset_dn"] == pytest.approx(30)
    np.testing.assert_allclose(result["corrected_target"][result["valid"]], fixed[result["valid"]], atol=1e-10)
    assert result["corrected_target"][result["valid"]].max() > 1000


def test_matching_rejects_unmeasurable_inputs_instead_of_inventing_identity() -> None:
    constant = np.ones((32, 32))
    with pytest.raises(ValueError, match="constant image"):
        estimate_geometry(constant, constant)
    with pytest.raises(ValueError, match="distinct low/high"):
        select_brightness_regions(constant, np.ones(constant.shape, dtype=bool))
    regions = np.ones(constant.shape, dtype=np.uint8)
    regions[16:] = 2
    with pytest.raises(ValueError, match="means are equal"):
        measure_brightness(constant, constant, regions, np.ones(constant.shape, dtype=bool))
    with pytest.raises(ValueError, match="singular"):
        pair_transform(identity_transform(), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="same shape"):
        estimate_geometry(constant, np.ones((33, 32)))
    with pytest.raises(ValueError, match="mask must match"):
        warp_target(constant, identity_transform(), invalid=np.zeros((16, 16), dtype=bool))


def test_full_image_quantile_fit_recovers_brightness_despite_pixel_permutation() -> None:
    rng = np.random.default_rng(77)
    target = np.tile(np.arange(256, dtype=np.uint8), (64, 1))
    target[:, :40], target[:, -40:] = 0, 255  # clipping bounds must contribute to full-image quantiles
    fixed = rng.permutation((0.8 * target + 12).ravel()).reshape(target.shape)
    originals = fixed.copy(), target.copy()
    fixed.flags.writeable = target.flags.writeable = False
    result = measure_quantile_brightness(fixed, target)
    assert result["gain"] == pytest.approx(0.8)
    assert result["offset_dn"] == pytest.approx(12)
    assert result["fit_rms_dn"] < 1e-12
    assert result["input_pixels"] == result["target_pixels"] == target.size
    assert result["target_quantiles_dn"][0] == 0 and result["target_quantiles_dn"][-1] == 255
    reverse = measure_quantile_brightness(target, fixed)
    assert reverse["gain"] == pytest.approx(1.25)
    assert reverse["offset_dn"] == pytest.approx(-15)
    np.testing.assert_array_equal(fixed, originals[0])
    np.testing.assert_array_equal(target, originals[1])


def test_quantile_fit_is_not_driven_by_extreme_tail_magnitudes() -> None:
    target = np.linspace(20, 200, 4096).reshape(64, 64)
    fixed = 0.9 * target + 7
    fixed[0], fixed[-1] = -1e6, 1e6  # only existing bottom/top tail pixels change
    result = measure_quantile_brightness(fixed, target)
    assert result["gain"] == pytest.approx(0.9)
    assert result["offset_dn"] == pytest.approx(7)


def test_quantile_fit_handles_two_independent_noisy_acquisitions() -> None:
    rng = np.random.default_rng(73)
    scene = specimen(256)
    fixed = scene + rng.normal(0, 10, scene.shape)
    target = (scene - 17 + rng.normal(0, 10, scene.shape)) / 0.8
    result = measure_quantile_brightness(fixed, target)
    assert result["gain"] == pytest.approx(0.8, abs=0.005)
    assert result["offset_dn"] == pytest.approx(17, abs=3)


def test_quantile_fit_reports_flat_central_distribution_as_unmeasurable() -> None:
    target = np.full((32, 32), 70, dtype=np.uint8)
    target[0, 0] = 250
    with pytest.raises(ValueError, match="percentiles are equal"):
        measure_quantile_brightness(target, target)


def test_unusable_geometry_is_distinct_from_invalid_arguments() -> None:
    constant = np.full((32, 32), 128, dtype=np.uint8)
    with pytest.raises(GeometryEstimationError, match="constant image"):
        estimate_geometry(constant, constant)
    with pytest.raises(GeometryEstimationError, match="too few"):
        check_geometry_reference(constant, invalid=np.ones_like(constant, dtype=bool))
    # A small nominally valid island has no pixels with full blur/gradient support.
    bad = np.ones_like(constant, dtype=bool)
    bad[10:18, 10:18] = False
    with pytest.raises(GeometryEstimationError, match="too few"):
        check_geometry_reference(constant, invalid=bad)
    for kwargs in ({"motion": "invalid"}, {"sigma": 0}, {"initial": np.zeros((2, 3))},
                   {"target_invalid": np.zeros((8, 8))}):
        with pytest.raises(ValueError) as caught:
            estimate_geometry(constant, constant, **kwargs)
        assert not isinstance(caught.value, GeometryEstimationError)


@pytest.mark.parametrize("code,expected", [(-7, GeometryEstimationError), (-5, Exception)])
def test_only_ecc_convergence_errors_are_skippable(monkeypatch, code, expected) -> None:
    import cv2

    error = cv2.error("synthetic ECC error")
    error.code, error.err = code, "synthetic ECC error"
    def fail(*args):
        raise error
    monkeypatch.setattr(cv2, "findTransformECCWithMask", fail)
    with pytest.raises(expected) as caught:
        estimate_geometry(specimen(64), specimen(64))
    if code != -7:
        assert caught.value is error


@pytest.mark.parametrize("score,matrix", [
    (float("nan"), np.eye(2, 3)), (1.0, np.zeros((2, 3))),
    (1.0, np.full((2, 3), np.nan)),
])
def test_unusable_ecc_results_never_become_measured_transforms(monkeypatch, score, matrix) -> None:
    import cv2

    monkeypatch.setattr(cv2, "findTransformECCWithMask", lambda *a: (score, matrix))
    with pytest.raises(GeometryEstimationError):
        estimate_geometry(specimen(64), specimen(64))
