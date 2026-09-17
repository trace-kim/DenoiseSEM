from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import ndimage

from sem_noise.registration import (PARAMETERS, _correlation_area, common_crop, fit_frame, identity_result,
                                    shift_only, translate, warp)

TRUTH = dict(dy_px=2.3, dx_px=-1.6, a11=0.0008, a12=-0.0012, a21=0.0015, a22=-0.0005, gain=0.9, offset_dn=20.0)
TOLERANCE = dict(dy_px=0.02, dx_px=0.02, a11=1e-4, a12=1e-4, a21=1e-4, a22=1e-4, gain=1e-3, offset_dn=0.5)


def specimen(size: int = 192, seed: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = 600 + 220 * (np.sin(xx / 12) > 0) + 150 * (np.cos(yy / 17 + xx / 45) > 0)
    image = ndimage.gaussian_filter(image.astype(float), 1.5)
    return image + 60 * ndimage.gaussian_filter(rng.normal(size=(size, size)), 3)


def moving_frame(reference: np.ndarray, truth: dict) -> np.ndarray:
    """Build M so that gain * M(W(x)) + offset == R(x) with W(x) = x + d + A (x - c)."""
    cy, cx = (reference.shape[0] - 1) / 2, (reference.shape[1] - 1) / 2
    linear_yx = np.array([[truth["a22"], truth["a21"]], [truth["a12"], truth["a11"]]])
    matrix = np.linalg.inv(np.eye(2) + linear_yx)
    centre, shift = np.array([cy, cx]), np.array([truth["dy_px"], truth["dx_px"]])
    offset = centre - matrix @ (shift + centre)
    return ndimage.affine_transform((reference - truth["offset_dn"]) / truth["gain"], matrix, offset,
                                    order=3, mode="reflect")


def noisy_pair(size: int, sigma: float, seed: int, truth: dict = TRUTH) -> tuple[np.ndarray, np.ndarray]:
    clean = specimen(size)
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0, sigma, clean.shape), moving_frame(clean, truth) + rng.normal(0, sigma, clean.shape)


def test_recovers_translation_affine_gain_offset_with_error_bars() -> None:
    reference, moving = noisy_pair(256, 3.0, 7)
    fit = fit_frame(reference, moving)
    assert fit["converged"]
    for key in PARAMETERS:
        assert fit[f"{key}_se"] > 0
        assert abs(fit[key] - TRUTH[key]) <= max(4 * fit[f"{key}_se"], TOLERANCE[key]), key
    assert fit["residual_rms_dn"] < 0.1 * fit["initial_rms_dn"]
    assert fit["valid_pixels"] > 0.9 * reference.size
    half = 127.5
    expected_corner = max(np.hypot(TRUTH["a11"] * u + TRUTH["a12"] * v, TRUTH["a21"] * u + TRUTH["a22"] * v)
                          for u in (-half, half) for v in (-half, half))
    assert fit["corner_max_px"] == pytest.approx(expected_corner, abs=0.02)
    assert fit["rotation_deg"] == pytest.approx(np.degrees((TRUTH["a21"] - TRUTH["a12"]) / 2), abs=0.005)
    assert fit["shear"] == pytest.approx((TRUTH["a12"] + TRUTH["a21"]) / 2, abs=1e-4)
    assert np.asarray(fit["covariance"]).shape == (8, 8)
    assert 0.1 < fit["downweighted_fraction"] < 0.3  # Huber at 1.345 sigma leaves ~18 % of Gaussian residuals


def test_error_bars_are_calibrated_over_noise_realizations() -> None:
    clean = specimen(160)
    moved = moving_frame(clean, TRUTH)
    z = {key: [] for key in ("dy_px", "dx_px", "gain", "offset_dn", "a11")}
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        fit = fit_frame(clean + rng.normal(0, 8, clean.shape), moved + rng.normal(0, 8, clean.shape))
        for key in z:
            z[key].append((fit[key] - TRUTH[key]) / fit[f"{key}_se"])
    for key, values in z.items():
        assert 0.4 < np.sqrt(np.mean(np.square(values))) < 2.5, key


def test_correlation_area_matches_blur_theory() -> None:
    rng = np.random.default_rng(1)
    white = rng.normal(size=(256, 256))
    everywhere = np.ones(white.shape, dtype=bool)
    assert _correlation_area(white, everywhere) < 1.3
    assert _correlation_area(ndimage.gaussian_filter(white, 1), everywhere) == pytest.approx(4 * np.pi, rel=0.15)


def test_clipped_pixels_and_their_blurred_neighbourhood_are_masked() -> None:
    truth = dict(TRUTH, a11=0, a12=0, a21=0, a22=0)
    reference, moving = noisy_pair(256, 3.0, 5, truth)
    moving[40:100, 60:140] = 4095
    fit = fit_frame(reference, moving, moving_invalid=moving >= 4095)
    assert abs(fit["dy_px"] - truth["dy_px"]) < 0.05 and abs(fit["dx_px"] - truth["dx_px"]) < 0.05
    assert fit["gain"] == pytest.approx(truth["gain"], abs=0.01)
    assert fit["offset_dn"] == pytest.approx(truth["offset_dn"], abs=1.0)
    assert fit["valid_pixels"] <= moving.size - 60 * 80
    unmasked = fit_frame(reference, moving)
    assert abs(unmasked["gain"] - truth["gain"]) > 0.1  # the block is not something the loss can absorb


def test_converges_from_zero_over_several_pixels() -> None:
    truth = dict(dy_px=7.0, dx_px=5.0, a11=0, a12=0, a21=0, a22=0, gain=1.0, offset_dn=0.0)
    reference, moving = noisy_pair(192, 8.0, 3, truth)
    fit = fit_frame(reference, moving)
    assert fit["converged"]
    assert fit["dy_px"] == pytest.approx(7.0, abs=0.05)
    assert fit["dx_px"] == pytest.approx(5.0, abs=0.05)


def test_identity_result_and_shift_only() -> None:
    result = identity_result((64, 80))
    assert result["gain"] == 1 and result["offset_dn"] == 0 and result["corner_max_px"] == 0
    assert all(result[f"{key}_se"] is None for key in PARAMETERS)
    assert result["corner_max_se"] is None and result["rotation_deg_se"] is None
    parameters = np.array([1.0, -2.0, 0.01, 0.02, 0.03, 0.04, 0.9, 5.0])
    reduced = shift_only(parameters)
    np.testing.assert_array_equal(reduced, [1.0, -2.0, 0, 0, 0, 0, 1.0, 0.0])
    np.testing.assert_array_equal(parameters[2:6], [0.01, 0.02, 0.03, 0.04])


def test_warp_applies_gain_offset_and_masks_footprint() -> None:
    yy, xx = np.indices((40, 50), dtype=float)
    ramp = 3.0 * xx + 2.0 * yy
    corrected, valid = warp(ramp, np.array([0.5, 1.25, 0, 0, 0, 0, 2.0, -1.0]))
    assert not valid[:, -4:].any() and not valid[-3:].any() and valid[10:-10, 10:-10].all()
    expected = 2.0 * (3.0 * (xx + 1.25) + 2.0 * (yy + 0.5)) - 1.0
    # The spline prefilter's constant boundary extension rings within a few
    # pixels of the edge; the interior reproduces the ramp exactly.
    np.testing.assert_allclose(corrected[valid], expected[valid], atol=0.05)
    interior = valid.copy()
    interior[:10] = interior[-10:] = interior[:, :10] = interior[:, -10:] = False
    assert interior.sum() > 500
    np.testing.assert_allclose(corrected[interior], expected[interior], atol=1e-5)
    invalid = np.zeros(ramp.shape, dtype=bool)
    invalid[10, 10] = True
    _, masked = warp(ramp, np.zeros(8) + [0, 0, 0, 0, 0, 0, 1, 0], invalid)
    assert not masked[10, 10] and masked[20, 20]


def test_common_crop_excludes_padding_for_both_domains() -> None:
    array = np.ones((40, 50), dtype=np.float32)
    shifts = np.array([[2.8, -4.2], [-3.4, 1.2]])
    crop = common_crop(array.shape, np.concatenate((shifts, np.rint(shifts))))
    for shift in shifts:
        for integer in (True, False):
            np.testing.assert_array_equal(translate(array, shift, integer=integer)[crop], 1)


def test_fit_frame_rejects_unusable_inputs() -> None:
    with pytest.raises(ValueError, match="equally shaped"):
        fit_frame(np.ones((32, 32)), np.ones((32, 31)))
    with pytest.raises(ValueError, match="finite"):
        fit_frame(np.full((32, 32), np.nan), np.ones((32, 32)))
    with pytest.raises(ValueError, match="16x16"):
        fit_frame(np.ones((8, 8)), np.ones((8, 8)))
    for sigma in (0, -1, np.nan):
        with pytest.raises(ValueError, match="sigma"):
            fit_frame(np.ones((32, 32)), np.ones((32, 32)), sigma=sigma)
    with pytest.raises(ValueError, match="masks must match"):
        fit_frame(np.ones((32, 32)), np.ones((32, 32)), moving_invalid=np.zeros((31, 32), dtype=bool))
    with pytest.raises(ValueError, match="fewer than 64"):
        fit_frame(np.ones((32, 32)), np.ones((32, 32)), moving_invalid=np.ones((32, 32), dtype=bool))


def test_constant_images_report_undetermined_error_bars_without_failing() -> None:
    fit = fit_frame(np.full((32, 32), 500.0), np.full((32, 32), 503.0))
    assert not fit["converged"]
    assert all(np.isnan(fit[f"{key}_se"]) for key in PARAMETERS)
    assert fit["dy_px"] == 0 and fit["gain"] == 1


def test_losing_pixels_cannot_by_itself_improve_the_objective() -> None:
    from sem_noise.registration import _improves_on_common_pixels

    residual = np.r_[np.ones(100), np.full(100, 20.0)]
    valid = np.ones(200, dtype=bool)
    cropped = np.arange(200) < 100
    # A trial with unchanged/worse residuals must not benefit from cropping
    # away the expensive half of the old objective.
    assert not _improves_on_common_pixels(residual, valid, residual, cropped, 2.0)
    assert not _improves_on_common_pixels(residual, valid, residual + 0.1, cropped, 2.0)
    assert _improves_on_common_pixels(residual, valid, residual * 0.5, cropped, 2.0)


def test_rejected_short_steps_are_not_reported_as_converged(monkeypatch) -> None:
    from sem_noise import registration

    monkeypatch.setattr(registration, "_improves_on_common_pixels", lambda *args: False)
    ref, mov = noisy_pair(96, 3, 72)
    result = fit_frame(ref, mov)
    assert not result["converged"]
    assert result["termination_reason"] == "line_search_stalled"
    assert result["dy_px"] == 0


def test_mask_covers_actual_gaussian_support_and_ignores_sentinel_values() -> None:
    from sem_noise.registration import prepare_fit_images

    ref, mov = noisy_pair(96, 3, 17)
    bad = np.zeros(ref.shape, dtype=bool)
    bad[40:46, 40:46] = True
    _, _, ref_bad, _ = prepare_fit_images(ref, mov, 2, bad, bad)
    assert ref_bad[32, 32] and not ref_bad[31, 32]
    a, b = mov.copy(), mov.copy()
    a[bad], b[bad] = 4095, 1e12
    first = fit_frame(ref, a, moving_invalid=bad)
    second = fit_frame(ref, b, moving_invalid=bad)
    for key in PARAMETERS:
        assert first[key] == second[key]


def test_cubic_mask_excludes_diagonal_support() -> None:
    image = specimen(64)
    bad = np.zeros(image.shape, dtype=bool)
    bad[30, 30] = True
    _, valid = warp(image, np.array([0.2, 0.2, 0, 0, 0, 0, 1, 0]), bad)
    assert not valid[28, 28]
    assert valid[24, 24]


def test_corner_magnitude_error_includes_cross_covariance() -> None:
    from sem_noise.registration import corner_effects

    p = np.array([0, 0, 0.02, 0, 0.03, 0, 1, 0], dtype=float)
    covariance = np.zeros((8, 8))
    covariance[2, 2] = covariance[4, 4] = 1e-4
    covariance[2, 4] = covariance[4, 2] = 0.8e-4
    result = corner_effects(p, covariance, (64, 64))
    gradient = np.zeros(8)
    magnitude = np.hypot(p[2], p[4])
    gradient[2], gradient[4] = 31.5 * p[2] / magnitude, 31.5 * p[4] / magnitude
    assert result["corner_bottom_right_se"] == pytest.approx(np.sqrt(gradient @ covariance @ gradient))


def test_report_pixels_reproduce_final_fit_residual_and_mask() -> None:
    from sem_noise.registration import fit_pixel_diagnostics, parameter_vector

    ref, mov = noisy_pair(96, 3, 9)
    bad = np.zeros(ref.shape, dtype=bool)
    bad[20:25, 40:45] = True
    mov[bad] = 4095
    row = fit_frame(ref, mov, moving_invalid=bad)
    pixels = fit_pixel_diagnostics(ref, mov, parameter_vector(row), moving_invalid=bad)
    valid = pixels["fit_valid"]
    assert valid.sum() == row["valid_pixels"]
    assert np.sqrt(np.mean(pixels["fit_residual"][valid] ** 2)) == pytest.approx(row["residual_rms_dn"], abs=1e-10)
    assert np.mean(pixels["huber_weights"][valid] < 1) == pytest.approx(row["downweighted_fraction"])


def test_correlation_area_matches_nonperiodic_direct_calculation() -> None:
    rng = np.random.default_rng(99)
    residual = ndimage.gaussian_filter(rng.normal(size=(16, 19)), 1)
    valid = np.ones(residual.shape, dtype=bool)
    valid[:3, :4] = False
    z = residual - residual[valid].mean()
    variance = np.mean(z[valid] ** 2)
    area = 0
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            a = (slice(max(0, dy), min(16, 16 + dy)), slice(max(0, dx), min(19, 19 + dx)))
            b = (slice(max(0, -dy), min(16, 16 - dy)), slice(max(0, -dx), min(19, 19 - dx)))
            both = valid[a] & valid[b]
            area += np.mean((z[a] * z[b])[both]) / variance
    assert _correlation_area(residual, valid) == pytest.approx(max(1, area))
