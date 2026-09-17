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
    np.testing.assert_array_equal(reduced, [1.0, -2.0, 0, 0, 0, 0, 0.9, 5.0])
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


def test_constant_images_report_undetermined_error_bars_without_failing() -> None:
    fit = fit_frame(np.full((32, 32), 500.0), np.full((32, 32), 503.0))
    assert not fit["converged"]
    assert all(np.isnan(fit[f"{key}_se"]) for key in PARAMETERS)
    assert fit["dy_px"] == 0 and fit["gain"] == 1
