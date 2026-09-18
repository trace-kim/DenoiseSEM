from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("scipy")

from sem_noise.registration import PARAMETERS
from sem_noise.site_registration import (INVALID_INDEX, PANEL_GAP_PX, PANELS, csv_rows, difference_image, difference_limit,
                                         difference_outputs, fit_rows, register_site, registration_summary)
from test_registration import moving_frame, specimen

ROTATED = {3, 5}


def site_truth(count: int = 6) -> list[dict]:
    rows = []
    for i in range(count):
        rotation = 0.007 if i in ROTATED else 0.0
        rows.append(dict(dy_px=2.0 * i / (count - 1), dx_px=-1.5 * i / (count - 1), a11=0.0, a12=-rotation,
                         a21=rotation, a22=0.0, gain=1 - 0.03 * i, offset_dn=5.0 * i))
    return rows


def site_stack(size: int = 128, noise: float = 4.0) -> tuple[np.ndarray, list[dict]]:
    clean = specimen(size)
    truth = site_truth()
    rng = np.random.default_rng(21)
    frames = [moving_frame(clean, t) + rng.normal(0, noise, clean.shape) for t in truth]
    return np.stack(frames), truth


def test_two_passes_recover_drift_rotation_and_brightness(tmp_path: Path) -> None:
    stack, truth = site_stack()
    included = np.ones(len(stack), dtype=bool)
    messages = []
    fit = register_site(stack, included, (None, None), sigma=1.0, progress=messages.append)
    assert fit["anchor"] == 0
    assert fit["pass1"][0]["iterations"] == 0  # the reference frame is the identity, never fitted
    assert fit["pass1"][0]["dy_px_se"] is None
    assert fit["mean_valid"].sum() > 0.85 * stack[0].size
    assert any("pass 1" in m for m in messages) and any("pass 2" in m for m in messages)
    for i, t in enumerate(truth):
        row = fit["pass2"][i]
        assert row["converged"], i
        assert row["dy_px"] == pytest.approx(t["dy_px"], abs=0.1)
        assert row["dx_px"] == pytest.approx(t["dx_px"], abs=0.1)
        assert row["gain"] == pytest.approx(t["gain"], abs=0.02)
        assert row["offset_dn"] == pytest.approx(t["offset_dn"], abs=2.5)
        expected_corner = 0.007 * np.hypot(63.5, 63.5) if i in ROTATED else 0.0
        assert row["corner_max_px"] == pytest.approx(expected_corner, abs=max(0.1, 4 * row["corner_max_se"]))
    indices = np.arange(len(stack)) * 2
    extras, regions = difference_outputs(stack, fit, (None, None), tmp_path / "differences", indices, lambda m: None)
    assert sorted(extras) == list(range(len(stack)))
    for i in range(len(stack)):
        with Image.open(tmp_path / "differences" / f"frame_{2 * i:04d}.png") as image:
            assert image.mode == "P"
            assert image.size == (3 * 128 + 2 * PANEL_GAP_PX, 128)
        # The reference frame itself is already aligned; interpolation at a
        # sub-0.01 px fitted shift changes its noise by a hair either way.
        assert extras[i]["full_fit_rms_dn"] <= 1.01 * extras[i]["before_rms_dn"]
        if i > 0:
            assert extras[i]["full_fit_rms_dn"] < 0.5 * extras[i]["before_rms_dn"]
    assert len(regions) == len(stack) * len(PANELS) * 16
    for i in ROTATED:
        shift_only = np.mean([r["rms_dn"] for r in regions if r["frame_position"] == i and r["panel"] == "shift_only"])
        full = np.mean([r["rms_dn"] for r in regions if r["frame_position"] == i and r["panel"] == "full_fit"])
        assert full < shift_only
    before = [extras[i]["mean_before_dn"] for i in range(len(stack))]
    after = [extras[i]["mean_after_dn"] for i in range(len(stack))]
    assert np.std(after) < 0.2 * np.std(before)
    rows = fit_rows(fit["pass2"], None, indices, extras)
    assert [r["frame_index"] for r in rows] == list(indices)
    assert all(r["role"] == "fitted" for r in rows)
    assert "covariance" in rows[0] and "covariance" not in csv_rows(rows)[0]
    pass1 = fit_rows(fit["pass1"], fit["anchor"], indices)
    assert pass1[0]["role"].startswith("reference")
    registration, brightness = registration_summary(rows, pass1, 0, 1.0)
    assert registration["max_drift_px"] == pytest.approx(2.5, abs=0.15)
    assert registration["unconverged_frames"] == 0
    assert registration["max_corner_effect_px"] > 0.5
    assert brightness["gain_min"] == pytest.approx(0.85, abs=0.03)
    assert abs(brightness["mean_after_slope_dn_per_frame"]) < abs(brightness["mean_before_slope_dn_per_frame"]) / 5


def test_excluded_frames_are_skipped_and_reference_is_first_included() -> None:
    stack, _ = site_stack(size=96)
    included = np.array([False, True, True, False, True, True])
    fit = register_site(stack, included, (None, None), sigma=1.0, progress=lambda m: None)
    assert fit["anchor"] == 1
    assert sorted(fit["pass1"]) == [1, 2, 4, 5] and sorted(fit["pass2"]) == [1, 2, 4, 5]
    assert fit["pass1"][1]["dy_px_se"] is None
    assert all(fit["pass1"][i]["dy_px_se"] > 0 for i in (2, 4, 5))
    registration, _ = registration_summary([], [], 0, 1.0, registration_enabled=False)
    assert registration == {"enabled": False}


def test_difference_image_indices_and_layout() -> None:
    valid = np.ones((4, 5), dtype=bool)
    valid[0, 0] = False
    panels = [np.zeros((4, 5)), np.full((4, 5), 10.0), np.full((4, 5), -10.0)]
    image = difference_image(panels, valid, 10.0)
    assert image.shape == (4, 3 * 5 + 2 * PANEL_GAP_PX)
    assert image[1, 1] == 127 and image[0, 0] == INVALID_INDEX
    assert image[1, 5 + PANEL_GAP_PX + 1] == 254
    assert image[1, 2 * (5 + PANEL_GAP_PX) + 1] == 0
    assert (image[:, 5:5 + PANEL_GAP_PX] == INVALID_INDEX).all()


def test_difference_scale_covers_corrected_maps_and_ignores_invalid_pixels() -> None:
    valid = np.ones((16, 16), dtype=bool)
    valid[0, 0] = False
    before = np.ones(valid.shape)
    corrected = np.full(valid.shape, -2.0)
    corrected[8, 8] = -30  # one real extreme must not be clipped by a percentile
    corrected[0, 0] = 1e9  # masked values do not set the display range
    failed = np.full(valid.shape, np.nan)
    assert difference_limit([before, corrected, failed], valid) == 30
    assert difference_limit([np.zeros(valid.shape), failed], valid) > 0


def test_parameter_names_are_stable() -> None:
    assert PARAMETERS == ("dy_px", "dx_px", "a11", "a12", "a21", "a22", "gain", "offset_dn")


def test_low_contrast_noise_exposes_gain_attenuation_in_both_passes() -> None:
    # Deliberately preserve the specified estimator's limitation: equal true
    # brightness does not imply gain=1 when the predictor contains noise.
    rng = np.random.default_rng(91)
    y, x = np.indices((128, 128))
    clean = 70 + 2 * np.sin(x / 9) + 2 * np.cos(y / 11)
    stack = np.stack([clean + rng.normal(0, 12, clean.shape) for _ in range(8)])
    fit = register_site(stack, np.ones(8, dtype=bool), (None, None), sigma=1, progress=lambda _: None)
    gain1 = np.median([row["gain"] for i, row in fit["pass1"].items() if i != 0])
    gain2 = np.median([row["gain"] for row in fit["pass2"].values()])
    assert 0.1 < gain1 < 0.5
    assert gain2 < gain1
    assert np.median([row["offset_dn"] for row in fit["pass2"].values()]) > 40


def test_shift_only_panel_does_not_apply_brightness() -> None:
    from sem_noise.site_registration import frame_differences

    reference = specimen(64)
    moving = (reference - 20) / 0.8
    p = np.array([0, 0, 0, 0, 0, 0, 0.8, 20])
    differences, valid, _, _ = frame_differences(moving, np.zeros(reference.shape, dtype=bool),
                                               reference, np.ones(reference.shape, dtype=bool), p)
    np.testing.assert_allclose(differences[0][valid], differences[1][valid], atol=1e-10)
    np.testing.assert_allclose(differences[2][valid], 0, atol=1e-10)


def test_legacy_difference_helper_accepts_uint8_without_wraparound() -> None:
    from sem_noise.site_registration import frame_differences

    reference = np.full((32, 32), 225, dtype=np.uint8)
    moving = np.full((32, 32), 5, dtype=np.uint8)
    parameters = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=float)
    differences, valid, limit, _ = frame_differences(moving, np.zeros(reference.shape, dtype=bool),
                                                   reference, np.ones(reference.shape, dtype=bool), parameters)
    for delta in differences:
        assert delta.dtype.kind == "f"
        np.testing.assert_allclose(delta[valid], -220)
    assert limit == pytest.approx(220)
