from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")
from scipy import ndimage

from sem_noise.config import AnalysisConfig
from sem_noise.registration import common_crop, estimate_translation, register_stack, translate


def _specimen(size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(12)
    pattern = ndimage.gaussian_filter(rng.normal(size=(size, size)), 2)
    return (500 + pattern * 150).astype(np.float32)


@pytest.mark.parametrize("displacement", [(2.35, -1.65), (-3.2, 2.7)])
def test_recovers_subpixel_displacement_with_noise(displacement: tuple) -> None:
    rng = np.random.default_rng(8)
    clean = _specimen()
    reference = clean + rng.normal(0, 2, clean.shape)
    moving = ndimage.shift(clean, displacement, order=3, mode="reflect") + rng.normal(0, 2, clean.shape)
    result = estimate_translation(reference, moving, AnalysisConfig())
    assert result["valid"]
    np.testing.assert_allclose(result["shift"], -np.array(displacement), atol=0.2)


def test_rejects_textureless_and_excessive_shift() -> None:
    config = AnalysisConfig(max_shift_px=2)
    assert not estimate_translation(np.ones((64, 64)), np.ones((64, 64)), config)["valid"]
    clean = _specimen()
    assert not estimate_translation(clean, np.roll(clean, 10, axis=0), config)["valid"]


def test_common_crop_excludes_padding_for_both_domains() -> None:
    array = np.ones((40, 50), dtype=np.float32)
    shifts = np.array([[2.8, -4.2], [-3.4, 1.2]])
    crop = common_crop(array.shape, np.concatenate((shifts, np.rint(shifts))))
    for shift in shifts:
        for integer in (True, False):
            np.testing.assert_array_equal(translate(array, shift, integer=integer)[crop], 1)


def test_refined_stack_keeps_anchor_and_exclusions() -> None:
    rng = np.random.default_rng(4)
    clean = _specimen()
    drift = np.column_stack((np.linspace(0, 2, 12), np.linspace(0, -2.5, 12)))
    stack = np.array([ndimage.shift(clean, d, order=3, mode="reflect") + rng.normal(0, 3, clean.shape) for d in drift])
    included = np.ones(12, dtype=bool)
    included[5] = False
    shifts, accepted, rows = register_stack(stack, included, AnalysisConfig(min_frames=4))
    assert not accepted[5]
    assert rows[5]["reason"] == "excluded in manifest"
    np.testing.assert_array_equal(shifts[0], 0)
    np.testing.assert_allclose(shifts[accepted], -drift[accepted], atol=0.25)


def test_repeated_patterns_with_offset_do_not_bias_refined_drift() -> None:
    rng = np.random.default_rng(23)
    yy, xx = np.mgrid[:128, :128]
    specimen = 600 + 220 * (np.sin(xx / 12) > 0) + 150 * (np.cos(yy / 17 + xx / 45) > 0)
    specimen = ndimage.gaussian_filter(specimen.astype(float), 1.5)
    drift = np.column_stack((np.linspace(0, 2.2, 24), np.linspace(0, -3.4, 24)))
    stack = np.array([ndimage.shift(specimen, d, order=3, mode="reflect") + 30 * i / 23 +
                      rng.normal(0, 12, specimen.shape) for i, d in enumerate(drift)])
    shifts, accepted, _ = register_stack(stack, np.ones(24, dtype=bool), AnalysisConfig())
    assert accepted.all()
    np.testing.assert_allclose(shifts, -drift, atol=0.15)


def test_failed_registration_retains_auditable_rows() -> None:
    stack = np.full((8, 32, 32), 500, dtype=np.float32)
    _, accepted, rows = register_stack(stack, np.ones(8, dtype=bool), AnalysisConfig())
    assert accepted.sum() == 1
    assert len(rows) == 8
    assert all("initial registration failed" in r["reason"] for r in rows[1:])
