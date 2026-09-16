from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("skimage")
from scipy import ndimage

from sem_noise.config import AnalysisConfig
from sem_noise.feature_registration import difference_examples, warp_to_reference
from sem_noise.intensity_registration import analyze_intensity, refine_affine
from sem_noise.registration import estimate_translation


def _pattern(size: int = 256) -> np.ndarray:
    y, x = np.indices((size, size))
    image = np.full((size, size), 1000.)
    for cy in range(24, size, 40):
        for cx in range(24, size, 40):
            image += 100 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 6**2))
    return image


def _pair(size: int = 256, angle: float = 0.35, noise: float = 5., *, affine: bool = False):
    rng = np.random.default_rng(26)
    clean = _pattern(size)
    theta = np.radians(angle)
    matrix = np.eye(3)
    matrix[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    if affine:
        matrix[:2, :2] = matrix[:2, :2] @ np.array([[1.003, 0.002], [0., 0.998]])
    center = (np.array(clean.shape[::-1]) - 1) / 2
    matrix[:2, 2] = center - matrix[:2, :2] @ center + [2.2, -1.4]
    moving = ndimage.affine_transform(clean, matrix[:2, :2][::-1, ::-1],
                                      matrix[:2, 2][::-1], order=3, mode="reflect")
    return clean + rng.normal(0, noise, clean.shape), moving + rng.normal(0, noise, clean.shape), matrix


@pytest.mark.parametrize("size,max_side,affine", [(256, 768, False), (256, 768, True), (512, 256, True)])
def test_recovers_small_affine_on_noisy_repeated_patterns(size, max_side, affine):
    reference, moving, truth = _pair(size, affine=affine)
    config = AnalysisConfig(registration_max_side=max_side, pixel_size_nm=2.)
    translation = estimate_translation(reference, moving, config)
    assert translation["valid"]
    fit = refine_affine(reference, moving, translation["shift"], config)
    assert fit["available"], fit
    assert fit["selected_model"] == "affine"
    assert fit["correction_rotation_deg"] == pytest.approx(0.35, abs=0.04)
    np.testing.assert_allclose(np.array(fit["matrix"])[:2, :2], truth[:2, :2], atol=0.0008)
    assert fit["correction_center_dx_px"] == pytest.approx(2.2, abs=0.1)
    assert fit["correction_center_dy_px"] == pytest.approx(-1.4, abs=0.1)
    assert fit["correction_center_dx_nm"] == 2 * fit["correction_center_dx_px"]
    assert fit["validation_relative_improvement"] > 0.02
    warped, valid = warp_to_reference(moving, np.asarray(fit["matrix"]))
    assert np.mean((warped[valid] - reference[valid])**2) < np.mean((moving[valid] - reference[valid])**2)


def test_brightness_changes_do_not_become_geometric_motion():
    reference, moving, _ = _pair(angle=0., noise=4.)
    moving = moving * 1.15 + 70
    config = AnalysisConfig()
    shift = estimate_translation(reference, moving, config)["shift"]
    fit = refine_affine(reference, moving, shift, config)
    assert not fit["available"], fit
    assert fit["selected_model"] == "translation"
    assert "does not improve" in fit["reason"]
    assert "matrix" not in fit


def test_small_rotation_with_brightness_changes():
    reference, moving, _ = _pair()
    moving = moving * 1.15 + 70
    config = AnalysisConfig()
    fit = refine_affine(reference, moving, estimate_translation(reference, moving, config)["shift"], config)
    assert fit["available"], fit
    assert fit["correction_rotation_deg"] == pytest.approx(0.35, abs=0.04)


@pytest.mark.parametrize("kind", ["flat", "parallel", "nonfinite", "small"])
def test_uninformative_images_never_report_affine(kind):
    image = _pattern()
    if kind == "flat":
        image[:] = 1000
    elif kind == "parallel":
        image = np.tile(1000 + 100 * np.sin(np.arange(256) / 10), (256, 1))
    elif kind == "nonfinite":
        image[0, 0] = np.nan
    else:
        image = image[:32, :32]
    fit = refine_affine(image, image.copy(), np.zeros(2), AnalysisConfig())
    assert not fit["available"]
    assert "matrix" not in fit


def test_motion_outside_bounds_is_rejected():
    reference, moving, _ = _pair(angle=2., noise=1.)
    config = AnalysisConfig(affine_refine_max_linear_change=0.01)
    shift = estimate_translation(reference, moving, config)["shift"]
    fit = refine_affine(reference, moving, shift, config)
    assert not fit["available"], fit
    assert "bounds" in fit["reason"]
    assert "matrix" not in fit


def test_spatially_inconsistent_motion_fails_validation():
    reference = _pattern()
    moving = ndimage.rotate(reference, 0.6, reshape=False, mode="reflect")
    y, x = np.indices(reference.shape)
    held = ((x // 64) + (y // 64)) % 3 == 0
    moving[held] = reference[held]
    fit = refine_affine(reference, moving, np.zeros(2), AnalysisConfig())
    assert not fit["available"], fit
    assert "matrix" not in fit


def test_spatial_validation_pixels_never_enter_optimization(monkeypatch):
    from sem_noise import intensity_registration

    optimize = intensity_registration._optimize
    calls = []

    def checked(initial, level, mask, bounds, *, affine):
        assert not np.any(mask & level["held"])
        calls.append(affine)
        return optimize(initial, level, mask, bounds, affine=affine)

    monkeypatch.setattr(intensity_registration, "_optimize", checked)
    reference, moving, _ = _pair()
    fit = refine_affine(reference, moving, np.array([-1.4, 2.2]), AnalysisConfig())
    assert fit["available"], fit
    assert True in calls and False in calls


@pytest.mark.parametrize("config,reason", [
    (AnalysisConfig(affine_max_stability_px=1e-5), "unstable across training regions"),
    (AnalysisConfig(affine_refine_min_overlap=0.999), "insufficient estimated image overlap"),
    (AnalysisConfig(registration_max_side=1), "resolution is too small"),
])
def test_refinement_quality_gates(config, reason):
    reference, moving, _ = _pair()
    fit = refine_affine(reference, moving, np.array([-1.4, 2.2]), config)
    assert not fit["available"], fit
    assert reason in fit["reason"]
    assert "matrix" not in fit


def test_sampling_rejected_initialization_and_translation_fallback():
    reference, moving, _ = _pair(angle=0.)
    stack = np.stack([reference, reference, moving, moving, moving])
    included = np.array([False, True, True, True, True])
    accepted = np.array([False, True, True, False, True])
    shift = estimate_translation(reference, moving, AnalysisConfig())["shift"]
    shifts = np.array([[0, 0], [0, 0], shift, shift, shift])
    config = AnalysisConfig(local_frames=2, diff_examples=1)
    summary, rows, matches, examples = analyze_intensity(stack, included, np.arange(5) * 2,
                                                        shifts, accepted, config, lambda _: None)
    assert summary["reference_frame_index"] == 2
    assert examples == [2]
    assert rows[0]["reason"] == "excluded from analysis"
    assert rows[3]["reason"] == "not sampled"
    assert rows[2]["selected_model"] == "translation"
    assert summary["retained_translation_frames"] == 2
    assert not matches
    pairs, maps = difference_examples(stack, rows, examples, shifts, accepted, True)
    assert pairs[0]["available_modes"] == ["raw", "translation"]
    assert "pair_00_affine_diff" not in maps
    for cfg, acceptance, reason in [
        (replace(config, local_frames=0), accepted, "translation initialization rejected"),
        (replace(config, local_frames=0, registration="none"), accepted, "translation initialization disabled"),
        (replace(config, local_frames=0), np.zeros(5, dtype=bool), "translation initialization rejected"),
    ]:
        _, rows, _, _ = analyze_intensity(stack, included, np.arange(5), shifts, acceptance, cfg, lambda _: None)
        assert reason in rows[3]["reason"]
        assert "matrix" not in rows[3]


@pytest.mark.parametrize("name,value", [("affine_method", "unknown"),
    ("affine_refine_max_linear_change", 0), ("affine_refine_max_linear_change", 1),
    ("affine_refine_max_translation_px", float("nan")), ("affine_refine_min_relative_improvement", True),
    ("affine_refine_min_overlap", 1)])
def test_intensity_config_validation(name, value):
    with pytest.raises(ValueError):
        replace(AnalysisConfig(), **{name: value})
