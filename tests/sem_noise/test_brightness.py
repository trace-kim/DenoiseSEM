from __future__ import annotations

import json

import numpy as np
import pytest

from sem_noise.brightness import analyze_brightness, apply_brightness, fit_brightness, match_target


def _scene() -> np.ndarray:
    y, x = np.mgrid[:128, :144]
    return 40 + 18 * np.sin(x / 15) + 12 * np.cos(y / 12)


def test_gain_offset_recovery_on_independently_noisy_low_dn_images() -> None:
    rng = np.random.default_rng(51)
    scene = _scene()
    reference = scene + rng.normal(0, 1, scene.shape)
    moving = 0.72 * scene + 3 + rng.normal(0, 1, scene.shape)
    fit = fit_brightness(reference, moving)
    assert fit["available"] and fit["model"] == "affine"
    assert fit["gain_to_reference"] == pytest.approx(1 / 0.72, abs=0.01)
    assert fit["offset_to_reference_dn"] == pytest.approx(-3 / 0.72, abs=0.3)
    assert fit["validation_after_rmse_dn"] < 0.05 * fit["validation_before_rmse_dn"]


def test_offset_only_and_flat_scene_do_not_invent_gain() -> None:
    for scene in (_scene(), np.full((64, 64), 40.0)):
        fit = fit_brightness(scene, scene - 8)
        assert fit["available"]
        assert fit["gain_to_reference"] == pytest.approx(1)
        assert fit["offset_to_reference_dn"] == pytest.approx(8)


def test_validation_rejects_spatially_inconsistent_correction() -> None:
    scene = _scene()
    yy, xx = np.indices(scene.shape)
    held = ((yy // 16 + xx // 16) % 3 == 0)
    moving = scene + np.where(held, -10, 10)
    fit = fit_brightness(scene, moving)
    assert not fit["available"]
    assert "worsens" in fit["reason"]


def test_target_mapping_preserves_input_scale_without_clipping_or_mutation() -> None:
    scene = _scene()
    question = 0.8 * scene + 2
    target = 0.6 * scene - 1
    before = target.copy()
    corrected = match_target(target, input_gain=1 / 0.8, input_offset_dn=-2 / 0.8,
                             target_gain=1 / 0.6, target_offset_dn=1 / 0.6)
    np.testing.assert_allclose(corrected, question)
    np.testing.assert_array_equal(target, before)
    np.testing.assert_array_equal(apply_brightness(np.array([-3, 260]), 2, -1), [-7, 519])
    with pytest.raises(ValueError):
        apply_brightness(scene, 0, 0)
    with pytest.raises(ValueError):
        match_target(target, input_gain=np.nan, input_offset_dn=0, target_gain=1, target_offset_dn=0)


def test_invalid_and_small_images_are_explicitly_unavailable() -> None:
    assert not fit_brightness(np.ones((8, 8)), np.ones((8, 8)))["available"]
    assert not fit_brightness(np.full((64, 64), np.nan), np.ones((64, 64)))["available"]
    with pytest.raises(ValueError):
        fit_brightness(np.ones((32, 32)), np.ones((32, 31)))


def test_analysis_common_support_gaps_and_signed_differences() -> None:
    scene = _scene()
    stack = np.stack([scene, np.roll(0.8 * scene + 1, 2, axis=1), scene, 0.6 * scene + 2])
    saved = stack.copy()
    shifts = np.array([[0, 0], [0, -2], [0, 0], [0, 0]])
    summary, rows, maps = analyze_brightness(stack, shifts, np.array([True, True, False, True]),
                                            np.array([2, 4, 7, 9]), (slice(4, -4), slice(4, -4)),
                                            registration_enabled=True)
    assert summary["reference_frame_index"] == 2
    assert summary["fitted_frames"] == 2
    assert not rows[2]["available"]
    assert abs(summary["after_slope_dn_per_frame"]) < 1e-8
    assert summary["before_slope_dn_per_frame"] < -1
    np.testing.assert_allclose(maps["frame_1_after"], maps["reference"], atol=1e-5)
    np.testing.assert_allclose(maps["frame_1_correction"], maps["frame_1_after"] - maps["frame_1_before"], atol=1e-5)
    np.testing.assert_array_equal(stack, saved)


def test_disabled_registration_never_claims_brightness_fit() -> None:
    stack = np.stack([_scene(), _scene() * 0.8])
    summary, rows, _ = analyze_brightness(stack, np.zeros((2, 2)), np.ones(2, dtype=bool),
                                          np.arange(2), (slice(None), slice(None)), registration_enabled=False)
    assert summary["fitted_frames"] == 0
    assert rows[1]["after_mean_dn"] is None
    assert "disabled" in rows[1]["reason"]


def test_pipeline_writes_brightness_comparison(tmp_path, monkeypatch) -> None:
    from sem_noise.config import AnalysisConfig
    from sem_noise.pipeline import analyze_dataset
    from sem_noise import pipeline, report
    from sem_noise.brightness_report import brightness_report

    source = tmp_path / "input"
    source.mkdir()
    rng = np.random.default_rng(12)
    for i in range(4):
        np.save(source / f"frame{i}.npy", (1 - i * 0.1) * _scene() + rng.normal(0, 0.2, _scene().shape))
    # Isolate integration from registration accuracy; keep the real brightness
    # estimator, serialization, and panel renderer in this test.
    monkeypatch.setattr(pipeline, "register_stack", lambda stack, included, config:
                        (np.zeros((4, 2)), included.copy(), [dict(frame_position=i, accepted=True,
                         peak_ratio=None, correlation=1.0, reason="") for i in range(4)]))
    monkeypatch.setattr(report, "site_report", lambda out, summary, *args:
                        (out / "report.html").write_text(brightness_report(out, summary), encoding="utf-8"))
    result = analyze_dataset(source, tmp_path / "out", config=AnalysisConfig(
        min_frames=4, expected_frames=4, sample_pixels=200, distribution_samples=500,
        spatial_pairs=1, local_frames=1, diff_examples=1))
    assert result["status"] == "complete"
    out = tmp_path / "out/site_001"
    saved = json.loads((out / "brightness.json").read_text())
    assert saved["summary"]["fitted_frames"] == 3
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Brightness correction comparison" in html and "data:image/png;base64," in html
    assert (out / "brightness_evolution.png").exists()
    assert (out / "brightness_difference_1.png").exists()
