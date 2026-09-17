from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from sem_noise.config import AnalysisConfig
from sem_noise.pair_diagnostics import diagnose_pairs
from sem_noise.pipeline import analyze_dataset
from sem_noise.registration import prepare_fit_images
from test_registration import moving_frame, specimen
from test_site_registration import site_truth


def test_raw_affine_pipeline_reports_both_pair_directions_and_preserves_inputs(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import registration

    def forbidden(*args, **kwargs):
        raise AssertionError("the new path must not use the legacy joint brightness fit")
    monkeypatch.setattr(registration, "fit_frame", forbidden)
    clean = specimen(128)
    rng = np.random.default_rng(55)
    truths = site_truth()
    stack = np.stack([moving_frame(clean, t) + rng.normal(0, 4, clean.shape) for t in truths])
    source, output = tmp_path / "repeats.npy", tmp_path / "report"
    np.save(source, stack)
    before = source.read_bytes()
    result = analyze_dataset(source, output, config=AnalysisConfig(min_frames=4, expected_frames=6,
                             sample_pixels=500, distribution_samples=3000, spatial_pairs=2))
    assert result["status"] == "complete"
    assert source.read_bytes() == before
    site = output / "site_001"
    saved = json.loads((site / "pair_registration.json").read_text(encoding="utf-8"))
    rows = saved["pair_rows"]
    assert len(rows) == 6 and all(row["status"] == "complete" for row in rows)
    assert {row["input_index"] for row in rows} == set(range(6))
    assert all(row["input_index"] != row["target_index"] for row in rows)
    assert len(saved["region_rows"]) == 6 * 4 * 16
    for row in rows:
        a, b = row["input_index"], row["target_index"]
        expected_gain = truths[b]["gain"] / truths[a]["gain"]
        expected_offset = (truths[b]["offset_dn"] - truths[a]["offset_dn"]) / truths[a]["gain"]
        assert row["gain"] == pytest.approx(expected_gain, abs=0.012)
        assert row["offset_dn"] == pytest.approx(expected_offset, abs=6)
        assert row["brightness_rms_dn"] < row["before_rms_dn"]
        with Image.open(site / row["difference_image"]) as image:
            assert image.size == (4 * 128 + 3 * 8, 128)
        if "example_arrays" in row:
            with np.load(site / row["example_arrays"]) as example:
                np.testing.assert_array_equal(example["input"], stack[a])
                np.testing.assert_array_equal(example["target"], stack[b])
                blurred_a, blurred_b, _, _ = prepare_fit_images(stack[a], stack[b], 1)
                np.testing.assert_array_equal(example["input_blurred"], blurred_a)
                np.testing.assert_array_equal(example["target_blurred"], blurred_b)
                np.testing.assert_allclose(example["corrected_target"], row["gain"] * example["aligned_target"] + row["offset_dn"])
                mask = example["difference_valid"]
                largest_difference = max(np.max(np.abs(example[key][mask] - example["input"][mask]))
                                         for key in ("target", "translated_target", "aligned_target", "corrected_target"))
                assert row["colour_limit_dn"] == pytest.approx(largest_difference)
                for label, name in ((1, "low"), (2, "high")):
                    mask = (example["regions"] == label) & example["brightness_valid"]
                    assert mask.sum() == row[name + "_pixels"]
                    assert example["input"][mask].mean() == pytest.approx(row["input_" + name + "_dn"])
                    assert example["aligned_target"][mask].mean() == pytest.approx(row["target_" + name + "_dn"])
    assert not (site / "registration_pass1.csv").exists()
    html = (site / "report.html").read_text(encoding="utf-8")
    assert "Raw target-to-input matching" in html
    assert "Neither image is assumed clean" in html
    assert "data:image/png;base64," in html
    assert "two measured region means" in html
    assert "one least-squares fit per frame" not in html
    assert 'href="pair_report_full.html"' in html
    full = (site / "pair_report_full.html").read_text(encoding="utf-8")
    assert html.count("<h3>Input A = ") == 3
    assert full.count("<h3>Input A = ") == 6
    for row in rows:
        heading = f'<h3>Input A = {row["input_index"]}, target B = {row["target_index"]}</h3>'
        assert heading in full
        assert (heading in html) == (row["input_index"] in {0, 3, 5})
    assert "no binning or subsampling" in html and "log scale" not in full
    assert (site / "geometry.csv").is_file() and (site / "target_pairs.csv").is_file()


def test_failed_affine_keeps_diagnostic_rows_without_fake_corrections(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import pair_diagnostics
    from sem_noise.pair_matching import identity_transform

    def estimate(*args, motion, **kwargs):
        if motion == "affine":
            raise ValueError("test affine failure")
        return identity_transform(), 0.9
    monkeypatch.setattr(pair_diagnostics, "estimate_geometry", estimate)
    stack = np.stack([specimen(64) + i for i in range(4)])
    result = diagnose_pairs(stack, np.ones(4, dtype=bool), np.arange(4), (None, None),
                            tmp_path / "pairs", sigma=1, progress=lambda _: None)
    assert len(result["geometry_rows"]) == len(result["pair_rows"]) == 4
    assert result["registration"]["affine_failures"] == 3
    assert all(row["status"] == "failed" and "gain" not in row for row in result["pair_rows"])
    assert "test affine failure" in result["geometry_rows"][1]["affine_error"]


def test_region_failure_and_exclusions_are_visible(tmp_path: Path) -> None:
    stack = np.stack([np.full((32, 32), 50 + i, dtype=float) for i in range(4)])
    result = diagnose_pairs(stack, np.array([False, True, True, True]), np.arange(4) * 2,
                            (None, None), tmp_path / "pairs", sigma=1, progress=lambda _: None)
    assert result["registration"]["anchor_index"] == 2
    assert {row["input_index"] for row in result["pair_rows"]} == {2, 4, 6}
    assert all(row["status"] == "failed" for row in result["pair_rows"])
    assert "no distinct" in result["registration"]["region_error"]


def test_failed_translation_does_not_discard_successful_affine_pair(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import pair_diagnostics
    from sem_noise.pair_matching import identity_transform
    from sem_noise.pair_report import pair_report
    from sem_noise.site_registration import INVALID_INDEX

    def estimate(*args, motion, initial, **kwargs):
        if motion == "translation":
            raise ValueError("test translation failure")
        assert initial is None  # full affine can still fit directly from identity
        return identity_transform(), 0.9

    monkeypatch.setattr(pair_diagnostics, "estimate_geometry", estimate)
    stack = np.stack([specimen(64) + 10 * i for i in range(4)])
    result = diagnose_pairs(stack, np.ones(4, dtype=bool), np.arange(4), (None, None),
                            tmp_path / "pairs", sigma=1, progress=lambda _: None)
    assert result["registration"]["translation_failures"] == 3
    assert result["registration"]["pair_failures"] == 0
    assert result["registration"]["drift_step_rms_px"] is None
    for row in result["pair_rows"]:
        assert row["status"] == "complete" and row["translation_status"] == "failed"
        assert row["gain"] == pytest.approx(1)
        assert row["offset_dn"] == pytest.approx(10 * (row["input_index"] - row["target_index"]))
        assert row["brightness_rms_dn"] < 1e-10 and row["translation_rms_dn"] is None
        with Image.open(tmp_path / row["difference_image"]) as image:
            assert np.all(np.asarray(image)[:, 72:136] == INVALID_INDEX)
    assert all(row["pixels"] == 0 and row["rms_dn"] is None for row in result["region_rows"]
               if row["panel"] == "translation")
    assert "Translation comparison failed" in pair_report(tmp_path, result)
