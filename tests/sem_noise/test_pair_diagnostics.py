from __future__ import annotations

import csv
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
from sem_noise.pair_matching import match_target, measure_quantile_brightness
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
    assert len(saved["region_rows"]) == 6 * 5 * 16
    assert len(saved["quantile_rows"]) == 6 * 17
    acquisition_rows = saved["acquisition_brightness_rows"]
    assert [row["frame_index"] for row in acquisition_rows] == list(range(6))
    assert {row["reference_index"] for row in acquisition_rows} == {0}
    with (site / "acquisition_brightness.csv").open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    for row, csv_row in zip(acquisition_rows, exported):
        q = measure_quantile_brightness(stack[0], stack[row["frame_position"]])
        assert row["quantile_gain"] == q["gain"]
        assert row["quantile_offset_dn"] == q["offset_dn"]
        for key in ("raw_mean_dn", "two_region_mean_dn", "quantile_mean_dn",
                    "two_region_gain", "two_region_offset_dn", "quantile_gain", "quantile_offset_dn"):
            assert float(csv_row[key]) == row[key]
        if "example_arrays" in row:
            with np.load(site / row["example_arrays"]) as example:
                for key, array in (("raw_mean_dn", "raw"), ("two_region_mean_dn", "two_region_corrected"),
                                   ("quantile_mean_dn", "quantile_corrected")):
                    assert row[key] == example[array].mean()
    for row in rows:
        a, b = row["input_index"], row["target_index"]
        expected_gain = truths[b]["gain"] / truths[a]["gain"]
        expected_offset = (truths[b]["offset_dn"] - truths[a]["offset_dn"]) / truths[a]["gain"]
        assert row["gain"] == pytest.approx(expected_gain, abs=0.012)
        assert row["offset_dn"] == pytest.approx(expected_offset, abs=6)
        assert row["brightness_rms_dn"] < row["before_rms_dn"]
        quantile = measure_quantile_brightness(stack[a], stack[b])
        assert row["quantile_status"] == "complete"
        assert row["quantile_gain"] == quantile["gain"]
        assert row["quantile_offset_dn"] == quantile["offset_dn"]
        assert row["quantile_input_pixels"] == row["quantile_target_pixels"] == stack[a].size
        with Image.open(site / row["difference_image"]) as image:
            assert image.size == (5 * 128 + 4 * 8, 128)
        if "example_arrays" in row:
            with np.load(site / row["example_arrays"]) as example:
                np.testing.assert_array_equal(example["input"], stack[a])
                np.testing.assert_array_equal(example["target"], stack[b])
                blurred_a, blurred_b, _, _ = prepare_fit_images(stack[a], stack[b], 1)
                np.testing.assert_array_equal(example["input_blurred"], blurred_a)
                np.testing.assert_array_equal(example["target_blurred"], blurred_b)
                np.testing.assert_allclose(example["corrected_target"], row["gain"] * example["aligned_target"] + row["offset_dn"])
                np.testing.assert_allclose(example["quantile_corrected_target"], row["quantile_gain"] * example["aligned_target"] + row["quantile_offset_dn"])
                original = match_target(stack[a], stack[b], example["matrix"], example["regions"])
                assert row["gain"] == original["brightness"]["gain"]
                assert row["offset_dn"] == original["brightness"]["offset_dn"]
                mask = example["difference_valid"]
                differences = [example[key][mask] - example["input"][mask] for key in
                               ("target", "translated_target", "aligned_target", "corrected_target", "quantile_corrected_target")]
                limit = max(np.max(np.abs(delta)) for delta in differences)
                assert row["colour_limit_dn"] == pytest.approx(limit)
                assert (site / row["difference_viewer"]).is_file()
                assert row["quantile_brightness_rms_dn"] == pytest.approx(np.sqrt(np.mean(differences[-1] ** 2)))
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
    assert 'id="acquisition-brightness"' in html and 'id="acquisition-brightness"' in full
    assert html.index('id="acquisition-brightness"') < html.index('id="raw-image-histograms"')
    assert "no resampling" in html and (site / "acquisition_brightness.png").is_file()
    assert html.count("<h3>Input A = ") == 3
    assert full.count("<h3>Input A = ") == 6
    for row in rows:
        heading = f'<h3>Input A = {row["input_index"]}, target B = {row["target_index"]}</h3>'
        assert heading in full
        assert (heading in html) == (row["input_index"] in {0, 3, 5})
    assert "no binning or subsampling" in html and "log scale" not in full
    assert 'id="raw-image-histograms"' in html
    assert "Signed minimum (DN)" in html and "P99 |difference| (DN)" in html
    assert "Percentile gain" in html and "Two-region gain" in html
    assert "Full-image brightness comparison" in html
    assert html.count('<iframe ') == full.count('<iframe ') == 3
    assert "chooseDifferenceScale" not in html
    assert 'id="acquisition-evolution"' in html and (site / "acquisition_evolution.png").is_file()
    assert html.index('id="acquisition-evolution"') < html.index('id="raw-image-histograms"')
    assert "Pair corrections B→A" in html and "Affine + percentiles" in html
    assert "Percentile gain range" in (output / "index.html").read_text(encoding="utf-8")
    assert saved["brightness"]["quantile_gain_min"] == min(r["quantile_gain"] for r in rows)
    assert all((site / row["difference_viewer"]).is_file() for row in rows)
    assert (site / "pair_quantiles.csv").is_file()
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
    assert result["registration"]["quantile_failures"] == 0
    for row in result["pair_rows"]:
        assert row["quantile_status"] == "complete"
        assert row["quantile_gain"] == pytest.approx(1)
        assert row["quantile_offset_dn"] == pytest.approx(row["input_index"] - row["target_index"])
    from sem_noise.pair_report import pair_report
    assert "Full-image brightness comparison" in pair_report(tmp_path, result)


def test_region_failure_and_exclusions_are_visible(tmp_path: Path) -> None:
    stack = np.stack([np.full((32, 32), 50 + i, dtype=float) for i in range(4)])
    result = diagnose_pairs(stack, np.array([False, True, True, True]), np.arange(4) * 2,
                            (None, None), tmp_path / "pairs", sigma=1, progress=lambda _: None)
    assert result["registration"]["anchor_index"] == 2
    assert {row["input_index"] for row in result["pair_rows"]} == {2, 4, 6}
    assert all(row["status"] == "failed" for row in result["pair_rows"])
    assert "no distinct" in result["registration"]["region_error"]
    assert result["registration"]["quantile_failures"] == 3
    assert all(row["quantile_status"] == "failed" and "quantile_gain" not in row for row in result["pair_rows"])


def test_quantile_failure_preserves_original_brightness_results(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import pair_diagnostics
    from sem_noise.pair_matching import identity_transform
    from sem_noise.pair_report import pair_report

    def fail(*args):
        raise ValueError("test percentile failure")

    monkeypatch.setattr(pair_diagnostics, "measure_quantile_brightness", fail)
    monkeypatch.setattr(pair_diagnostics, "estimate_geometry", lambda *a, **k: (identity_transform(), 1.0))
    stack = np.stack([specimen(64) + i for i in range(4)])
    result = diagnose_pairs(stack, np.ones(4, dtype=bool), np.arange(4), (None, None),
                            tmp_path / "pairs", sigma=1, progress=lambda _: None)
    assert result["registration"]["pair_failures"] == 0
    assert result["registration"]["quantile_failures"] == 4
    assert result["quantile_rows"] == []
    for row in result["pair_rows"]:
        assert row["quantile_status"] == "failed" and "quantile_gain" not in row
        assert row["gain"] == pytest.approx(1)
        assert row["offset_dn"] == pytest.approx(row["input_index"] - row["target_index"])
        assert row["quantile_brightness_rms_dn"] is None
        with Image.open(tmp_path / row["difference_image"]) as image:
            assert np.all(np.asarray(image)[:, 4 * (64 + 8):] == 255)
    assert "test percentile failure" in pair_report(tmp_path, result)


def test_region_failure_keeps_geometry_and_percentile_differences(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import pair_diagnostics
    from sem_noise.pair_matching import identity_transform
    from test_difference_viewer import read_viewer

    def fail(*args):
        raise ValueError("test region means failure")

    monkeypatch.setattr(pair_diagnostics, "measure_brightness", fail)
    monkeypatch.setattr(pair_diagnostics, "estimate_geometry", lambda *a, **k: (identity_transform(), 1.0))
    stack = np.stack([specimen(64) + i for i in range(4)])
    result = diagnose_pairs(stack, np.ones(4, dtype=bool), np.arange(4), (None, None),
                            tmp_path / "pairs", sigma=1, progress=lambda _: None)
    for row in result["pair_rows"]:
        assert row["status"] == "failed" and "gain" not in row
        assert row["difference_status"] == row["affine_status"] == row["quantile_status"] == "complete"
        assert row["brightness_rms_dn"] is None and row["quantile_brightness_rms_dn"] < 1e-10
        _, values = read_viewer(tmp_path / row["difference_viewer"])
        assert np.isnan(values[3]).all()
        assert np.isfinite(values[0]).any() and np.isfinite(values[2]).any() and np.isfinite(values[4]).any()
    from sem_noise.pair_report import pair_report
    html = pair_report(tmp_path, result)
    assert "test region means failure" in html and html.count('<iframe ') == 3


def test_affine_drift_is_measured_at_centre_not_matrix_origin(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import pair_diagnostics
    matrix = np.array([[1.02, 0.03, 2], [-0.01, 0.99, -3]])
    monkeypatch.setattr(pair_diagnostics, "estimate_geometry", lambda *a, **k: (matrix.copy(), 1.0))
    stack = np.stack([specimen(64) + i for i in range(4)])
    result = diagnose_pairs(stack, np.ones(4, dtype=bool), np.arange(4), (None, None),
                            tmp_path / "pairs", sigma=1, progress=lambda _: None)
    for row in result["geometry_rows"][1:]:
        centre = matrix @ [31.5, 31.5, 1] - [31.5, 31.5]
        assert [row["affine_dx_px"], row["affine_dy_px"]] == pytest.approx(centre)

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


def test_uint8_png_differences_preserve_negative_dn_and_explain_extreme_scale(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import pair_diagnostics
    from sem_noise.pair_matching import identity_transform

    yy, xx = np.indices((64, 64))
    base = (20 + 3 * xx + yy % 7).astype(np.uint8)
    first, target = base.copy(), base.copy()
    first[20, 20], first[20, 21] = 225, 5
    target[20, 20], target[20, 21] = 5, 225
    first[8, 8], first[8, 9] = 0, 255
    target[8, 8], target[8, 9] = 0, 255
    stack = np.stack([first, base + 1, target, base + 2])
    source = tmp_path / "raw"
    source.mkdir()
    for i, frame in enumerate(stack):
        Image.fromarray(frame).save(source / f"frame_{i:04d}.png")
    monkeypatch.setattr(pair_diagnostics, "estimate_geometry", lambda *a, **k: (identity_transform(), 1.0))
    captured = {}
    write_image = pair_diagnostics.write_difference_png

    def capture(path, differences, valid, limit):
        captured[path.name] = ([d.copy() for d in differences], valid.copy(), limit)
        write_image(path, differences, valid, limit)

    monkeypatch.setattr(pair_diagnostics, "write_difference_png", capture)
    output = tmp_path / "report"
    result = analyze_dataset(source, output, config=AnalysisConfig(min_frames=4, expected_frames=4,
                             sample_pixels=500, distribution_samples=3000, spatial_pairs=2))
    assert result["status"] == "complete"
    differences, valid, limit = captured["input_0000_target_0002_differences.png"]
    assert all(delta.dtype.kind == "f" for delta in differences)
    expected = target.astype(np.float64) - first.astype(np.float64)
    np.testing.assert_array_equal(differences[0], expected)
    assert valid[20, 20] and valid[20, 21]
    assert differences[0][20, 20] == -220  # uint8 subtraction would incorrectly give +36
    assert differences[0][20, 21] == 220
    assert limit >= 220  # full range is only the initial view; the viewer accepts any DN limit
    site = output / "site_001"
    saved = json.loads((site / "pair_registration.json").read_text(encoding="utf-8"))
    row = saved["pair_rows"][0]
    assert row["before_min_dn"] == -220 and row["before_max_dn"] == 220
    assert row["before_abs_p99_dn"] == 0  # two pixels set the range; most differences are zero
    assert row["colour_limit_full_dn"] >= 220
    assert row["before_rms_dn"] == pytest.approx(np.sqrt(np.mean(expected[valid] ** 2)))
    assert (site / row["difference_viewer"]).is_file()
    html = (site / "report.html").read_text(encoding="utf-8")
    assert "Original dtype: <b>uint8</b>" in html
    assert "One bin per integer DN" in html
