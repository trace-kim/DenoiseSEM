from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import erf

from edge_denoise.uint8_output import average_uint8
from tools import real_sem_averages as averages
from tools import real_sem_compare as compare
from test_real_sem_viewer import saved_record


def test_registered_drifting_burst_is_sharper_and_keeps_blank_frames():
    yy, xx = np.mgrid[:96, :104]
    shifts = [-3, -2, -1, 0, 1, 2, 3, 0]
    frames = np.array([np.rint(120 + 80 * erf((np.hypot(yy - 48, xx - 52 - shift) - 18) / 1.4))
                       for shift in shifts], dtype=np.uint8)
    output, support, record = averages.registered_average(frames)
    native = average_uint8(frames)
    # The edge width proxy is inverse maximum slope on the right disk edge.
    assert np.max(np.diff(output[48, 60:80].astype(float))) > np.max(np.diff(native[48, 63:83].astype(float))) * 1.4
    assert output.shape == frames.shape[1:] and output.dtype == np.uint8 and support.dtype == bool
    assert record["reference_frame"] == 1
    assert all(r["status"] in {"reference", "registered"} for r in record["frames"])
    blank = np.full((3, 32, 40), 101, dtype=np.uint8)
    blank[1] = 100
    unchanged, _, skipped = averages.registered_average(blank)
    np.testing.assert_array_equal(unchanged, average_uint8(blank))
    assert skipped["reference_frame"] is None
    assert all(r["status"] == "skipped_low_contrast" and r["reason"] for r in skipped["frames"])


def test_saved_average_names_provenance_and_remainder(tmp_path, monkeypatch):
    source = saved_record(tmp_path / "source", count=9)
    record = json.loads(source.read_text())
    site = record["sites"][0]
    before = {p: p.read_bytes() for p in source.parent.rglob("*.png")}
    added = averages.add_average_series(source.parent, site, [2, 4, 8], device="cpu", model="model")
    assert set(added) == {"average2", "average2_registered", "average4", "average4_registered",
                          "average8_registered", "model_average2", "model_average4", "model_average8"}
    assert all(p.read_bytes() == data for p, data in before.items())
    for name in added:
        series = site["series"][name]
        assert series["remainder_frames"] == 1
        assert len(series["frames"]) == 9 // series["frames_per_output"]
        for frame in series["frames"]:
            assert compare.read_uint8(source.parent / frame["path"]).dtype == np.uint8
            if series["registered"]:
                assert frame["averaging_registration"]["source"] == "decoded saved uint8 images"
    monkeypatch.setattr(averages, "registered_average", lambda *a, **k: pytest.fail("reuse must not repeat registration"))
    assert averages.add_average_series(source.parent, site, [2, 4, 8], device="cpu", model="model") == []


def test_contours_only_adds_missing_registered_series_without_reanalyzing_old(tmp_path, monkeypatch):
    source = saved_record(tmp_path / "source")
    first = compare.rebuild(source, tmp_path / "first", contour_method="otsu_refined", tensorboard=False)
    path = tmp_path / "first/comparison.json"
    first["sites"][0]["series"].pop("average8_registered")
    path.write_text(json.dumps(first))
    calls = []
    real_analysis = compare.analyze_series
    def analyze(root, name, *a, **k):
        calls.append(name)
        return real_analysis(root, name, *a, **k)
    monkeypatch.setattr(compare, "analyze_series", analyze)
    result = compare.rebuild(path, tmp_path / "second", contours_only=True)
    assert calls == ["average8_registered"]
    assert "average8_registered" in result["sites"][0]["series"]
    assert (tmp_path / "second/site/frames_vs_precision.csv").is_file()


def test_raw_average_repeatability_decreases_with_acquisitions(tmp_path):
    yy, xx = np.mgrid[:72, :72]
    truth = 120 + 75 * erf((np.hypot(yy - 35.5, xx - 35.5) - 15.3) / 1.4)
    rng = np.random.default_rng(41)
    frames = np.rint(truth + rng.normal(0, 6, (128, 72, 72))).clip(0, 255).astype(np.uint8)
    from sem_segment.config import RefineConfig
    from sem_segment.otsu_measurement import measure_saved_otsu

    deviations = []
    for k in (1, 2, 4, 8):
        values = []
        for start in range(0, 128, k):
            pixels = average_uint8(frames[start:start+k]) if k > 1 else frames[start]
            path = tmp_path / f"{k}_{start}.png"
            compare.save_rgb(path, pixels)
            result = measure_saved_otsu(path, refine=RefineConfig())
            values.append(result.regions[0].refined.equivalent_diameter_px)
        deviations.append(np.std(values, ddof=1))
    assert all(b < a for a, b in zip(deviations, deviations[1:]))
