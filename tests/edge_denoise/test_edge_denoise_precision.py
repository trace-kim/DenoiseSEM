from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import erf

from sem_segment.config import RefineConfig
from sem_segment.otsu_baseline import OtsuSettings
from sem_segment.otsu_measurement import measure_saved_otsu
from sem_segment.repeatability import summarize_observations
from tools import real_sem_compare as compare
from test_real_sem_viewer import saved_record


@pytest.mark.parametrize("estimator", ["gradient_peak", "threshold", "erf"])
def test_refined_otsu_measures_saved_subpixel_disk(tmp_path, estimator):
    yy, xx = np.mgrid[:96, :96]
    radius = 18.37
    rng = np.random.default_rng(0)
    pixels = np.rint(120 + 80 * erf((np.hypot(yy - 47.3, xx - 48.1) - radius) / 1.3 / np.sqrt(2))
                     + rng.normal(0, .3, (96, 96))).astype(np.uint8)
    path = tmp_path / "disk.png"
    compare.save_rgb(path, pixels)
    result = measure_saved_otsu(path, OtsuSettings(), refine=RefineConfig(estimator=estimator))
    region = result.regions[0]
    assert region.valid_fraction > .95
    assert abs(region.coarse.equivalent_diameter_px - 2 * radius) > .1
    assert abs(region.refined.equivalent_diameter_px - 2 * radius) < .05
    assert result.diagnostics.refinement["estimator"] == estimator


def test_refined_rebuild_and_contours_only_record_estimator(tmp_path, monkeypatch):
    source = saved_record(tmp_path / "source")
    first = compare.rebuild(source, tmp_path / "first", contour_method="otsu", tensorboard=False)
    monkeypatch.setattr(compare, "analyze_series", lambda *a, **k: pytest.fail("native analysis must be reused"))
    monkeypatch.setattr(compare, "registration_tracks", lambda *a, **k: pytest.fail("registration must be reused"))
    result = compare.rebuild(tmp_path / "first/comparison.json", tmp_path / "refined",
                             contours_only=True, contour_method="otsu_refined", refine_estimator="threshold")
    assert result["refine_estimator"] == result["settings"]["refine_estimator"] == "threshold"
    assert {r["method"] for r in result["sites"][0]["observations"]} == {"coarse", "refined"}
    assert any(r["status"] == "valid" for r in result["sites"][0]["observations"] if r["method"] == "refined")
    viewer = json.loads((tmp_path / "refined/viewer/data.js").read_text().split(" = ", 1)[1].rstrip(";\n"))
    assert viewer["contour_method"] == "otsu_refined" and viewer["refine_estimator"] == "threshold"
    assert viewer["sites"][0]["per_hole"]


def test_precision_decomposition_recovers_noise_and_records_intervals():
    rng = np.random.default_rng(751)
    rows = []
    for hole in range(8):
        noise = rng.normal(0, .2, 800)
        for i, epsilon in enumerate(noise):
            cd = 36 + hole + i * .01 + epsilon
            rows.append(dict(series="model", hole=hole, method="refined", status="valid", order=i,
                             cd=cd, major_axis=cd, minor_axis=cd, mean_dn=cd * 3, otsu_threshold_dn=200-cd))
    per_hole, summaries = summarize_observations(rows, ["model"])
    for row in per_hole:
        assert row["cd_std"] > 2
        assert row["cd_std_detrended"] == pytest.approx(.2, rel=.1)
        assert row["cd_std_successive"] == pytest.approx(.2, rel=.1)
        assert row["cd_brightness_correlation"] == pytest.approx(1)
        assert row["cd_threshold_correlation"] == pytest.approx(-1)
    summary = next(r for r in summaries if r["method"] == "refined")
    low, high = summary["median_cd_std_detrended_ci95"]
    assert low <= summary["median_cd_std_detrended"] <= high
    assert set(summary["observations_per_hole"].values()) == {800}
    short, _ = summarize_observations(rows[:2], ["model"])
    assert short[0]["cd_std_detrended"] is short[0]["cd_std_successive"] is None
