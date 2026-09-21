import numpy as np
import pytest

from sem_segment.repeatability import correspondence_gate, match_centroids, summarize_observations


def test_translation_missing_and_ambiguous_correspondence():
    template = np.array([[10, 10], [10, 40], [40, 20]])
    gate = correspondence_gate(template)
    assert gate < 15
    observed = template[[2, 0]] + [2.3, -1.5]
    matches = match_centroids(template, observed, np.array([2.3, -1.5]), gate)
    assert [m["region_index"] for m in matches] == [1, None, 0]
    assert matches[1]["match_status"] == "missing"
    duplicated = np.vstack([observed, observed[1] + [.1, 0]])
    assert match_centroids(template, duplicated, np.array([2.3, -1.5]), gate)[0]["match_status"] == "ambiguous"
    assert match_centroids(template, observed, np.array([np.nan, np.nan]), gate)[0]["match_status"] == "registration_failed"
    with pytest.raises(ValueError, match="below half"):
        correspondence_gate(template, 20)


def test_repeatability_is_within_hole_sample_sd_on_common_holes():
    rows = []
    for series, offset in (("average8", 0), ("model", 10)):
        for hole, base in ((1, 20), (2, 100)):
            for index, delta in enumerate([-1, 0, 1]):
                rows.append({"series": series, "hole": hole, "method": "refined", "status": "valid",
                             "cd": base + offset + delta, "major_axis": base + delta, "minor_axis": base + delta,
                             "clipped": series == "model"})
    rows[-2]["status"] = rows[-1]["status"] = "missing"
    holes, summaries = summarize_observations(rows, ["average8", "model"])
    summary = next(r for r in summaries if r["series"] == "model" and r["method"] == "refined")
    assert summary["common_holes"] == [1]
    assert summary["median_cd_std"] == 1
    assert summary["median_cd_3sigma"] == 3
    assert summary["failed_count"] == 2 and summary["contributing_observations"] == 3
    missing = next(r for r in holes if r["hole"] == 2 and r["series"] == "model")
    assert missing["valid_count"] == 1 and missing["cd_std"] is None
    assert missing["clipped_observations"] == 3


def test_native_saved_holes_known_diameter_variation_and_translation(tmp_path):
    from scipy.special import erf
    from sem_segment.config import Config
    from tools.real_sem_compare import save_rgb, measure_series, registration_tracks

    yy, xx = np.mgrid[:96, :96]
    frames = []
    radii = [11, 12, 13]
    drift = [(0., 0.), (1.2, -.8), (2., -1.1)]
    for i, (radius, shift) in enumerate(zip(radii, drift)):
        image = 220 - 170 * .5 * (1 + erf((radius - np.hypot(yy - 48 - shift[0], xx - 48 - shift[1])) / 1.4))
        path = tmp_path / f"{i}.png"
        save_rgb(path, np.rint(image).astype(np.uint8))
        frames.append({"index": i + 1, "order": i + 1, "timestamp_s": None, "path": path.name,
                       "dy_px": shift[0], "dx_px": shift[1], "clipped": i == 1})
    config = Config(segmentation={"backend": "classical", "polarity": "dark", "contrast_stretch": None})
    rows, contours = measure_series(tmp_path, "model", {"frames": frames}, np.array([[48., 48.]]), 10, config)
    valid = [r for r in rows if r["method"] == "refined" and r["status"] == "valid"]
    assert len(valid) == 3
    np.testing.assert_allclose([r["cd"] for r in valid], 2 * np.array(radii), atol=.6)
    holes, summary = summarize_observations(rows, ["model"])
    refined = next(r for r in summary if r["method"] == "refined")
    assert refined["median_cd_std"] == pytest.approx(2, abs=.15)
    assert refined["median_cd_3sigma"] == pytest.approx(6, abs=.45)
    assert len(contours) == 3 and valid[1]["clipped"]
    # Estimate translation using the existing estimator on saved pixels.
    from tools.real_sem_compare import read_uint8
    shifted = 220 - 170 * .5 * (1 + erf((11 - np.hypot(yy - 50, xx - 47)) / 1.4))
    save_rgb(tmp_path / "translated.png", np.rint(shifted).astype(np.uint8))
    tracks = registration_tracks(read_uint8(tmp_path / "0.png"), [tmp_path / "translated.png"], 1.)
    assert tracks[0]["registration_status"] == "registered"
    assert tracks[0]["dy_px"] == pytest.approx(2, abs=.1)
    assert tracks[0]["dx_px"] == pytest.approx(-1, abs=.1)
