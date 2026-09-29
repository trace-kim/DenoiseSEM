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


def _positions(shifts, offsets, noise, rng, *, series="model", method="refined"):
    rows = []
    for frame, shift in enumerate(shifts, 1):
        for hole, offset in enumerate(offsets, 1):
            y, x = offset + shift + noise[hole - 1] * rng.standard_normal(2)
            rows.append({"series": series, "method": method, "frame": frame, "hole": hole, "y": y, "x": x})
    return rows


def test_placement_removes_common_image_shift_exactly_even_with_missing_holes():
    from sem_segment.repeatability import placement_per_hole, placement_residuals

    rng = np.random.default_rng(0)
    shifts = rng.uniform(-6, 6, size=(12, 2)) + np.linspace(0, 4, 12)[:, None]  # jitter + drift
    offsets = np.array([[20 * i, 30 * j] for i in range(3) for j in range(3)], dtype=float)
    rows = _positions(shifts, offsets, np.zeros(9), rng)
    rows = [r for r in rows if not (r["hole"] == 4 and r["frame"] % 3 == 0)]  # a hole sometimes missing
    residuals = placement_residuals(rows)
    assert len(residuals) == len(rows)
    assert np.allclose([[r["residual_y"], r["residual_x"]] for r in residuals], 0, atol=1e-7)
    stats = placement_per_hole(residuals)
    assert stats["model", 4, "refined"]["placement_count"] == 8
    assert stats["model", 1, "refined"]["placement_3sigma_y"] == pytest.approx(0, abs=1e-6)


def test_placement_measures_relative_hole_motion_without_shrinkage():
    from sem_segment.repeatability import placement_per_hole, placement_residuals

    rng = np.random.default_rng(1)
    shifts = rng.uniform(-5, 5, size=(400, 2))
    offsets = np.array([[25 * i, 25 * j] for i in range(2) for j in range(2)], dtype=float)
    stats = placement_per_hole(placement_residuals(_positions(shifts, offsets, np.full(4, .1), rng)))
    # Four holes: without the sqrt(N/(N-1)) correction a 0.1 px hole would read 0.087 px.
    for hole in range(1, 5):
        assert stats["model", hole, "refined"]["placement_std_y"] == pytest.approx(.1, rel=.12)
    # A wandering hole stands out; it also moves the common shift by 1/N, as a
    # whole-image registration would.
    noise = np.array([.1, .1, .1, .4])
    stats = placement_per_hole(placement_residuals(_positions(shifts, offsets, noise, rng)))
    assert stats["model", 4, "refined"]["placement_std_x"] > 2 * stats["model", 1, "refined"]["placement_std_x"]


def test_placement_skips_frames_with_too_few_holes_and_joins_the_summary():
    from sem_segment.repeatability import placement_per_hole, placement_residuals

    rng = np.random.default_rng(2)
    offsets = np.array([[0, 0], [0, 30], [30, 0]], dtype=float)
    rows = _positions(np.zeros((5, 2)), offsets, np.full(3, .2), rng)
    rows = [r for r in rows if not (r["frame"] == 5 and r["hole"] == 3)]
    residuals = placement_residuals(rows)
    assert {r["frame"] for r in residuals} == {1, 2, 3, 4}
    observations = [{"series": "model", "hole": h, "method": "refined", "status": "valid", "cd": 10 + .1 * f,
                     "major_axis": 10, "minor_axis": 10, "order": f} for h in (1, 2, 3) for f in range(1, 6)]
    per_hole, summaries = summarize_observations(observations, ["model"], placement=placement_per_hole(residuals))
    assert all(r["placement_count"] == 4 and r["placement_3sigma_y"] > 0 for r in per_hole)
    summary = next(s for s in summaries if s["method"] == "refined")
    assert summary["median_placement_3sigma_x"] == pytest.approx(
        np.median([r["placement_3sigma_x"] for r in per_hole]))
    coarse = next(s for s in summaries if s["method"] == "coarse")
    assert coarse["median_placement_3sigma_y"] is None
