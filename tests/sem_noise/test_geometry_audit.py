from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from sem_noise.config import AnalysisConfig
from sem_noise.geometry_audit import audit_pairs, comparison_metrics, compare_direct_registration
from sem_noise.pair_matching import estimate_geometry
from sem_noise.pipeline import analyze_dataset
from test_registration import moving_frame, specimen


def test_selection_covers_every_input_and_target_without_self_pairs() -> None:
    positions = [0, 2, 3, 7, 10, 11]
    pairs = audit_pairs(positions, "sampled")
    assert len(pairs) == 18 and len(set(pairs)) == 18
    assert all(a != b for a, b in pairs)
    assert all((b, a) in pairs for a, b in pairs)
    assert {a for a, _ in pairs} == {b for _, b in pairs} == set(positions)
    assert len(audit_pairs(positions, "all")) == 30
    assert audit_pairs([2, 9], "sampled") == [(2, 9), (9, 2)]
    with pytest.raises(ValueError):
        AnalysisConfig(compare_direct_registration="all", registration="none")
    with pytest.raises(ValueError):
        AnalysisConfig(compare_direct_registration="unknown")


def test_metrics_have_independent_sign_oracle_and_identical_pixel_support() -> None:
    yy, xx = np.mgrid[:64, :72]
    image = xx + 10.0 * yy
    # Integer translations give an exact pixel oracle independent of cubic kernel bias.
    direct = np.array([[1, 0, 1.], [0, 1, -1.]])
    composed = np.array([[1, 0, -2.], [0, 1, 1.]])
    bad = np.zeros(image.shape, dtype=bool)
    bad[30, 35] = True
    result, arrays = comparison_metrics(image, image, direct, composed, bad, bad,
                                        {"gain": 2, "offset_dn": 7})
    valid = arrays["measurement_valid"]
    assert result["comparison_pixels"] == valid.sum()
    assert not valid[22:39, 27:44].any()
    assert result["coordinate_rms_px"] == pytest.approx(np.hypot(3, -2))
    assert result["coordinate_max_px"] == pytest.approx(result["coordinate_rms_px"])
    for name, expected in (("direct", 9.), ("composed", 8.), ("disagreement", 17.)):
        assert result[f"{name}_rms_dn"] == pytest.approx(expected, abs=1e-10)
        assert result[f"{name}_blurred_rms_dn"] == pytest.approx(expected, abs=1e-10)
    assert result["percentile_disagreement_rms_dn"] == pytest.approx(34.)
    assert result["percentile_disagreement_blurred_rms_dn"] == pytest.approx(34.)
    with pytest.raises(ValueError, match="no shared pixels"):
        comparison_metrics(image, image, direct, composed, np.ones_like(bad), bad, None)


def test_direct_fits_do_not_use_composed_seed_and_failures_stay_visible(tmp_path, monkeypatch) -> None:
    from sem_noise import geometry_audit

    yy, xx = np.mgrid[:40, :40]
    stack = np.stack([xx + yy + i for i in range(3)])
    calls = []
    def fit(a, b, **kwargs):
        calls.append(kwargs)
        if a[0, 0] == 2:
            raise ValueError("synthetic direct failure")
        if kwargs["motion"] == "translation":
            assert kwargs["initial"] is None
        else:
            np.testing.assert_array_equal(kwargs["initial"], np.eye(2, 3))
        return np.eye(2, 3), .99
    monkeypatch.setattr(geometry_audit, "estimate_geometry", fit)
    # Frame 1 has no reference fit. Its direct fit must still be attempted.
    cached = {0: np.eye(2, 3), 2: np.array([[1, 0, 1], [0, 1, -1]])}
    rows = compare_direct_registration(stack, np.ones(3, dtype=bool), np.array([10, 20, 30]),
                                      (None, None), cached, cached, tmp_path / "geometry_audit",
                                      mode="all", sigma=1, progress=lambda _: None)
    assert len(rows) == 12
    missing_anchor = next(r for r in rows if r["input_index"] == 20 and r["target_index"] == 10 and r["motion"] == "translation")
    assert missing_anchor["direct_status"] == "complete" and missing_anchor["composed_status"] == "failed"
    failed_direct = next(r for r in rows if r["input_index"] == 30 and r["target_index"] == 10 and r["motion"] == "translation")
    assert failed_direct["direct_status"] == "failed" and failed_direct["composed_status"] == "complete"
    assert "coordinate_rms_px" not in failed_direct
    assert calls


def test_complete_report_exports_actual_direct_and_composed_affine_fits(tmp_path: Path) -> None:
    clean = specimen(192)
    rng = np.random.default_rng(89)
    truths = [dict(dy_px=dy, dx_px=dx, a11=a11, a12=a12, a21=a21, a22=a22,
                   gain=1, offset_dn=0) for dy, dx, a11, a12, a21, a22 in
              [(0, 0, 0, 0, 0, 0), (1.2, -.7, .002, -.004, .003, -.001),
               (-.8, 1.5, -.003, .002, -.002, .001), (.6, .4, .001, .001, -.003, .002)]]
    stack = np.stack([moving_frame(clean, truth) + rng.normal(0, 2, clean.shape) for truth in truths])
    source, out = tmp_path / "repeats.npy", tmp_path / "report"
    np.save(source, stack)
    before = source.read_bytes()
    result = analyze_dataset(source, out, config=AnalysisConfig(expected_frames=4, min_frames=4,
                             compare_direct_registration="all", sample_pixels=256,
                             distribution_samples=256, spatial_pairs=1))
    assert result["status"] == "complete" and source.read_bytes() == before
    site = out / "site_001"
    payload = json.loads((site / "pair_registration.json").read_text(encoding="utf-8"))
    rows = payload["geometry_comparison_rows"]
    assert len(rows) == 24 and all(r["comparison_status"] == "complete" for r in rows)
    # Verify recovered A->B coordinates against independently known physical points.
    reference_points = np.array([[25, 25], [165, 25], [25, 165], [165, 165]], dtype=float)
    def camera_points(i):
        t = truths[i]
        u, v = (reference_points - 95.5).T
        return reference_points + np.column_stack((t["dx_px"] + t["a11"] * u + t["a12"] * v,
                                                   t["dy_px"] + t["a21"] * u + t["a22"] * v))
    for row in rows:
        if row["motion"] != "affine":
            continue
        a, b = row["input_index"], row["target_index"]
        inputs = np.column_stack((camera_points(a), np.ones(4)))
        for route in ("direct", "composed"):
            matrix = np.array([[row[f"{route}_m{r}{c}"] for c in range(3)] for r in range(2)])
            np.testing.assert_allclose(inputs @ matrix.T, camera_points(b), atol=.15)
        assert row["coordinate_rms_px"] < .12
    with (site / "geometry_comparison.csv").open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == len(rows)
    assert float(exported[0]["direct_blurred_rms_dn"]) == rows[0]["direct_blurred_rms_dn"]
    for name in ("report.html", "pair_report_full.html"):
        html = (site / name).read_text(encoding="utf-8")
        assert "Direct versus composed registration" in html
        assert "geometry_comparison.csv" in html
    assert len(list((site / "geometry_audit").glob("*.npz"))) == 6
    assert len(list((site / "geometry_audit").glob("*.html"))) == 6
