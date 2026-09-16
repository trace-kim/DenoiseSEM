from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("skimage")
pytest.importorskip("tifffile")
pytest.importorskip("matplotlib")

from sem_noise.cli import main
from sem_noise.config import AnalysisConfig
from sem_noise.pipeline import analyze_dataset


def _site(path: Path, seed: int = 3, count: int = 8) -> list[Path]:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        image = rng.normal(1000, 10, (48, 48))
        frame = path / f"frame{i}.png"
        Image.fromarray(image.astype(np.uint16)).save(frame)
        paths.append(frame)
    return paths


def _config() -> AnalysisConfig:
    return AnalysisConfig(registration="none", min_frames=4, expected_frames=8,
                          sample_pixels=500, distribution_samples=3000, spatial_pairs=2)


def test_end_to_end_png_report_provenance_and_duplicates(tmp_path: Path) -> None:
    source, output = tmp_path / "input", tmp_path / "output"
    paths = _site(source / "site")
    (source / "site/frame8.png").write_bytes(paths[1].read_bytes())
    before = {p.name: p.read_bytes() for p in paths}
    result = analyze_dataset(source, output, config=_config())
    assert result["status"] == "complete"
    summary = result["sites"][0]
    assert summary["accepted_frames"] == 8
    assert summary["modes"]["native"]["flat_temporal_sigma_dn"] == pytest.approx(10, rel=0.08)
    assert summary["max_drift_px"] is None
    assert summary["duration_s"] is None
    assert summary["interval_s"] is None
    assert "data:image/png;base64," in (output / "site_001/report.html").read_text(encoding="utf-8")
    assert (output / "index.html").exists()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["config"]["registration"] == "none"
    assert "metrics.py" in provenance["source_sha256"]
    audit = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    assert audit[-1]["duplicate_of_frame_index"] == 1
    assert not audit[-1]["included"]
    assert len(audit[0]["file_sha256"]) == 64
    assert not list((output / "site_001").glob(".cache-*"))
    assert {p.name: p.read_bytes() for p in paths} == before
    with np.load(output / "site_001/maps.npz", allow_pickle=False) as maps:
        assert maps["native_mean"].shape == (48, 48)


def test_failed_site_does_not_block_good_site(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import report
    monkeypatch.setattr(report, "site_report", lambda *args: None)
    source, output = tmp_path / "input", tmp_path / "output"
    bad = _site(source / "01_bad")
    Image.fromarray(np.ones((32, 32), dtype=np.uint16)).save(bad[-1])
    _site(source / "02_good", seed=5)
    result = analyze_dataset(source, output, config=_config())
    assert result["status"] == "partial_failure"
    assert result["successful_sites"] == 1
    assert "identical shape and dtype" in result["sites"][0]["error"]
    assert not list(output.rglob("raw.dat"))


def test_existing_output_and_nested_output_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "input"
    _site(source / "site")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        analyze_dataset(source, output, config=_config())
    with pytest.raises(ValueError, match="outside"):
        analyze_dataset(source, source / "report", config=_config())


def test_irregular_manifest_timing_and_exclusions(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import report
    monkeypatch.setattr(report, "site_report", lambda *args: None)
    source = tmp_path / "input"
    paths = _site(source / "site")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["site", "path", "frame_index", "timestamp_s", "include"])
        writer.writeheader()
        for i, p in enumerate(paths):
            writer.writerow(dict(site="label", path=p.relative_to(source).as_posix(), frame_index=i,
                                 timestamp_s=i**2, include="false" if i == 3 else "true"))
    result = analyze_dataset(source, tmp_path / "result", config=_config(), manifest=manifest)
    site = result["sites"][0]
    assert site["accepted_frames"] == 7
    assert site["interval_s"] is None
    assert site["duration_s"] == 49
    assert site["brightness_slope_time_unit"] == "second"
    temporal = site["modes"]["native"]["temporal"]
    assert not temporal["periodogram_available"]
    assert temporal["acf"][0]["pairs"] == 5


def test_cli_partial_failure_has_nonzero_exit_code(tmp_path: Path) -> None:
    _site(tmp_path / "input/site", count=4)
    assert main(["analyze", "--input", str(tmp_path / "input"), "--output", str(tmp_path / "result")]) == 1
    assert (tmp_path / "result/index.html").exists()


def test_cross_site_duplicates_are_flagged_without_pooling(tmp_path: Path, monkeypatch) -> None:
    from sem_noise import report
    monkeypatch.setattr(report, "site_report", lambda *args: None)
    source = tmp_path / "input"
    a = _site(source / "site_a", seed=10)
    b = _site(source / "site_b", seed=11)
    b[0].write_bytes(a[0].read_bytes())
    result = analyze_dataset(source, tmp_path / "result", config=_config())
    assert result["successful_sites"] == 2
    assert len(result["cross_site_duplicate_groups"]) == 1
    assert len(result["sites"]) == 2


def test_packed_integer_clipping_bounds_are_respected(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace
    from sem_noise import report
    monkeypatch.setattr(report, "site_report", lambda *args: None)
    source = tmp_path / "input"
    paths = _site(source / "site")
    for path in paths:
        with Image.open(path) as image:
            array = np.array(image)
        array[:4, :4] = 4095
        Image.fromarray(array).save(path)
    output = tmp_path / "result"
    result = analyze_dataset(source, output, config=replace(_config(), white_level=4095))
    assert result["sites"][0]["white_level_dn"] == 4095
    with np.load(output / "site_001/maps.npz") as maps:
        assert not maps["native_valid_mask"][:4, :4].any()


def test_failed_registration_writes_shift_audit(tmp_path: Path) -> None:
    source = tmp_path / "input"
    paths = _site(source / "site")
    for i, path in enumerate(paths):
        Image.fromarray(np.full((48, 48), 500 + i, dtype=np.uint16)).save(path)
    output = tmp_path / "result"
    result = analyze_dataset(source, output, config=AnalysisConfig())
    assert result["status"] == "partial_failure"
    assert (output / "site_001/registration.csv").exists()
    assert "only 1 frames pass registration" in result["sites"][0]["error"]


@pytest.mark.parametrize("method", ["intensity", "features"])
def test_affine_cli_report_and_unchanged_noise_statistics(tmp_path: Path, monkeypatch, method: str) -> None:
    from dataclasses import replace
    from scipy import ndimage
    from sem_noise import report

    rng = np.random.default_rng(55)
    reference = 1000 + 200 * ndimage.gaussian_filter(rng.normal(size=(192, 192)), 1.5)
    stack = np.stack([ndimage.rotate(reference, angle, reshape=False, mode="reflect")
                      + rng.normal(0, 1, reference.shape) for angle in (0, 0.2, 0.4, 0.6)])
    source = tmp_path / "repeats.npy"
    np.save(source, stack.astype(np.float32))
    config = tmp_path / "config.yml"
    config.write_text(f"affine_method: {method}\nmin_frames: 4\nexpected_frames: 4\nsample_pixels: 500\ndistribution_samples: 3000\nspatial_pairs: 2\n", encoding="utf-8")
    output = tmp_path / "affine"
    assert main(["analyze", "--input", str(source), "--output", str(output), "--config", str(config),
                 "--affine-diagnostics", "--local-frames", "0", "--local-grid", "3"]) == 0
    diagnostic = json.loads((output / "site_001/affine.json").read_text(encoding="utf-8"))
    assert len(diagnostic["frames"]) == 4
    assert len(diagnostic["models"]) == 16
    assert diagnostic["summary"]["reliable_affine_frames"] == 4
    html = (output / "site_001/report.html").read_text(encoding="utf-8")
    assert "Approximate tile-based affine diagnostics" in html
    assert ("Translation-initialized affine registration" if method == "intensity" else "Feature-based affine registration") in html
    assert "Frame-pair difference comparison" in html
    feature = json.loads((output / "site_001/feature_affine.json").read_text(encoding="utf-8"))
    assert feature["summary"]["estimated_frames"] == 3
    assert feature["summary"]["estimator"] == method
    assert feature["frames"][-1]["correction_rotation_deg"] == pytest.approx(0.6, abs=0.08)
    pairs = json.loads((output / "site_001/difference_examples.json").read_text(encoding="utf-8"))
    assert len(pairs) == 3
    assert all(r["available_modes"] == ["raw", "translation", "affine"] for r in pairs)
    assert pairs[-1]["affine_rms_dn"] < pairs[-1]["translation_rms_dn"] / 3
    assert (output / "site_001/difference_pair_02.png").exists()
    assert "Within-site affine parameter distributions" in html
    assert (output / "site_001/affine.png").exists()
    assert "Motion model support by site" in (output / "index.html").read_text(encoding="utf-8")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["config"]["local_frames"] == 0
    monkeypatch.setattr(report, "site_report", lambda *args: None)
    baseline = analyze_dataset(source, tmp_path / "baseline", config=replace(
        _config(), registration="translation", min_frames=4, expected_frames=4, local_frames=0, local_grid=3))
    with (output / "site_001/summary.json").open(encoding="utf-8") as stream:
        enriched = json.load(stream)
    assert enriched["modes"] == baseline["sites"][0]["modes"]


def test_affine_without_registration_reports_unavailable(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace
    from sem_noise import report
    monkeypatch.setattr(report, "site_report", lambda *args: None)
    _site(tmp_path / "input/site")
    output = tmp_path / "result"
    result = analyze_dataset(tmp_path / "input", output, config=replace(_config(), affine_diagnostics=True))
    summary = result["sites"][0]["affine_diagnostics"]
    assert summary["supported_frames"] == 0
    assert summary["not_assessed_frames"] == 8
    frames = json.loads((output / "site_001/affine.json").read_text(encoding="utf-8"))["frames"]
    assert all(r["selected_model"] is None and r["reason"] == "registration disabled" for r in frames)
