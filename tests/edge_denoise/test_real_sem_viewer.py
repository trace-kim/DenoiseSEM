import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from edge_denoise.uint8_output import RANGE_WARNING
from sem_noise.config import AnalysisConfig
from sem_noise.pipeline import write_json
from sem_segment.config import Config
from sem_segment.contours import polygon_area
from sem_segment.repeatability import summarize_observations
from tools import real_sem_compare as compare


def saved_record(root: Path, count: int = 8) -> Path:
    """A small old-format report, with actual saved pixels and no checkpoints."""
    root.mkdir()
    yy, xx = np.mgrid[:64, :64]
    pixels = np.where(np.hypot(yy - 32, xx - 32) < 12, 40, 180).astype(np.uint8)
    raw, model = [], []
    for i in range(count):
        for name, rows, image in (("raw", raw, pixels + i), ("model", model, pixels + i + 5)):
            path = f"site/{name}/frame_{i + 1:03d}.png"
            compare.save_rgb(root / path, image)
            rows.append({"index": i + 1, "order": i + 1, "timestamp_s": None, "path": path, "clipped": False})
    average = "site/average8/block.png"
    full = "site/reference/full.png"
    compare.save_rgb(root / average, pixels + 4)
    compare.save_rgb(root / full, pixels + 4)
    config = Config(segmentation={"backend": "classical", "polarity": "dark"})
    record = {"schema_version": 2, "status": "complete", "unit": "px", "settings": {},
              "segmentation_settings": config.model_dump(mode="json"), "prediction_ranges": [],
              "range_warning": RANGE_WARNING, "warnings": [],
              "models": {"model": {"arm": {"registration": "none", "brightness": "none", "settings_source": "synthetic fixture"},
                                     "step": 123, "ema": True, "checkpoint": "absent.pt", "sha256": "unused"}},
              "sites": [{"name": "site", "full_average": full, "contours": [], "observations": [], "series": {
                  "raw": {"frames": raw, "step": 0}, "model": {"frames": model, "step": 123},
                  "average8": {"step": 0, "frames": [{"path": average, "index": 1, "order": 4.5,
                    "first_acquisition": 1, "last_acquisition": 8, "timestamp_s": None, "clipped": False}]}}}]}
    write_json(root / "comparison.json", record)
    return root / "comparison.json"


def test_rebuild_and_render_only_use_saved_uint8_without_inference_or_correction(tmp_path, monkeypatch, capsys):
    from edge_denoise.infer import Denoiser
    from sem_noise import pipeline, comparison_report

    source = saved_record(tmp_path / "old")
    saved = json.loads(source.read_text(encoding="utf-8"))
    series = saved["sites"][0]["series"]
    saved["sites"][0]["series"] = {name: series[name] for name in ("model", "average8", "raw")}
    saved["timings_s"] = {"save_comparison_json": 1e9}
    saved["sites"][0]["timings_s"] = {"export_contours_json": 1e9}
    write_json(source, saved)  # JSON key order cannot decide the drift reference.

    def forbidden(*args, **kwargs):
        raise AssertionError("inference/acquisition corrections must not run")

    monkeypatch.setattr(Denoiser, "from_checkpoint", forbidden)
    monkeypatch.setattr(pipeline, "analyze_dataset", forbidden)
    monkeypatch.setattr(comparison_report, "write_tensorboard", lambda *args: None)
    record = compare.rebuild(source, tmp_path / "new", metrology_device="cpu")
    assert record["timings_s"]["save_comparison_json"] < 1e9
    assert record["sites"][0]["timings_s"]["export_contours_json"] < 1e9
    assert record["schema_version"] == 3
    site = record["sites"][0]
    assert site["contour_status"] == "available"
    assert all(f["brightness_delta_dn"] == 5 for f in site["series"]["model"]["frames"])
    assert site["series"]["average128"]["native"]["temporal_rms_dn"] is None
    assert record["segmentation_settings"]["segmentation"]["contrast_stretch"] is None
    viewer = json.loads((tmp_path / "new/viewer/data.js").read_text().split(" = ", 1)[1].rstrip(";\n"))
    assert viewer["sites"][0]["series"]["model"]["frames"][7]["brightness_delta_dn"] == 5
    band = viewer["sites"][0]["series"]["model"]["contour_bands"]["coarse"]
    groups = ET.parse(tmp_path / "new" / band).findall(".//{*}g[@data-frame]")
    assert [int(g.attrib["data-frame"]) for g in groups] == list(range(1, 9))
    original_pngs = {p.relative_to(source.parent): p.read_bytes() for p in source.parent.rglob("*.png")}
    monkeypatch.setattr(compare, "measure_series", forbidden)
    monkeypatch.setattr(compare, "registration_tracks", forbidden)
    from sem_noise import comparison_storage
    monkeypatch.setattr(comparison_storage, "finish_contours", forbidden)
    compare.rebuild(tmp_path / "new/comparison.json", tmp_path / "rendered", render_only=True)
    assert (tmp_path / "new/site/contours.json").read_bytes() == (tmp_path / "rendered/site/contours.json").read_bytes()
    progress = capsys.readouterr().out
    assert "Loading saved comparison" in progress
    assert "site: loading saved contours: starting" in progress
    assert "Copying saved comparison assets: starting" in progress
    assert "MiB copied" in progress
    assert (tmp_path / "new" / band).read_bytes() == (tmp_path / "rendered" / band).read_bytes()
    for relative, pixels in original_pngs.items():
        assert (source.parent / relative).read_bytes() == pixels
        assert (tmp_path / "rendered" / relative).read_bytes() == pixels
    with pytest.raises(ValueError, match="older report"):
        compare.rebuild(source, tmp_path / "invalid", render_only=True)
    with pytest.raises(ValueError, match="new --output-dir"):
        compare.rebuild(source, source.parent)


def test_render_only_recovers_incremental_contour_parts_without_remeasurement(tmp_path, monkeypatch):
    from sem_noise.comparison_storage import load_contours

    source = saved_record(tmp_path / "source", count=2)
    completed = tmp_path / "measured"
    record = compare.rebuild(source, completed, contour_method="otsu", metrology_device="cpu", tensorboard=False)
    expected = record["sites"][0]["contours"]
    # Reproduce a running snapshot taken after measurements, before finalization.
    record["sites"][0].pop("contours_path")
    record["status"] = "running"
    compare.save_record(completed, record)
    snapshot = json.loads((completed / "comparison.json").read_text())
    assert "contours_parts" in snapshot["sites"][0] and "contours" not in snapshot["sites"][0]

    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery must reuse saved measurements")

    monkeypatch.setattr(compare, "measure_series", forbidden)
    monkeypatch.setattr(compare, "registration_tracks", forbidden)
    monkeypatch.setattr(compare, "analyze_series", forbidden)
    recovered = tmp_path / "recovered"
    result = compare.rebuild(completed / "comparison.json", recovered, render_only=True, tensorboard=False)
    assert result["status"] == "complete"
    saved = json.loads((recovered / "comparison.json").read_text())
    assert load_contours(recovered, saved["sites"][0]) == expected
    assert saved["sites"][0]["observations"] == snapshot["sites"][0]["observations"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for asynchronous viewer regression checks")
def test_comparison_viewer_keeps_decoded_images_contours_and_statistics_in_sync(tmp_path):
    source = saved_record(tmp_path / "source")
    report = tmp_path / "report"
    compare.rebuild(source, report, metrology_device="cpu", tensorboard=False)
    result = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("comparison_controls.cjs")), str(report)],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("method", ["coarse", "refined"])
def test_contour_band_keeps_all_frames_holes_unmatched_regions_and_open_paths(tmp_path, method):
    from sem_noise.comparison_report import _contour_band

    region = {"hole": None, "coarse": [[1.25, 2.5], [3, 4], [5, 6]],
              "holes": [[[2, 3], [2, 4], [3, 4]]], "open_paths": [[[6, 7], [7, 8]]],
              "refined": [[1.5, 2.75], [3, 4.25], [5, 6.25]], "refined_valid": [True, True, False],
              "status": {"coarse": "valid", "refined": "insufficient_refinement"}}
    frames = [{"index": 9, "order": 9}, {"index": 29, "order": 29}]
    groups = {("model", 9): [region], ("model", 29): [region, region]}
    destination = tmp_path / "band.svg"
    _contour_band(destination, [64, 64], "model", frames, groups, method)
    svg = ET.parse(destination)
    acquisitions = svg.findall(".//{*}g[@data-frame]")
    assert [g.attrib["data-frame"] for g in acquisitions] == ["9", "29"]
    for group, count in zip(acquisitions, [1, 2]):
        paths = group.findall("{*}path")
        data = " ".join(p.attrib["d"] for p in paths)
        assert data.count("M3,2L4,2L4,3Z") == count  # Interior rings, with no ID requirement.
        assert data.count("M7,6L8,7") == count
        assert "M7,6L8,7Z" not in data  # Never close a border path.
        if method == "coarse":
            assert data.count("M2.5,1.25L4,3L6,5Z") == count
        else:
            solid = next(p.attrib["d"] for p in paths if "stroke-dasharray" not in p.attrib)
            assert solid.count("M2.75,1.5L4.25,3") == count
            assert "6.25,5" not in solid  # Failed refinement remains dashed.
    assert "Acquisition / block center" in destination.read_text(encoding="utf-8")


def test_all_contours_survive_without_template_and_crop_offset_is_applied_once(tmp_path):
    yy, xx = np.mgrid[:144, :192]
    points = [(40, 35), (40, 90), (40, 150), (100, 35), (100, 90), (100, 150), (70, 0)]
    image = np.full(yy.shape, 210, dtype=np.uint8)
    for y, x in points:
        image[np.hypot(yy - y, xx - x) < 12] = 40
    compare.save_rgb(tmp_path / "frame.png", image)
    series = {"frames": [{"path": "frame.png", "index": 1, "order": 1, "timestamp_s": None, "dy_px": None, "dx_px": None}]}
    config = Config(input={"crop": [10, 134, 0, 192]},
                    segmentation={"backend": "classical", "polarity": "dark", "contrast_stretch": None})
    observations, contours = compare.measure_series(tmp_path, "raw", series, np.empty((0, 2)), 10, config)
    assert not observations
    assert len(contours) == 7
    assert all(c["hole"] is None for c in contours)
    assert any(c["status"]["coarse"] == "border" for c in contours)
    complete = [c for c in contours if c["status"]["coarse"] == "valid"]
    assert len(complete) == 6
    for c in complete:
        polygon = np.asarray(c["coarse"])
        center = polygon.mean(axis=0)
        assert min(np.linalg.norm(center - p) for p in points) < 1
        area = abs(polygon_area(polygon)) - sum(abs(polygon_area(np.array(r))) for r in c["holes"])
        assert c["measures"]["coarse"]["area_px2"] == pytest.approx(area)
        assert c["measures"]["coarse"]["ecd"] == pytest.approx(2 * np.sqrt(area / np.pi))
        assert len(c["refined_valid"]) == len(c["refined"])
    assert series["frames"][0]["correspondence_status"] == "unavailable"


def test_failed_raw_measurements_do_not_remove_model_repeatability():
    rows = []
    for name in ("raw", "model_a", "model_b"):
        for i in range(3):
            rows.append({"series": name, "hole": 1, "method": "coarse", "status": "missing" if name == "raw" else "valid",
                         "cd": 20 + i, "major_axis": 20 + i, "minor_axis": 20 + i})
    _, summaries = summarize_observations(rows, ["raw", "model_a", "model_b"], comparison_series=["model_a", "model_b"])
    for row in summaries:
        if row["method"] == "coarse":
            assert row["common_hole_count"] == (0 if row["series"] == "raw" else 1)
            assert row["median_cd_std"] == (None if row["series"] == "raw" else 1)


def test_cli_directory_overrides_follow_remote_conventions(tmp_path):
    args = compare.build_parser().parse_args([
        "--experiment-prefix", "260921_real_n2n", "--runs-dir", str(tmp_path / "runs"),
        "--site-dir", "/data/260904_raw_data/test/260904_0947-13", "--output-dir", str(tmp_path / "out"),
        "--checkpoint", f"translation_none={tmp_path / 'older.pt'}", "--metrology-device", "cuda:0"])
    config = compare.configure_run(args)
    assert config.output_dir == tmp_path / "out"
    assert next(iter(config.sites)) == "260904_0947-13"
    assert config.checkpoints["affine_percentile"].checkpoint == tmp_path / "runs/260921_real_n2n_affine_percentile/ckpt_latest.pt"
    assert config.checkpoints["translation_none"].checkpoint == tmp_path / "older.pt"
    assert all(a.prepared_manifest is None for a in config.checkpoints.values())
    with pytest.raises(ValueError, match="Unknown model"):
        compare.configure_run(compare.build_parser().parse_args(["--model", "missing"]))
