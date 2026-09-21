from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from sem_segment.otsu_baseline import OtsuSettings, otsu_baseline
from tools import preview_sem_contours as preview

NS = {"s": "http://www.w3.org/2000/svg"}


def saved_report(root, *, count=3, external=True):
    root.mkdir()
    pixels = np.full((64, 80), 170, dtype=np.uint8)
    pixels[20:46, 28:58] = 40
    series = {}
    for name, offset, indices in [
        ("model_b", 20, [9]), ("raw", 0, [9, 29, 103] if count == 3 else list(range(1, count + 1))),
        ("average8", 5, [2, 4]), ("average128", 8, [1]), ("model_a", 10, [9, 29]),
    ]:
        path = name + ".png"
        Image.fromarray(np.repeat((pixels + offset)[..., None], 3, axis=2)).save(root / path)
        frames = [{"path": path, "index": index, "order": index + .25} for index in indices]
        if name.startswith("average"):
            for frame in frames:
                first = 1 if name == "average128" else (frame["index"] - 1) * 8 + 1
                frame.update(first_acquisition=first, last_acquisition=128 if name == "average128" else first + 7)
        series[name] = {"frames": frames}
    series["raw"]["frames"][0]["contour_counts"] = {"detected": 0}
    site = {"name": "fixture", "series": series, "full_average": "average128.png"}
    contours = [{"series": "raw", "frame": 29, "coarse": [[30, 40], [32, 40], [32, 42]],
                 "holes": [[[30.5, 40.5], [31, 40.5], [31, 41]]],
                 "refined": [[30.1, 40.1], [32.1, 40.1], [32.1, 42.1]],
                 "refined_valid": [True, True, False]}]
    if external:
        site["contours_path"] = "contours.json"
        (root / "contours.json").write_text(json.dumps(contours), encoding="utf-8")
    else:
        site["contours"] = contours
    record = {"schema_version": 3, "synthetic": True, "study": "Synthetic test",
              "segmentation_settings": {"input": {"crop": [10, 60, 15, 75]}}, "sites": [site]}
    path = root / "comparison.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


@pytest.mark.parametrize("external", [False, True])
def test_preview_uses_saved_pixels_stored_contours_and_one_config(tmp_path, monkeypatch, external):
    import sem_segment.backends
    import sem_segment.pipeline
    import sem_segment.refine

    def forbidden(*args, **kwargs):
        raise AssertionError("The preview must not run the current detector or refinement")

    monkeypatch.setattr(sem_segment.backends, "build_segmenter", forbidden)
    monkeypatch.setattr(sem_segment.pipeline, "segment_image", forbidden)
    monkeypatch.setattr(sem_segment.refine, "refine_all", forbidden)
    source = saved_report(tmp_path / "source", external=external)
    before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
    output = tmp_path / "preview"
    config = OtsuSettings()
    metadata = preview.build_preview(source, output, config)
    assert {p.name: p.read_bytes() for p in source.parent.iterdir()} == before
    assert metadata["settings"] == config.model_dump()
    assert metadata["measurement_crop_y0_y1_x0_x1"] == (10, 60, 15, 75)
    series = metadata["sites"][0]["series"]
    assert list(series) == ["raw", "average8", "average128", "model_b", "model_a"]
    assert [f["index"] for f in series["raw"]["frames"]] == [9, 29, 103]
    assert "no contours detected" in series["raw"]["frames"][0]["current_status"]
    assert "unavailable" in series["raw"]["frames"][2]["current_status"]
    for value in series.values():
        for frame in value["frames"]:
            expected = before[frame["source_image"]]
            assert (output / frame["original"]).read_bytes() == expected
            assert frame["shape"] == [64, 80]
            assert all(seconds >= 0 for seconds in frame["timings_s"].values())
            for name in ("current", "baseline"):
                svg = ET.parse(output / frame[name]).getroot()
                data = svg.find("s:image", NS).attrib["href"]
                assert base64.b64decode(data.split(",", 1)[1]) == expected
    thresholds = [series[name]["frames"][0]["threshold_dn"] for name in series]
    assert len(set(thresholds)) == 5  # Each shifted image gets its own threshold.
    current = ET.parse(output / series["raw"]["frames"][1]["current"])
    points = current.find("s:g/s:polyline", NS).attrib["points"].split()
    assert points[0] == "40.500000,30.500000"  # Existing full-image crop offset applied once.
    assert points[0] == points[-1]  # Stored rings have implicit closure.
    assert len(current.findall("s:g/s:polyline", NS)) == 4  # Outer, hole, fallback, valid segment.
    baseline = ET.parse(output / series["raw"]["frames"][0]["baseline"])
    points = baseline.find("s:g/s:polyline", NS).attrib["points"]
    drawn_xy = np.array([[float(v) for v in pair.split(",")] for pair in points.split()])
    result = otsu_baseline(source.parent / "raw.png", crop=(10, 60, 15, 75))
    np.testing.assert_allclose(drawn_xy, result.outlines[0][:, ::-1] + .5)
    assert "ecd" not in (output / "metadata.json").read_text().lower()
    assert "fetch(" not in (output / "preview.js").read_text(encoding="utf-8")


def test_missing_contours_file_is_explicit_and_old_full_average_is_selectable(tmp_path):
    source = saved_report(tmp_path / "source")
    record = json.loads(source.read_text())
    record["sites"][0]["contours_path"] = "not_saved.json"
    del record["sites"][0]["series"]["average128"]
    source.write_text(json.dumps(record), encoding="utf-8")
    data = preview.build_preview(source, tmp_path / "preview")
    series = data["sites"][0]["series"]
    assert len(series["average128"]["frames"]) == 1
    assert all("unavailable" in frame["current_status"] for value in series.values() for frame in value["frames"])


def test_native_tiff_pixels_and_source_bytes_are_preserved(tmp_path):
    source = saved_report(tmp_path / "source")
    original = source.parent / "raw.png"
    with Image.open(original) as image:
        pixels = np.asarray(image).copy()
        image.save(source.parent / "raw.tif")
    record = json.loads(source.read_text())
    record["sites"][0]["series"]["raw"]["frames"][0]["path"] = "raw.tif"
    source.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "preview"
    metadata = preview.build_preview(source, output)
    frame = metadata["sites"][0]["series"]["raw"]["frames"][0]
    assert (output / frame["original"]).read_bytes() == (source.parent / "raw.tif").read_bytes()
    with Image.open(output / frame["display"]) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels)


def test_preview_draws_open_border_path_without_closing_it(tmp_path):
    source = saved_report(tmp_path / "source")
    pixels = np.full((64, 80), 170, dtype=np.uint8)
    pixels[:, :40] = 40
    Image.fromarray(pixels).save(source.parent / "raw.png")
    output = tmp_path / "preview"
    data = preview.build_preview(source, output, OtsuSettings(sigma_px=0))
    frame = data["sites"][0]["series"]["raw"]["frames"][0]
    svg = ET.parse(output / frame["baseline"])
    points = svg.find("s:g/s:polyline", NS).attrib["points"].split()
    assert points[0] != points[-1]
    assert set([points[0], points[-1]]) == {"40.000000,10.500000", "40.000000,59.500000"}


def test_source_and_existing_output_are_protected(tmp_path):
    source = saved_report(tmp_path / "source")
    for target in (source.parent, source.parent / "nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            preview.build_preview(source, target)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        preview.build_preview(source, existing)
    record = json.loads(source.read_text())
    record["sites"][0]["series"]["raw"]["frames"][0]["path"] = "../outside.png"
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        preview.build_preview(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_duplicate_indices_and_missing_images_fail_before_writing(tmp_path):
    source = saved_report(tmp_path / "source")
    record = json.loads(source.read_text())
    frames = record["sites"][0]["series"]["raw"]["frames"]
    frames[1]["index"] = frames[0]["index"]
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate acquisition"):
        preview.build_preview(source, tmp_path / "duplicate")
    frames[1]["index"] = 29
    frames[0]["path"] = "missing.png"
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Saved image unavailable"):
        preview.build_preview(source, tmp_path / "missing")


def test_preview_rejects_non_uint8_and_cli_validates_settings(tmp_path):
    source = saved_report(tmp_path / "source")
    Image.fromarray(np.full((64, 80), 200, dtype=np.uint16)).save(source.parent / "raw.png")
    with pytest.raises(ValueError, match="saved uint8"):
        preview.build_preview(source, tmp_path / "bad_uint8")
    config = tmp_path / "settings.yml"
    config.write_text("sigma_px: -1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        preview.main(["--config", str(config), "--from-comparison", str(source),
                      "--output-dir", str(tmp_path / "bad_config")])
    assert error.value.code == 2
    assert not (tmp_path / "bad_config").exists()


def test_cli_needs_no_training_package_or_current_detector(tmp_path):
    source = saved_report(tmp_path / "source")
    script = """
import sys
for name in ('torch', 'edge_denoise', 'burst_diffusion', 'ddim', 'runctl', 'sem_noise'):
    sys.modules[name] = None
from tools.preview_sem_contours import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run([sys.executable, "-c", script, "--from-comparison", str(source),
                             "--output-dir", str(tmp_path / "cli"), "--device", "cpu"],
                            cwd=preview.ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli/index.html").is_file()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to exercise viewer controls")
def test_viewer_source_selection_and_all_acquisition_positions(tmp_path):
    source = saved_report(tmp_path / "source", count=128)
    record = json.loads(source.read_text(encoding="utf-8"))
    record["sites"].append({"name": "empty site", "series": {}})
    source.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "preview"
    preview.build_preview(source, output)
    # Execute the actual UI script with small DOM controls. A browser smoke
    # check separately verifies native image loading and appearance.
    result = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("contour_preview_controls.cjs")),
                             str(output)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_report_with_no_saved_images_fails_before_writing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "comparison.json"
    path.write_text('{"sites": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no saved images"):
        preview.build_preview(path, tmp_path / "empty")
    assert not (tmp_path / "empty").exists()
