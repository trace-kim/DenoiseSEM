from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from edge_denoise.infer import Denoiser
from sem_segment.otsu_baseline import OtsuSettings
from tools import real_sem_compare as compare
from test_real_sem_compare import CaptureWriter, _arm_fixture, _inputs
from test_real_sem_viewer import saved_record


def forbid_legacy(monkeypatch):
    from sem_segment import backends, cuda, pipeline, refine

    def forbidden(*args, **kwargs):
        raise AssertionError("Otsu must not invoke the legacy detector, edge refinement or CUDA refiner")

    monkeypatch.setattr(backends, "build_segmenter", forbidden)
    monkeypatch.setattr(pipeline, "segment_image", forbidden)
    monkeypatch.setattr(refine, "refine_all", forbidden)
    monkeypatch.setattr(cuda, "CudaRefiner", forbidden)
    return forbidden


def test_rebuild_uses_otsu_in_main_report_without_inference(tmp_path, monkeypatch):
    from sem_noise import comparison_report
    from sem_segment import otsu_measurement

    source = saved_record(tmp_path / "source")
    before = {p.relative_to(source.parent): p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
    forbidden = forbid_legacy(monkeypatch)
    monkeypatch.setattr(Denoiser, "from_checkpoint", forbidden)
    writer = CaptureWriter()
    write_tensorboard = comparison_report.write_tensorboard
    monkeypatch.setattr(comparison_report, "write_tensorboard",
                        lambda root, record: write_tensorboard(root, record, writer_factory=lambda **k: writer))
    output = tmp_path / "otsu"
    record = compare.rebuild(source, output, contour_method="otsu", metrology_device="cpu")
    assert record["contour_method"] == "otsu"
    assert record["otsu_settings"] == OtsuSettings().model_dump()
    assert {p.relative_to(source.parent): p.read_bytes() for p in source.parent.rglob("*") if p.is_file()} == before
    site = record["sites"][0]
    assert len(site["template_centroids"]) == 1
    assert len(site["contours"]) == 18  # 8 raw, 8 model, average8, average128.
    for series in site["series"].values():
        assert series["segmentation_backend"]["backend"] == "otsu"
        assert series["segmentation_backend"]["settings"] == OtsuSettings().model_dump()
        assert series["refinement_backend"]["backend"] == "disabled"
        for frame in series["frames"]:
            assert frame["contour_counts"]["complete"] == 1
            assert frame["contour_counts"]["refined"] == 0
            assert frame["gaussian_backend"] == "scipy"
            assert "mask_metrology" in frame["segmentation_timings_s"]
            assert "otsu_threshold_dn" in frame
            assert (output / frame["path"]).read_bytes() == before[Path(frame["path"])]
    for raw, model in zip(site["series"]["raw"]["frames"], site["series"]["model"]["frames"]):
        assert model["brightness_delta_dn"] == 5
        assert model["otsu_threshold_dn"] - raw["otsu_threshold_dn"] == pytest.approx(5)
    assert all(c["status"]["coarse"] == "valid" and c["status"]["refined"] == "not_run" for c in site["contours"])
    assert all(c["measures"]["coarse"]["ecd"] > 0 and c["measures"]["refined"] is None for c in site["contours"])
    assert any(row["method"] == "coarse" and row["status"] == "valid" for row in site["observations"])
    assert not any(row["method"] == "refined" for row in site["observations"])
    viewer = json.loads((output / "viewer/data.js").read_text(encoding="utf-8").split(" = ", 1)[1].rstrip(";\n"))
    assert viewer["contour_method"] == "otsu"
    assert viewer["sites"][0]["series"]["raw"]["frames"][0]["otsu_threshold_dn"] is not None
    assert ("acquisitions/site/model/contours", 8) in writer.images
    # Rendering the new report reuses the new data; it cannot remeasure either method.
    monkeypatch.setattr(otsu_measurement, "measure_saved_otsu", forbidden)
    rendered = compare.rebuild(output / "comparison.json", tmp_path / "rendered", render_only=True)
    assert rendered["contour_method"] == "otsu"
    with pytest.raises(ValueError, match="render-only"):
        compare.rebuild(output / "comparison.json", tmp_path / "invalid", render_only=True, contour_method="current")


def test_new_comparison_runs_otsu_on_reference_raw_averages_and_quantized_model(tmp_path, monkeypatch):
    from sem_noise import comparison_report
    from sem_segment import otsu_measurement

    source, _ = _inputs(tmp_path)
    yy, xx = np.mgrid[:48, :48]
    for i, path in enumerate(sorted(source.glob("*.png"))):
        compare.save_rgb(path, (np.where(np.hypot(yy - 24, xx - 24) < 10, 40, 180) + i % 5).astype(np.uint8))
    arm, config, digest = _arm_fixture(tmp_path)
    denoiser = SimpleNamespace(config=config, dataset_fingerprint=digest, image_size=16, checkpoint_step=5,
                               default_margin=0, denoise_full=lambda pixels, **kwargs: pixels + 5.4 / 255)
    monkeypatch.setattr(Denoiser, "from_checkpoint", lambda *args, **kwargs: denoiser)
    forbid_legacy(monkeypatch)
    monkeypatch.setattr(compare, "registration_tracks", lambda reference, paths, sigma: [
        {"dy_px": 0., "dx_px": 0., "registration_status": "registered", "registration_score": 1., "registration_error": ""}
        for path in paths])
    monkeypatch.setattr(comparison_report, "write_tensorboard", lambda *args: None)
    calls, actual_measure = [], otsu_measurement.measure_saved_otsu

    def measured(path, settings, **kwargs):
        assert compare.read_uint8(path).dtype == np.uint8
        calls.append((path, settings))
        return actual_measure(path, settings, **kwargs)

    monkeypatch.setattr(otsu_measurement, "measure_saved_otsu", measured)
    settings = compare.ComparisonSettings(checkpoints={"model": arm},
        sites={"site": compare.SiteSettings(source_dir=source)}, output_dir=tmp_path / "new",
        contour_method="otsu", metrology_device="cpu", otsu={"polarity": "dark", "sigma_px": 1., "min_area_px": 25})
    record = compare.run(settings)
    assert record["status"] == "complete" and record["contour_method"] == "otsu"
    assert len(calls) == 1 + 128 + 16 + 1 + 128
    assert all(config == settings.otsu for _, config in calls)
    assert all(frame["brightness_delta_dn"] == 5 for frame in record["sites"][0]["series"]["model"]["frames"])
    assert all(frame["contour_counts"]["complete"] == 1
               for series in record["sites"][0]["series"].values() for frame in series["frames"])


def test_otsu_cuda_uses_only_gaussian_and_records_actual_backend(tmp_path, monkeypatch):
    from scipy.ndimage import gaussian_filter
    from sem_noise import comparison_report

    source = saved_record(tmp_path / "source", count=2)
    forbid_legacy(monkeypatch)
    monkeypatch.setattr(comparison_report, "write_tensorboard", lambda *args: None)
    calls = []

    class Device:
        def __init__(self, index):
            calls.append(index)
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(Device=Device), asarray=np.asarray, asnumpy=np.asarray))
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", SimpleNamespace(gaussian_filter=gaussian_filter))
    record = compare.rebuild(source, tmp_path / "cuda", contour_method="otsu", metrology_device="cuda:1")
    assert calls == [1] * 7  # Reference plus six saved images.
    assert all(f["gaussian_backend"] == "cupy" for s in record["sites"][0]["series"].values() for f in s["frames"])


def test_main_report_preserves_border_paths_and_crop_offsets(tmp_path):
    from sem_segment.config import Config
    from sem_noise.comparison_report import _overlay

    pixels = np.full((80, 100), 210, dtype=np.uint8)
    pixels[:, :45] = 40
    compare.save_rgb(tmp_path / "frame.png", pixels)
    series = {"frames": [{"path": "frame.png", "index": 9, "order": 9, "timestamp_s": None, "dy_px": 0., "dx_px": 0.}]}
    config = Config(input={"crop": [10, 70, 20, 90]})
    observations, contours = compare.measure_series(tmp_path, "raw", series, np.empty((0, 2)), 10, config,
                                                     contour_method="otsu", otsu=OtsuSettings(sigma_px=0))
    assert not observations
    assert len(contours) == 1 and contours[0]["status"]["coarse"] == "border"
    assert contours[0]["coarse"] == [] and contours[0]["measures"]["coarse"] is None
    path = np.array(contours[0]["open_paths"][0])
    assert np.all(path[:, 1] == 44.5) and set(path[[0, -1], 0]) == {10., 69.}
    # A deliberately bent open path would acquire a false chord if closed.
    contours[0]["open_paths"] = [[[5, 5], [10, 15], [15, 5]]]
    overlay = np.asarray(_overlay(Image.fromarray(pixels), contours))
    np.testing.assert_array_equal(overlay[10, 5], [40, 40, 40])
    assert not np.array_equal(overlay[10, 15], [40, 40, 40])


def test_cli_defaults_and_saved_otsu_config_are_reusable(tmp_path, monkeypatch):
    from sem_noise import comparison_report

    default = compare.configure_run(compare.build_parser().parse_args([]))
    assert default.contour_method == "otsu"
    assert default.otsu == OtsuSettings()
    assert compare.configure_run(compare.build_parser().parse_args(["--contour-method", "current"])).contour_method == "current"
    source = saved_record(tmp_path / "source", count=2)
    config = tmp_path / "otsu.yml"
    config.write_text("polarity: dark\nsigma_px: 0\nmin_area_px: 30\n", encoding="utf-8")
    output = tmp_path / "cli"
    forbid_legacy(monkeypatch)
    monkeypatch.setattr(comparison_report, "write_tensorboard", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["real_sem_compare.py", "--from-comparison", str(source),
        "--output-dir", str(output), "--contour-method", "otsu", "--otsu-config", str(config), "--metrology-device", "cpu"])
    assert compare.main() == 0
    record = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert record["otsu_settings"] == {"polarity": "dark", "sigma_px": 0., "min_area_px": 30}
    remeasured = compare.rebuild(output / "comparison.json", tmp_path / "again", metrology_device="cpu")
    assert remeasured["contour_method"] == "otsu" and remeasured["otsu_settings"] == record["otsu_settings"]
    with pytest.raises(ValueError, match="requires.*otsu"):
        compare.rebuild(source, tmp_path / "bad", otsu_config=config, contour_method="current")


def test_switching_back_to_current_clears_active_otsu_frame_metadata(tmp_path, monkeypatch):
    from sem_noise import comparison_report

    source = saved_record(tmp_path / "source", count=2)
    monkeypatch.setattr(comparison_report, "write_tensorboard", lambda *args: None)
    compare.rebuild(source, tmp_path / "otsu", contour_method="otsu", metrology_device="cpu")
    current = compare.rebuild(tmp_path / "otsu/comparison.json", tmp_path / "current", contour_method="current")
    assert current["contour_method"] == "current"
    for series in current["sites"][0]["series"].values():
        assert series["segmentation_backend"]["backend"] == "classical"
        assert all("otsu_threshold_dn" not in f and "gaussian_backend" not in f for f in series["frames"])
    assert any(c["status"]["refined"] == "valid" for c in current["sites"][0]["contours"])
