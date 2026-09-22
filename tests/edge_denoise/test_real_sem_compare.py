from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from conftest import make_config
from edge_denoise.infer import Denoiser
from edge_denoise.uint8_output import RANGE_WARNING, average_uint8, prediction_uint8
from tools import real_sem_compare as compare


def test_range_audit_precedes_rounding_and_clipping():
    values = np.array([[-1e-10, 0, 0.5, 1.5, 254.5, 255, 255 + 1e-10]])
    pixels, stats = prediction_uint8(values / 255, 0, 255)
    np.testing.assert_array_equal(pixels, [[0, 0, 0, 2, 254, 255, 255]])
    assert stats["below_zero"] == stats["above_255"] == 1
    assert stats["minimum_dn"] == pytest.approx(-1e-10)
    assert stats["maximum_dn"] > 255
    assert stats["below_percent"] == pytest.approx(100 / 7)
    assert stats["out_of_range_fraction"] == pytest.approx(2 / 7)
    assert stats["clipped"] and stats["warning"] == RANGE_WARNING
    _, boundary = prediction_uint8(np.array([0.0, 1.0]), 0, 255)
    assert not boundary["clipped"]
    pixels, stats = prediction_uint8(np.array([-0.5, 1.5]), 50, 150)
    assert not stats["clipped"]
    np.testing.assert_array_equal(pixels, [0, 200])
    from sem_noise.comparison_report import _warning
    rendered = _warning({"range_warning": RANGE_WARNING}, [stats | {"maximum_dn": 255 + 1e-10}])
    assert repr(255 + 1e-10) in rendered


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_fails_explicitly(bad):
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        prediction_uint8(np.array([bad]), 0, 255)


def test_rgb_and_averages_are_exact(tmp_path):
    stack = np.arange(128, dtype=np.uint8)[:, None, None] * np.ones((1, 16, 16), dtype=np.uint8)
    for block in range(16):
        pixels = average_uint8(stack[block * 8:(block + 1) * 8])
        assert np.all(pixels == np.rint(block * 8 + 3.5))
        path = tmp_path / f"block_{block}.png"
        compare.save_rgb(path, pixels)
        np.testing.assert_array_equal(compare.read_uint8(path), pixels)
        with Image.open(path) as image:
            assert image.mode == "RGB"
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[0, 0, 1] = 1
    Image.fromarray(rgb).save(tmp_path / "color.png")
    with pytest.raises(ValueError, match="identical RGB channels"):
        compare.read_uint8(tmp_path / "color.png")


def test_unclipped_predictions_survive_patch_and_tiled_inference(tmp_path):
    class Excursions(torch.nn.Module):
        def predict_image(self, x):
            return x * 3 + .25

    denoiser = Denoiser(Excursions(), config=make_config(tmp_path, tmp_path / "run"), device=torch.device("cpu"))
    batch = torch.linspace(-1, 1, 256).reshape(1, 1, 16, 16)
    unclipped = denoiser.denoise(batch, clip_output=False)
    assert unclipped.min() < -1 and unclipped.max() > 1
    assert denoiser.denoise(batch).min() == -1
    frame = np.linspace(0, 1, 32 * 33).reshape(32, 33)
    prediction = denoiser.denoise_full(frame, stride=8, tile_batch=3, clip_output=False)
    np.testing.assert_allclose(prediction, frame * 3 - .875, atol=2e-7)
    assert prediction.min() < 0 and prediction.max() > 1
    default = denoiser.denoise_full(frame, stride=8, tile_batch=3)
    assert default.min() >= 0 and default.max() <= 1


def test_smallest_float32_upper_excursion_survives_tile_range_mapping(tmp_path):
    value = np.nextafter(np.float32(1), np.float32(2)).item()

    class JustOutside(torch.nn.Module):
        def predict_image(self, x):
            return torch.full_like(x, value)

    denoiser = Denoiser(JustOutside(), config=make_config(tmp_path, tmp_path / "run"), device=torch.device("cpu"))
    output = denoiser.denoise_full(np.zeros((32, 33)), stride=8, tile_batch=3, clip_output=False)
    _, stats = prediction_uint8(output, 0, 255)
    assert stats["above_255"] == output.size
    assert stats["minimum_dn"] > 255


def _inputs(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    for i in range(128):
        compare.save_rgb(source / f"frame_{i+1}.png", np.full((16, 16), i, dtype=np.uint8))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    return source, checkpoint


def test_validation_comparison_requires_recorded_holdout_and_keeps_test_exclusions():
    metadata = {"train_val_content_hashes": ["train", "val"], "validation_content_hashes": ["val"]}
    compare.validate_evaluation_content(metadata, {"val"}, "val")
    compare.validate_evaluation_content(metadata, {"unseen-test"}, "test")
    for content in ({"train"}, {"unseen-test"}, {"train", "val"}, set()):
        with pytest.raises(ValueError, match="recorded validation"):
            compare.validate_evaluation_content(metadata, content, "val")
    for content in ({"train"}, {"val"}):
        with pytest.raises(ValueError, match="train/validation"):
            compare.validate_evaluation_content(metadata, content, "test")
    args = compare.build_parser().parse_args(["--split", "val"])
    assert compare.configure_run(args).evaluation_split == "val"


TREATMENTS = [("affine", "percentile"), ("translation", "percentile"), ("none", "percentile"),
              ("affine", "none"), ("translation", "none"), ("none", "none")]


def _arm_fixture(tmp_path, registration="none", brightness="none", *, legacy=False):
    from edge_denoise.config import RealMatchingConfig
    from sem_noise.io import file_hash

    checkpoint = tmp_path / f"{registration}_{brightness}.pt"
    checkpoint.write_bytes(f"mock {registration} {brightness}".encode())
    manifest_path = tmp_path / f"{registration}_{brightness}_dataset.json"
    manifest_path.write_text(json.dumps({"kind": "real_sem", "normalization": {"black": 0, "white": 255},
        "registration": {"mode": registration if legacy else "none"},
        "sites": [{"split": split, "frames": [{"sha256": split + "-content"}]} for split in ("train", "val", "test")]}))
    config = make_config(tmp_path, tmp_path / "run", representation="image", lambda_gradient=0)
    config.data.black_level, config.data.white_level = 0, 255
    config.data.real_matching = None if legacy else RealMatchingConfig(registration=registration, brightness=brightness)
    arm = compare.ComparisonArm(checkpoint=checkpoint, registration=registration, brightness=brightness,
                                prepared_manifest=manifest_path)
    return arm, config, file_hash(manifest_path)


class CaptureWriter:
    def __init__(self, **kwargs):
        self.scalars, self.texts, self.images = {}, {}, {}

    def add_scalar(self, key, value, step):
        self.scalars[key, step] = value

    def add_text(self, key, value, step):
        self.texts[key, step] = value

    def add_image(self, key, value, step, **kwargs):
        self.images[key, step] = value

    def close(self):
        pass


@pytest.mark.parametrize("mixed_objectives", [False, True])
def test_mocked_six_model_workflow_saved_pixels_and_exports(tmp_path, monkeypatch, capsys, mixed_objectives):
    from sem_noise import pipeline
    from sem_noise.comparison_report import write_tensorboard
    from sem_segment import pipeline as segmentation

    source, checkpoint = _inputs(tmp_path)
    measured = []
    fixtures = [_arm_fixture(tmp_path, reg, bright, legacy=i >= 4) for i, (reg, bright) in enumerate(TREATMENTS)]
    if mixed_objectives:
        from edge_denoise.config import ObjectiveConfig

        for (_, config, _), objective in zip(fixtures, [
            {"representation": "image", "lambda_gradient": 0},
            {"representation": "image"},
            {"representation": "image", "lambda_consistency": 1},
            {"representation": "image", "target": "noisy_mean", "lambda_consistency": 1},
            {"representation": "hybrid"},
            {"representation": "gradient", "lambda_image": 0, "lambda_gradient": 1},
        ]):
            config.objective = ObjectiveConfig(**objective)
    arms = {f"{arm.registration}_{arm.brightness}": arm for arm, _, _ in fixtures}
    by_path = {arm.checkpoint: (config, digest) for arm, config, digest in fixtures}

    class MockDenoiser:
        image_size, checkpoint_step, default_margin = 16, 1234, 0

        def __init__(self, path):
            self.config, self.dataset_fingerprint = by_path[path]

        def denoise_full(self, frame, **kwargs):
            assert kwargs["clip_output"] is False
            result = frame.copy() + .49 / 255  # Fractional information must disappear before analysis.
            result[0, 0], result[0, 1] = -1 / 255, 256 / 255
            return result

    monkeypatch.setattr(Denoiser, "from_checkpoint", lambda path, **k: MockDenoiser(path))

    def forbidden(*args, **kwargs):
        raise AssertionError("Acquisition correction must never run on comparison outputs")

    from sem_segment.config import Config as SegmentConfig
    yy, xx = np.mgrid[:16, :16]
    hole_image = np.where((yy - 8)**2 + (xx - 8)**2 < 16, 50, 210) / 255.
    fixed_segmentation = segmentation.segment_image(hole_image, SegmentConfig(
        segmentation={"backend": "classical", "polarity": "dark"}))
    assert fixed_segmentation.regions

    used_backends = []

    def segment(image, config, *, segmenter=None):
        np.testing.assert_allclose(image * 255, np.rint(image * 255), atol=1e-12)
        assert config.segmentation.contrast_stretch is None
        measured.append(image.copy())
        used_backends.append(segmenter)
        return fixed_segmentation

    monkeypatch.setattr(pipeline, "analyze_dataset", forbidden)
    from sem_noise import pair_diagnostics
    monkeypatch.setattr(pair_diagnostics, "diagnose_pairs", forbidden)
    monkeypatch.setattr(pair_diagnostics, "acquisition_brightness", forbidden)
    monkeypatch.setattr(segmentation, "segment_image", segment)
    monkeypatch.setattr(compare, "registration_tracks", lambda reference, paths, sigma: [
        {"dy_px": .25 if p.parent.name in arms else 0., "dx_px": 0.,
         "registration_status": "registered", "registration_score": 1.,
         "registration_error": "", "mean_dn": float(compare.read_uint8(p).mean())} for p in paths])
    writer = CaptureWriter()
    import torch.utils.tensorboard
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", lambda **k: writer)
    config = compare.ComparisonSettings(checkpoints=arms,
                                        sites={"site": compare.SiteSettings(source_dir=source)},
                                        output_dir=tmp_path / "output", frame_interval_s=.5)
    result = compare.run(config)
    assert result["status"] == "complete"
    assert not list(config.output_dir.glob("site/noise_*"))
    assert len(measured) == 1 + 128 + 16 + 1 + 6 * 128  # Template plus each saved uint8 image.
    assert used_backends[0] is None  # Template; each measurement series shares one backend.
    start = 1
    for count in (128, 16, 1, *([128] * 6)):
        group = used_backends[start:start + count]
        assert group[0] is not None and all(backend is group[0] for backend in group)
        start += count
    progress = capsys.readouterr().out
    assert progress.index("raw: measuring native uint8") < progress.index("site/raw: contours/CD starting")
    assert "site/raw: contours/CD 1/128" in progress
    assert "site/raw: contours/CD 128/128" in progress
    assert "site/average8: contours/CD 16/16" in progress
    assert "Combined comparison report finished" in progress
    assert "TensorBoard comparison finished" in progress
    after_contours = progress[progress.index("site/none_none: contours/CD complete"):]
    stages = ["Saving comparison.json", "Finalizing comparison", "repeatability summary",
              "site/observations.csv", "site/contours.json", "Rendering interactive comparison report",
              "site/raw: viewer contour JSON/JS", "site/raw: coarse contour SVG",
              "site/raw: TensorBoard decode/overlay/encode", "TensorBoard: flushing and closing",
              "Comparison finalization complete"]
    positions = [after_contours.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "Saved comparison.json:" in progress
    for series in result["sites"][0]["series"].values():
        assert series["timings_s"]["contours_cd"] >= 0
        assert series["timings_s"]["native_analysis"] >= 0
        assert "edge_strength" in series["segmentation_stage_totals_s"]
        assert "segmentation_timings_s" in series["frames"][0]
    averages = result["sites"][0]["series"]["average8"]["frames"]
    for block, frame in enumerate(averages):
        assert (frame["first_acquisition"], frame["last_acquisition"]) == (block * 8 + 1, block * 8 + 8)
        assert frame["order"] == block * 8 + 4.5
        assert frame["timestamp_s"] == (block * 8 + 3.5) * .5
        assert np.all(compare.read_uint8(config.output_dir / frame["path"]) == np.rint(block * 8 + 3.5))
    assert len(result["prediction_ranges"]) == 768
    assert all(r["clipped"] and r["below_zero"] == r["above_255"] == 1 for r in result["prediction_ranges"])
    assert all(r["minimum_dn"] == pytest.approx(-1) and r["maximum_dn"] == pytest.approx(256) for r in result["prediction_ranges"])
    assert any(a[0, 0] == 0 and a[0, 1] == 1 for a in measured)
    html = (config.output_dir / "index.html").read_text(encoding="utf-8")
    assert html.count(RANGE_WARNING) == 1  # One expandable range audit, not repeated warning walls.
    assert "not ground truth" in html
    viewer_data = json.loads((config.output_dir / "viewer/data.js").read_text().split(" = ", 1)[1].rstrip(";\n"))
    assert len(viewer_data["sites"][0]["series"]["none_none"]["frames"]) == 128
    assert viewer_data["sites"][0]["series"]["none_none"]["frames"][-1]["path"].endswith("frame_128.png")
    assert "legacy prepared dataset" in html
    assert "Link acquisitions" in html and "Overlapping wipe" in html
    assert "below_zero" in (config.output_dir / "prediction_ranges.csv").read_text()
    saved = json.loads((config.output_dir / "comparison.json").read_text())
    assert "contours" not in saved["sites"][0] and "contours_parts" not in saved["sites"][0]
    assert json.loads((config.output_dir / saved["sites"][0]["contours_path"]).read_text()) == result["sites"][0]["contours"]
    timings = json.loads((config.output_dir / "timings.json").read_text())
    assert timings["timings_s"]["save_comparison_json"] == result["timings_s"]["save_comparison_json"]
    assert timings["timings_s"]["save_comparison_json"] >= saved["timings_s"]["save_comparison_json"]
    assert timings["sites"]["site"]["timings_s"]["export_contours_json"] >= 0
    assert timings["sites"]["site"]["series"]["none_none"]["report_timings_s"]["tensorboard_images"] >= 0
    assert saved["prediction_ranges"][0]["clipped"]
    assert [(r["registration"], r["brightness"]) for r in saved["arms"]] == TREATMENTS
    assert "training_registration" in (config.output_dir / "metrics.csv").read_text()
    for name in arms:
        model_frames = saved["sites"][0]["series"][name]["frames"]
        assert all(f["dy_px"] == 0 and f["output_dy_px"] == .25 for f in model_frames)
        assert writer.scalars[f"summary/site/{name}/output_vs_raw_translation_rms_px", 1234] == .25
        for frame in model_frames:
            raw_mean = frame["index"] - 1
            expected_mean = (254 * raw_mean + 255) / 256
            assert frame["mean_dn"] == expected_mean
            assert frame["brightness_delta_dn"] == expected_mean - raw_mean
        assert (f"acquisitions/site/{name}/pixels", 128) in writer.images
        assert (f"acquisitions/site/{name}/contours", 128) in writer.images
        assert (f"comparison/{name}/training_treatment", 1234) in writer.texts
    observations = saved["sites"][0]["observations"]
    assert sum(r["clipped"] and r["status"] == "valid" and r["method"] == "coarse" for r in observations) == 768
    assert "True" in (config.output_dir / "site/observations.csv").read_text()
    for row in saved["metrics"]:
        for key, value in row["values"].items():
            if value is not None:
                assert writer.scalars[f"summary/{row['site']}/{row['series']}/{key}", row["step"]] == value
    assert any(RANGE_WARNING in value for value in writer.texts.values())
    assert not any(Path(a["path"]).name.startswith("block_") for a in result["artifacts"])
    assert result["sites"][0]["correspondence_reference"] == "average128"
    assert result["sites"][0]["series"]["average128"]["native"]["temporal_rms_dn"] is None


def test_save_logs_before_serialization_and_names_failed_stage(tmp_path, monkeypatch, capsys):
    from sem_noise import comparison_storage

    def failed_write(path, value):
        assert path.name == "comparison.json"
        output = capsys.readouterr().out
        assert "Saving comparison.json (serialize/write; 0 embedded contours): starting" in output
        raise OSError("disk full")

    monkeypatch.setattr(comparison_storage, "write_record", failed_write)
    with pytest.raises(OSError, match="disk full"):
        compare.save_record(tmp_path, {"sites": [], "prediction_ranges": []})
    output = capsys.readouterr().out
    assert "Saving comparison.json" in output and "failed after" in output
    assert "Saved comparison.json:" not in output


def test_intermediate_record_references_saved_contours_and_does_not_rewrite_them(tmp_path, monkeypatch):
    from sem_noise import comparison_storage

    site = {"name": "site", "series": {}, "contours": [{"coarse": [[1., .5]]}]}
    record = {"sites": [site], "prediction_ranges": []}
    compare.save_record(tmp_path, record)
    saved = json.loads((tmp_path / "comparison.json").read_text())
    assert "contours" not in saved["sites"][0]
    assert comparison_storage.load_contours(tmp_path, saved["sites"][0]) == site["contours"]

    def forbidden(*args, **kwargs):
        raise AssertionError("An unchanged contour part was written again")

    part_path = tmp_path / site["contours_parts"][0]["path"]
    before = part_path.read_bytes()
    atomic = comparison_storage.atomic_binary

    def metadata_only(path):
        if path == part_path:
            forbidden()
        return atomic(path)

    monkeypatch.setattr(comparison_storage, "atomic_binary", metadata_only)
    compare.save_record(tmp_path, record)
    assert part_path.read_bytes() == before


def test_incomplete_site_rejected_before_outputs(tmp_path):
    source, checkpoint = _inputs(tmp_path)
    (source / "frame_128.png").unlink()
    config = compare.ComparisonSettings(checkpoints={"model": compare.ComparisonArm(checkpoint=checkpoint, registration="none", brightness="none")},
                                        sites={"site": compare.SiteSettings(source_dir=source)}, output_dir=tmp_path / "output")
    with pytest.raises(ValueError, match="exactly 128"):
        compare.run(config)
    assert not config.output_dir.exists()


def test_nonfinite_workflow_records_failed_prediction_without_png(tmp_path, monkeypatch):
    from sem_segment import pipeline

    source, checkpoint = _inputs(tmp_path)
    arm, training_config, digest = _arm_fixture(tmp_path)
    denoiser = SimpleNamespace(config=training_config, dataset_fingerprint=digest, image_size=16,
                               checkpoint_step=5, default_margin=0,
                               denoise_full=lambda frame, **kwargs: np.full_like(frame, np.nan))
    monkeypatch.setattr(Denoiser, "from_checkpoint", lambda *args, **kwargs: denoiser)
    monkeypatch.setattr(pipeline, "segment_image", lambda *args: SimpleNamespace(regions=[]))
    monkeypatch.setattr(compare, "registration_tracks", lambda reference, paths, sigma: [{} for _ in paths])
    monkeypatch.setattr(compare, "analyze_series", lambda *args: None)
    monkeypatch.setattr(compare, "measure_series", lambda *args: ([], []))
    config = compare.ComparisonSettings(checkpoints={"bad": arm},
                                        sites={"site": compare.SiteSettings(source_dir=source)}, output_dir=tmp_path / "output")
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        compare.run(config)
    audit = json.loads((config.output_dir / "comparison.json").read_text())
    assert audit["status"] == "failed"
    assert audit["prediction_ranges"][0]["status"] == "failed"
    assert "NaN" in audit["prediction_ranges"][0]["error"]
    assert not (config.output_dir / "site/bad/frame_001.png").exists()


@pytest.mark.parametrize("registration,brightness", TREATMENTS)
def test_checkpoint_treatment_checked_against_inline_settings(tmp_path, registration, brightness):
    arm, config, digest = _arm_fixture(tmp_path, registration, brightness)
    denoiser = SimpleNamespace(config=config, dataset_fingerprint=digest)
    metadata = compare.checkpoint_arm_metadata(arm, denoiser)
    assert (metadata["registration"], metadata["brightness"]) == (registration, brightness)
    assert metadata["settings_source"] == "checkpoint inline matching"
    incorrect = arm.model_copy(update={"brightness": "none" if brightness == "percentile" else "percentile"})
    with pytest.raises(ValueError, match="conflicts"):
        compare.checkpoint_arm_metadata(incorrect, denoiser)


def test_legacy_translation_not_mislabelled_as_none(tmp_path):
    arm, config, digest = _arm_fixture(tmp_path, "translation", "none", legacy=True)
    denoiser = SimpleNamespace(config=config, dataset_fingerprint=digest)
    metadata = compare.checkpoint_arm_metadata(arm, denoiser)
    assert metadata["registration"] == "translation"
    assert metadata["settings_source"] == "legacy prepared dataset"
    with pytest.raises(ValueError, match="conflicts"):
        compare.checkpoint_arm_metadata(arm.model_copy(update={"registration": "none"}), denoiser)
    denoiser.dataset_fingerprint = "different-manifest"
    with pytest.raises(ValueError, match="hash differs"):
        compare.checkpoint_arm_metadata(arm, denoiser)
    with pytest.raises(ValueError, match="prepared_manifest"):
        compare.checkpoint_arm_metadata(arm.model_copy(update={"prepared_manifest": None}), denoiser)


def test_different_models_keep_training_content_and_normalization_checks(tmp_path):
    arm, config, digest = _arm_fixture(tmp_path)
    denoiser = SimpleNamespace(config=config, dataset_fingerprint=digest)
    metadata = compare.checkpoint_arm_metadata(arm, denoiser)
    previous = {"none_none": {"config": config.model_dump(mode="json"), "arm": metadata}}
    altered = dict(metadata, raw_content_split_sha256="other")
    with pytest.raises(ValueError, match="same raw acquisition"):
        compare.validate_comparable_arm(altered, config.model_dump(mode="json"), previous)
    altered_config = config.model_copy(deep=True)
    altered_config.data.image_size = 32
    altered_config.model.ch = 16
    altered_config.objective.lambda_consistency = 1
    compare.validate_comparable_arm(metadata, altered_config.model_dump(mode="json"), previous)
    for field, value in (("channels", 3), ("black_level", 1), ("white_level", 200)):
        invalid = altered_config.model_dump(mode="json")
        invalid["data"][field] = value
        with pytest.raises(ValueError, match=field):
            compare.validate_comparable_arm(metadata, invalid, previous)
    config.objective.lambda_gradient = 4
    assert compare.checkpoint_arm_metadata(arm, denoiser)["raw_content_split_sha256"] == metadata["raw_content_split_sha256"]


@pytest.mark.parametrize("representation,target,image,gradient,consistency", [
    ("image", "noisy", 1, 4, 0), ("image", "noisy", 1, 4, 1),
    ("image", "noisy_mean", 1, 4, 1), ("gradient", "noisy", 0, 1, 0),
    ("hybrid", "noisy", 1, 4, 1),
])
def test_non_n2n_treatments_and_objectives_are_exported(tmp_path, representation, target, image, gradient, consistency):
    from edge_denoise.config import ObjectiveConfig
    from sem_noise.comparison_report import comparison_arms

    arm, config, digest = _arm_fixture(tmp_path, "affine", "percentile")
    config.objective = ObjectiveConfig(representation=representation, target=target,
        lambda_image=image, lambda_gradient=gradient, lambda_consistency=consistency)
    metadata = compare.checkpoint_arm_metadata(compare.ComparisonArm(checkpoint=arm.checkpoint,
        prepared_manifest=arm.prepared_manifest), SimpleNamespace(config=config, dataset_fingerprint=digest))
    assert (metadata["registration"], metadata["brightness"]) == ("affine", "percentile")
    metadata["evaluation_split"] = "val"
    exported = comparison_arms({"models": {"candidate": {"config": config.model_dump(mode="json"),
        "arm": metadata, "checkpoint": str(arm.checkpoint), "sha256": "weights", "step": 100, "ema": True}}})[0]
    assert (exported["representation"], exported["target"]) == (representation, target)
    assert exported["evaluation_split"] == "val"
    assert (exported["lambda_image"], exported["lambda_gradient"], exported["lambda_consistency"]) == (image, gradient, consistency)


def test_burst_model_cannot_silently_receive_single_frame_inputs(tmp_path):
    from edge_denoise.config import FusionConfig

    arm, config, digest = _arm_fixture(tmp_path)
    config.objective.fusion = FusionConfig(align="none")
    with pytest.raises(ValueError, match="burst fusion"):
        compare.checkpoint_arm_metadata(arm, SimpleNamespace(config=config, dataset_fingerprint=digest))


def test_shipped_comparison_is_the_six_preprocessing_treatments():
    config = compare.load_settings(compare.ROOT / "edge_denoise/configs/sem_real_compare.yml")
    assert [(a.registration, a.brightness) for a in config.checkpoints.values()] == TREATMENTS
    assert all("260921_real_n2n_" in str(a.checkpoint) for a in config.checkpoints.values())
    assert config.metrology_device == "cuda:0"
    assert str(next(iter(config.sites.values())).source_dir).replace("\\", "/").endswith("/data/260904_raw_data/test/260904_0947-13")


def test_output_registration_failure_never_changes_raw_correspondence(tmp_path, monkeypatch):
    frames = [{"path": "a.png", "dy_px": 2., "dx_px": -1., "registration_status": "registered"},
              {"path": "b.png", "dy_px": None, "dx_px": None, "registration_status": "failed"}]
    monkeypatch.setattr(compare, "registration_tracks", lambda *args: [
        {"dy_px": None, "dx_px": None, "registration_status": "failed", "registration_score": None, "registration_error": "flat"},
        {"dy_px": 3., "dx_px": 4., "registration_status": "registered", "registration_score": .9, "registration_error": ""}])
    compare.output_registration_diagnostics(tmp_path, np.zeros((16, 16)), {"frames": frames}, 1.)
    assert frames[0]["dy_px"] == 2 and frames[1]["dy_px"] is None
    assert all(f["output_minus_raw_dy_px"] is None for f in frames)
    assert frames[0]["output_registration_error"] == "flat"


def test_live_examples_leave_training_rng_and_test_sites_untouched(tmp_path):
    from edge_denoise.real_comparison import prepare_examples, log_examples

    class ForbiddenTest:
        @property
        def frames(self):
            raise AssertionError("test pixels were accessed")

    raw = np.arange(8 * 24 * 24, dtype=np.uint8).reshape(8, 24, 24)
    cache = SimpleNamespace(real_metadata={"sites": [{"source_index": 0, "bounds": [0, 0, 24, 24]},
                                                     {"source_index": 1, "bounds": [0, 0, 24, 24]}]},
                            train_sources=[SimpleNamespace(source_index=0, frames=raw)],
                            val_sources=[SimpleNamespace(source_index=1, frames=raw)], test_sources=[ForbiddenTest()])
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    examples = prepare_examples(cache, 16)
    assert len(examples) == 8
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch_state, torch.get_rng_state())
    expected = prepare_examples(cache, 16)
    for left, right in zip(examples, expected):
        np.testing.assert_array_equal(left["average8"], right["average8"])

    class RandomModel(torch.nn.Module):
        def predict_image(self, x):
            return x * 3 + torch.rand_like(x)  # Deliberately consume RNG to verify isolation.

    model, writer = RandomModel(), CaptureWriter()
    log_examples(model, examples, writer, 99, device="cpu", black=0, white=255)
    assert model.training
    assert torch.equal(torch_state, torch.get_rng_state())
    assert len(writer.images) == 2
    assert any(RANGE_WARNING in value for value in writer.texts.values())


def test_live_panels_preserve_actual_training_samples_and_weights(tmp_path):
    from edge_denoise.real_data import prepare_real_dataset
    from edge_denoise.train import Trainer, load_checkpoint

    rng = np.random.default_rng(7)
    source = tmp_path / "source"
    for site in range(2):
        for index in range(8):
            compare.save_rgb(source / f"site{site}" / f"frame{index}.png",
                             rng.integers(30, 220, (24, 24), dtype=np.uint8))
    dataset = tmp_path / "prepared"
    prepare_real_dataset(source, dataset, image_size=16, align="none", val_fraction=.5, test_fraction=0)
    states = []
    for enabled in (False, True):
        config = make_config(dataset, tmp_path / f"run_{enabled}", representation="image", lambda_gradient=0, max_steps=2)
        config.training.val_every = 1
        config.training.real_comparison_images = enabled
        trainer = Trainer(config)
        trainer.run()
        states.append(load_checkpoint(trainer.latest_checkpoint_path, map_location="cpu"))
    for key in states[0]["model"]:
        assert torch.equal(states[0]["model"][key], states[1]["model"][key]), key
    assert states[0]["factory"] == states[1]["factory"]
    assert torch.equal(states[0]["torch_rng"], states[1]["torch_rng"])
