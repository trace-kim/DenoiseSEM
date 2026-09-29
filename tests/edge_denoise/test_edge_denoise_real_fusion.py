"""Real-SEM burst fusion (objective.fusion.align: matched): sampler, training,
inference and the saved-uint8 comparison series."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from burst_diffusion.data import BurstCache
from burst_diffusion.real_data import normalize_native
from edge_denoise.cli import app
from edge_denoise.config import Config, FusionConfig, load_config
from edge_denoise.infer import Denoiser
from edge_denoise.real_data import prepare_real_dataset
from edge_denoise.real_fusion import (MatchedRealFusionFactory, fused_input, registered_mean,
                                      registered_mean_frame)
from edge_denoise.real_matching import MatchedRealPairFactory
from edge_denoise.train import Trainer, load_checkpoint
from edge_denoise.uint8_output import prediction_uint8
from sem_noise.pair_matching import measure_quantile_brightness
from test_edge_denoise_real import prepared, real_config  # noqa: F401 -- fixture
from test_real_matching import matching_config
from tools import real_sem_averages as averages
from tools import real_sem_compare as compare


def fusion_config(dataset, run, *, levels=(1, 2, 3), registration="none", brightness="percentile",
                  **fusion) -> Config:
    raw = real_config(dataset, run).model_dump()
    raw["data"]["real_matching"] = {"registration": registration, "brightness": brightness}
    raw["objective"]["fusion"] = {"align": "matched", "frames_per_burst": max(levels) + 1,
                                  "levels": list(levels), **fusion}
    return Config.model_validate(raw)


def structured(shape=(40, 48), offset=0) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (60 + yy * 2 + xx + offset) % 256


@pytest.fixture
def prepared_uint8(tmp_path: Path) -> Path:
    rng = np.random.default_rng(3)
    for site in range(4):
        folder = tmp_path / "raw8" / f"site_{site}"
        folder.mkdir(parents=True)
        for frame in range(8):
            pixels = np.clip(structured(offset=site * 20) + rng.integers(-12, 13, (40, 48)), 0, 255)
            Image.fromarray(pixels.astype(np.uint8)).save(folder / f"frame_{frame}.png")
    prepare_real_dataset(tmp_path / "raw8", tmp_path / "prepared8", image_size=16, align="none")
    return tmp_path / "prepared8"


@pytest.fixture
def fusion_checkpoint(prepared_uint8: Path, tmp_path: Path) -> Path:
    return Trainer(fusion_config(prepared_uint8, tmp_path / "fusion_run")).run()


def test_single_frame_level_is_exactly_the_matched_n2n_pair(prepared, tmp_path) -> None:
    cache = BurstCache(prepared)
    fusion = MatchedRealFusionFactory(cache, fusion_config(prepared, tmp_path / "f"), seed=3)
    pair = MatchedRealPairFactory(cache, matching_config(prepared, tmp_path / "p"), seed=3)
    source = cache.train_sources[0]
    for a, b in ((0, 5), (3, 1), (7, 0)):
        batch, infos = fusion._batch([fusion._fusion_sample(source, (8, 8), (a,), b)])
        expected, _ = pair._batch([pair._pair(source, (8, 8), a, b, None)])
        assert torch.equal(batch.inputs, expected.inputs)
        assert torch.equal(batch.targets, expected.targets)
        assert torch.equal(batch.target_valid, expected.target_valid)
        assert batch.levels.tolist() == [1.0] and infos[0].anchor_replica == a and infos[0].target_replica == b


def test_m_frame_input_is_brightness_matched_mean_of_consecutive_frames(prepared, tmp_path) -> None:
    cache = BurstCache(prepared)
    factory = MatchedRealFusionFactory(cache, fusion_config(prepared, tmp_path / "run"), seed=0)
    source = cache.train_sources[0]
    frames = source.frames
    batch, infos = factory._batch([factory._fusion_sample(source, (8, 8), (2, 3, 4), 6)])
    crop = np.s_[8:24, 8:24]
    expected = [normalize_native(frames[2][crop], 0, 4095)]
    for j in (3, 4):
        fit = measure_quantile_brightness(frames[2], frames[j])
        expected.append((fit["gain"] * frames[j][crop].astype(float) + fit["offset_dn"]) / 4095)
    np.testing.assert_allclose(batch.inputs[0, 0], np.mean(expected, axis=0) * 2 - 1, atol=2e-6)
    assert batch.levels.tolist() == [3.0] and batch.target_valid.all()
    assert (infos[0].anchor_replica, infos[0].target_replica, infos[0].members) == (2, 6, (2, 3, 4))


def test_sampled_subsets_are_consecutive_exclude_target_and_resume(prepared, tmp_path) -> None:
    cache = BurstCache(prepared)
    factory = MatchedRealFusionFactory(cache, fusion_config(prepared, tmp_path / "run"), seed=11)
    batch, infos = factory.sample_batch(count=64, return_info=True)
    assert {len(info.members) for info in infos} == {1, 2, 3}
    for info, level in zip(infos, batch.levels.tolist()):
        assert list(info.members) == list(range(info.members[0], info.members[0] + len(info.members)))
        assert info.target_replica not in info.members and level == len(info.members) == info.level
        assert info.anchor_replica == info.members[0]
    state = factory.state_dict()
    expected = factory.sample_batch()
    factory.load_state_dict(state)
    actual = factory.sample_batch()
    for name in ("inputs", "targets", "target_valid", "levels"):
        assert torch.equal(getattr(expected, name), getattr(actual, name))


def test_dose_blind_ablation_feeds_the_single_frame_constant(prepared, tmp_path) -> None:
    cfg = fusion_config(prepared, tmp_path / "run", condition_on_level=False)
    factory = MatchedRealFusionFactory(BurstCache(prepared), cfg, seed=0)
    assert set(factory.sample_batch(count=16).levels.tolist()) == {1.0}


def test_anchor_is_first_member_with_usable_geometry(prepared, tmp_path) -> None:
    cache = BurstCache(prepared)
    factory = MatchedRealFusionFactory(cache, fusion_config(prepared, tmp_path / "run"), seed=0)
    source = cache.train_sources[0]
    factory.geometry_available[source.source_index][2] = False
    spec, *_, info = factory._fusion_sample(source, (8, 8), (2, 3, 4), 6)
    assert info.anchor_replica == 3 and spec.anchor == 1
    np.testing.assert_array_equal(spec.matrices[0], np.eye(2, 3))  # Unmeasured member stays native.
    factory.geometry_available[source.source_index][:] = False
    assert factory._fusion_sample(source, (8, 8), (2, 3, 4), 6)[-1].anchor_replica == 2


def _shifted_members(shifts):
    canvas = np.random.default_rng(4).integers(0, 4096, (60, 70)).astype(np.uint16)
    anchor = canvas[8:48, 8:56]
    frames, matrices = [anchor], [np.eye(2, 3)]
    for dy, dx in shifts:
        y, x = round(dy), round(dx)  # Content only needs the right shape for fractional matrices.
        frames.append(canvas[8 - y:48 - y, 8 - x:56 - x])
        matrices.append(np.array([[1.0, 0, dx], [0, 1.0, dy]]))
    return frames, np.stack(matrices)


def test_registered_mean_of_aligned_copies_is_the_anchor_and_marks_missing_support() -> None:
    frames, matrices = _shifted_members([(2, 1), (-3, 2)])
    brightness = np.tile([1.0, 0.0], (3, 1))
    mean, valid = registered_mean(frames, 0, matrices, brightness, (8, 8), 16, 0, 4095, "cpu")
    np.testing.assert_allclose(mean, normalize_native(frames[0][8:24, 8:24], 0, 4095), atol=1e-6)
    assert valid.all()
    mean, valid = registered_mean(frames, 0, matrices, brightness, (0, 0), 16, 0, 4095, "cpu")
    # The (-3, 2) member has no cubic support in the top rows, the (2, 1) one
    # none in the first column: those pixels average the members that do and
    # are excluded from the full-support mask.
    np.testing.assert_allclose(mean, normalize_native(frames[0][:16, :16], 0, 4095), atol=1e-6)
    assert not valid[:5].any() and not valid[:, 0].any() and valid[5:, 1:].all()


def test_full_frame_mean_matches_training_crops_across_tiles() -> None:
    frames, matrices = _shifted_members([(0.25, -0.5), (1, 2)])
    brightness = np.array([[1.0, 0.0], [1.05, -0.01], [0.97, 0.02]])
    full, support = registered_mean_frame(frames, 0, matrices, brightness, 0, 4095, "cpu", tile=16)
    for origin in ((0, 0), (8, 8), (24, 32), (13, 5)):
        crop, valid = registered_mean(frames, 0, matrices, brightness, origin, 16, 0, 4095, "cpu")
        window = np.s_[origin[0]:origin[0] + 16, origin[1]:origin[1] + 16]
        np.testing.assert_allclose(full[window], crop, atol=1e-6)
        np.testing.assert_array_equal(support[window], valid)


def test_fused_input_anchors_on_first_textured_frame_and_matches_input_brightness(monkeypatch) -> None:
    from sem_noise import pair_matching

    monkeypatch.setattr(pair_matching, "estimate_geometry", lambda *a, **k: (np.eye(2, 3), 1.0))
    frames = np.stack([np.full((40, 48), 100, np.uint8), structured(), structured() // 2 + 20]).astype(np.uint8)
    mean, valid, record = fused_input(frames, black=0, white=255, registration="affine",
                                      brightness="percentile", device="cpu")
    assert record["anchor_frame"] == 2 and record["geometry"][0]["status"] == "skipped_low_contrast"
    assert record["brightness_to_anchor"][1] == [1.0, 0.0]
    assert record["brightness_to_anchor"][0] == [1.0, 0.0]  # Flat frame: gain unmeasurable, native kept.
    gain, offset = record["brightness_to_anchor"][2]
    assert gain == pytest.approx(2, rel=0.02)
    expected = (100 / 255 + structured() / 255 + gain * (frames[2] / 255) + offset) / 3
    np.testing.assert_allclose(mean[valid], expected[valid], atol=1e-6)
    with pytest.raises(ValueError, match="affine or none"):
        fused_input(frames, black=0, white=255, registration="translation", brightness="none", device="cpu")


def test_config_rejects_unusable_or_silently_ignored_real_fusion_settings(prepared, tmp_path) -> None:
    with pytest.raises(ValueError, match="requires data.real_matching"):
        Config.model_validate({**real_config(prepared, tmp_path / "r").model_dump(), "objective": {
            **real_config(prepared, tmp_path / "r").model_dump()["objective"],
            "fusion": {"align": "matched", "frames_per_burst": 4, "levels": [1, 2, 3]}}})
    with pytest.raises(ValueError, match="does not use warp_margin"):
        FusionConfig(align="matched", frames_per_burst=4, levels=[1, 2, 3], warp_margin=3)
    saved = FusionConfig(align="matched", frames_per_burst=4, levels=[1, 2, 3]).model_dump(mode="json")
    assert FusionConfig.model_validate(saved).align == "matched"  # Checkpoint configs spell out defaults.
    with pytest.raises(ValueError, match="target 'frame' only"):
        FusionConfig(align="matched", frames_per_burst=4, levels=[1, 2, 3], target="complement_mean")
    unmatched = fusion_config(prepared, tmp_path / "run")
    unmatched.objective.fusion = FusionConfig(align="none", frames_per_burst=4, levels=[1, 2, 3])
    with pytest.raises(ValueError, match="requires objective.fusion.align 'matched'"):
        Trainer(unmatched)


def test_shipped_recipe_is_fresh_t16_l2_on_the_real_n2n_backbone() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "edge_denoise/configs/sem_real_burst_t16.yml")
    teacher = load_config(root / "edge_denoise/configs/sem_real_n2n.yml")
    fusion = cfg.objective.fusion
    assert fusion.align == "matched" and fusion.levels == list(range(1, 17)) and cfg.min_replicas == 17
    assert cfg.training.init_checkpoint is None and cfg.training.max_steps == teacher.training.max_steps
    assert (cfg.objective.lambda_image, cfg.objective.lambda_gradient) == (1.0, 0.0)
    assert cfg.model == teacher.model and cfg.data.image_size == teacher.data.image_size
    assert (cfg.data.real_matching.registration, cfg.data.real_matching.brightness) == ("affine", "percentile")
    assert cfg.training.batch_size * cfg.training.accumulation_steps == 16


class CaptureWriter:
    def __init__(self, **kwargs):
        self.scalars, self.images = {}, {}

    def add_scalar(self, key, value, step):
        self.scalars[key, step] = value

    def add_image(self, key, value, step, **kwargs):
        self.images[key, step] = value

    def close(self):
        pass


@pytest.mark.parametrize("registration", ["affine", "none"])
def test_trains_validates_per_level_and_resumes(prepared, tmp_path, monkeypatch, registration) -> None:
    from edge_denoise import train as train_module
    from sem_noise import pair_matching

    monkeypatch.setattr(pair_matching, "estimate_geometry", lambda *a, **k: (np.eye(2, 3), 1.0))
    writer = CaptureWriter()
    monkeypatch.setattr(train_module, "SummaryWriter", lambda **k: writer)
    cfg = fusion_config(prepared, tmp_path / "run", registration=registration)
    trainer = Trainer(cfg)
    checkpoint = trainer.run()
    status = json.loads((cfg.training.run_dir / "training_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and status["step"] == 2
    assert {("val/loss_m01", 2), ("val/loss_m03", 2), ("val/loss_image_m03", 2)} <= writer.scalars.keys()
    assert ("val/input_pred_target_m03", 2) in writer.images
    assert "fusion_input" in json.loads((cfg.training.run_dir / "timings.json").read_text(encoding="utf-8"))["stage_seconds"]
    saved = load_checkpoint(checkpoint)
    assert saved["config"]["objective"]["fusion"]["align"] == "matched"
    resumed = Trainer(cfg, resume_from=checkpoint)
    expected, actual = trainer.factory.sample_batch(), resumed.factory.sample_batch()
    for name in ("inputs", "targets", "levels"):
        assert torch.equal(getattr(expected, name), getattr(actual, name))


def test_denoiser_conditions_on_frame_count_and_refuses_untrained_levels(fusion_checkpoint, prepared_uint8) -> None:
    denoiser = Denoiser.from_checkpoint(fusion_checkpoint, device="cpu")
    assert denoiser.is_fusion and denoiser.level_for(3) == 3.0
    with pytest.raises(ValueError, match="exceeds the trained levels"):
        denoiser.level_for(4)
    seen = []
    predict = denoiser.model.predict_image
    denoiser.model.predict_image = lambda x, t=None: (seen.append(None if t is None else set(t.tolist())), predict(x, t))[1]
    frames = np.asarray(BurstCache(prepared_uint8).train_sources[0].frames[:3])
    single, valid, record = denoiser.denoise_frames(frames[:1], stride=8, tile_batch=4)
    assert valid.all() and record["level"] == 1.0 and seen[-1] == {1.0}
    np.testing.assert_array_equal(single, denoiser.denoise_full(frames[0] / 255, stride=8, tile_batch=4, level=1.0))
    fused, valid, record = denoiser.denoise_frames(frames, stride=8, tile_batch=4)
    assert fused.shape == frames.shape[1:] and record["level"] == 3.0 and seen[-1] == {3.0}
    assert record["frames"] == 3 and record["anchor_frame"] == 1


def test_denoise_cli_fuses_repeated_inputs_into_one_saved_uint8_image(fusion_checkpoint, prepared_uint8, tmp_path) -> None:
    raw = sorted((prepared_uint8.parent / "raw8" / "site_0").glob("*.png"))[:2]
    out = tmp_path / "fused"
    result = CliRunner().invoke(app, ["denoise", "--checkpoint", str(fusion_checkpoint), "--out", str(out),
                                      "--input", str(raw[0]), "--input", str(raw[1]), "--device", "cpu",
                                      "--stride", "8"])
    assert result.exit_code == 0, (result.output, result.exception)
    record = json.loads((out / "frame_0_fuse2_fusion.json").read_text(encoding="utf-8"))
    assert record["level"] == 2.0 and record["inputs"] == [str(raw[0]), str(raw[1])]
    denoiser = Denoiser.from_checkpoint(fusion_checkpoint, device="cpu")
    frames = np.stack([np.asarray(Image.open(path)) for path in raw])
    prediction, _, _ = denoiser.denoise_frames(frames, stride=8, tile_batch=4, clip_output=False)
    expected, _ = prediction_uint8(prediction, 0, 255)
    np.testing.assert_array_equal(np.asarray(Image.open(out / "frame_0_fuse2_denoised.png"))[..., 0], expected)


def test_single_frame_checkpoint_refuses_multiple_inputs(prepared_uint8, tmp_path) -> None:
    raw = real_config(prepared_uint8, tmp_path / "n2n").model_dump()
    raw["data"]["real_matching"] = {"registration": "none", "brightness": "none"}
    checkpoint = Trainer(Config.model_validate(raw)).run()
    paths = sorted((prepared_uint8.parent / "raw8" / "site_0").glob("*.png"))[:2]
    result = CliRunner().invoke(app, ["denoise", "--checkpoint", str(checkpoint), "--out", str(tmp_path / "o"),
                                      "--input", str(paths[0]), "--input", str(paths[1]), "--device", "cpu"])
    assert result.exit_code != 0 and "burst-fusion checkpoint" in result.output
    assert not compare.checkpoint_is_fusion(checkpoint)


def _raw_site(root: Path, frames: np.ndarray) -> dict:
    rows = []
    for i, pixels in enumerate(frames):
        path = Path("site") / "raw" / f"frame_{i + 1:03d}.png"
        compare.save_rgb(root / path, pixels)
        rows.append({"index": i + 1, "order": i + 1, "timestamp_s": None, "path": path.as_posix()})
    return {"name": "site", "series": {"raw": {"frames": rows, "step": 0}}}


def test_fused_series_use_average_blocks_and_saved_uint8_outputs(fusion_checkpoint, prepared_uint8, tmp_path) -> None:
    denoiser = Denoiser.from_checkpoint(fusion_checkpoint, device="cpu")
    assert compare.checkpoint_is_fusion(fusion_checkpoint)
    frames = np.asarray(BurstCache(prepared_uint8).train_sources[0].frames)
    root = tmp_path / "report"
    site = _raw_site(root, frames)
    ranges = []
    added = averages.add_fused_series(root, site, "burst", denoiser, [2, 3], stride=8, tile_batch=4,
                                      prediction_ranges=ranges)
    assert added == ["burst_fuse2", "burst_fuse3"]
    fuse3 = site["series"]["burst_fuse3"]
    assert (fuse3["frames_per_output"], fuse3["family"], fuse3["remainder_frames"], len(fuse3["frames"])) == (3, "burst", 2, 2)
    assert [(f["first_acquisition"], f["last_acquisition"]) for f in fuse3["frames"]] == [(1, 3), (4, 6)]
    assert len(ranges) == 4 + 2 and all(r["status"] == "complete" for r in ranges)
    block = frames[3:6]
    prediction, _, record = denoiser.denoise_frames(block, stride=8, tile_batch=4, clip_output=False)
    expected, _ = prediction_uint8(prediction, 0, 255)
    np.testing.assert_array_equal(compare.read_uint8(root / fuse3["frames"][1]["path"]), expected)
    assert fuse3["frames"][1]["fusion_input"]["level"] == record["level"] == 3.0
    assert averages.add_fused_series(root, site, "burst", denoiser, [2], stride=8, tile_batch=4,
                                     prediction_ranges=ranges) == []  # Existing series are kept.
    site["series"]["raw"]["frames"][0]["common_support_path"] = "support.png"
    with pytest.raises(ValueError, match="raw acquisitions"):
        averages.add_fused_series(root, site, "other", denoiser, [2], stride=8, tile_batch=4, prediction_ranges=[])


def test_comparison_settings_accept_matched_fusion_arms_and_reserve_fused_names(tmp_path) -> None:
    from test_real_sem_compare import _arm_fixture

    arm, config, digest = _arm_fixture(tmp_path, "affine", "percentile")
    config.objective.fusion = FusionConfig(align="matched", frames_per_burst=4, levels=[1, 2, 3])
    metadata = compare.checkpoint_arm_metadata(arm, type("D", (), {"config": config, "dataset_fingerprint": digest})())
    assert (metadata["registration"], metadata["brightness"]) == ("affine", "percentile")
    args = compare.build_parser().parse_args(["--fusion-frames", "16,2,8,4"])
    assert compare.configure_run(args).fusion_frames == [2, 4, 8, 16]
    with pytest.raises(ValueError, match="fusion_frames"):
        compare.ComparisonSettings(checkpoints={}, sites={}, output_dir=tmp_path / "o", fusion_frames=[1])
    site = tmp_path / "site"
    site.mkdir()
    settings = compare.ComparisonSettings(checkpoints={"burst_fuse4": arm}, sites={"s": site}, output_dir=tmp_path / "o")
    with pytest.raises(ValueError, match="reserved"):
        compare.validate_inputs(settings)


def test_equivalence_check_builds_one_fused_output_per_backend(fusion_checkpoint, prepared_uint8, tmp_path) -> None:
    from tools import check_real_sem_equivalence as equivalence

    paths = sorted((prepared_uint8.parent / "raw8" / "site_0").glob("*.png"))[:3]
    result = equivalence.check_checkpoint(fusion_checkpoint, paths, tmp_path / "eq", device="cpu", fuse_frames=3)
    assert result["passed"] and len(result["frames"]) == 1
    assert result["frames"][0]["max_abs_difference_dn"] == 0
    assert (tmp_path / "eq" / "candidate" / "fuse3.png").is_file()
    with pytest.raises(ValueError, match="--frames >= --fuse-frames"):
        equivalence.check_checkpoint(fusion_checkpoint, paths[:2], tmp_path / "eq2", device="cpu", fuse_frames=3)
