from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from typer.testing import CliRunner

from burst_diffusion.data import BurstCache, BatchFactory
from burst_diffusion.real_data import REAL_MANIFEST, normalize_native
from edge_denoise.cli import app
from edge_denoise.config import Config
from edge_denoise.real_data import RealPairFactory, estimate_translations, prepare_real_dataset, sample_region
from edge_denoise.real_evaluate import _metrics, frame_pools
from edge_denoise.train import Trainer, load_checkpoint


def write_raw(root: Path, *, sites: int = 4, frames: int = 8, high: bool = False) -> Path:
    rng = np.random.default_rng(5)
    for site in range(sites):
        folder = root / f"site_{site}"
        folder.mkdir(parents=True)
        yy, xx = np.mgrid[:40, :48]
        signal = (60000 if high else 500) + yy * 5 + xx * 10 + site * 250
        for frame in range(frames):
            pixels = (signal + rng.integers(-80, 81, signal.shape) + frame).astype(np.uint16)
            Image.fromarray(pixels).save(folder / f"frame_{frame}.tif")
    return root


def real_config(dataset: Path, run: Path, *, target: str = "noisy", consistency: float = 0, gradient: float = 0) -> Config:
    return Config.model_validate({
        "data": {"dataset_dir": dataset, "image_size": 16},
        "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
        "objective": {"representation": "image", "target": target, "lambda_image": 1,
                      "lambda_gradient": gradient, "lambda_consistency": consistency},
        "training": {"run_dir": run, "batch_size": 2, "max_steps": 2, "device": "cpu",
                     "val_every": 2, "val_images": 2, "checkpoint_every": 2},
    })


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    raw = write_raw(tmp_path / "raw")
    prepare_real_dataset(raw, tmp_path / "prepared", image_size=16, align="none", white_level=4095)
    return tmp_path / "prepared"


def test_prepare_preserves_native_values_and_freezes_content_splits(prepared: Path) -> None:
    cache = BurstCache(prepared, min_size=16)
    assert cache.real_metadata is not None
    assert (len(cache.train_sources), len(cache.val_sources), len(cache.test_sources)) == (2, 1, 1)
    assert cache.summary()["ram_bytes"] == 0
    source = cache.all_sources[0]
    assert source.clean is None
    assert isinstance(source.frames, np.memmap)
    assert source.frames.dtype == np.uint16
    expected = np.asarray(Image.open(prepared.parent / "raw" / f"site_{source.source_index}" / "frame_0.tif"))
    np.testing.assert_array_equal(source.frames[0], expected)
    assert (prepared / "qc.csv").is_file()
    assert len(list((prepared / "previews").glob("*.png"))) == 4
    # Training config fractions cannot accidentally redraw a prepared split.
    other = BurstCache(prepared, val_fraction=0, test_fraction=0, split_seed=99)
    assert [s.source_index for s in cache.test_sources] == [s.source_index for s in other.test_sources]
    with pytest.raises(ValueError, match="registration-aware"):
        BatchFactory(cache, num_steps=1, image_size=16, batch_size=1)


def test_native_patch_and_leave_one_out_mean_match_measured_pixels(prepared: Path, tmp_path: Path) -> None:
    cache = BurstCache(prepared)
    cfg = real_config(prepared, tmp_path / "run", target="noisy_mean", consistency=1, gradient=4)
    factory = RealPairFactory(cache, cfg, seed=10)
    batch, info = factory.sample_batch(count=5, return_info=True)
    for index, sample in enumerate(info):
        source = next(s for s in cache.train_sources if s.source_index == sample.source_index)
        y, x = sample.crop_yx
        crops = normalize_native(source.frames[:, y:y + 16, x:x + 16], 0, 4095)
        np.testing.assert_allclose(batch.inputs[index, 0], crops[sample.input_replica] * 2 - 1, atol=1e-7)
        expected = np.mean(np.delete(crops, sample.input_replica, axis=0), axis=0) * 2 - 1
        np.testing.assert_allclose(batch.targets[index, 0], expected, atol=3e-7)
        assert sample.input_replica != sample.second_replica
    state = factory.state_dict()
    expected = factory.sample_batch()
    factory.load_state_dict(state)
    actual = factory.sample_batch()
    assert torch.equal(expected.inputs, actual.inputs)
    assert torch.equal(expected.targets, actual.targets)
    assert factory.val_batch(count=3).clean is None


def test_128_bright_16bit_frames_do_not_overflow(tmp_path: Path) -> None:
    raw = write_raw(tmp_path / "raw", sites=1, frames=128, high=True)
    prepare_real_dataset(raw, tmp_path / "prepared", image_size=16, align="none", val_fraction=0, test_fraction=0)
    cache = BurstCache(tmp_path / "prepared")
    factory = RealPairFactory(cache, real_config(tmp_path / "prepared", tmp_path / "run", target="noisy_mean"), seed=1)
    batch, info = factory.sample_batch(count=1, return_info=True)
    sample, source = info[0], cache.train_sources[0]
    y, x = sample.crop_yx
    crops = source.frames[:, y:y + 16, x:x + 16].astype(np.float64) / 65535
    expected = np.delete(crops, sample.input_replica, axis=0).mean(axis=0) * 2 - 1
    np.testing.assert_allclose(batch.targets[0, 0], expected, atol=1e-6)


def test_registered_pairs_keep_inputs_native_and_align_target_and_second(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw" / "site"
    raw.mkdir(parents=True)
    yy, xx = np.mgrid[:40, :48]
    shifts = np.array([[0, 0], [0.25, -0.5], [-0.25, 0.5]])
    for index, (dy, dx) in enumerate(shifts):
        Image.fromarray(np.rint(10000 + 400 * (yy - dy) + 100 * (xx - dx)).astype(np.uint16)).save(raw / f"{index}.tif")
    monkeypatch.setattr("edge_denoise.real_data.estimate_translations", lambda *args, **kwargs: (shifts, np.zeros_like(shifts)))
    prepare_real_dataset(raw.parent, tmp_path / "prepared", image_size=16, val_fraction=0, test_fraction=0)
    cache = BurstCache(tmp_path / "prepared")
    cfg = real_config(tmp_path / "prepared", tmp_path / "run", consistency=1)
    factory = RealPairFactory(cache, cfg, seed=0)
    sample = factory._pair(cache.train_sources[0], (10, 12), 0, 1, 2)
    inputs, target, second, shift, info = sample
    expected = normalize_native(cache.train_sources[0].frames[0][10:26, 12:28], 0, 65535) * 2 - 1
    np.testing.assert_array_equal(inputs[0], expected)
    # Bicubic's fractional coordinate interpolation is approximate; check
    # the shift sign and subpixel geometry against this analytic plane.
    np.testing.assert_allclose(target[0, 3:-3, 3:-3], expected[3:-3, 3:-3], atol=0.001)
    np.testing.assert_allclose(shift, -shifts[2])
    assert info.input_replica != info.target_replica != info.second_replica
    mean_factory = RealPairFactory(cache, real_config(tmp_path / "prepared", tmp_path / "mean", target="noisy_mean"), seed=0)
    mean_pair = mean_factory._pair(cache.train_sources[0], (10, 12), 1, 0, None)
    registered = mean_factory.aligned[0]
    expected_reference = np.mean(registered[[0, 2]], axis=0)
    origin = np.rint(np.array([10, 12]) + shifts[1]) - shifts[1]
    expected_target = sample_region(expected_reference, *origin, 16) * 2 - 1
    np.testing.assert_allclose(mean_pair[1][0], expected_target, atol=3e-7)


@pytest.mark.parametrize("target,gradient,consistency", [("noisy", 0, 0), ("noisy_mean", 0, 0), ("noisy_mean", 4, 0), ("noisy_mean", 4, 1)])
def test_real_objectives_train_validate_and_resume(prepared: Path, tmp_path: Path, target, gradient, consistency) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = real_config(prepared, tmp_path / "run", target=target, gradient=gradient, consistency=consistency)
    trainer = Trainer(cfg)
    checkpoint = trainer.run()
    payload = load_checkpoint(checkpoint)
    assert payload["config"]["data"]["white_level"] == 4095
    assert payload["dataset_fingerprint"] == trainer.cache.real_fingerprint
    events = EventAccumulator(str(tmp_path / "run" / "tb")).Reload()
    assert "val/loss" in events.Tags()["scalars"]
    assert "val/psnr" not in events.Tags()["scalars"]
    assert "val/input_pred_target" in events.Tags()["images"]
    resumed = Trainer(cfg, resume_from=checkpoint)
    assert resumed.step == trainer.step
    assert torch.equal(trainer.factory.sample_batch().inputs, resumed.factory.sample_batch().inputs)


def test_real_cli_preparation_training_inference_and_evaluation(tmp_path: Path) -> None:
    runner = CliRunner()
    raw = write_raw(tmp_path / "raw")
    result = runner.invoke(app, ["prepare-real", "--source-dir", str(raw), "--out", str(tmp_path / "prepared"),
                                 "--image-size", "16", "--align", "none", "--white-level", "4095"])
    assert result.exit_code == 0, (result.output, result.exception)
    cfg = real_config(tmp_path / "prepared", tmp_path / "unused")
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(config_file), "--run-dir", str(tmp_path / "run"),
                                 "--max-steps", "1", "--accumulation-steps", "2"])
    assert result.exit_code == 0, (result.output, result.exception)
    checkpoint = tmp_path / "run" / "ckpt_latest.pt"
    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text())
    assert provenance["dataset"]["kind"] == "real_sem"
    assert provenance["config"]["data"]["white_level"] == 4095
    for mode in ([], ["--center-crop"]):
        out = tmp_path / ("crop" if mode else "full")
        result = runner.invoke(app, ["denoise", "--checkpoint", str(checkpoint), "--input", str(raw / "site_0" / "frame_0.tif"),
                                     "--out", str(out), "--device", "cpu", *mode])
        assert result.exit_code == 0, (result.output, result.exception)
        with Image.open(out / "frame_0_denoised.tif") as image:
            assert image.mode == "F"
            assert image.size == ((16, 16) if mode else (48, 40))
    result = runner.invoke(app, ["evaluate-real", "--config", str(config_file), "--checkpoint", f"n2n={checkpoint}",
                                 "--out", str(tmp_path / "eval"), "--input-frames", "3", "--rois", "2", "--device", "cpu"])
    assert result.exit_code == 0, (result.output, result.exception)
    results = json.loads((tmp_path / "eval" / "results.json").read_text())
    assert results["site_count"] == 1
    assert set(results["methods"]) == {"single_frame", "n2n"}
    for site in results["sites"]:
        assert not set(site["input_indices"]) & set(site["reference_indices"])
        assert len(site["rois"]) == 2
    assert "not clean ground truth" in results["reference"]


def test_real_data_rejects_changed_pixels_and_overwrite(prepared: Path) -> None:
    with pytest.raises(ValueError, match="new directory"):
        prepare_real_dataset(prepared.parent / "raw", prepared)
    path = prepared / "arrays" / "00000_raw.npy"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="content changed"):
        BurstCache(prepared)


def test_duplicate_content_cannot_cross_manual_split(tmp_path: Path) -> None:
    raw = write_raw(tmp_path / "raw", sites=3)
    (raw / "site_1" / "frame_0.tif").write_bytes((raw / "site_0" / "frame_0.tif").read_bytes())
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"site_0": "train", "site_1": "val", "site_2": "test"}))
    with pytest.raises(ValueError, match="duplicate image content"):
        prepare_real_dataset(raw, tmp_path / "prepared", image_size=16, align="none", split_file=splits)
    assert not (tmp_path / "prepared").exists()


def test_unsupported_real_objectives_fail_before_training(prepared: Path, tmp_path: Path) -> None:
    cfg = real_config(prepared, tmp_path / "run")
    raw = cfg.model_dump()
    raw["objective"]["target"] = "clean"
    with pytest.raises(ValueError, match="synthetic oracle"):
        Trainer(Config.model_validate(raw))


def test_reference_frame_pools_and_erased_feature_failures() -> None:
    from burst_diffusion.repeatability import find_cd_sites

    inputs, reference = frame_pools(128, 32, 9)
    assert len(inputs) == 32 and len(reference) == 96
    assert set(inputs).isdisjoint(reference)
    assert (inputs, reference) == frame_pools(128, 32, 9)
    with pytest.raises(ValueError, match="disjoint reference"):
        frame_pools(4, 3, 0)
    image = np.full((48, 48), 0.2, dtype=np.float32)
    image[:, 16:30] = 0.8
    sites = find_cd_sites(image)
    assert sites
    scores = _metrics([np.full_like(image, 0.5)] * 3, image, sites)
    assert scores["cd_failure_fraction"] == 1
    assert scores["cd_3sigma_px"] is None


def test_sample_region_normalizes_before_interpolation() -> None:
    values = np.arange(30 * 30, dtype=np.uint16).reshape(30, 30)
    a = sample_region(values, 5.25, 6.5, 16, normalization=(100, 600))
    b = sample_region(normalize_native(values, 100, 600), 5.25, 6.5, 16)
    np.testing.assert_array_equal(a, b)


def test_failed_preparation_closes_mmaps_before_cleanup(tmp_path: Path) -> None:
    raw = write_raw(tmp_path / "raw", sites=1)
    (raw / "site_0" / "frame_1.tif").write_bytes((raw / "site_0" / "frame_0.tif").read_bytes())
    with pytest.raises(ValueError, match="duplicate frame pixels"):
        prepare_real_dataset(raw, tmp_path / "prepared", image_size=16, align="none")
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".sem-prepare-*"))


def test_streamed_registration_recovers_known_motion() -> None:
    from edge_denoise.register import warp_frame, warp_scene

    yy, xx = np.mgrid[:96, :96]
    scene = (0.12 + 0.45 * np.exp(-((yy - 28) ** 2 + (xx - 35) ** 2) / 200)
             + 0.3 * np.exp(-((yy - 65) ** 2 + (xx - 60) ** 2) / 98))
    expected = np.array([[0, 0], [1.25, -0.5], [2.5, -1.0]])
    rng = np.random.default_rng(4)
    raw = np.stack([np.rint((warp_scene(scene, shift) + rng.normal(0, 0.001, scene.shape)) * 65535).astype(np.uint16)
                    for shift in expected])
    shifts, uncertainty = estimate_translations(raw, black=0, white=65535, sigma=1.5,
                                                radius=4, max_shift=8, device="cpu")
    np.testing.assert_allclose(shifts, expected, atol=0.1)
    assert np.isfinite(uncertainty).all()
    unit = normalize_native(raw[-1], 0, 65535)
    aligned = warp_frame(unit, shifts[-1], clip=False)[0]
    interior = np.s_[8:-8, 8:-8]
    assert np.mean((aligned[interior] - scene[interior]) ** 2) < np.mean((unit[interior] - scene[interior]) ** 2) / 10
    with pytest.raises(ValueError, match="implausible shift"):
        estimate_translations(raw, black=0, white=65535, sigma=1.5, radius=4, max_shift=0.2, device="cpu")


def test_real_recipes_share_teacher_backbone_and_expected_objectives() -> None:
    from edge_denoise.config import load_config

    directory = Path(__file__).resolve().parents[2] / "edge_denoise" / "configs"
    teacher = load_config(directory / "sem_real_n2n.yml")
    assert teacher.data.image_size == 512
    objectives = {"n2n": ("noisy", 0, 0), "mean": ("noisy_mean", 0, 0),
                  "avgfull": ("noisy_mean", 4, 0), "avgfull_consist": ("noisy_mean", 4, 1)}
    for name, (target, gradient, consistency) in objectives.items():
        cfg = load_config(directory / f"sem_real_ft_{name}.yml")
        assert cfg.model == teacher.model and cfg.data == teacher.data
        assert cfg.objective.target == target
        assert cfg.objective.lambda_image == 1
        assert (cfg.objective.lambda_gradient, cfg.objective.lambda_consistency) == (gradient, consistency)
        assert cfg.training.init_checkpoint == teacher.training.run_dir / "ckpt_latest.pt"


def test_resume_rejects_changed_prepared_metadata(prepared: Path, tmp_path: Path) -> None:
    cfg = real_config(prepared, tmp_path / "run")
    checkpoint = Trainer(cfg).run()
    path = prepared / REAL_MANIFEST
    metadata = json.loads(path.read_text(encoding="utf-8"))
    # Array hashes are still correct; the coordinate convention has changed.
    metadata["sites"][0]["shifts"][1][0] += 0.1
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="resume dataset differs"):
        Trainer(cfg, resume_from=checkpoint)


@pytest.mark.parametrize("change,error", [("bounds", "common overlap"), ("path", "escapes dataset")])
def test_invalid_prepared_metadata_is_rejected(prepared: Path, change: str, error: str) -> None:
    path = prepared / REAL_MANIFEST
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if change == "bounds":
        metadata["sites"][0]["bounds"][0] = -4
    else:
        metadata["sites"][0]["raw"]["path"] = "../outside.npy"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        BurstCache(prepared)
