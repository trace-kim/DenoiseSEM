from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from conftest import make_config
from burst_diffusion.data import BurstCache, content_key
from burst_diffusion.real_data import file_digest
from edge_denoise.infer import Denoiser
from edge_denoise.real_data import RealPairFactory, prepare_real_dataset, read_native
from tools import prepare_real_sem_blocks as blocks
from tools import real_sem_averages as averages
from tools import real_sem_comparison as compare


def fixture_dataset(tmp_path: Path, count: int = 8) -> tuple[Path, Path]:
    """Independent noisy acquisitions whose pair/block means are exact blanks."""
    raw = tmp_path / "raw"
    rng = np.random.default_rng(391)
    for split, name, level in (("train", "site2", 80), ("train", "site10", 110), ("test", "heldout", 140)):
        folder = raw / split / name
        folder.mkdir(parents=True)
        for i in range(0, count, 2):
            noise = rng.integers(-2, 3, (32, 40))
            for j, sign in enumerate((1, -1)):
                compare.save_rgb(folder / f"frame_{i + j + 1}.png", (level + sign * noise).astype(np.uint8))
        shutil.copytree(folder, raw / "all" / name)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"site2": "train", "site10": "val"}))
    prepared = tmp_path / "prepared"
    prepare_real_dataset(raw / "train", prepared, image_size=16, align="none", split_file=splits)
    return raw, prepared


def test_all_four_variants_counts_pixels_splits_training_and_resume(tmp_path, monkeypatch):
    raw, prepared = fixture_dataset(tmp_path, 128)
    before = {p: file_digest(p) for base in (raw, prepared) for p in base.rglob("*") if p.is_file()}
    output = tmp_path / "blocks"
    calls = []
    real_average = blocks.registered_average

    def registered(frames, **kwargs):
        calls.append(len(frames))
        return real_average(frames, **kwargs)

    monkeypatch.setattr(blocks, "registered_average", registered)
    result = blocks.build_datasets(raw, prepared, output, image_size=16)
    assert json.loads(result.read_text())["outputs_per_site"] == {
        "average2": 64, "average4": 32, "average2_registered": 64, "average4_registered": 32}
    # all/ reuses byte-identical inputs rather than fitting them again.
    assert calls.count(2) == 3 * 64 and calls.count(4) == 3 * 32
    for variant, (count, registered) in blocks.VARIANTS.items():
        root = output / variant
        for split, name, level in (("train", "site2", 80), ("train", "site10", 110), ("test", "heldout", 140)):
            folder = root / "raw" / split / name
            files = blocks.image_files(folder)
            assert len(files) == 128 // count
            record = averages.load_block_manifest(folder, files)
            assert record["frames"][-1]["last_acquisition"] == 128
            assert record["frames"][1]["source_files"] == [f"frame_{i}.png" for i in range(count + 1, 2 * count + 1)]
            for file in files:
                with Image.open(file) as image:
                    assert image.mode == "RGB" and np.asarray(image).dtype == np.uint8
                    np.testing.assert_array_equal(np.asarray(image), level)
                assert file.read_bytes() == (root / "raw" / "all" / name / file.name).read_bytes()
            if registered:
                assert all(r["status"].startswith("skipped_") for f in record["frames"] for r in f["registration"]["frames"])
        cache = BurstCache(root / "train_align_none", min_size=16)
        assert {s["name"]: s["split"] for s in cache.real_metadata["sites"]} == {"site2": "train", "site10": "val"}
        assert cache.real_metadata["registration"]["mode"] == "none"
        assert len(cache.train_sources[0].frames) == 128 // count
        assert cache.real_metadata["block_averaging"]["variant"] == variant
        for site in cache.real_metadata["sites"]:
            for frame in site["frames"]:
                assert len(frame["source_sha256"]) == count
                assert frame["sha256"] == content_key(read_native(root / "raw" / "train" / frame["name"]))
        cfg = make_config(root / "train_align_none", tmp_path / "run")
        batch = RealPairFactory(cache, cfg, seed=11).sample_batch()
        assert batch.inputs.shape == batch.targets.shape == (2, 1, 16, 16)
    output_before = {p: file_digest(p) for p in output.rglob("*") if p.is_file() and p.name != "block_datasets.json"}
    monkeypatch.setattr(blocks, "registered_average", lambda *a, **k: pytest.fail("resume must reuse completed blocks"))
    blocks.build_datasets(raw, prepared, output, image_size=16, resume=True)
    assert all(file_digest(p) == digest for p, digest in output_before.items())
    assert all(file_digest(p) == digest for p, digest in before.items())


@pytest.mark.parametrize("bad", ["count", "raw_changed", "overlap", "uint16", "size", "wrong_split"])
def test_invalid_sources_fail_without_publishing_dataset(tmp_path, bad):
    raw, prepared = fixture_dataset(tmp_path)
    if bad == "count":
        (raw / "test/heldout/frame_8.png").unlink()
    elif bad == "raw_changed":
        compare.save_rgb(raw / "train/site2/frame_1.png", np.full((32, 40), 99, dtype=np.uint8))
    elif bad == "overlap":
        shutil.copyfile(raw / "train/site2/frame_1.png", raw / "test/heldout/frame_1.png")
    elif bad == "uint16":
        Image.fromarray(np.ones((32, 40), dtype=np.uint16)).save(raw / "test/heldout/frame_1.png")
    elif bad == "wrong_split":
        metadata = json.loads((prepared / "real_dataset.json").read_text())
        metadata["sites"][0]["split"] = "invalid"
        blocks.write_json(prepared / "real_dataset.json", metadata)
    with pytest.raises(ValueError):
        blocks.build_datasets(raw, prepared, tmp_path / "out", image_size=64 if bad == "size" else 16,
                              expected_frames=8, variants=["average2"])
    assert not (tmp_path / "out").exists()


def test_resume_verifies_images_and_settings_and_requires_explicit_flag(tmp_path):
    raw, prepared = fixture_dataset(tmp_path)
    out = tmp_path / "out"
    options = {"expected_frames": 8, "image_size": 16, "variants": ["average4"]}
    blocks.build_datasets(raw, prepared, out, **options)
    with pytest.raises(ValueError, match="output must be new"):
        blocks.build_datasets(raw, prepared, out, **options)
    with pytest.raises(ValueError, match="settings"):
        blocks.build_datasets(raw, prepared, out, **{**options, "image_size": 8}, resume=True)
    image = blocks.image_files(out / "average4/raw/train/site2")[0]
    compare.save_rgb(image, np.ones((32, 40), dtype=np.uint8))
    with pytest.raises(ValueError, match="block image changed"):
        blocks.build_datasets(raw, prepared, out, **options, resume=True)


def test_interrupted_preparation_reuses_finished_sites(tmp_path, monkeypatch):
    raw, prepared = fixture_dataset(tmp_path)
    out = tmp_path / "out"
    options = {"expected_frames": 8, "image_size": 16, "variants": ["average4_registered"]}
    original = blocks.prepare_real_dataset
    monkeypatch.setattr(blocks, "prepare_real_dataset", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        blocks.build_datasets(raw, prepared, out, **options)
    assert not (out / "block_datasets.json").exists()
    # A scheduler SIGKILL can leave an incomplete staging directory. Staging
    # lives outside raw/train, so the next preparation never treats it as a site.
    orphan = out / "average4_registered/.blocks-interrupted"
    orphan.mkdir()
    (orphan / "incomplete.txt").write_text("partial export")
    monkeypatch.setattr(blocks, "prepare_real_dataset", original)
    monkeypatch.setattr(blocks, "registered_average", lambda *a, **k: pytest.fail("reuse finished sites"))
    blocks.build_datasets(raw, prepared, out, **options, resume=True)
    assert (out / "block_datasets.json").is_file()


@pytest.mark.parametrize("count,variant", [(64, "average2"), (32, "average4_registered")])
def test_comparison_accepts_block_folders_and_uses_actual_count(tmp_path, monkeypatch, count, variant):
    raw, prepared = fixture_dataset(tmp_path, 128)
    out = tmp_path / "blocks"
    blocks.build_datasets(raw, prepared, out, image_size=16, variants=[variant])
    dataset = out / variant / "train_align_none"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    cfg = make_config(dataset, tmp_path / "run", representation="image")
    cfg.data.black_level, cfg.data.white_level = 0, 255

    class Model:
        config = cfg
        dataset_fingerprint = file_digest(dataset / "real_dataset.json")
        image_size, checkpoint_step, default_margin = 16, 1, 0

        def denoise_full(self, pixels, **kwargs):
            return pixels

    monkeypatch.setattr(Denoiser, "from_checkpoint", lambda *a, **k: Model())
    settings = compare.ComparisonSettings(checkpoints={"n2n": compare.ComparisonArm(checkpoint=checkpoint)},
        sites={"heldout": compare.SiteSettings(source_dir=out / variant / "raw/test/heldout")},
        output_dir=tmp_path / "report", device="cpu", metrology_device="cpu", contour_method="otsu",
        frame_interval_s=.5, tensorboard=False)
    result = compare.run(settings)
    site = result["sites"][0]
    assert len(site["series"]["raw"]["frames"]) == len(site["series"]["n2n"]["frames"]) == count
    reference = site["series"]["average128"]["frames"][0]
    assert reference["last_acquisition"] == count and reference["order"] == (count + 1) / 2
    np.testing.assert_array_equal(compare.read_uint8(settings.output_dir / site["full_average"]), 140)
    assert site["series"]["raw"]["frames"][0]["timestamp_s"] == (128 / count - 1) / 2 * .5
    assert site["input_block_averaging"]["frames_per_output"] == 128 // count
    if "registered" in variant:
        assert site["series"]["n2n"]["frames"][0]["common_support_path"] == site["series"]["raw"]["frames"][0]["common_support_path"]
        assert site["series"]["average8"]["frames"][0]["common_support_path"]
    # Ancestors keep averaged train/val pixels out of test reports.
    arm = compare.ComparisonArm(checkpoint=checkpoint)
    metadata = compare.checkpoint_arm_metadata(arm, Model())
    parent_cfg = cfg.model_copy(deep=True)
    parent_cfg.data.dataset_dir = prepared
    parent_meta = compare.checkpoint_arm_metadata(arm, SimpleNamespace(config=parent_cfg, dataset_fingerprint=None))
    assert metadata["raw_content_split_sha256"] == parent_meta["raw_content_split_sha256"]
    train_folder = out / variant / "raw/train/site2"
    record = averages.load_block_manifest(train_folder, blocks.image_files(train_folder))
    with pytest.raises(ValueError, match="train/validation"):
        compare.validate_evaluation_content(metadata, set(record["frames"][0]["source_sha256"]), "test")
    # Full remeasurement keeps the count and saved support without raw files.
    rebuilt = compare.rebuild(settings.output_dir / "comparison.json", tmp_path / "rebuilt", contours_only=True, tensorboard=False)
    assert rebuilt["sites"][0]["series"]["average128"]["frames"][0]["last_acquisition"] == count


def test_registered_reaveraging_preserves_previous_invalid_support():
    pixels = np.full((2, 40, 48), 100, dtype=np.uint8)
    supports = np.ones_like(pixels, dtype=bool)
    supports[0, 16:24, 20:28] = False
    output, valid, _ = averages.registered_average(pixels, input_supports=supports)
    assert not valid[14:26, 18:30].any()
    assert valid[6:10, 6:10].all()
    np.testing.assert_array_equal(output, 100)


def test_derived_identical_pixels_use_ancestors_for_split_checks(tmp_path):
    raw = tmp_path / "derived"
    origins = {}
    for name in ("a", "b"):
        for i in range(2):
            relative = f"{name}/{i}.png"
            compare.save_rgb(raw / relative, np.full((32, 40), 100, dtype=np.uint8))
            origins[relative] = [content_key(np.full((8, 8), i + (0 if name == "a" else 10), dtype=np.uint8))]
    splits = tmp_path / "split.json"
    splits.write_text(json.dumps({"a": "train", "b": "val"}))
    prepare_real_dataset(raw, tmp_path / "good", image_size=16, align="none", split_file=splits, frame_sources=origins)
    origins["b/0.png"] = origins["a/0.png"]
    with pytest.raises(ValueError, match="duplicate image content crosses"):
        prepare_real_dataset(raw, tmp_path / "bad", image_size=16, align="none", split_file=splits, frame_sources=origins)


def test_cli_generation_training_and_equivalence(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    import yaml

    from edge_denoise.cli import app
    from tools import check_real_sem_averages as check

    raw, prepared = fixture_dataset(tmp_path)
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["prepare_real_sem_blocks.py", "--raw-dir", str(raw),
        "--prepared-dir", str(prepared), "--output-dir", str(output), "--device", "cpu",
        "--image-size", "16", "--expected-frames", "8", "--variants", "average2"])
    assert blocks.main() == 0
    config = make_config(output / "average2/train_align_none", tmp_path / "run",
                         representation="image", lambda_gradient=0, max_steps=1)
    config_path = tmp_path / "train.yml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    result = CliRunner().invoke(app, ["train", "--config", str(config_path), "--cpu-threads", "2"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run/ckpt_latest.pt").is_file()
    monkeypatch.setattr("sys.argv", ["check_real_sem_averages.py", "--site-dir", str(raw / "test/heldout"),
        "--output-dir", str(tmp_path / "equivalence"), "--average-frames", "4", "--device", "cpu"])
    assert check.main() == 0
    assert json.loads((tmp_path / "equivalence/equivalence.json").read_text())["passed"]


def test_registration_uses_first_structured_anchor_and_preserves_seed_on_failure(monkeypatch):
    yy, xx = np.mgrid[:40, :48]
    structured = np.where((yy - 20)**2 + (xx - 24)**2 < 100, 40, 180).astype(np.uint8)
    frames = np.stack([np.full_like(structured, 100), *[structured + i for i in range(4)]])
    calls = []

    def estimate(reference, moving, *, motion, initial, **kwargs):
        index = next(i for i, frame in enumerate(frames) if np.array_equal(frame, moving))
        calls.append((index, motion, initial.copy()))
        assert np.array_equal(reference, frames[1])
        if index == 3 and motion == "affine":
            raise averages.GeometryEstimationError("synthetic failed fit")
        matrix = np.eye(2, 3)
        matrix[0, 2] = index
        return matrix, 1.

    monkeypatch.setattr("sem_noise.pair_matching.estimate_geometry", estimate)
    matrices, diagnostics, reference = averages.block_geometry(frames)
    assert reference == 1 and diagnostics[0]["status"] == "skipped_low_contrast"
    assert diagnostics[3]["status"] == "skipped_failed_registration"
    np.testing.assert_array_equal(matrices[3], np.eye(2, 3))
    following_seed = next(seed for i, motion, seed in calls if i == 4 and motion == "translation")
    assert following_seed[0, 2] == 2
