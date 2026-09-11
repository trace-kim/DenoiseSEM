from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

tifffile = pytest.importorskip("tifffile")

from sem_noise.cli import main
from sem_noise.config import AnalysisConfig, load_config
from sem_noise.io import Frame, discover_sites, pixel_hash, read_frame


def _png(path: Path, value: int = 2000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 40), value, dtype=np.uint16)).save(path)


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_natural_order_uint16_and_content_identity(tmp_path: Path) -> None:
    for name in ("scan10.png", "scan2.png", "scan1.PNG"):
        _png(tmp_path / "site" / name)
    frames = discover_sites(tmp_path)["site"]
    assert [f.path.name for f in frames] == ["scan1.PNG", "scan2.png", "scan10.png"]
    array = read_frame(frames[0])
    assert array.dtype == np.uint16
    assert array.min() == array.max() == 2000
    assert pixel_hash(array) == pixel_hash(read_frame(frames[1]))
    assert pixel_hash(array) != pixel_hash(array + 1)


def test_manifest_controls_arbitrary_layout_and_metadata(tmp_path: Path) -> None:
    _png(tmp_path / "arbitrary/deep/a.png", 1000)
    _png(tmp_path / "elsewhere/b.png", 1100)
    manifest = tmp_path / "order.csv"
    _csv(manifest, [
        {"site": "pattern", "path": "elsewhere/b.png", "frame_index": 8, "timestamp_s": 3.2, "include": "false", "voltage_kv": 5},
        {"site": "pattern", "path": "arbitrary/deep/a.png", "frame_index": 7, "timestamp_s": 1.2, "include": "true", "voltage_kv": 5},
    ])
    frames = discover_sites(tmp_path, manifest)["pattern"]
    assert [f.index for f in frames] == [7, 8]
    assert frames[0].metadata == {"voltage_kv": "5"}
    assert frames[1].timestamp_s == 3.2
    assert not frames[1].include


@pytest.mark.parametrize("change", [
    {"frame_index": 0}, {"timestamp_s": ""}, {"timestamp_s": 0},
    {"timestamp_s": "nan"}, {"include": "maybe"}, {"path": "missing.png"},
])
def test_invalid_manifest_is_rejected(tmp_path: Path, change: dict) -> None:
    _png(tmp_path / "a.png")
    _png(tmp_path / "b.png")
    rows = [dict(site="a", path="a.png", frame_index=0, timestamp_s=0, include="true"),
            dict(site="a", path="b.png", frame_index=1, timestamp_s=1, include="true")]
    rows[1].update(change)
    _csv(tmp_path / "manifest.csv", rows)
    with pytest.raises(ValueError):
        discover_sites(tmp_path, tmp_path / "manifest.csv")


def test_manifest_cannot_escape_input(tmp_path: Path) -> None:
    _png(tmp_path / "outside.png")
    (tmp_path / "input").mkdir()
    _csv(tmp_path / "manifest.csv", [dict(site="a", path="../outside.png", frame_index=0)])
    with pytest.raises(ValueError, match="invalid site or input path"):
        discover_sites(tmp_path / "input", tmp_path / "manifest.csv")


def test_stack_pages_and_roi(tmp_path: Path) -> None:
    array = np.arange(3 * 32 * 40, dtype=np.uint16).reshape(3, 32, 40)
    path = tmp_path / "site.tif"
    tifffile.imwrite(path, array, photometric="minisblack")
    frames = discover_sites(path)["site"]
    assert len(frames) == 3
    np.testing.assert_array_equal(read_frame(frames[2], (3, 25, 5, 30)), array[2, 3:25, 5:30])
    with pytest.raises(ValueError, match="outside"):
        read_frame(frames[0], (0, 33, 0, 40))


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".png", ".bmp"])
def test_identical_rgb_channels_preserve_decoded_grayscale(tmp_path: Path, suffix: str) -> None:
    gray = np.random.default_rng(42).integers(0, 256, (32, 40), dtype=np.uint8)
    path = tmp_path / f"scan{suffix}"
    Image.fromarray(np.repeat(gray[..., None], 3, axis=2)).save(path)
    with Image.open(path) as image:
        assert image.mode == "RGB"
        expected = np.asarray(image)[..., 0].copy()
    frame = discover_sites(path)["scan"][0]
    actual = read_frame(frame)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    assert pixel_hash(actual) == pixel_hash(expected)
    np.testing.assert_array_equal(read_frame(frame, (3, 25, 5, 30)), expected[3:25, 5:30])


@pytest.mark.parametrize("channel", [1, 2])
def test_rejects_rgb_channel_difference_even_outside_roi(tmp_path: Path, channel: int) -> None:
    color = tmp_path / "color.png"
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[0, 0, channel] = 1
    Image.fromarray(array).save(color)
    with pytest.raises(ValueError, match="RGB channels must be identical"):
        read_frame(Frame("site", color, "color.png", 0), (3, 25, 5, 30))


def test_rejects_nonfinite_inputs(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.full((32, 32), np.nan))
    with pytest.raises(ValueError, match="NaN"):
        read_frame(Frame("site", path, "bad.npy", 0))


def test_inventory_is_recursive_editable_and_never_overwrites(tmp_path: Path) -> None:
    _png(tmp_path / "input/a/deep/frame10.png")
    _png(tmp_path / "input/a/deep/frame2.png")
    output = tmp_path / "manifest.csv"
    args = ["inventory", "--input", str(tmp_path / "input"), "--output", str(output)]
    assert main(args) == 0
    frames = discover_sites(tmp_path / "input", output)["a/deep"]
    assert frames[0].path.name == "frame2.png"
    before = output.read_bytes()
    assert main(args) == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize("kwargs", [
    {"min_frames": 2}, {"max_shift_px": float("nan")}, {"frame_interval_s": 0},
    {"roi": (0, 1, 2, 1)}, {"white_level": 0, "black_level": 1},
    {"sample_pixels": True}, {"registration": "affine"}, {"flat_fraction": 1},
    {"max_shift_px": None}, {"registration_sigma": "1"}, {"min_correlation": None},
])
def test_invalid_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AnalysisConfig(**kwargs)


def test_unknown_config_key_is_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("max_shfit_px: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
