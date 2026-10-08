from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tools import average_site_groups as groups


def _write_sites(folder, count, seed=0):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    frames = []
    for index in range(1, count + 1):
        frame = rng.integers(0, 256, (24, 32), dtype=np.uint8)
        Image.fromarray(frame).save(folder / f"img_{index}.png")
        frames.append(frame)
    return np.array(frames, dtype=np.float64)


def test_consecutive_groups_in_natural_order_are_averaged_from_originals(tmp_path):
    frames = _write_sites(tmp_path / "src", 16)
    assert groups.main(["--source", str(tmp_path / "src"), "--output", str(tmp_path / "out")]) == 0
    out = tmp_path / "out"
    assert len(list((out / "average2").iterdir())) == 8
    assert len(list((out / "average4").iterdir())) == 4
    assert len(list((out / "average8").iterdir())) == 2
    # img_9..img_16 is site 2; natural order keeps img_10 after img_9.
    saved = np.asarray(Image.open(out / "average8" / "site002_avg8_1_img_9-img_16.png"))
    np.testing.assert_array_equal(saved, np.rint(frames[8:16].mean(axis=0)).astype(np.uint8))
    saved = np.asarray(Image.open(out / "average4" / "site001_avg4_2_img_5-img_8.png"))
    np.testing.assert_array_equal(saved, np.rint(frames[4:8].mean(axis=0)).astype(np.uint8))


def test_subfolders_are_ignored(tmp_path):
    frames = _write_sites(tmp_path / "src", 8)
    # Unrelated images in a subfolder, in a count that would break the grouping if read.
    _write_sites(tmp_path / "src" / "other", 3, seed=1)
    assert groups.main(["--source", str(tmp_path / "src"), "--output", str(tmp_path / "out")]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["average2", "average4", "average8"]
    assert all(p.is_file() for p in (tmp_path / "out" / "average8").iterdir())
    saved = np.asarray(Image.open(tmp_path / "out" / "average8" / "site001_avg8_1_img_1-img_8.png"))
    np.testing.assert_array_equal(saved, np.rint(frames.mean(axis=0)).astype(np.uint8))


def test_partial_site_is_rejected_before_writing(tmp_path):
    _write_sites(tmp_path / "src", 12)
    with pytest.raises(SystemExit, match="not a multiple of 8"):
        groups.main(["--source", str(tmp_path / "src"), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_existing_output_is_refused(tmp_path):
    _write_sites(tmp_path / "src", 8)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        groups.main(["--source", str(tmp_path / "src"), "--output", str(tmp_path / "out")])
