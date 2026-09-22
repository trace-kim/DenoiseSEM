from __future__ import annotations

import threading

import numpy as np
from PIL import Image
import pytest
from scipy import ndimage
from skimage.filters import threshold_otsu

from sem_segment.otsu_baseline import OtsuSettings, otsu_baseline
from sem_segment.otsu_cuda import batch_capacity, histogram_thresholds, iter_saved_otsu
from sem_segment.otsu_measurement import iter_measure_saved_otsu, measure_otsu_result
from synthetic_images import visual_qa_frames


@pytest.mark.parametrize("sigma", [0, 1])
def test_batched_histogram_is_ordinary_per_image_otsu(sigma):
    rng = np.random.default_rng(904)
    pixels = rng.integers(0, 256, (12, 64, 96), dtype=np.uint8)
    pixels[0] = 128
    pixels[1] = 0
    pixels[2] = 255
    pixels[3] = np.arange(96)[None, :] % 2 * 170 + 40
    pixels[4] = np.arange(96)[None, :] + 20
    pixels[5] = pixels[4] + 35
    pixels[6] = pixels[4] * 2
    values = ndimage.gaussian_filter(pixels.astype(float), (0, sigma, sigma)) if sigma else pixels.astype(float)
    actual, constant = histogram_thresholds(values, np)
    expected = np.array([threshold_otsu(p) for p in values])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(constant, [True, True, True] + [False] * 9)
    assert actual[4] != actual[5] != actual[6]


def fixture_paths(tmp_path):
    paths = []
    images = [frames["raw"] for i, frames in visual_qa_frames(29) if i in (1, 9, 29)]
    shape = images[0].shape
    yy, xx = np.indices(shape)
    radius = np.hypot(yy - 80, xx - 120)
    mask = ((radius < 45) & (radius > 20)) | (radius < 6)
    mask[:, 230:240] = True
    pixels = np.where(mask, 40, 210).astype(np.uint8)
    pixels[5:45:3, 5:180:3] = 40
    images += [pixels, np.full(shape, 73, dtype=np.uint8), 255 - pixels]
    for i, pixels in enumerate(images):
        path = tmp_path / f"{i}.png"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("sigma", [0, 1])
@pytest.mark.parametrize("polarity", ["dark", "bright"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_cuda_batches_preserve_masks_metrology_and_all_paths(tmp_path, fake_cupy, sigma, polarity, batch_size):
    paths = fixture_paths(tmp_path)
    before = [p.read_bytes() for p in paths]
    settings = OtsuSettings(sigma_px=sigma, polarity=polarity)
    crop = (5, 218, 3, 314)
    expected = [otsu_baseline(path, settings, crop=crop) for path in paths]
    actual = list(iter_saved_otsu(paths, settings, crop=crop, device="cuda:2", batch_size=batch_size))
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a.mask, b.mask)
        assert a.threshold_dn == b.threshold_dn
        assert (a.component_count, a.retained_count) == (b.component_count, b.retained_count)
        assert len(a.outlines) == len(b.outlines)
        for p, q in zip(a.outlines, b.outlines):
            np.testing.assert_array_equal(p, q)
        ma, mb = [measure_otsu_result(r, settings, crop=crop) for r in (a, b)]
        for ra, rb in zip(ma.regions, mb.regions):
            for key, value in ra.to_row().items():
                other = rb.to_row()[key]
                if isinstance(value, str):
                    assert value == other
                else:
                    np.testing.assert_allclose(value, other, rtol=0, atol=0, equal_nan=True)
    assert [p.read_bytes() for p in paths] == before
    assert set(fake_cupy.devices) == {2}
    assert all(len(s) == 2 for s in fake_cupy.labels)
    assert all(sigma_axes[0] == 0 for _, sigma_axes in fake_cupy.filters)
    # Whole floating-point images must never be downloaded for CPU thresholding.
    assert all(len(shape) < 3 or dtype == np.int32 for shape, dtype in fake_cupy.downloads)
    assert not any(t.name.startswith(("sem-decode", "sem-otsu")) for t in threading.enumerate())


def test_budget_limits_batches_and_early_close_joins_workers(tmp_path, fake_cupy):
    paths = fixture_paths(tmp_path)
    assert batch_capacity((224, 320), 16, 16) == 2
    results = iter_measure_saved_otsu(paths, OtsuSettings(), device="cuda", batch_size=16, memory_mb=16)
    result = next(results)
    assert result.diagnostics.backend["execution"]["batch_size"] == 2
    results.close()
    assert not any(t.name.startswith(("sem-decode", "sem-otsu")) for t in threading.enumerate())
    with pytest.raises(ValueError, match="memory budget"):
        batch_capacity((4096, 4096), 16, 1)
    for requested, budget in [(0, 64), (4, 0)]:
        with pytest.raises(ValueError, match="positive"):
            batch_capacity((32, 32), requested, budget)


def test_decode_failure_and_wrong_dimensions_propagate_without_worker_leaks(tmp_path, fake_cupy):
    paths = fixture_paths(tmp_path)
    with pytest.raises(FileNotFoundError):
        list(iter_saved_otsu(paths[:1] + [tmp_path / "absent.png"], OtsuSettings()))
    wrong = tmp_path / "wrong.tif"
    Image.fromarray(np.zeros((32, 32), np.uint16)).save(wrong)
    with pytest.raises(ValueError, match="saved uint8"):
        list(iter_saved_otsu([wrong], OtsuSettings()))
    Image.fromarray(np.zeros((32, 32), np.uint8)).save(wrong)
    with pytest.raises(ValueError, match="identical dimensions"):
        list(iter_saved_otsu(paths[:1] + [wrong], OtsuSettings()))
    assert not any(t.name.startswith(("sem-decode", "sem-otsu")) for t in threading.enumerate())
