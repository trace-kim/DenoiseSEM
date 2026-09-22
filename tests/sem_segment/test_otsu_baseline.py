from __future__ import annotations

import sys

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from sem_segment.otsu_baseline import OtsuSettings, otsu_baseline
from synthetic_images import visual_qa_frames


def save(path, pixels):
    Image.fromarray(pixels).save(path)
    return path


@pytest.mark.parametrize("polarity", ["dark", "bright"])
@pytest.mark.parametrize("sigma", [0, 1])
def test_disk_polarity_and_boundary_placement(tmp_path, polarity, sigma):
    yy, xx = np.mgrid[:96, :112]
    inside = np.hypot(yy - 43, xx - 61) <= 20
    pixels = np.where(inside, 40, 210).astype(np.uint8)
    if polarity == "bright":
        pixels = 255 - pixels
    path = save(tmp_path / "disk.png", np.repeat(pixels[..., None], 3, axis=2))
    before = path.read_bytes()
    result = otsu_baseline(path, OtsuSettings(polarity=polarity, sigma_px=sigma))
    assert result.retained_count == len(result.outlines) == 1
    ring = result.outlines[0]
    assert np.array_equal(ring[0], ring[-1])
    radial_error = np.abs(np.hypot(ring[:, 0] - 43, ring[:, 1] - 61) - 20)
    assert radial_error.max() <= .65
    assert path.read_bytes() == before
    if sigma == 0:
        np.testing.assert_array_equal(result.mask, inside)
        # Ordinary Otsu's first split of a two-level float image is near the
        # low level, not a custom foreground/background midpoint.
        assert result.threshold_dn < pixels.min() + 1


@pytest.mark.parametrize("sigma", [0, 1])
def test_ellipse_boundary_position(tmp_path, sigma):
    yy, xx = np.mgrid[:112, :128]
    distance = np.sqrt(((yy - 53) / 17) ** 2 + ((xx - 69) / 30) ** 2)
    path = save(tmp_path / "ellipse.png", np.where(distance <= 1, 35, 215).astype(np.uint8))
    result = otsu_baseline(path, OtsuSettings(sigma_px=sigma))
    assert result.retained_count == len(result.outlines) == 1
    ring = result.outlines[0]
    errors = np.abs(np.sqrt(((ring[:, 0] - 53) / 17) ** 2 + ((ring[:, 1] - 69) / 30) ** 2) - 1)
    assert errors.max() < .04


@pytest.mark.parametrize("sigma", [0, 1])
@pytest.mark.parametrize("shape", ["concave", "annulus", "connected"])
def test_concavity_holes_and_connected_shapes(tmp_path, sigma, shape):
    yy, xx = np.mgrid[:112, :128]
    if shape == "annulus":
        radius = np.hypot(yy - 56, xx - 64)
        inside = (radius >= 14) & (radius <= 30)
    elif shape == "concave":
        inside = (yy >= 20) & (yy < 90) & (xx >= 20) & (xx < 100) & ((xx < 40) | (yy >= 70))
    else:
        inside = (np.hypot(yy - 56, xx - 36) < 20) | (np.hypot(yy - 56, xx - 92) < 20)
        inside |= (yy >= 52) & (yy < 61) & (xx >= 36) & (xx <= 92)
    path = save(tmp_path / "shape.png", np.where(inside, 40, 210).astype(np.uint8))
    result = otsu_baseline(path, OtsuSettings(sigma_px=sigma))
    assert result.retained_count == 1
    assert len(result.outlines) == (2 if shape == "annulus" else 1)
    if sigma == 0:
        np.testing.assert_array_equal(result.mask, inside)
    if shape == "annulus":
        assert not result.mask[56, 64]
        for ring in result.outlines:
            radii = np.hypot(ring[:, 0] - 56, ring[:, 1] - 64)
            expected = 30 if np.median(radii) > 22 else 14
            assert np.max(np.abs(radii - expected)) <= .65
    elif shape == "concave":
        assert not result.mask[40, 70]  # Preserve the concavity, not its hull.
        ring = result.outlines[0]
        notch = ring[(ring[:, 0] > 25) & (ring[:, 0] < 65) & (ring[:, 1] > 30)]
        assert np.all(notch[:, 1] == 39.5)
    else:
        assert result.mask[56, 64]  # No watershed division across the neck.


def test_smoothing_removes_a_thin_protrusion(tmp_path):
    pixels = np.full((96, 96), 210, dtype=np.uint8)
    pixels[24:72, 24:72] = 40
    pixels[12:24, 48] = 40
    path = save(tmp_path / "thin.png", pixels)
    sharp = otsu_baseline(path, OtsuSettings(sigma_px=0))
    smooth = otsu_baseline(path)
    assert sharp.mask[16, 48] and not smooth.mask[16, 48]
    assert sharp.retained_count == smooth.retained_count == 1
    assert sharp.outlines[0][:, 0].min() == 11.5
    assert smooth.outlines[0][:, 0].min() > 20


def test_area_filter_sees_all_components_without_maximum_cutoff(tmp_path):
    pixels = np.full((300, 300), 210, dtype=np.uint8)
    pixels[1:100:3, 1:299:3] = 40  # More than 3,000 specks precede real regions.
    pixels[150:290, 5:280] = 40  # A large connected foreground must survive.
    pixels[110:115, 5:10] = 40  # Exactly the minimum area.
    pixels[110:114, 30:36] = 40  # One pixel below it.
    path = save(tmp_path / "specks.png", pixels)
    result = otsu_baseline(path, OtsuSettings(sigma_px=0))
    assert result.component_count > 3000
    assert result.retained_count == len(result.outlines) == 2
    assert result.mask[200, 200] and result.mask[112, 7]
    assert not result.mask[112, 32]
    almost_full = np.full((64, 64), 40, dtype=np.uint8)
    almost_full[-5:, -5:] = 210
    result = otsu_baseline(save(tmp_path / "large.png", almost_full), OtsuSettings(sigma_px=0))
    assert result.retained_count == 1 and result.mask.mean() > .99


def test_four_connectivity_in_area_filter_and_contour_tracing(tmp_path):
    pixels = np.full((48, 48), 210, dtype=np.uint8)
    pixels[10:15, 10:15] = 40
    pixels[15:20, 15:20] = 40
    path = save(tmp_path / "diagonal.png", pixels)
    result = otsu_baseline(path, OtsuSettings(sigma_px=0))
    assert result.retained_count == len(result.outlines) == 2
    assert not otsu_baseline(path, OtsuSettings(sigma_px=0, min_area_px=26)).mask.any()


@pytest.mark.parametrize("sigma", [0, 1])
@pytest.mark.parametrize("axis", [0, 1])
def test_border_paths_are_open_and_crop_offsets_restored(tmp_path, sigma, axis):
    pixels = np.full((96, 112), 210, dtype=np.uint8)
    if axis == 0:
        pixels[:, :60] = 40
    else:
        pixels[:45, :] = 40
    path = save(tmp_path / "border.png", pixels)
    result = otsu_baseline(path, OtsuSettings(sigma_px=sigma), crop=(10, 86, 20, 102))
    assert result.retained_count == len(result.outlines) == 1
    line = result.outlines[0]
    assert not np.array_equal(line[0], line[-1])
    # With smoothing, ordinary Otsu's histogram split in this planar two-level
    # fixture moves the binary boundary inward by one pixel. Keep that visible
    # rather than assuming the detector is an unbiased edge estimator.
    expected = (59.5 if axis == 0 else 44.5) - sigma
    assert np.all(line[:, 1 - axis] == expected)
    np.testing.assert_array_equal(np.sort(line[[0, -1], axis]), [10, 85] if axis == 0 else [20, 101])


@pytest.mark.parametrize("level", [0, 128, 255])
@pytest.mark.parametrize("polarity", ["dark", "bright"])
def test_constant_images_are_empty(tmp_path, level, polarity):
    path = save(tmp_path / "constant.png", np.full((32, 32), level, dtype=np.uint8))
    result = otsu_baseline(path, OtsuSettings(polarity=polarity))
    assert result.threshold_dn == pytest.approx(level)
    assert not result.mask.any() and not result.outlines
    assert result.component_count == result.retained_count == 0


def test_enforces_saved_uint8_and_grayscale_with_crop(tmp_path):
    path = save(tmp_path / "sixteen.tif", np.full((32, 32), 40, dtype=np.uint16))
    with pytest.raises(ValueError, match="saved uint8"):
        otsu_baseline(path)
    path = save(tmp_path / "float.tif", np.full((32, 32), .3, dtype=np.float32))
    with pytest.raises(ValueError, match="saved uint8"):
        otsu_baseline(path)
    rgb = np.full((48, 48, 3), 200, dtype=np.uint8)
    rgb[20:30, 20:30] = 40
    rgb[:3, :, 0] = 255
    path = save(tmp_path / "color.png", rgb)
    with pytest.raises(ValueError, match="grayscale required"):
        otsu_baseline(path)
    assert otsu_baseline(path, crop=(3, 48, 0, 48)).retained_count == 1
    for crop in [(-1, 32, 0, 32), (0, 20, 0, 99), (20, 10, 0, 32)]:
        with pytest.raises(ValueError, match="crop"):
            otsu_baseline(path, crop=crop)


@pytest.mark.parametrize("setting", [{"sigma_px": -1}, {"sigma_px": float("nan")},
                                      {"sigma_px": float("inf")}, {"min_area_px": 0},
                                      {"min_area_px": 1.5}, {"polarity": "auto"}, {"watershed": True}])
def test_invalid_settings_fail(setting):
    with pytest.raises(ValueError):
        OtsuSettings(**setting)


def test_cuda_detector_selects_visible_device_and_matches_cpu(tmp_path, monkeypatch, fake_cupy):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    pixels = np.full((48, 48), 200, dtype=np.uint8)
    pixels[10:30, 10:30] = 40
    path = save(tmp_path / "cuda.png", pixels)
    cpu = otsu_baseline(path)
    cuda = otsu_baseline(path, device="cuda:1")
    np.testing.assert_array_equal(cpu.mask, cuda.mask)
    assert cuda.threshold_dn == cpu.threshold_dn
    assert fake_cupy.devices == [1]
    assert fake_cupy.labels == [(48, 48)]
    assert fake_cupy.downloads[0] == ((1, 48, 48), np.dtype("int32"))
    assert cuda.gaussian_backend == "cupy"
    assert all(t >= 0 for t in cuda.timings_s.values())


def test_cuda_unavailable_fails_even_when_smoothing_is_disabled(tmp_path, monkeypatch):
    path = save(tmp_path / "cuda.png", np.full((32, 32), 40, dtype=np.uint8))
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="requires CuPy"):
        otsu_baseline(path, device="cuda:0")
    with pytest.raises(RuntimeError, match="requires CuPy"):
        otsu_baseline(path, OtsuSettings(sigma_px=0), device="cuda:0")
    with pytest.raises(ValueError, match="device"):
        otsu_baseline(path, device="auto")


@pytest.mark.parametrize("frame_id,unsmoothed_loops,old_count", [(9, 678, 0), (29, 11, 8)])
def test_original_high_noise_and_false_split_cases(tmp_path, frame_id, unsmoothed_loops, old_count):
    from sem_segment.backends import build_segmenter
    from sem_segment.config import Config
    from sem_segment.masks import postprocess

    frames = dict(visual_qa_frames(frame_id))
    pixels = frames[frame_id]["raw"]
    path = save(tmp_path / f"frame_{frame_id}.png", pixels)
    sharp = otsu_baseline(path, OtsuSettings(sigma_px=0))
    smooth = otsu_baseline(path)
    assert sharp.retained_count == smooth.retained_count == 7
    assert len(sharp.outlines) == unsmoothed_loops
    assert len(smooth.outlines) == 7
    assert sum(not np.array_equal(p[0], p[-1]) for p in smooth.outlines) == 1
    config = Config(segmentation={"backend": "classical", "polarity": "dark", "contrast_stretch": None},
                    masks={"min_area_px": 60})
    raw_instances = build_segmenter(config).segment(np.repeat(pixels[..., None], 3, axis=2))
    instances, _ = postprocess(raw_instances, pixels.shape, config.masks)
    assert len(instances) == old_count
    # Six complete circular boundaries plus one partial at the left edge.
    i = frame_id - 1
    centers = [(y + .8 * np.sin(i / 17), x + 1.2 * np.sin(i / 23))
               for y in (60, 166) for x in (55, 160, 265)]
    radius = 28 + .2 * np.sin(i / 9)
    for ring in smooth.outlines:
        if np.array_equal(ring[0], ring[-1]):
            center = min(centers, key=lambda point: np.linalg.norm(ring.mean(axis=0) - point))
            errors = np.abs(np.linalg.norm(ring - center, axis=1) - radius)
            assert np.median(errors) < .6
            assert errors.max() < 2
