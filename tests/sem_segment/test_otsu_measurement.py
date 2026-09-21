from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from sem_segment.contours import polygon_area
from sem_segment.otsu_baseline import OtsuSettings, otsu_baseline
from sem_segment.otsu_measurement import measure_saved_otsu


def test_otsu_measurement_preserves_holes_nested_objects_and_open_paths(tmp_path):
    yy, xx = np.mgrid[:112, :180]
    radius = np.hypot(yy - 55, xx - 70)
    mask = ((radius >= 12) & (radius <= 26)) | (radius <= 4)
    mask |= np.hypot(yy - 55, xx - 10) <= 18  # Crosses the crop's left border.
    mask[:, 138:151] = True  # Two separate open paths belong to one region.
    path = tmp_path / "regions.png"
    Image.fromarray(np.where(mask, 40, 210).astype(np.uint8)).save(path)
    before = path.read_bytes()
    settings = OtsuSettings(sigma_px=0)
    crop = (5, 107, 5, 175)
    baseline = otsu_baseline(path, settings, crop=crop)
    measured = measure_saved_otsu(path, settings, crop=crop)
    np.testing.assert_array_equal(measured.label_map() > 0, baseline.mask)
    assert measured.region_count == 4
    assert sum(r.touches_border for r in measured.regions) == 2
    assert sorted(len(paths) for paths in measured.open_paths) == [0, 0, 1, 2]
    assert not measured.refined
    exported = []
    for outer, holes, opened, region in zip(measured.coarse, measured.holes, measured.open_paths, measured.regions):
        if region.touches_border:
            assert region.coarse is None and not len(outer)
        else:
            area = abs(polygon_area(outer.points)) - sum(abs(polygon_area(h.points)) for h in holes)
            assert region.coarse.area_px2 == pytest.approx(area)
            assert region.coarse.equivalent_diameter_px == pytest.approx(2 * np.sqrt(area / np.pi))
        assert region.refined is None
        exported.extend(p.points for p in [outer, *holes] if len(p))
        exported.extend(opened)
    # Grouping must not move, resample, fill or lose any baseline boundary.
    actual_points = set(map(tuple, np.concatenate(exported) + [crop[0], crop[2]]))
    expected_points = set(map(tuple, np.concatenate(baseline.outlines)))
    assert actual_points == expected_points
    assert sorted(r.n_holes for r in measured.regions) == [0, 0, 0, 1]
    assert path.read_bytes() == before


def test_closed_hole_inside_border_region_is_not_measured_as_outer(tmp_path):
    yy, xx = np.mgrid[:80, :96]
    foreground = (np.hypot(yy - 40, xx - 10) <= 28) & (np.hypot(yy - 40, xx - 18) >= 7)
    path = tmp_path / "partial_ring.png"
    Image.fromarray(np.where(foreground, 35, 215).astype(np.uint8)).save(path)
    result = measure_saved_otsu(path, OtsuSettings(sigma_px=0))
    assert result.region_count == 1
    assert result.regions[0].touches_border
    assert result.regions[0].coarse is None
    assert len(result.holes[0]) == len(result.open_paths[0]) == 1
    assert result.regions[0].hole_area_px2 > 0


@pytest.mark.parametrize("polarity", ["dark", "bright"])
def test_measurement_keeps_four_connected_regions_and_empty_results(tmp_path, polarity):
    image = np.full((48, 48), 210, dtype=np.uint8)
    image[10:15, 10:15] = image[15:20, 15:20] = 40
    if polarity == "bright":
        image = 255 - image
    path = tmp_path / "diagonal.png"
    Image.fromarray(image).save(path)
    result = measure_saved_otsu(path, OtsuSettings(polarity=polarity, sigma_px=0))
    assert result.region_count == 2
    assert all(r.coarse.area_px2 == 24.5 for r in result.regions)
    empty = measure_saved_otsu(path, OtsuSettings(polarity=polarity, sigma_px=0, min_area_px=26))
    assert empty.region_count == 0
    assert empty.image_stats["region_count"] == 0
