"""Method 1: polygon geometry, orientation, and the (y, x) convention."""

from __future__ import annotations

import numpy as np
import pytest

from sem_segment.backends import InstanceMask
from sem_segment.config import ContoursConfig
from sem_segment.contours import (
    outward_normals,
    polygon_area,
    polygon_centroid,
    polygon_perimeter,
    resample_closed,
    trace_instance,
    trace_all,
)


def disk_mask(shape=(128, 128), centre=(64.0, 64.0), radius=24.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot(yy - centre[0], xx - centre[1]) <= radius


def unit_square_points():
    """A 1x1 square as (y, x) vertices, counter-clockwise in display orientation."""
    return np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


def test_polygon_area_of_a_unit_square_is_one():
    assert abs(polygon_area(unit_square_points())) == pytest.approx(1.0)


def test_polygon_area_sign_flips_with_winding():
    points = unit_square_points()
    assert polygon_area(points) == pytest.approx(-polygon_area(points[::-1]))


def test_polygon_perimeter_of_a_unit_square_is_four():
    assert polygon_perimeter(unit_square_points()) == pytest.approx(4.0)


def test_polygon_centroid_of_a_unit_square_is_its_middle():
    cy, cx = polygon_centroid(unit_square_points())
    assert (cy, cx) == pytest.approx((0.5, 0.5))


def test_polygon_measures_match_a_regular_polygon_closed_form():
    """A regular n-gon inscribed in radius R has exact area and perimeter."""
    n, radius = 256, 10.0
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = np.column_stack([radius * np.sin(angle), radius * np.cos(angle)])
    expected_area = 0.5 * n * radius**2 * np.sin(2.0 * np.pi / n)
    expected_perimeter = 2.0 * n * radius * np.sin(np.pi / n)
    assert abs(polygon_area(points)) == pytest.approx(expected_area, rel=1e-9)
    assert polygon_perimeter(points) == pytest.approx(expected_perimeter, rel=1e-9)


def test_polygon_centroid_is_area_weighted_not_vertex_mean():
    """Densely sampling one side must not drag the centroid toward it."""
    dense_side = np.column_stack([np.zeros(50), np.linspace(0.0, 1.0, 50)])
    rest = np.array([[1.0, 1.0], [1.0, 0.0]])
    points = np.vstack([dense_side, rest])
    cy, cx = polygon_centroid(points)
    assert (cy, cx) == pytest.approx((0.5, 0.5), abs=1e-6)
    assert points[:, 0].mean() < 0.2  # the vertex mean is badly biased


def test_resample_gives_uniform_spacing_including_the_wraparound():
    angle = np.linspace(0.0, 2.0 * np.pi, 37, endpoint=False)
    points = np.column_stack([20.0 * np.sin(angle), 20.0 * np.cos(angle)])
    resampled = resample_closed(points, spacing_px=1.0)
    closed = np.vstack([resampled, resampled[:1]])
    steps = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    # Every step, the seam included, is the same length.
    assert steps.std() < 0.01 * steps.mean()


def test_traced_disk_area_matches_the_true_area():
    radius = 24.0
    instance = InstanceMask.from_mask(disk_mask(radius=radius), score=1.0, backend="t")
    outer, holes = trace_instance(instance, ContoursConfig())
    assert holes == []
    area = abs(polygon_area(outer.points))
    # A pixelated disk traced at its 0.5 level lands within a percent of pi r^2.
    assert area == pytest.approx(np.pi * radius**2, rel=0.02)


def test_traced_disk_centroid_matches_the_true_centre():
    instance = InstanceMask.from_mask(
        disk_mask(centre=(70.0, 55.0), radius=20.0), score=1.0, backend="t"
    )
    outer, _ = trace_instance(instance, ContoursConfig())
    assert polygon_centroid(outer.points) == pytest.approx((70.0, 55.0), abs=0.15)


def test_annulus_yields_an_outer_contour_and_a_hole():
    yy, xx = np.mgrid[0:128, 0:128]
    radius = np.hypot(yy - 64, xx - 64)
    ring = (radius < 30) & (radius > 15)
    instance = InstanceMask.from_mask(ring, score=1.0, backend="t")
    outer, holes = trace_instance(instance, ContoursConfig())
    assert len(holes) == 1
    assert abs(polygon_area(outer.points)) > abs(polygon_area(holes[0].points))
    assert abs(polygon_area(holes[0].points)) == pytest.approx(np.pi * 15.0**2, rel=0.1)


def test_border_touching_region_still_yields_a_closed_ring():
    """Without the pre-trace pad this returns an open polyline and every measure breaks."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[0:40, 0:40] = True
    instance = InstanceMask.from_mask(mask, score=1.0, backend="t")
    outer, _ = trace_instance(instance, ContoursConfig())
    assert abs(polygon_area(outer.points)) == pytest.approx(40.0 * 40.0, rel=0.1)


def test_normals_point_outward_from_a_disk():
    """The orientation contract: normals leave the region, not enter it."""
    mask = disk_mask(radius=24.0)
    instance = InstanceMask.from_mask(mask, score=1.0, backend="t")
    outer, _ = trace_instance(instance, ContoursConfig())
    normals = outward_normals(outer.points, mask, smooth_px=2.0)

    radial = outer.points - np.array([64.0, 64.0])
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    # Outward normal and outward radial direction agree everywhere on a circle.
    assert float(np.mean(np.sum(normals * radial, axis=1))) > 0.97


def test_normals_are_unit_length():
    mask = disk_mask(radius=20.0)
    instance = InstanceMask.from_mask(mask, score=1.0, backend="t")
    outer, _ = trace_instance(instance, ContoursConfig())
    normals = outward_normals(outer.points, mask, smooth_px=2.0)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)


def test_hole_normals_point_into_the_material():
    """"Out of the mask" on an annulus means toward the hole centre."""
    yy, xx = np.mgrid[0:128, 0:128]
    radius = np.hypot(yy - 64, xx - 64)
    ring = (radius < 30) & (radius > 15)
    instance = InstanceMask.from_mask(ring, score=1.0, backend="t")
    outer, holes = trace_instance(instance, ContoursConfig())
    hole_normals = outward_normals(holes[0].points, ring, smooth_px=2.0)

    radial = holes[0].points - np.array([64.0, 64.0])
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    # The hole boundary's outward normal points toward the disk centre.
    assert float(np.mean(np.sum(hole_normals * radial, axis=1))) < -0.9


def test_coordinates_are_row_then_column():
    """Pins the (y, x) convention: a wide flat region must be wide in x."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:68, 20:100] = True  # 8 tall, 80 wide
    instance = InstanceMask.from_mask(mask, score=1.0, backend="t")
    outer, _ = trace_instance(instance, ContoursConfig())
    extent_y = np.ptp(outer.points[:, 0])
    extent_x = np.ptp(outer.points[:, 1])
    assert extent_x > extent_y * 5


def test_trace_all_attaches_normals_to_every_contour():
    instances = [
        InstanceMask.from_mask(disk_mask(centre=(30.0, 30.0), radius=12.0), score=1.0, backend="t"),
        InstanceMask.from_mask(disk_mask(centre=(90.0, 90.0), radius=12.0), score=1.0, backend="t"),
    ]
    outers, holes = trace_all(instances, (128, 128), ContoursConfig())
    assert len(outers) == 2 and len(holes) == 2
    for index, contour in enumerate(outers, start=1):
        assert contour.normals is not None
        assert contour.normals.shape == contour.points.shape
        assert contour.region_id == index


def test_tiny_region_below_min_vertices_is_dropped():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:12, 10:12] = True
    instance = InstanceMask.from_mask(mask, score=1.0, backend="t")
    outer, _ = trace_instance(instance, ContoursConfig(min_vertices=32))
    assert outer is None
