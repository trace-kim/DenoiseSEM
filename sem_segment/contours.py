"""Method 1: the contour that wraps a segmented region.

Marching squares on the mask at its 0.5 level, resampled to uniform arc length,
with an outward normal at every vertex.  That parametrisation is what method 2
then refines, and keeping both contours on the *same* vertices is what makes the
two directly comparable.

Coordinates are ``(y, x)`` - row then column - everywhere in this module,
matching ``skimage.measure.find_contours``.  Mixing the two conventions is the
single easiest way to produce a plausible-looking but transposed measurement, so
the order is stated in every signature and pinned by a test.

One thing worth knowing about the boundary this produces: it is sub-pixel in
form but not in accuracy.  It interpolates the 0.5 level of a *binary* mask, so
its precision is bounded by the resolution the mask was produced at.  For SAM 3
on a frame larger than 1008 px that is roughly two original pixels, which is why
``refine.py`` exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends import InstanceMask
from .config import ContoursConfig


@dataclass
class Contour:
    """A closed polygon in full-image ``(y, x)`` coordinates."""

    #: (N, 2) array of (y, x) vertices; the ring is closed implicitly (no repeat).
    points: np.ndarray
    #: (N, 2) outward unit normals, or None until computed.
    normals: np.ndarray | None = None
    is_hole: bool = False
    region_id: int = 0
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def closed(self) -> np.ndarray:
        """Vertices with the first point repeated at the end."""
        return np.vstack([self.points, self.points[:1]])


def polygon_area(points: np.ndarray) -> float:
    """Signed shoelace area of a closed ring given as (N, 2) ``(y, x)`` vertices.

    Positive means counter-clockwise in ``(x, y)`` display orientation.  Callers
    that only want size should take the absolute value; the sign is what
    distinguishes an outer boundary from a hole.
    """
    y = np.asarray(points[:, 0], dtype=np.float64)
    x = np.asarray(points[:, 1], dtype=np.float64)
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_perimeter(points: np.ndarray) -> float:
    """Closed-ring arc length."""
    deltas = np.diff(np.vstack([points, points[:1]]), axis=0)
    return float(np.hypot(deltas[:, 0], deltas[:, 1]).sum())


def polygon_centroid(points: np.ndarray) -> tuple[float, float]:
    """Area centroid of the polygon (not the mean of its vertices).

    The vertex mean is biased toward wherever vertices happen to be dense, which
    after arc-length resampling is nearly uniform but not exactly so.
    """
    area = polygon_area(points)
    if abs(area) < 1e-12:
        return (float(points[:, 0].mean()), float(points[:, 1].mean()))
    y = np.asarray(points[:, 0], dtype=np.float64)
    x = np.asarray(points[:, 1], dtype=np.float64)
    y_next, x_next = np.roll(y, -1), np.roll(x, -1)
    cross = x * y_next - x_next * y
    cy = float(np.dot(y + y_next, cross) / (6.0 * area))
    cx = float(np.dot(x + x_next, cross) / (6.0 * area))
    return (cy, cx)


def resample_closed(points: np.ndarray, spacing_px: float) -> np.ndarray:
    """Resample a closed ring to approximately uniform arc-length spacing.

    The wraparound segment from the last vertex back to the first is included,
    so vertex density does not dip across the seam - an artefact that would
    otherwise show up as a periodic wobble in any along-contour statistic.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return points
    closed = np.vstack([points, points[:1]])
    segment = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    total = float(arc[-1])
    if total <= 0:
        return points
    count = max(3, int(round(total / float(spacing_px))))
    targets = np.linspace(0.0, total, count, endpoint=False)
    return np.column_stack(
        [np.interp(targets, arc, closed[:, 0]), np.interp(targets, arc, closed[:, 1])]
    )


def _smooth_closed(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along a closed ring (wrap boundary)."""
    from scipy.ndimage import gaussian_filter1d

    if sigma <= 0:
        return values
    return gaussian_filter1d(values, sigma=sigma, mode="wrap")


def tangents(points: np.ndarray, smooth_px: float) -> np.ndarray:
    """Unit tangents along a closed ring.

    The vertex sequence is smoothed before differentiating.  A marching-squares
    polygon traced from a *binary* mask is a staircase, so an unsmoothed central
    difference oscillates between axis-aligned directions and the normals built
    from it are useless for sampling an intensity profile.  Too much smoothing
    biases a genuinely curved boundary, which is why the default is small.
    """
    y = _smooth_closed(points[:, 0].astype(np.float64), smooth_px)
    x = _smooth_closed(points[:, 1].astype(np.float64), smooth_px)
    dy = 0.5 * (np.roll(y, -1) - np.roll(y, 1))
    dx = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
    norm = np.hypot(dy, dx)
    norm[norm < 1e-12] = 1.0
    return np.column_stack([dy / norm, dx / norm])


def outward_normals(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    smooth_px: float = 2.0,
    probe_px: float = 1.5,
) -> np.ndarray:
    """Unit normals pointing out of the mask.

    Orientation is decided by a **majority vote** over all vertices, not per
    vertex.  A per-vertex test flips exactly where the mask is ragged, which is
    where the geometry is already least reliable, and a single flipped normal
    puts that vertex's refined position on the wrong side of the edge.

    The vote probes the *mask*, never the intensity: SEM contrast polarity
    varies between detectors and between feature types, so "inside is brighter"
    is not a fact this package may assume.

    Hole boundaries need no special case.  "Out of the mask" on an annulus means
    *toward* the hole centre, which is what the same vote produces: probing
    outward in radius from the inner boundary lands inside the ring, so the
    normal is flipped inward.  Both boundaries therefore traverse their
    intensity profile material-first, which is what ``refine`` relies on.
    """
    tangent = tangents(points, smooth_px)
    # Rotate the tangent by -90 degrees in (y, x).
    candidate = np.column_stack([tangent[:, 1], -tangent[:, 0]])

    probe = points + probe_px * candidate
    rows = np.clip(np.round(probe[:, 0]).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.round(probe[:, 1]).astype(int), 0, mask.shape[1] - 1)
    inside_fraction = float(np.mean(mask[rows, cols]))

    if inside_fraction > 0.5:
        candidate = -candidate
    return candidate


def trace_instance(
    instance: InstanceMask,
    config: ContoursConfig,
    *,
    region_id: int = 0,
) -> tuple[Contour | None, list[Contour]]:
    """Trace the outer boundary and any surviving holes of one instance.

    The mask crop is padded by one pixel before tracing so that a region running
    up to the image border still yields a *closed* ring; without the pad
    ``find_contours`` returns an open polyline there and every downstream
    polygon measure silently becomes wrong.
    """
    from skimage.measure import find_contours

    padded = np.pad(instance.crop, 1, mode="constant", constant_values=False)
    rings = find_contours(padded.astype(np.float64), 0.5)
    if not rings:
        return None, []

    y0, _, x0, _ = instance.bbox
    offset = np.array([y0 - 1.0, x0 - 1.0])
    polygons = [ring + offset for ring in rings]
    # find_contours repeats the first vertex on a closed ring; drop it so the
    # closure stays implicit and resampling does not see a zero-length segment.
    polygons = [p[:-1] if np.allclose(p[0], p[-1]) else p for p in polygons]
    polygons = [p for p in polygons if p.shape[0] >= 3]
    if not polygons:
        return None, []

    areas = [abs(polygon_area(p)) for p in polygons]
    outer_index = int(np.argmax(areas))

    outer_points = resample_closed(polygons[outer_index], config.spacing_px)
    if outer_points.shape[0] < config.min_vertices:
        return None, []
    outer = Contour(points=outer_points, is_hole=False, region_id=region_id)

    holes: list[Contour] = []
    for index, polygon in enumerate(polygons):
        if index == outer_index:
            continue
        hole_points = resample_closed(polygon, config.spacing_px)
        if hole_points.shape[0] >= config.min_vertices:
            holes.append(Contour(points=hole_points, is_hole=True, region_id=region_id))
    return outer, holes


def attach_normals(
    contour: Contour,
    mask: np.ndarray,
    config: ContoursConfig,
) -> Contour:
    """Compute and store outward normals on a contour."""
    contour.normals = outward_normals(
        contour.points, mask, smooth_px=config.normal_smooth_px
    )
    return contour


def trace_all(
    instances: list[InstanceMask],
    shape: tuple[int, int],
    config: ContoursConfig,
) -> tuple[list[Contour], list[list[Contour]]]:
    """Trace every instance, returning outer contours and per-region holes."""
    outers: list[Contour] = []
    holes: list[list[Contour]] = []
    for region_id, instance in enumerate(instances, start=1):
        mask = instance.full_mask(shape)
        outer, region_holes = trace_instance(instance, config, region_id=region_id)
        if outer is None:
            outers.append(Contour(points=np.zeros((0, 2)), region_id=region_id))
            holes.append([])
            continue
        attach_normals(outer, mask, config)
        for hole in region_holes:
            attach_normals(hole, mask, config)
        outers.append(outer)
        holes.append(region_holes)
    return outers, holes
