"""Shape-agnostic measurements on a closed contour.

Everything here works on an arbitrary closed polygon: no assumption that a
feature is a line, a hole, or a particle.  Measurements are computed from the
polygon rather than by counting pixels, because the point of the refinement step
is a boundary located between pixel centres, and a pixel count throws that away.

Two choices are worth stating plainly.

*The critical dimension defaults to the equivalent circular diameter, not the
minimum Feret caliper.*  A minimum taken over many caliper angles is an
extreme-value statistic: skewed, biased low, and noisier than the shape it
describes.  It has no business as the headline number in a repeatability table,
though it remains available for cases where the narrowest width is the quantity
of interest.

*Line-edge roughness is measured from the refined boundary's deviation about its
own smooth trend - never from the refinement displacement.*  The displacement
says how far the segmentation's boundary was from the truth, which is a property
of the model.  Roughness is a property of the specimen.  Conflating them
produces a number that improves when the segmentation gets worse.

All values are in pixels.  Conversion to nanometres happens at the reporting
boundary, from an operator-supplied pixel size; nothing here reads a scale from
an image file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import MetrologyConfig
from .contours import Contour, polygon_area, polygon_centroid, polygon_perimeter
from .refine import RefinedContour


@dataclass
class ShapeMeasures:
    """Geometry of one closed polygon."""

    area_px2: float
    perimeter_px: float
    centroid_y: float
    centroid_x: float
    equivalent_diameter_px: float
    circularity: float
    solidity: float
    convexity: float
    feret_min_px: float
    feret_max_px: float
    feret_mean_px: float
    aspect_ratio: float
    major_axis_px: float
    minor_axis_px: float
    orientation_deg: float
    eccentricity: float
    bbox_height_px: float
    bbox_width_px: float
    bbox_fill: float
    chord_min_px: float
    chord_max_px: float
    chord_mean_px: float
    cd_px: float


def polygon_moments(points: np.ndarray) -> tuple[float, float, float, float, float]:
    """Area, centroid and central second moments of a polygon.

    Returns ``(area, cy, cx, mu_yy, mu_xx, mu_yx)`` normalised by area, i.e. the
    covariance of the enclosed region - the quantities an equivalent ellipse is
    built from.
    """
    y = np.asarray(points[:, 0], dtype=np.float64)
    x = np.asarray(points[:, 1], dtype=np.float64)
    y1, x1 = np.roll(y, -1), np.roll(x, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        return 0.0, float(y.mean()), float(x.mean()), 0.0, 0.0, 0.0

    cx = float(np.dot(x + x1, cross) / (6.0 * area))
    cy = float(np.dot(y + y1, cross) / (6.0 * area))
    # Second moments about the origin, then shifted to the centroid.
    i_yy = float(np.dot(y**2 + y * y1 + y1**2, cross) / 12.0)
    i_xx = float(np.dot(x**2 + x * x1 + x1**2, cross) / 12.0)
    i_xy = float(np.dot(x * y1 + 2.0 * x * y + 2.0 * x1 * y1 + x1 * y, cross) / 24.0)

    mu_yy = i_yy / area - cy * cy
    mu_xx = i_xx / area - cx * cx
    mu_yx = i_xy / area - cx * cy
    return abs(area), cy, cx, mu_yy, mu_xx, mu_yx


def equivalent_ellipse(points: np.ndarray) -> tuple[float, float, float, float]:
    """Major axis, minor axis, orientation (degrees) and eccentricity."""
    _, _, _, mu_yy, mu_xx, mu_yx = polygon_moments(points)
    covariance = np.array([[mu_xx, mu_yx], [mu_yx, mu_yy]])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    minor, major = 4.0 * np.sqrt(eigenvalues[0]), 4.0 * np.sqrt(eigenvalues[1])
    principal = eigenvectors[:, 1]  # (x, y) of the largest eigenvalue
    orientation = float(np.degrees(np.arctan2(principal[1], principal[0])))
    if major <= 0:
        eccentricity = 0.0
    else:
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))
    return float(major), float(minor), orientation, eccentricity


def convex_hull_points(points: np.ndarray) -> np.ndarray:
    """Convex hull vertices in order, as (M, 2) ``(y, x)``."""
    from scipy.spatial import ConvexHull

    if points.shape[0] < 3:
        return points
    try:
        hull = ConvexHull(points[:, ::-1])  # ConvexHull wants (x, y)
    except Exception:  # pragma: no cover - degenerate (collinear) polygons
        return points
    return points[hull.vertices]


def feret_diameters(points: np.ndarray, *, hull: np.ndarray | None = None) -> tuple[float, float, float]:
    """Minimum, maximum and mean caliper width.

    The minimum width of a convex body is attained perpendicular to one of its
    hull edges (the rotating-calipers theorem), so checking each edge normal is
    exact rather than a sampled approximation.  The mean follows from Cauchy's
    formula - the mean caliper width of a convex body is its perimeter divided
    by pi - which is also a useful internal consistency check.
    Callers that already measured the hull of these points may supply it.
    """
    if hull is None:
        hull = convex_hull_points(points)
    if hull.shape[0] < 2:
        return 0.0, 0.0, 0.0

    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    keep = lengths > 1e-12
    normals = np.column_stack([-edges[keep, 1], edges[keep, 0]]) / lengths[keep, None]

    projections = hull @ normals.T  # (hull points, edges)
    widths = projections.max(axis=0) - projections.min(axis=0)
    feret_min = float(widths.min()) if widths.size else 0.0

    differences = hull[:, None, :] - hull[None, :, :]
    feret_max = float(np.hypot(differences[..., 0], differences[..., 1]).max())

    hull_perimeter = polygon_perimeter(hull)
    return feret_min, feret_max, float(hull_perimeter / np.pi)


def chord_widths(points: np.ndarray, centroid: tuple[float, float], angles: int) -> np.ndarray:
    """Span of the polygon through its centroid, sampled at ``angles`` directions.

    A shape-agnostic stand-in for "width": for a line it recovers the line width
    at its narrowest direction, for a hole the diameter, and for an arbitrary
    blob a well-defined extent - without ever branching on feature type.
    """
    cy, cx = centroid
    theta = np.linspace(0.0, np.pi, angles, endpoint=False)
    starts = points
    ends = np.roll(points, -1, axis=0)

    widths = np.zeros(angles)
    # Bound temporary arrays even for unusually long contours or many angles.
    batch = max(1, min(angles, 262144 // max(1, len(points))))
    for first in range(0, angles, batch):
        dy, dx = np.sin(theta[first:first + batch, None]), np.cos(theta[first:first + batch, None])
        side_start = (starts[:, 0] - cy) * dx - (starts[:, 1] - cx) * dy
        side_end = (ends[:, 0] - cy) * dx - (ends[:, 1] - cx) * dy
        rows, edges = np.nonzero((side_start <= 0) != (side_end <= 0))
        denominator = side_start[rows, edges] - side_end[rows, edges]
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        fraction = side_start[rows, edges] / denominator
        hits = starts[edges] + fraction[:, None] * (ends[edges] - starts[edges])
        t = (hits[:, 0] - cy) * dy[rows, 0] + (hits[:, 1] - cx) * dx[rows, 0]
        count = len(dy)
        lo, hi = np.full(count, np.inf), np.full(count, -np.inf)
        np.minimum.at(lo, rows, t)
        np.maximum.at(hi, rows, t)
        present = np.isfinite(lo)
        widths[first + np.flatnonzero(present)] = (hi - lo)[present]
    return widths


def measure_shape(
    contour: Contour | np.ndarray,
    config: MetrologyConfig,
    *,
    holes: list[np.ndarray] | None = None,
) -> ShapeMeasures | None:
    """Measure one closed polygon, subtracting any holes from its area."""
    points = contour.points if isinstance(contour, Contour) else np.asarray(contour)
    if points.shape[0] < 3 or not np.isfinite(points).all():
        return None

    area = abs(polygon_area(points))
    perimeter = polygon_perimeter(points)
    hole_area = sum(abs(polygon_area(h)) for h in (holes or []) if h.shape[0] >= 3)
    net_area = max(area - hole_area, 0.0)
    if net_area <= 0 or perimeter <= 0:
        return None

    cy, cx = polygon_centroid(points)
    hull = convex_hull_points(points)
    hull_area = abs(polygon_area(hull)) if hull.shape[0] >= 3 else area
    hull_perimeter = polygon_perimeter(hull) if hull.shape[0] >= 3 else perimeter

    feret_min, feret_max, feret_mean = feret_diameters(points, hull=hull)
    major, minor, orientation, eccentricity = equivalent_ellipse(points)
    chords = chord_widths(points, (cy, cx), config.chord_angles)

    height = float(np.ptp(points[:, 0]))
    width = float(np.ptp(points[:, 1]))
    equivalent_diameter = 2.0 * float(np.sqrt(net_area / np.pi))

    cd = {
        "equivalent_diameter": equivalent_diameter,
        "feret_min": feret_min,
        "feret_mean": feret_mean,
        "chord_mean": float(chords.mean()) if chords.size else 0.0,
    }[config.cd_definition]

    return ShapeMeasures(
        area_px2=net_area,
        perimeter_px=perimeter,
        centroid_y=cy,
        centroid_x=cx,
        equivalent_diameter_px=equivalent_diameter,
        circularity=float(4.0 * np.pi * net_area / (perimeter**2)),
        solidity=float(net_area / hull_area) if hull_area > 0 else 0.0,
        convexity=float(hull_perimeter / perimeter) if perimeter > 0 else 0.0,
        feret_min_px=feret_min,
        feret_max_px=feret_max,
        feret_mean_px=feret_mean,
        aspect_ratio=float(feret_max / feret_min) if feret_min > 1e-9 else float("inf"),
        major_axis_px=major,
        minor_axis_px=minor,
        orientation_deg=orientation,
        eccentricity=eccentricity,
        bbox_height_px=height,
        bbox_width_px=width,
        bbox_fill=float(net_area / (height * width)) if height * width > 0 else 0.0,
        chord_min_px=float(chords.min()) if chords.size else 0.0,
        chord_max_px=float(chords.max()) if chords.size else 0.0,
        chord_mean_px=float(chords.mean()) if chords.size else 0.0,
        cd_px=cd,
    )


def edge_roughness(
    refined: RefinedContour,
    config: MetrologyConfig,
    *,
    spacing_px: float = 1.0,
) -> dict:
    """Line-edge roughness of the measured boundary about its own smooth trend.

    The residual is taken between the refined boundary and a low-pass version of
    *itself*, so the number describes the specimen.  Using the refinement
    displacement instead would describe how wrong the segmentation was, and
    would get "better" as the segmentation got worse.
    """
    from scipy.ndimage import gaussian_filter1d

    valid = refined.valid
    if config.roughness_detrend == "none" or int(np.count_nonzero(valid)) < 8:
        return {"ler_3sigma_px": float("nan"), "ler_rms_px": float("nan"), "ler_points": 0}

    points = refined.refined_points
    normals = refined.normals
    # Work on the closed sequence with invalid vertices bridged, so the smoother
    # sees no discontinuity, then measure only where the data is real.
    filled_y = np.interp(
        np.arange(points.shape[0]), np.flatnonzero(valid), points[valid, 0], period=points.shape[0]
    )
    filled_x = np.interp(
        np.arange(points.shape[0]), np.flatnonzero(valid), points[valid, 1], period=points.shape[0]
    )
    sigma_samples = max(config.ler_highpass_cutoff_px / max(spacing_px, 1e-9), 0.5)
    trend_y = gaussian_filter1d(filled_y, sigma=sigma_samples, mode="wrap")
    trend_x = gaussian_filter1d(filled_x, sigma=sigma_samples, mode="wrap")

    residual = (filled_y - trend_y) * normals[:, 0] + (filled_x - trend_x) * normals[:, 1]
    residual = residual[valid]
    return {
        "ler_3sigma_px": float(3.0 * np.std(residual, ddof=1)),
        "ler_rms_px": float(np.sqrt(np.mean(residual**2))),
        "ler_points": int(residual.size),
    }


def _nanmean_or_nan(values: np.ndarray) -> float:
    """Mean of the finite entries, or nan when there are none."""
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if finite.any() else float("nan")


def _to_nanometres(row: dict, pixel_size_nm: float) -> dict:
    """Nanometre twins of every pixel-unit column.

    The marker can sit anywhere in the name, not just at the end - a column is
    called ``area_px2_coarse``, not ``area_coarse_px2`` - so the match is on the
    substring. ``_px2`` is checked first, since ``_px`` is a prefix of it.
    """
    scaled: dict = {}
    for key, value in row.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if "_nm" in key:
            continue
        if "_px2" in key:
            scaled[key.replace("_px2", "_nm2")] = value * pixel_size_nm**2
        elif "_px" in key:
            scaled[key.replace("_px", "_nm")] = value * pixel_size_nm
    return scaled


@dataclass
class RegionMetrology:
    """One row of the metrology table: both contour methods plus diagnostics."""

    region_id: int
    coarse: ShapeMeasures | None
    refined: ShapeMeasures | None
    score: float
    backend: str
    prompt: str | None
    touches_border: bool
    n_holes: int
    hole_area_px2: float
    valid_fraction: float
    shift_mean_px: float
    shift_abs_mean_px: float
    shift_3sigma_px: float
    edge_width_px: float
    edge_contrast: float
    roughness: dict = field(default_factory=dict)
    reject_counts: dict = field(default_factory=dict)

    def to_row(self, *, pixel_size_nm: float | None = None) -> dict:
        """Flatten to a CSV row, with method-suffixed geometry columns."""
        row: dict = {
            "region_id": self.region_id,
            "backend": self.backend,
            "prompt": self.prompt or "",
            "score": self.score,
            "touches_border": int(self.touches_border),
            "n_holes": self.n_holes,
            "hole_area_px2": self.hole_area_px2,
            "valid_fraction": self.valid_fraction,
            "shift_mean_px": self.shift_mean_px,
            "shift_abs_mean_px": self.shift_abs_mean_px,
            "shift_3sigma_px": self.shift_3sigma_px,
            "edge_width_px": self.edge_width_px,
            "edge_contrast": self.edge_contrast,
        }
        for method, measures in (("coarse", self.coarse), ("refined", self.refined)):
            values = asdict(measures) if measures is not None else {}
            for key in ShapeMeasures.__dataclass_fields__:
                row[f"{key}_{method}"] = values.get(key, float("nan"))
        row.update(self.roughness)
        for reason, count in self.reject_counts.items():
            row[f"reject_{reason}"] = count

        if pixel_size_nm:
            row.update(_to_nanometres(row, pixel_size_nm))
        return row


def measure_region(
    region_id: int,
    instance,
    coarse: Contour,
    holes: list[Contour],
    refined: RefinedContour | None,
    config: MetrologyConfig,
    *,
    spacing_px: float = 1.0,
) -> RegionMetrology:
    """Measure one region by both contour methods."""
    hole_polygons = [h.points for h in holes]
    coarse_measures = measure_shape(coarse, config, holes=hole_polygons)
    refined_measures = None
    roughness: dict = {}
    valid_fraction = 0.0
    shift_mean = shift_abs = shift_3sigma = float("nan")
    edge_width = edge_contrast = float("nan")
    reject_counts: dict = {}

    if refined is not None and refined.valid.any():
        refined_measures = measure_shape(refined.as_contour(), config, holes=hole_polygons)
        roughness = edge_roughness(refined, config, spacing_px=spacing_px)
        valid_fraction = refined.valid_fraction
        shifts = refined.displacement[refined.valid]
        shift_mean = float(np.mean(shifts))
        shift_abs = float(np.mean(np.abs(shifts)))
        shift_3sigma = float(3.0 * np.std(shifts, ddof=1)) if shifts.size > 1 else float("nan")
        # Only the erf estimator produces an edge width; the others leave it all
        # nan, and nanmean of an all-nan slice warns rather than returning nan.
        edge_width = _nanmean_or_nan(refined.edge_width)
        edge_contrast = _nanmean_or_nan(refined.contrast)
        reject_counts = refined.reason_counts()

    return RegionMetrology(
        region_id=region_id,
        coarse=coarse_measures,
        refined=refined_measures,
        score=float(instance.score),
        backend=instance.backend,
        prompt=instance.prompt,
        touches_border=bool(instance.meta.get("touches_border", False)),
        n_holes=int(instance.meta.get("n_holes", 0)),
        hole_area_px2=float(instance.meta.get("hole_area_px", 0.0)),
        valid_fraction=valid_fraction,
        shift_mean_px=shift_mean,
        shift_abs_mean_px=shift_abs,
        shift_3sigma_px=shift_3sigma,
        edge_width_px=edge_width,
        edge_contrast=edge_contrast,
        roughness=roughness,
        reject_counts=reject_counts,
    )


def _nearest_neighbour_distances(centroids: np.ndarray, k: int) -> np.ndarray:
    """Distance to the k-th nearest other centroid, for each point."""
    if centroids.shape[0] < 2:
        return np.zeros(centroids.shape[0])
    differences = centroids[:, None, :] - centroids[None, :, :]
    distances = np.hypot(differences[..., 0], differences[..., 1])
    np.fill_diagonal(distances, np.inf)
    order = np.sort(distances, axis=1)
    return order[:, min(k, order.shape[1]) - 1]


def summarise_image(
    regions: list[RegionMetrology],
    shape: tuple[int, int],
    config: MetrologyConfig,
    *,
    include_border_regions: bool = False,
    pixel_size_nm: float | None = None,
) -> dict:
    """Image-level aggregates.

    Border-touching regions are measured and reported individually but excluded
    from these aggregates by default: their visible extent is real, their true
    size is not, and averaging truncated features into a size distribution is
    how a population quietly acquires a low-side tail.
    """
    usable = [r for r in regions if include_border_regions or not r.touches_border]
    summary: dict = {
        "region_count": len(regions),
        "region_count_aggregated": len(usable),
        "region_count_border": sum(1 for r in regions if r.touches_border),
        "image_height_px": int(shape[0]),
        "image_width_px": int(shape[1]),
    }
    if not usable:
        return summary

    def stats(name: str, values: list[float]) -> None:
        array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if array.size == 0:
            return
        summary[f"{name}_mean"] = float(array.mean())
        summary[f"{name}_median"] = float(np.median(array))
        summary[f"{name}_std"] = float(array.std(ddof=1)) if array.size > 1 else 0.0
        summary[f"{name}_min"] = float(array.min())
        summary[f"{name}_max"] = float(array.max())

    for method in ("coarse", "refined"):
        measures = [getattr(r, method) for r in usable]
        present = [m for m in measures if m is not None]
        if not present:
            continue
        stats(f"cd_px_{method}", [m.cd_px for m in present])
        stats(f"area_px2_{method}", [m.area_px2 for m in present])
        stats(f"equivalent_diameter_px_{method}", [m.equivalent_diameter_px for m in present])
        stats(f"circularity_{method}", [m.circularity for m in present])
        summary[f"coverage_fraction_{method}"] = float(
            sum(m.area_px2 for m in present) / (shape[0] * shape[1])
        )

    stats("shift_abs_mean_px", [r.shift_abs_mean_px for r in usable])
    stats("valid_fraction", [r.valid_fraction for r in usable])
    stats("ler_3sigma_px", [r.roughness.get("ler_3sigma_px", float("nan")) for r in usable])
    stats("edge_width_px", [r.edge_width_px for r in usable])

    centroids = np.array(
        [(r.refined or r.coarse).centroid_y for r in usable if (r.refined or r.coarse)],
    )
    if centroids.size:
        points = np.array(
            [
                ((r.refined or r.coarse).centroid_y, (r.refined or r.coarse).centroid_x)
                for r in usable
                if (r.refined or r.coarse)
            ]
        )
        stats("nearest_neighbour_px", list(_nearest_neighbour_distances(points, config.nearest_neighbours)))

    if pixel_size_nm:
        summary.update(_to_nanometres(summary, pixel_size_nm))
    return summary
