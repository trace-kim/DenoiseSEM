"""Correspondence and within-hole repeatability on native-pixel measurements.

Translations here only move centroids into the template coordinate system.
They never resample images or change the measured dimensions.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def correspondence_gate(centroids: np.ndarray, requested: float | None = None) -> float:
    """Keep the gate strictly below half the median nearest-neighbour spacing."""
    points = np.asarray(centroids, dtype=float).reshape(-1, 2)
    if requested is not None and (not np.isfinite(requested) or requested <= 0):
        raise ValueError("match_gate_px must be finite and positive")
    if len(points) < 2:
        return requested or 10.0  # No inter-hole spacing exists for a single hole.
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(distances.min(axis=1)))
    if spacing <= 0:
        raise ValueError("template contains coincident centroids")
    if requested is not None and requested >= spacing / 2:
        raise ValueError("match_gate_px must be below half the template median nearest-neighbour spacing")
    return requested if requested is not None else 0.45 * spacing


def match_centroids(template: np.ndarray, observed: np.ndarray, shift_yx: np.ndarray,
                    gate: float) -> list[dict]:
    """One-to-one nearest matches; competing candidates are explicit ambiguities.

    ``shift_yx`` is the content drift in the native observation relative to the
    template. Reject competing candidates rather than choosing an arbitrary ID.
    """
    template = np.asarray(template, dtype=float).reshape(-1, 2)
    observed = np.asarray(observed, dtype=float).reshape(-1, 2)
    if not np.isfinite(shift_yx).all():
        return [{"hole": i + 1, "match_status": "registration_failed", "region_index": None}
                for i in range(len(template))]
    distances = np.linalg.norm(template[:, None] - (observed - shift_yx)[None, :], axis=2)
    candidates = distances < gate
    result = []
    for i in range(len(template)):
        indices = np.flatnonzero(candidates[i])
        status, chosen = "missing", None
        if len(indices):
            nearest = int(indices[np.argmin(distances[i, indices])])
            if len(indices) != 1 or candidates[:, nearest].sum() != 1:
                status = "ambiguous"
            else:
                status, chosen = "matched", nearest
        result.append({"hole": i + 1, "match_status": status, "region_index": chosen})
    return result


def placement_residuals(positions: list[dict], *, min_holes: int = 3,
                        iterations: int = 50) -> list[dict]:
    """Hole position relative to the other holes in the same image.

    Each position row names series, method, frame, hole and a centroid (y, x).
    Per series and method, fit position = hole mean + frame shift by
    alternating means (missing holes allowed), then return the residuals. A
    common image shift is removed because it will be corrected; what remains
    is how the distances between contours change. Frames with fewer than
    ``min_holes`` holes are skipped. Residuals are scaled by sqrt(N/(N-1)): the
    shift estimate includes the hole itself, which otherwise shrinks it.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in positions:
        if np.isfinite([row["y"], row["x"]]).all():
            groups[row["series"], row["method"]].append(row)
    result = []
    for (series, method), rows in groups.items():
        frames = sorted({r["frame"] for r in rows})
        holes = sorted({r["hole"] for r in rows})
        f_index = {f: i for i, f in enumerate(frames)}
        h_index = {h: i for i, h in enumerate(holes)}
        grid = np.full((len(frames), len(holes), 2), np.nan)
        for r in rows:
            grid[f_index[r["frame"]], h_index[r["hole"]]] = r["y"], r["x"]
        counts = np.isfinite(grid[..., 0]).sum(axis=1)
        grid[counts < min_holes] = np.nan
        present = np.isfinite(grid[..., 0])
        if not present.any():
            continue
        shift = np.zeros((len(frames), 1, 2))
        for _ in range(iterations):
            mean = np.nanmean(np.where(present[..., None], grid - shift, np.nan), axis=0, keepdims=True)
            updated = np.nanmean(np.where(present[..., None], grid - mean, np.nan), axis=1, keepdims=True)
            updated = np.nan_to_num(updated)
            converged = np.allclose(updated, shift, atol=1e-9, rtol=0)
            shift = updated
            if converged:
                break
        counts = present.sum(axis=1)
        scale = np.sqrt(counts / np.maximum(counts - 1, 1))[:, None, None]
        residual = (grid - mean - shift) * scale
        for i, frame in enumerate(frames):
            for j, hole in enumerate(holes):
                if present[i, j]:
                    result.append({"series": series, "method": method, "frame": frame, "hole": hole,
                                   "residual_y": float(residual[i, j, 0]), "residual_x": float(residual[i, j, 1]),
                                   "frame_shift_y": float(shift[i, 0, 0]), "frame_shift_x": float(shift[i, 0, 1]),
                                   "holes_in_frame": int(counts[i])})
    return result


def placement_per_hole(residuals: list[dict]) -> dict[tuple, dict]:
    """Sample SD of the relative position per hole, per axis."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in residuals:
        groups[row["series"], row["hole"], row["method"]].append(row)
    result = {}
    for key, rows in groups.items():
        values = {}
        for axis in ("y", "x"):
            data = np.asarray([r[f"residual_{axis}"] for r in rows], dtype=float)
            sd = float(data.std(ddof=1)) if len(data) >= 2 else None
            values[f"placement_std_{axis}"] = sd
            values[f"placement_3sigma_{axis}"] = 3 * sd if sd is not None else None
        values["placement_count"] = len(rows)
        result[key] = values
    return result


def summarize_observations(rows: list[dict], series_names: list[str], *,
                           comparison_series: list[str] | None = None,
                           placement: dict[tuple, dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Sample SD per hole, then median SD over common comparison holes.

    Never pool dimensions across different holes. A hole needs two finite valid
    observations to contribute. Missing observations still count as attempts.
    By default every series contributes to the common-hole intersection. With
    comparison_series, other series retain their own independent coverage.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["series"], row["hole"], row["method"]].append(row)
    per_hole = []
    for (series, hole, method), observations in groups.items():
        valid = [r for r in observations if r["status"] == "valid" and r.get("cd") is not None and np.isfinite(r["cd"])]
        result = {"series": series, "hole": hole, "method": method,
                  "unit": observations[0].get("unit", "px"),
                  "attempted_count": len(observations), "valid_count": len(valid),
                  "failed_count": len(observations) - len(valid),
                  "clipped_observations": sum(bool(r.get("clipped")) for r in observations)}
        for measure in ("cd", "major_axis", "minor_axis"):
            values = np.asarray([r[measure] for r in valid], dtype=float)
            result[f"{measure}_mean"] = float(values.mean()) if len(values) else None
            sd = float(values.std(ddof=1)) if len(values) >= 2 else None
            result[f"{measure}_std"] = sd
            result[f"{measure}_3sigma"] = 3 * sd if sd is not None else None
        result.update(precision_components(observations))
        result.update((placement or {}).get((series, hole, method), {
            "placement_std_y": None, "placement_3sigma_y": None,
            "placement_std_x": None, "placement_3sigma_x": None, "placement_count": 0}))
        per_hole.append(result)
    summaries = []
    for method in ("coarse", "refined"):
        compared = series_names if comparison_series is None else comparison_series
        usable = [{r["hole"] for r in per_hole if r["series"] == name and r["method"] == method
                   and r["cd_std"] is not None} for name in compared]
        common = set.intersection(*usable) if usable else set()
        for name in series_names:
            all_holes = [r for r in per_hole if r["series"] == name and r["method"] == method]
            # Baselines have their own coverage; a failed raw segmentation must
            # not remove valid holes from a comparison between model outputs.
            selected = [r for r in all_holes if r["cd_std"] is not None and
                        (name not in compared or r["hole"] in common)]
            sd = float(np.median([r["cd_std"] for r in selected])) if selected else None
            selected_holes = sorted(r["hole"] for r in selected)
            summaries.append({"series": name, "method": method, "common_holes": selected_holes,
                              "comparison_series": list(compared) if name in compared else [name],
                              "unit": all_holes[0]["unit"] if all_holes else None,
                              "common_hole_count": len(selected_holes), "median_cd_std": sd,
                              "median_cd_3sigma": 3 * sd if sd is not None else None,
                              "contributing_observations": sum(r["valid_count"] for r in selected),
                              "valid_count": sum(r["valid_count"] for r in all_holes),
                              "failed_count": sum(r["failed_count"] for r in all_holes)})
            summary = summaries[-1]
            summary["observations_per_hole"] = {str(r["hole"]): r["valid_count"] for r in selected}
            for field in ("cd_std", "cd_std_detrended", "cd_std_successive"):
                values = [r[field] for r in selected if r[field] is not None]
                summary[f"median_{field}"] = float(np.median(values)) if values else None
                summary[f"median_{field}_ci95"] = bootstrap_median(values)
            # Same holes as the CD summary, so both precisions describe one population.
            for axis in ("y", "x"):
                values = [r[f"placement_std_{axis}"] for r in selected if r[f"placement_std_{axis}"] is not None]
                sd = float(np.median(values)) if values else None
                summary[f"median_placement_3sigma_{axis}"] = 3 * sd if sd is not None else None
    return per_hole, summaries


def precision_components(observations: list[dict]) -> dict:
    """Separate drift from jitter; gaps never become consecutive observations."""
    ordered = sorted(enumerate(observations), key=lambda pair: pair[1].get("order", pair[0]))
    result = {"cd_std_detrended": None, "cd_std_successive": None,
              "cd_brightness_correlation": None, "cd_threshold_correlation": None}
    usable = [(i, r) for i, r in ordered if r["status"] == "valid" and r.get("cd") is not None and np.isfinite(r["cd"])]
    if len(usable) < 3:
        return result
    values = np.array([r["cd"] for _, r in usable])
    order = np.array([r.get("order", i) for i, r in usable], dtype=float)
    design = np.column_stack((order - order.mean(), np.ones(len(order))))
    if np.linalg.matrix_rank(design) == 2:
        residuals = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        # Two fitted parameters; unbiased white-noise variance after detrending.
        result["cd_std_detrended"] = float(np.sqrt(np.sum(residuals**2) / (len(values) - 2)))
    adjacent = []
    for (_, a), (_, b) in zip(ordered, ordered[1:]):
        if all(r["status"] == "valid" and r.get("cd") is not None and np.isfinite(r["cd"]) for r in (a, b)):
            adjacent.append(b["cd"] - a["cd"])
    if len(adjacent) >= 2:
        result["cd_std_successive"] = float(np.std(adjacent, ddof=1) / np.sqrt(2))
    for field, output in (("mean_dn", "cd_brightness_correlation"), ("otsu_threshold_dn", "cd_threshold_correlation")):
        pairs = [(r["cd"], r.get(field)) for _, r in usable if r.get(field) is not None and np.isfinite(r[field])]
        if len(pairs) >= 3:
            x, y = np.asarray(pairs).T
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                result[output] = float(np.corrcoef(x, y)[0, 1])
    return result


def bootstrap_median(values: list[float], *, samples: int = 2000) -> list[float] | None:
    """Deterministic percentile interval, resampling holes, never frames."""
    if len(values) < 2:
        return None
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    medians = []
    # Bound bootstrap storage even on sites with many thousands of holes.
    for start in range(0, samples, 64):
        draws = rng.choice(data, size=(min(64, samples - start), len(data)))
        medians.extend(np.median(draws, axis=1))
    return np.percentile(medians, [2.5, 97.5]).tolist()
