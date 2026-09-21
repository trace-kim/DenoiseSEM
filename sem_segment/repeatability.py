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


def summarize_observations(rows: list[dict], series_names: list[str]) -> tuple[list[dict], list[dict]]:
    """Sample SD per hole, then median SD over holes usable in every series.

    Never pool dimensions across different holes. A hole needs two finite valid
    observations to contribute. Missing observations still count as attempts.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["series"], row["hole"], row["method"]].append(row)
    per_hole = []
    for (series, hole, method), observations in groups.items():
        valid = [r for r in observations if r["status"] == "valid"]
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
        per_hole.append(result)
    summaries = []
    for method in ("coarse", "refined"):
        usable = [{r["hole"] for r in per_hole if r["series"] == name and r["method"] == method
                   and r["cd_std"] is not None} for name in series_names]
        common = set.intersection(*usable) if usable else set()
        for name in series_names:
            all_holes = [r for r in per_hole if r["series"] == name and r["method"] == method]
            selected = [r for r in all_holes if r["hole"] in common]
            sd = float(np.median([r["cd_std"] for r in selected])) if selected else None
            summaries.append({"series": name, "method": method, "common_holes": sorted(common),
                              "unit": all_holes[0]["unit"] if all_holes else None,
                              "common_hole_count": len(common), "median_cd_std": sd,
                              "median_cd_3sigma": 3 * sd if sd is not None else None,
                              "contributing_observations": sum(r["valid_count"] for r in selected),
                              "valid_count": sum(r["valid_count"] for r in all_holes),
                              "failed_count": sum(r["failed_count"] for r in all_holes)})
    return per_hole, summaries
