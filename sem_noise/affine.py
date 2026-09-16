"""Small-motion affine diagnostics from tiles; never warp measurement pixels.

Local translations approximate point correspondences at tile centers. Matrices
map native ROI pixel centers (x right, y down) to the leave-one-out mean's
coordinates. They are diagnostic estimates, not calibrated stage motion.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .config import AnalysisConfig

MODELS = ("translation", "rigid", "similarity", "affine")


def _solve(points: np.ndarray, targets: np.ndarray, weights: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    weights = weights / weights.sum()
    origin = weights @ points
    destination = weights @ targets
    p, q = points - origin, targets - destination
    if model == "translation":
        matrix = np.eye(2)
    elif model == "affine":
        matrix, _, rank, _ = np.linalg.lstsq(p * np.sqrt(weights[:, None]),
                                            q * np.sqrt(weights[:, None]), rcond=None)
        if rank < 2:
            raise ValueError("tile centers are collinear")
        matrix = matrix.T
    else:
        u, singular, vt = np.linalg.svd(p.T @ (weights[:, None] * q))
        sign = np.array([1.0, np.linalg.det(vt.T @ u.T)])
        matrix = vt.T @ np.diag(sign) @ u.T
        if model == "similarity":
            matrix *= np.dot(singular, sign) / max(np.sum(weights[:, None] * p * p), 1e-12)
    return matrix, destination - matrix @ origin


def _fit(points: np.ndarray, targets: np.ndarray, quality: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    weights = quality.copy()
    for _ in range(12):
        matrix, offset = _solve(points, targets, weights, model)
        residual = np.linalg.norm(points @ matrix.T + offset - targets, axis=1)
        cutoff = max(0.02, 1.5 * float(np.median(residual)))
        updated = quality * np.minimum(1.0, cutoff / np.maximum(residual, 1e-12))
        if np.max(np.abs(updated - weights)) < 1e-6:
            break
        weights = updated
    return _solve(points, targets, weights, model)


def _geometry(points: np.ndarray, shape: tuple[int, int]) -> bool:
    scaled = points / np.array(shape[::-1])
    centered = scaled - scaled.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return bool(np.all(np.ptp(scaled, axis=0) >= 0.25) and singular[-1] > 0
                and singular[0] / singular[-1] < 20)


def _parameters(matrix: np.ndarray, offset: np.ndarray, shape: tuple[int, int]) -> dict:
    """QR-style A = R(theta) [[sx, shear*sy], [0, sy]], positive scales."""
    sx = float(np.linalg.norm(matrix[:, 0]))
    sy = float(np.linalg.det(matrix) / sx)
    angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    shear = float(np.dot(matrix[:, 0] / sx, matrix[:, 1]) / sy)
    center = (np.array(shape[::-1]) - 1) / 2
    displacement = matrix @ center + offset - center
    corners = np.array([[0, 0], [shape[1] - 1, 0], [0, shape[0] - 1],
                        [shape[1] - 1, shape[0] - 1]], dtype=float)
    max_displacement = float(np.max(np.linalg.norm(corners @ matrix.T + offset - corners, axis=1)))
    # Diagnostic overlap estimate on a fixed grid, explicitly not a valid mask.
    yy, xx = np.meshgrid(np.linspace(0, shape[0] - 1, 64), np.linspace(0, shape[1] - 1, 64), indexing="ij")
    mapped = np.column_stack((xx.ravel(), yy.ravel())) @ matrix.T + offset
    overlap = float(np.mean(np.all((mapped >= 0) & (mapped <= np.array(shape[::-1]) - 1), axis=1)))
    return {"correction_rotation_deg": float(np.degrees(angle)),
            "correction_scale_x_percent": 100 * (sx - 1), "correction_scale_y_percent": 100 * (sy - 1),
            "correction_shear": shear, "correction_center_dx_px": float(displacement[0]),
            "correction_center_dy_px": float(displacement[1]), "max_corner_displacement_px": max_displacement,
            "sampled_overlap_fraction": overlap,
            "m00": float(matrix[0, 0]), "m01": float(matrix[0, 1]), "m02": float(offset[0]),
            "m10": float(matrix[1, 0]), "m11": float(matrix[1, 1]), "m12": float(offset[1])}


def compare_models(local: list[dict], shifts: np.ndarray, shape: tuple[int, int],
                   config: AnalysisConfig) -> tuple[list[dict], list[dict], dict]:
    """Compare models by leave-one-tile-out prediction and deletion stability.

    At least eight informative tiles with 2-D coverage are required. The
    held-out median error limits outlier influence; untrimmed RMS is also saved.
    Deletion spread is sensitivity to tile choice, NOT a confidence interval.
    """
    candidates, frames = [], []
    for position in sorted({r["frame_position"] for r in local}):
        measured = [r for r in local if r["frame_position"] == position]
        usable = [r for r in measured if r["valid"]
                  and r.get("texture_ratio", 0) >= config.affine_min_texture_ratio
                  and (r["peak_ratio"] is None or r["peak_ratio"] >= 1.05)
                  and np.isfinite([r["x_px"], r["y_px"], r["residual_dx_px"], r["residual_dy_px"], r["correlation"]]).all()]
        frame = {"frame_position": position, "frame_index": measured[0]["frame_index"],
                 "usable_tiles": len(usable), "measured_tiles": len(measured),
                 "selected_model": None, "reason": "insufficient informative tiles (need at least 8)"}
        frames.append(frame)
        if len(usable) < 8:
            continue
        points = np.array([[r["x_px"], r["y_px"]] for r in usable])
        if not _geometry(points, shape):
            frame["reason"] = "insufficient two-dimensional tile coverage"
            continue
        targets = points + np.array([[r["residual_dx_px"], r["residual_dy_px"]] for r in usable])
        quality = np.maximum([r["correlation"] for r in usable], 0.01) ** 2
        corners = np.array([[0, 0], [shape[1] - 1, 0], [0, shape[0] - 1],
                            [shape[1] - 1, shape[0] - 1]], dtype=float)
        fits = []
        for model in MODELS:
            row = {"frame_position": position, "frame_index": frame["frame_index"], "model": model,
                   "usable_tiles": len(usable), "reliable": False, "selected": False, "reason": ""}
            candidates.append(row)
            try:
                matrix, offset = _fit(points, targets, quality, model)
                errors, predictions, angles = [], [], []
                for held in range(len(points)):
                    keep = np.arange(len(points)) != held
                    if not _geometry(points[keep], shape):
                        raise ValueError("unstable tile coverage after holding out a tile")
                    m, t = _fit(points[keep], targets[keep], quality[keep], model)
                    errors.append(np.linalg.norm(m @ points[held] + t - targets[held]))
                    predictions.append(corners @ m.T + t)
                    angles.append(np.arctan2(m[1, 0], m[0, 0]))
                cv = float(np.median(errors))
                stability = float(np.max(np.sqrt(np.sum(np.var(predictions, axis=0), axis=1))))
                row.update(cv_median_error_px=cv, cv_rms_error_px=float(np.sqrt(np.mean(np.square(errors)))),
                           deletion_corner_spread_px=stability,
                           deletion_rotation_spread_deg=float(np.degrees(np.std(np.unwrap(angles)))),
                           fit_rms_error_px=float(np.sqrt(np.mean(np.sum((points @ matrix.T + offset - targets)**2, axis=1)))))
                if not np.isfinite(matrix).all() or np.linalg.det(matrix) <= 0:
                    raise ValueError("nonfinite or reflected transform")
                # Native -> translation-aligned -> local mean coordinates.
                total_offset = matrix @ shifts[position, ::-1] + offset
                row.update(_parameters(matrix, total_offset, shape))
                if config.pixel_size_nm is not None:
                    row.update(correction_center_dx_nm=row["correction_center_dx_px"] * config.pixel_size_nm,
                               correction_center_dy_nm=row["correction_center_dy_px"] * config.pixel_size_nm)
                row["reliable"] = cv <= config.affine_max_cv_error_px and stability <= config.affine_max_stability_px
                if not row["reliable"]:
                    row["reason"] = "held-out error or deletion sensitivity exceeds configured limit"
                fits.append(row)
            except (ValueError, np.linalg.LinAlgError) as error:
                row["reason"] = str(error)
        # Walk from simple to complex, requiring both absolute and relative
        # improvement over the current simpler candidate. Reliability is a
        # prerequisite for the reported choice, including translation.
        best = None
        for row in fits:
            if best is None:
                best = row
                continue
            improvement = best["cv_median_error_px"] - row["cv_median_error_px"]
            if (row["reliable"] and improvement >= config.affine_min_improvement_px
                    and improvement >= config.affine_min_relative_improvement * best["cv_median_error_px"]):
                best = row
        if best is not None and best["reliable"]:
            best["selected"] = True
            frame.update(selected_model=best["model"], reason="simplest model with supported held-out improvement")
        else:
            frame["reason"] = "no model passes fit quality and improvement requirements"
    counts = Counter(r["selected_model"] for r in frames if r["selected_model"] is not None)
    available = [r for r in candidates if r["model"] == "affine" and r["reliable"]]
    distributions = {}
    for key in ("correction_rotation_deg", "correction_scale_x_percent", "correction_scale_y_percent",
                "correction_shear", "correction_center_dx_px", "correction_center_dy_px"):
        values = [r[key] for r in available]
        distributions[key] = {"min": float(np.min(values)), "median": float(np.median(values)),
                              "max": float(np.max(values)), "std": float(np.std(values))} if values else None
    summary = {"enabled": True, "measured_frames": len(frames), "supported_frames": sum(counts.values()),
               "unavailable_frames": len(frames) - sum(counts.values()),
               "selected_model_counts": {model: counts[model] for model in MODELS},
               "reliable_affine_frames": len(available), "affine_parameter_distributions": distributions,
               "roi_shape_yx": list(shape),
               "selection_thresholds": {"min_improvement_px": config.affine_min_improvement_px,
                                        "min_relative_improvement": config.affine_min_relative_improvement,
                                        "max_cv_median_error_px": config.affine_max_cv_error_px,
                                        "max_deletion_corner_spread_px": config.affine_max_stability_px,
                                        "min_texture_ratio": config.affine_min_texture_ratio},
               "reference": "per-frame leave-one-out translation-aligned repeat mean",
               "coordinates": "native ROI pixel centers, x right/y down; homogeneous matrix maps native to mean; positive rotation clockwise",
               "decomposition": "A = R(theta) [[sx, shear*sy], [0, sy]]; scales reported as 100*(s-1)",
               "limitations": "Small-motion tile approximation; local shifts limited to 3 px per axis. Deletion spreads are sensitivity, not calibrated uncertainty. Shared reference noise makes tile errors dependent. No affine correction is applied."}
    return candidates, frames, summary
