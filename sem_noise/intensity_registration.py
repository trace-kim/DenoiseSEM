"""Small affine corrections from translation-initialized intensity registration.

Only smoothed detector copies enter fitting. Spatial validation cells never
enter optimization; native measurement pixels are resampled separately once.
The fixed native reference defines the same coordinates for every frame.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from .affine import _geometry, _parameters
from .config import AnalysisConfig


def _matrix(parameters: np.ndarray, shift: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert centered native-pixel pull parameters to moving->reference."""
    radius = max(shape) / 2
    center = (np.array(shape[::-1]) - 1) / 2
    linear = np.eye(2) + parameters[2:6].reshape(2, 2) / radius
    pull = np.eye(3)
    pull[:2, :2] = linear
    pull[:2, 2] = -shift[::-1] + parameters[:2] + center - linear @ center
    return np.linalg.inv(pull)


def _level(reference: np.ndarray, moving: np.ndarray, shift: np.ndarray,
           stride: int, config: AnalysisConfig) -> dict:
    sigma = max(config.registration_sigma, stride / 2)
    ref = ndimage.gaussian_filter(reference, sigma)[::stride, ::stride]
    mov = ndimage.gaussian_filter(moving, sigma)[::stride, ::stride]
    scale = max(float(np.std(ref)), 1e-12)
    # Normalize each copy separately so a harmless export gain/black-level
    # difference cannot exhaust the nuisance-parameter bounds.
    ref = (ref - float(np.mean(ref))) / scale
    mov = (mov - float(np.mean(mov))) / max(float(np.std(mov)), 1e-12)
    # Fixed interior valid for EVERY allowed candidate, so a fit cannot improve
    # its score by moving difficult pixels outside the image.
    margin = (np.max(np.abs(shift)) + config.affine_refine_max_translation_px
              + config.affine_refine_max_linear_change * sum(reference.shape) / 2
              + 3 * sigma + 2 * stride)
    step = max(stride, int(np.ceil(np.sqrt(reference.size / 12000))))
    yy, xx = np.mgrid[margin:reference.shape[0] - margin:step,
                      margin:reference.shape[1] - margin:step]
    x, y = xx.ravel(), yy.ravel()
    cells_x = np.minimum((x / reference.shape[1] * 4).astype(int), 3)
    cells_y = np.minimum((y / reference.shape[0] * 4).astype(int), 3)
    # Omit a smoothing-width buffer around region boundaries at every level.
    dx = np.abs(x[:, None] - np.arange(1, 4) * reference.shape[1] / 4).min(axis=1)
    dy = np.abs(y[:, None] - np.arange(1, 4) * reference.shape[0] / 4).min(axis=1)
    keep = (dx > 3 * sigma) & (dy > 3 * sigma)
    x, y, cells_x, cells_y = x[keep], y[keep], cells_x[keep], cells_y[keep]
    center = (np.array(reference.shape[::-1]) - 1) / 2
    radius = max(reference.shape) / 2
    gy, gx = np.gradient(mov)
    return dict(x=x, y=y, u=(x - center[0]) / radius, v=(y - center[1]) / radius,
                target=ndimage.map_coordinates(ref, [y / stride, x / stride], order=3),
                coefficients=ndimage.spline_filter(mov), gx=gx, gy=gy,
                cells=cells_y * 4 + cells_x, held=(cells_x + cells_y) % 3 == 0,
                stride=stride, shift=shift, scale=scale)


def _evaluate(parameters: np.ndarray, level: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, u, v = (level[key][mask] for key in ("x", "y", "u", "v"))
    stride = level["stride"]
    coordinates = [(y - level["shift"][0] + parameters[1] + parameters[4] * u + parameters[5] * v) / stride,
                   (x - level["shift"][1] + parameters[0] + parameters[2] * u + parameters[3] * v) / stride]
    warped = ndimage.map_coordinates(level["coefficients"], coordinates, order=3, prefilter=False)
    gx = ndimage.map_coordinates(level["gx"], coordinates, order=1, prefilter=False) * parameters[6] / stride
    gy = ndimage.map_coordinates(level["gy"], coordinates, order=1, prefilter=False) * parameters[6] / stride
    jacobian = np.column_stack((gx, gy, gx * u, gx * v, gy * u, gy * v, warped, np.ones_like(x)))
    return parameters[6] * warped + parameters[7] - level["target"][mask], jacobian


def _optimize(initial: np.ndarray, level: dict, mask: np.ndarray,
              bounds: tuple[np.ndarray, np.ndarray], *, affine: bool) -> tuple[np.ndarray, bool]:
    active = np.arange(8) if affine else np.array([0, 1, 6, 7])

    def expand(values: np.ndarray) -> np.ndarray:
        full = initial.copy()
        full[active] = values
        return full

    fit = least_squares(lambda values: _evaluate(expand(values), level, mask)[0], initial[active],
                        jac=lambda values: _evaluate(expand(values), level, mask)[1][:, active],
                        bounds=(bounds[0][active], bounds[1][active]), loss="soft_l1", f_scale=0.2,
                        x_scale="jac", max_nfev=80, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    return expand(fit.x), bool(fit.success and np.isfinite(fit.x).all())


def refine_affine(reference: np.ndarray, moving: np.ndarray, shift: np.ndarray,
                  config: AnalysisConfig) -> dict:
    """Validate an affine candidate against a separately refined translation.

    ``shift`` is the accepted moving->reference correction in native (dy, dx).
    Unavailable candidates carry no matrix; the caller retains its translation.
    Validation is conditional on the existing whole-image translation initializer,
    not a statistically independent uncertainty estimate.
    """
    result = dict(available=False, selected_model="translation", reason="insufficient intensity texture")
    reference, moving = np.asarray(reference, dtype=float), np.asarray(moving, dtype=float)
    shift = np.asarray(shift, dtype=float)
    if reference.ndim != 2 or reference.shape != moving.shape or shift.shape != (2,):
        raise ValueError("affine refinement requires equally shaped 2-D images and a (dy, dx) shift")
    if not (np.isfinite(reference).all() and np.isfinite(moving).all() and np.isfinite(shift).all()):
        result["reason"] = "nonfinite image or initial translation"
        return result
    if min(reference.shape) < 48 or min(np.std(reference), np.std(moving)) < 1e-7:
        return result
    radius = max(reference.shape) / 2
    limits = np.array([config.affine_refine_max_translation_px] * 2
                      + [config.affine_refine_max_linear_change * radius] * 4 + [4., 5.])
    lower = -limits
    lower[6] = 0.25
    bounds = lower, limits
    translation = np.array([0., 0., 0., 0., 0., 0., 1., 0.])
    candidate = translation.copy()
    finest = max(1, int(np.ceil(max(reference.shape) / config.registration_max_side)))
    strides = sorted({finest, max(finest, int(np.ceil(max(reference.shape) / 384))),
                      max(finest, int(np.ceil(max(reference.shape) / 192)))}, reverse=True)
    for stride in strides:
        if min((dimension + stride - 1) // stride for dimension in reference.shape) < 4:
            result["reason"] = "registration resolution is too small for intensity refinement"
            return result
        level = _level(reference, moving, shift, stride, config)
        train, held = ~level["held"], level["held"]
        if min(train.sum(), held.sum()) < 64:
            result["reason"] = "insufficient spatial training/validation pixels"
            return result
        points = np.column_stack((level["x"], level["y"]))
        if not all(_geometry(points[mask], reference.shape) for mask in (train, held)):
            result["reason"] = "insufficient spatial intensity coverage"
            return result
        translation, ok_translation = _optimize(translation, level, train, bounds, affine=False)
        if stride == strides[0]:
            candidate = translation.copy()
        candidate, ok_affine = _optimize(candidate, level, train, bounds, affine=True)
        if not (ok_translation and ok_affine):
            result["reason"] = "intensity optimization did not converge"
            return result
    result.update(training_pixels=int(train.sum()), validation_pixels=int(held.sum()))
    # Condition the full photometric/geometric design, not just a count of
    # pixels: straight parallel edges cannot identify a full affine transform.
    _, jac = _evaluate(candidate, level, train)
    eigenvalues = np.linalg.eigvalsh(jac[:, :2].T @ jac[:, :2])
    texture_ratio = float(max(0, eigenvalues[0]) / max(eigenvalues[1], 1e-12))
    norms = np.linalg.norm(jac, axis=0)
    condition = float(np.linalg.cond(jac / np.maximum(norms, 1e-12)))
    result["design_condition"] = condition
    result["texture_ratio"] = texture_ratio
    if np.min(norms) < 1e-7 or condition > 1000 or texture_ratio < config.affine_min_texture_ratio:
        result["reason"] = "affine motion is underconstrained by image texture"
        return result
    if np.any(np.minimum(candidate - lower, limits - candidate)[:6] < 0.01):
        result["reason"] = "affine refinement reached small-motion bounds"
        return result
    baseline, _ = _evaluate(translation, level, held)
    residual, _ = _evaluate(candidate, level, held)
    base_mse, affine_mse = float(np.mean(baseline**2)), float(np.mean(residual**2))
    improvement = 1 - np.sqrt(affine_mse / max(base_mse, 1e-20))
    cells = level["cells"][held]
    improved = [np.mean(residual[cells == cell]**2) < np.mean(baseline[cells == cell]**2)
                for cell in np.unique(cells)]
    result.update(validation_translation_rms_dn=float(np.sqrt(base_mse) * level["scale"]),
                  validation_affine_rms_dn=float(np.sqrt(affine_mse) * level["scale"]),
                  validation_relative_improvement=float(improvement),
                  validation_improved_regions=int(sum(improved)), validation_regions=len(improved))
    matrix = _matrix(candidate, shift, reference.shape)
    baseline_matrix = _matrix(translation, shift, reference.shape)
    corners = np.array([[0, 0, 1], [reference.shape[1] - 1, 0, 1],
                        [0, reference.shape[0] - 1, 1], [reference.shape[1] - 1, reference.shape[0] - 1, 1]])
    effect = float(np.max(np.linalg.norm((corners @ (matrix - baseline_matrix).T)[:, :2], axis=1)))
    result["additional_corner_displacement_px"] = effect
    if (improvement < config.affine_refine_min_relative_improvement or sum(improved) <= len(improved) / 2
            or effect < config.affine_min_improvement_px):
        result["reason"] = "affine does not improve held-out regions enough; retained translation"
        return result
    # Refit on two disjoint sets of TRAINING cells, never validation cells.
    # Stable predictions are needed across the whole field, not just near its center.
    matrices = []
    for parity in (0, 1):
        subset = train & (level["cells"] % 2 == parity)
        if subset.sum() < 64 or not _geometry(points[subset], reference.shape):
            result["reason"] = "insufficient training regions for stability check"
            return result
        alternate, ok = _optimize(candidate, level, subset, bounds, affine=True)
        if not ok:
            result["reason"] = "training-region stability fit did not converge"
            return result
        matrices.append(_matrix(alternate, shift, reference.shape))
    stability = float(np.max(np.linalg.norm((corners @ (matrices[0] - matrices[1]).T)[:, :2], axis=1)))
    result["training_region_corner_spread_px"] = stability
    if stability > config.affine_max_stability_px:
        result["reason"] = "affine estimates are unstable across training regions"
        return result
    parameters = _parameters(matrix[:2, :2], matrix[:2, 2], reference.shape)
    if parameters["sampled_overlap_fraction"] < config.affine_refine_min_overlap:
        result["reason"] = "insufficient estimated image overlap"
        return result
    result.update(available=True, selected_model="affine", reason="validated intensity affine refinement",
                  matrix=matrix.tolist(), **parameters)
    if config.pixel_size_nm is not None:
        result.update(correction_center_dx_nm=parameters["correction_center_dx_px"] * config.pixel_size_nm,
                      correction_center_dy_nm=parameters["correction_center_dy_px"] * config.pixel_size_nm)
    return result


def analyze_intensity(stack: np.ndarray, included: np.ndarray, indices: np.ndarray,
                      shifts: np.ndarray, accepted: np.ndarray, config: AnalysisConfig,
                      progress: Callable[[str], None]) -> tuple[dict, list[dict], list[dict], list[int]]:
    """Analyze sampled frames in the first included native frame's coordinates."""
    positions = np.flatnonzero(included)
    anchor = int(positions[0])
    others = positions[1:]
    examples = others[np.unique(np.linspace(0, len(others) - 1, min(config.diff_examples, len(others))).astype(int))].tolist()
    sampled = positions if config.local_frames == 0 else positions[
        np.unique(np.linspace(0, len(positions) - 1, min(config.local_frames, len(positions))).astype(int))]
    selected = set(sampled.tolist()) | set(examples) | {anchor}
    rows = []
    for i in range(len(stack)):
        row = dict(frame_position=i, frame_index=int(indices[i]), reference_position=anchor,
                   reference_frame_index=int(indices[anchor]), available=False, selected_model=None,
                   reason="excluded from analysis")
        if included[i] and i not in selected:
            row["reason"] = "not sampled"
        elif i == anchor:
            row.update(available=True, reason="reference identity (not an estimated fit)",
                       matrix=np.eye(3).tolist(), **_parameters(np.eye(2), np.zeros(2), stack.shape[1:]))
        elif included[i]:
            if config.registration == "none":
                row["reason"] = "translation initialization disabled"
            elif not (accepted[anchor] and accepted[i]):
                row["reason"] = "translation initialization rejected for frame or reference"
            else:
                progress(f"Intensity affine: frame {indices[i]} against reference {indices[anchor]}")
                row.update(refine_affine(stack[anchor], stack[i], shifts[i] - shifts[anchor], config))
        rows.append(row)
    reliable = [r for r in rows if r["available"] and r["frame_position"] != anchor]
    summary = dict(enabled=True, method="translation-initialized intensity affine refinement",
                   reference_position=anchor, reference_frame_index=int(indices[anchor]),
                   estimated_frames=len(reliable), attempted_frames=len(selected) - 1,
                   retained_translation_frames=sum(r["selected_model"] == "translation" for r in rows),
                   coordinates="native ROI (x right, y down) -> fixed native reference ROI; positive rotation clockwise",
                   validation="4x4 reference cells with (column+row)%3 == 0 held out; RMS compared with training-refined translation; stability across disjoint training regions",
                   limitations="Small-motion fit; validation is conditional on the whole-image translation initializer. Training-region spread is sensitivity, not calibrated uncertainty. Noise statistics remain translation-based.",
                   decomposition="A = R(theta) [[sx, shear*sy], [0, sy]]; scale changes are 100*(s-1) percent",
                   parameters={key: {"min": float(np.min([r[key] for r in reliable])),
                                     "median": float(np.median([r[key] for r in reliable])),
                                     "max": float(np.max([r[key] for r in reliable]))} for key in
                               ("correction_rotation_deg", "correction_scale_x_percent", "correction_scale_y_percent",
                                "correction_shear", "correction_center_dx_px", "correction_center_dy_px")} if reliable else {})
    return summary, rows, [], examples
