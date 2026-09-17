"""One least-squares fit per frame: translation, affine, gain, and offset.

Every frame is fitted against a fixed reference on native-resolution pixels.
Both copies are lightly blurred, nothing is subsampled, there is no search and
no pyramid, and the fit starts from zero shift. Nothing here decides whether a
parameter is real: each number is reported with its standard error, and the
reader compares the two.

Conventions
-----------
Coordinates are ROI pixels, x right and y down, with ``u = x - cx`` and
``v = y - cy`` measured from the image centre. The corrected copy of a moving
frame ``M`` on the reference grid is::

    corrected(x, y) = gain * M(x + dx + a11 u + a12 v, y + dy + a21 u + a22 v) + offset

so ``(dy, dx)`` is where the reference centre's content sits in the moving
frame (its drift), and the affine terms add a displacement that grows linearly
away from the centre. ``translate`` and ``common_crop`` apply the centre shift
alone to the noise statistics; they never warp measurement pixels affinely.
"""

from __future__ import annotations

import numpy as np
from scipy import fft, ndimage

PARAMETERS = ("dy_px", "dx_px", "a11", "a12", "a21", "a22", "gain", "offset_dn")
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
HUBER_C = 1.345          # 95 % efficiency for Gaussian residuals
CORRELATION_LAGS = 8     # residual autocorrelation window (+/- lags) for error bars
MAX_ITERATIONS = 40
STEP_TOLERANCE_PX = 1e-3


def translate(image: np.ndarray, shift: np.ndarray, *, integer: bool = False) -> np.ndarray:
    """Shift with constant padding, without wraparound; callers crop valid overlap."""
    if integer:
        shift = np.rint(shift)
    return ndimage.shift(image, shift, order=0 if integer else 1,
                         mode="constant", cval=0, prefilter=False)


def common_crop(shape: tuple[int, int], shifts: np.ndarray, margin: int = 2) -> tuple[slice, slice]:
    starts = np.ceil(np.maximum(0, shifts.max(axis=0))).astype(int) + margin
    stops = np.floor(np.minimum(np.asarray(shape) - 1, np.asarray(shape) - 1 + shifts.min(axis=0))).astype(int) + 1 - margin
    if np.any(stops - starts < 16):
        raise ValueError("registration leaves less than 16x16 pixels of common valid overlap")
    return slice(starts[0], stops[0]), slice(starts[1], stops[1])


def clip_mask(image: np.ndarray, levels: tuple[float | None, float | None]) -> np.ndarray:
    """Pixels at or beyond the configured black/white levels."""
    mask = np.zeros(image.shape, dtype=bool)
    if levels[0] is not None:
        mask |= image <= levels[0]
    if levels[1] is not None:
        mask |= image >= levels[1]
    return mask


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    yy, xx = np.indices(shape, dtype=np.float64)
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    radius = max(shape) / 2
    return yy, xx, (xx - cx) / radius, (yy - cy) / radius, radius


def warp_coordinates(parameters: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sampling positions (y', x') in the moving frame for every reference pixel.

    ``parameters`` are the eight fit values in the order of ``PARAMETERS``.
    The affine terms are dimensionless; internally they are scaled by half the
    larger image side so that every geometric unknown is in pixels.
    """
    yy, xx, u, v, radius = _grid(shape)
    dy, dx, a11, a12, a21, a22 = parameters[:6]
    y = yy + dy + (a21 * u + a22 * v) * radius
    x = xx + dx + (a11 * u + a12 * v) * radius
    # Three pixels of margin keep the cubic support, and the prefilter's
    # boundary ringing, inside the moving frame.
    inside = (y >= 3) & (y <= shape[0] - 4) & (x >= 3) & (x <= shape[1] - 4)
    return y, x, inside


def warp(image: np.ndarray, parameters: np.ndarray, invalid: np.ndarray | None = None,
         *, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``image`` once onto the reference grid and apply gain and offset.

    Returns the corrected copy and its valid mask: inside the moving frame with
    full interpolation support and, when ``invalid`` is given, not touching any
    invalid (clipped) moving pixel.
    """
    y, x, valid = warp_coordinates(parameters, image.shape)
    coordinates = [y.ravel(), x.ravel()]
    source = np.asarray(image, dtype=np.float64)
    if invalid is not None:
        source = _fill_invalid(source, invalid)
    if order > 1:
        source = ndimage.spline_filter(source, order=order, mode="nearest")
    resampled = ndimage.map_coordinates(source, coordinates, order=order, prefilter=False,
                                        mode="nearest").reshape(image.shape)
    if invalid is not None:
        # Cubic interpolation uses more than the two bilinear neighbours.
        support_bad = ndimage.binary_dilation(invalid, structure=np.ones((5, 5), dtype=bool)) if order > 1 else invalid
        touched = ndimage.map_coordinates(support_bad.astype(np.float32), coordinates, order=1,
                                          mode="nearest").reshape(image.shape)
        valid &= touched <= 0
    return parameters[6] * resampled + parameters[7], valid


def _fill_invalid(image: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Prevent masked sentinel values entering the global spline prefilter.

    Filled pixels remain excluded by the support mask; this is only a numerical
    extension, never a measured value or an extra fit observation.
    """
    if not invalid.any():
        return image
    if invalid.all():
        return np.zeros_like(image)
    nearest = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return image[tuple(nearest)]


def _correlation_area(residual: np.ndarray, valid: np.ndarray, lags: int = CORRELATION_LAGS) -> float:
    """Sum of the normalized residual autocorrelation over a small lag window.

    A Gaussian blur of the fitted copies (and any intrinsic noise correlation)
    makes neighbouring residuals dependent. The least-squares covariance counts
    every pixel as independent, so it is multiplied by this area (in pixels).
    """
    z = np.where(valid, residual - np.mean(residual[valid]), 0.0).astype(np.float64)
    m = valid.astype(np.float64)
    # Zero padding prevents opposite borders being counted as neighbours.
    lags = min(lags, min(z.shape) - 1)
    # Only the requested lag window needs padding, not a full 2H x 2W FFT.
    padded = tuple(fft.next_fast_len(side + lags) for side in z.shape)
    acf = fft.irfft2(np.abs(fft.rfft2(z, s=padded)) ** 2, s=padded)
    counts = fft.irfft2(np.abs(fft.rfft2(m, s=padded)) ** 2, s=padded)
    index = np.r_[0:lags + 1, -lags:0]
    acf = acf[np.ix_(index, index)]
    counts = np.maximum(counts[np.ix_(index, index)], 1.0)
    normalized = (acf / counts) / max(acf[0, 0] / counts[0, 0], 1e-300)
    return float(max(1.0, normalized.sum()))


def prepare_fit_images(reference: np.ndarray, moving: np.ndarray, sigma: float,
                       reference_invalid: np.ndarray | None = None,
                       moving_invalid: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """Blur and mask exactly the images used by the fit and its report plots."""
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    masks = []
    for invalid in (reference_invalid, moving_invalid):
        mask = np.zeros(reference.shape, dtype=bool) if invalid is None else np.asarray(invalid, dtype=bool)
        if mask.shape != reference.shape:
            raise ValueError("invalid masks must match the image shape")
        masks.append(mask)
    # scipy's default Gaussian kernel is truncated at four sigma.
    radius = int(4 * sigma + 0.5)
    structure = np.ones((2 * radius + 1,) * 2, dtype=bool)
    ref_bad, mov_bad = [ndimage.binary_dilation(mask, structure) for mask in masks]
    return (ndimage.gaussian_filter(_fill_invalid(reference, masks[0]), sigma),
            ndimage.gaussian_filter(_fill_invalid(moving, masks[1]), sigma),
            ref_bad, mov_bad)


def _huber_cost(residual: np.ndarray, valid: np.ndarray, scale: float) -> float:
    magnitude = np.abs(residual[valid])
    cutoff = HUBER_C * scale
    return float(np.mean(np.where(magnitude <= cutoff, 0.5 * magnitude ** 2,
                                  cutoff * (magnitude - 0.5 * cutoff))))


def _improves_on_common_pixels(residual: np.ndarray, valid: np.ndarray,
                              trial: np.ndarray, trial_valid: np.ndarray, scale: float) -> bool:
    common = valid & trial_valid
    return bool(common.sum() >= 64 and _huber_cost(trial, common, scale) < _huber_cost(residual, common, scale))


def _full_rank(normal: np.ndarray) -> bool:
    norms = np.sqrt(np.maximum(np.diag(normal), 0))
    if np.any(norms == 0):
        return False
    return bool(np.linalg.matrix_rank(normal / np.outer(norms, norms)) == len(normal))


def fit_pixel_diagnostics(reference: np.ndarray, moving: np.ndarray, parameters: np.ndarray, *,
                          sigma: float = 1.0, reference_invalid: np.ndarray | None = None,
                          moving_invalid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Exact final fit pixels for plotting/export; performs no additional fit."""
    ref, mov, ref_bad, mov_bad = prepare_fit_images(reference, moving, sigma, reference_invalid, moving_invalid)
    geometry = parameters.copy()
    geometry[6:] = (1.0, 0.0)
    # Do not replace blurred invalid values here: the solver uses these same
    # coefficients, with masking performed separately.
    warped, valid = warp(mov, geometry)
    y, x, _ = warp_coordinates(parameters, reference.shape)
    support_bad = ndimage.binary_dilation(mov_bad, structure=np.ones((5, 5), dtype=bool))
    touched = ndimage.map_coordinates(support_bad.astype(float), [y, x], order=1, mode="nearest")
    valid &= ~ref_bad & (touched <= 0)
    residual = parameters[6] * warped + parameters[7] - ref
    r = residual[valid]
    scale = max(float(1.4826 * np.median(np.abs(r - np.median(r)))), 1e-9 * max(float(np.std(ref)), 1e-12), 1e-300)
    weights = np.zeros(reference.shape)
    weights[valid] = np.minimum(1.0, HUBER_C * scale / np.maximum(np.abs(r), 1e-300))
    return {"reference_blurred": ref, "moving_blurred": mov, "moving_blurred_warped": warped,
            "fit_valid": valid, "fit_residual": residual, "huber_weights": weights}


def fit_frame(reference: np.ndarray, moving: np.ndarray, *, sigma: float = 1.0,
              reference_invalid: np.ndarray | None = None,
              moving_invalid: np.ndarray | None = None) -> dict:
    """Fit dy, dx, a11, a12, a21, a22, gain, offset with standard errors.

    Both copies are blurred by ``sigma`` pixels. Pixels marked invalid in either
    image are excluded, together with the full Gaussian-kernel neighbourhood that
    the blur contaminates. The loss is Huber with a scale re-estimated from the
    residual's median absolute deviation on every iteration. Standard errors
    are the weighted least-squares covariance scaled by the measured residual
    correlation area; they describe the fit, not calibrated stage motion.
    """
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.ndim != 2 or reference.shape != moving.shape:
        raise ValueError("fit_frame requires two equally shaped 2-D images")
    if min(reference.shape) < 16:
        raise ValueError("fit_frame requires images of at least 16x16 pixels")
    if not (np.isfinite(reference).all() and np.isfinite(moving).all()):
        raise ValueError("fit_frame requires finite images; mask clipped pixels instead")
    shape = reference.shape
    ref, mov, ref_bad, mov_bad = prepare_fit_images(reference, moving, sigma, reference_invalid, moving_invalid)
    mov_bad = ndimage.binary_dilation(mov_bad, structure=np.ones((5, 5), dtype=bool))
    coefficients = ndimage.spline_filter(mov, order=3, mode="nearest")
    yy, xx, u, v, radius = _grid(shape)
    u_flat, v_flat = u.ravel(), v.ravel()
    ref_flat, ref_ok = ref.ravel(), (~ref_bad).ravel()
    mov_bad32 = mov_bad.astype(np.float32) if mov_bad.any() else None
    scale = max(float(np.std(ref)), 1e-12)

    def evaluate(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y, x, inside = warp_coordinates(p, shape)
        coordinates = [y.ravel(), x.ravel()]
        warped = ndimage.map_coordinates(coefficients, coordinates, order=3, prefilter=False, mode="nearest")
        valid = inside.ravel() & ref_ok
        if mov_bad32 is not None:
            touched = ndimage.map_coordinates(mov_bad32, coordinates, order=1, mode="nearest")
            valid &= touched <= 0
        residual = p[6] * warped + p[7] - ref_flat
        return residual, warped, coordinates, valid

    def huber_weights(residual: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
        r = residual[valid]
        centre = np.median(r)
        s = 1.4826 * np.median(np.abs(r - centre))
        s = max(float(s), 1e-9 * scale, 1e-300)
        c = HUBER_C * s
        w = np.zeros(len(residual))
        w[valid] = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-300))
        return w, s

    def derivatives(coordinates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Differentiate the actual cubic interpolant, rather than interpolating
        # a separate finite-difference gradient with a different kernel.
        epsilon = 1e-3
        y, x = coordinates
        def sample(y, x):
            return ndimage.map_coordinates(coefficients, [y, x], order=3, prefilter=False, mode="nearest")
        return ((sample(y + epsilon, x) - sample(y - epsilon, x)) / (2 * epsilon),
                (sample(y, x + epsilon) - sample(y, x - epsilon)) / (2 * epsilon))

    # ``p`` holds the reported units (affine terms dimensionless). The solver
    # works on scaled columns: geometry in pixels, with the affine terms as
    # pixels at half the larger side, and gain in units of reference contrast.
    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    residual, warped, coordinates, valid = evaluate(p)
    if valid.sum() < 64:
        raise ValueError("fewer than 64 usable pixels; check clipping levels and ROI")
    initial_rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
    weights, robust_scale = huber_weights(residual, valid)
    converged, iterations = False, 0
    termination = "iteration_limit"
    normal = np.eye(8)
    column_scale = np.ones(8)
    for iterations in range(1, MAX_ITERATIONS + 1):
        gyw, gxw = derivatives(coordinates)
        g = p[6]
        jacobian = np.column_stack((g * gyw, g * gxw, g * gxw * u_flat * radius, g * gxw * v_flat * radius,
                                    g * gyw * u_flat * radius, g * gyw * v_flat * radius, warped,
                                    np.ones_like(warped)))
        column_scale = np.array([1.0, 1.0, radius, radius, radius, radius, scale, 1.0])
        jacobian /= column_scale
        jw = jacobian * weights[:, None]
        normal = jacobian.T @ jw
        gradient = jw.T @ residual
        try:
            if np.ptp(moving) == 0 or not _full_rank(normal):
                raise np.linalg.LinAlgError("unidentifiable parameters")
            step = -np.linalg.solve(normal, gradient)
        except np.linalg.LinAlgError:
            termination = "singular_system"
            break
        if not np.isfinite(step).all():
            termination = "nonfinite_step"
            break

        def negligible(s: np.ndarray) -> bool:
            # Scaled steps: the first six are pixels, the gain step is relative
            # to the reference contrast, the offset step is in DN.
            return bool(np.abs(s[:6]).max() < STEP_TOLERANCE_PX and abs(s[6]) < 1e-5 * scale
                        and abs(s[7]) < 1e-4 * scale)

        if negligible(step):
            converged = True
            termination = "step_tolerance"
            break
        # Backtrack on the Huber loss with its current scale frozen.
        accepted = False
        for _ in range(8):
            candidate = p + step / column_scale
            trial_residual, trial_warped, trial_coordinates, trial_valid = evaluate(candidate)
            # Compare BOTH costs on the same pixels. Previously lost pixels
            # disappeared only from the trial cost, rewarding loss of overlap.
            if _improves_on_common_pixels(residual, valid, trial_residual, trial_valid, robust_scale):
                accepted = True
                break
            step /= 2
            if negligible(step):
                break
        if not accepted:
            termination = "line_search_stalled"
            break
        p, residual, warped, coordinates, valid = candidate, trial_residual, trial_warped, trial_coordinates, trial_valid
        weights, robust_scale = huber_weights(residual, valid)
        # Only an unshortened normal-equation step establishes convergence.
    # Covariance at the solution, with the weights of the final residual.
    gyw, gxw = derivatives(coordinates)
    g = p[6]
    jacobian = np.column_stack((g * gyw, g * gxw, g * gxw * u_flat * radius, g * gxw * v_flat * radius,
                                g * gyw * u_flat * radius, g * gyw * v_flat * radius, warped,
                                np.ones_like(warped))) / column_scale
    normal = jacobian.T @ (jacobian * weights[:, None])
    effective = float(weights.sum())
    variance = float(np.sum(weights * residual ** 2)) / max(effective - 8, 1.0)
    area = _correlation_area((residual * weights).reshape(shape), valid.reshape(shape))
    try:
        if np.ptp(moving) == 0 or not _full_rank(normal):
            raise np.linalg.LinAlgError("unidentifiable parameters")
        covariance = np.linalg.inv(normal) * variance * area
    except np.linalg.LinAlgError:
        covariance = np.full((8, 8), np.nan)
    # Undo the column scaling: ``p`` already holds the reported units.
    unscale = 1 / column_scale
    covariance = covariance * np.outer(unscale, unscale)
    parameters = p.copy()
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    result = {name: float(parameters[i]) for i, name in enumerate(PARAMETERS)}
    result.update({f"{name}_se": float(errors[i]) for i, name in enumerate(PARAMETERS)})
    result.update(residual_rms_dn=float(np.sqrt(np.mean(residual[valid] ** 2))), initial_rms_dn=initial_rms,
                  robust_scale_dn=float(robust_scale), downweighted_fraction=float(np.mean(weights[valid] < 1)),
                  valid_pixels=int(valid.sum()), residual_correlation_area_px2=area,
                  iterations=iterations, converged=bool(converged), blur_sigma_px=float(sigma))
    result["termination_reason"] = termination
    result.update(corner_effects(parameters, covariance, shape))
    result.update(small_motion_decomposition(parameters, covariance))
    result["covariance"] = covariance.tolist()
    return result


def corner_effects(parameters: np.ndarray, covariance: np.ndarray, shape: tuple[int, int]) -> dict:
    """Displacement the four affine terms add at each corner, in pixels, with errors."""
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    a11, a12, a21, a22 = parameters[2:6]
    out = {}
    largest = 0.0
    for name, (y, x) in zip(CORNERS, ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1))):
        u, v = x - cx, y - cy
        dx_c, dy_c = a11 * u + a12 * v, a21 * u + a22 * v
        # Linear propagation: dx_c depends on (a11, a12), dy_c on (a21, a22).
        gx_vec, gy_vec = np.array([u, v]), np.array([u, v])
        var_x = float(gx_vec @ covariance[2:4, 2:4] @ gx_vec)
        var_y = float(gy_vec @ covariance[4:6, 4:6] @ gy_vec)
        cov_yx = float(gy_vec @ covariance[4:6, 2:4] @ gx_vec)
        magnitude = float(np.hypot(dx_c, dy_c))
        if magnitude > 0:
            direction = np.array([dy_c, dx_c]) / magnitude
            se_mag = float(np.sqrt(max(direction[0] ** 2 * var_y + direction[1] ** 2 * var_x
                                      + 2 * direction[0] * direction[1] * cov_yx, 0)))
        else:
            se_mag = float(np.sqrt(max(var_x + var_y, 0)))
        out[f"corner_{name}_dy_px"] = float(dy_c)
        out[f"corner_{name}_dy_se"] = float(np.sqrt(max(var_y, 0)))
        out[f"corner_{name}_dx_px"] = float(dx_c)
        out[f"corner_{name}_dx_se"] = float(np.sqrt(max(var_x, 0)))
        out[f"corner_{name}_px"] = magnitude
        out[f"corner_{name}_se"] = se_mag
        if magnitude >= largest:
            largest = magnitude
            out["corner_max_px"], out["corner_max_se"], out["corner_max_name"] = magnitude, se_mag, name
    return out


def small_motion_decomposition(parameters: np.ndarray, covariance: np.ndarray) -> dict:
    """Rotation, shear and scale as linear combinations of the affine terms.

    Valid for the small motions this fit targets; each carries a propagated
    standard error. Positive rotation is clockwise with x right and y down.
    """
    a = parameters[2:6]
    c = covariance[2:6, 2:6]
    combos = {"rotation_deg": (np.array([0, -0.5, 0.5, 0]), np.degrees(1.0)),
              "shear": (np.array([0, 0.5, 0.5, 0]), 1.0),
              "scale_x_percent": (np.array([1, 0, 0, 0]), 100.0),
              "scale_y_percent": (np.array([0, 0, 0, 1]), 100.0)}
    out = {}
    for name, (vector, factor) in combos.items():
        out[name] = float(vector @ a * factor)
        out[f"{name}_se"] = float(np.sqrt(max(vector @ c @ vector, 0)) * factor)
    return out


def identity_result(shape: tuple[int, int]) -> dict:
    """The reference frame's own row in pass 1: identity by definition, not a fit."""
    parameters = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    covariance = np.zeros((8, 8))
    result = {name: float(parameters[i]) for i, name in enumerate(PARAMETERS)}
    result.update({f"{name}_se": None for name in PARAMETERS})
    result.update(residual_rms_dn=0.0, initial_rms_dn=0.0, robust_scale_dn=None, downweighted_fraction=None,
                  valid_pixels=int(np.prod(shape)), residual_correlation_area_px2=None, iterations=0,
                  converged=True, blur_sigma_px=None)
    result.update(corner_effects(parameters, covariance, shape))
    result.update(small_motion_decomposition(parameters, covariance))
    for key in [k for k in result if k.endswith("_se")]:
        result[key] = None
    result["covariance"] = covariance.tolist()
    return result


def parameter_vector(row: dict) -> np.ndarray:
    return np.array([row[name] for name in PARAMETERS], dtype=np.float64)


def shift_only(parameters: np.ndarray) -> np.ndarray:
    """Only the centre translation, without affine or brightness correction."""
    reduced = parameters.copy()
    reduced[2:6] = 0.0
    reduced[6:] = (1.0, 0.0)
    return reduced
