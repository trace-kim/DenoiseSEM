"""Global brightness matching on registered copies, without a noise model.

Fits use disjoint block means to reduce errors in both noisy images. Validation
blocks select offset versus gain+offset; they are never used to fit parameters.
The first accepted frame defines an arbitrary, explicitly recorded DN scale.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .registration import translate


def apply_brightness(image: np.ndarray, gain: float, offset_dn: float) -> np.ndarray:
    """Return a floating copy on the reference scale; never clip or quantize."""
    if not np.isfinite([gain, offset_dn]).all() or gain <= 0:
        raise ValueError("brightness gain must be positive and parameters finite")
    return np.asarray(image, dtype=np.float64) * gain + offset_dn


def match_target(target: np.ndarray, *, input_gain: float, input_offset_dn: float,
                 target_gain: float, target_offset_dn: float) -> np.ndarray:
    """Map an already registered target into the untouched input's DN scale.

    Parameters map each frame TO the same reference: C_i = a_i I_i + b_i.
    The input image is deliberately not an argument. Reject unavailable fits
    before calling; identity fallback is not evidence of successful calibration.
    """
    apply_brightness(np.empty(0), input_gain, input_offset_dn)
    return (apply_brightness(target, target_gain, target_offset_dn) - input_offset_dn) / input_gain


def fit_brightness(reference: np.ndarray, moving: np.ndarray, *, block_size: int = 16) -> dict:
    """Estimate moving->reference offset or positive gain+offset on block means.

    Low texture permits only offset. No correction is accepted when its
    held-out block RMSE exceeds the identity RMSE. Validation is conditional
    on registration and is not independent of its estimation.
    """
    reference, moving = np.asarray(reference, dtype=float), np.asarray(moving, dtype=float)
    if reference.ndim != 2 or reference.shape != moving.shape:
        raise ValueError("brightness fitting requires equally shaped 2-D images")
    if type(block_size) is not int or block_size < 2:
        raise ValueError("block_size must be an integer >= 2")
    result = dict(available=False, model="unavailable", gain_to_reference=None,
                  offset_to_reference_dn=None, reason="insufficient finite blocks")
    if not (np.isfinite(reference).all() and np.isfinite(moving).all()):
        result["reason"] = "nonfinite image"
        return result
    h, w = reference.shape
    size = min(block_size, h // 6, w // 6)
    if size < 2:
        return result
    ny, nx = h // size, w // size

    def blocks(image: np.ndarray) -> np.ndarray:
        return image[:ny * size, :nx * size].reshape(ny, size, nx, size).transpose(0, 2, 1, 3).reshape(-1, size * size)

    rb, mb = blocks(reference), blocks(moving)
    x, y = mb.mean(axis=1), rb.mean(axis=1)
    yy, xx = np.indices((ny, nx))
    held = ((yy + xx) % 3 == 0).ravel()
    train = ~held
    offset = float(np.median(y[train] - x[train]))
    rmse = lambda residual: float(np.sqrt(np.mean(residual**2)))
    before = rmse(x[held] - y[held])
    offset_error = rmse(x[held] + offset - y[held])
    gain, bias, model = 1.0, offset, "offset"
    # Within-block structure makes this a conservative noise-floor proxy,
    # not a claim about the detector's noise distribution.
    floor = float(np.median(mb.var(axis=1) + rb.var(axis=1)) / (size * size))
    texture = float(min(np.var(x[train]), np.var(y[train])))
    affine_error = None
    if texture > max(9 * floor, 1e-12):
        center = float(np.mean(x[train]))
        scale = max(float(np.std(x[train])), 1e-6)
        z = (x[train] - center) / scale
        initial = np.linalg.lstsq(np.column_stack((z, np.ones(len(z)))), y[train], rcond=None)[0]
        residual = initial[0] * z + initial[1] - y[train]
        robust_scale = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-6)
        fit = least_squares(lambda p: p[0] * z + p[1] - y[train], initial,
                            loss="soft_l1", f_scale=robust_scale)
        a = float(fit.x[0] / scale)
        b = float(fit.x[1] - a * center)
        if fit.success and np.isfinite([a, b]).all() and 0.25 < a < 4:
            affine_error = rmse(a * x[held] + b - y[held])
            if affine_error < 0.98 * offset_error:
                gain, bias, model = a, b, "affine"
    after = rmse(gain * x[held] + bias - y[held])
    result.update(block_size=size, training_blocks=int(train.sum()), validation_blocks=int(held.sum()),
                  validation_before_rmse_dn=before, validation_offset_rmse_dn=offset_error,
                  validation_affine_rmse_dn=affine_error, validation_after_rmse_dn=after,
                  block_texture_variance_dn2=texture, block_noise_proxy_dn2=floor)
    if after > before + 1e-12:
        result["reason"] = "correction worsens held-out block error; no correction applied"
        return result
    result.update(available=True, model=model, gain_to_reference=gain,
                  offset_to_reference_dn=bias, reason="held-out block validation passed")
    return result


def analyze_brightness(stack: np.ndarray, shifts: np.ndarray, accepted: np.ndarray,
                       indices: np.ndarray, crop: tuple[slice, slice], *,
                       registration_enabled: bool, example_count: int = 3) -> tuple[dict, list[dict], dict]:
    """Measure accepted frames on one common support; preserve rejected gaps.

    Raw noise statistics are intentionally not replaced by calibrated statistics:
    calibration scales noise and fitted parameters induce dependence.
    """
    positions = np.flatnonzero(accepted)
    if len(positions) == 0:
        raise ValueError("brightness analysis requires an accepted reference")
    anchor = int(positions[0])
    reference = translate(stack[anchor], shifts[anchor])[crop].astype(float)
    candidates = positions[1:]
    examples = set(candidates[np.unique(np.linspace(0, len(candidates) - 1,
                   min(example_count, len(candidates))).astype(int))]) if len(candidates) else set()
    rows, maps = [], {}
    for i in range(len(stack)):
        row = dict(frame_position=i, frame_index=int(indices[i]), available=False,
                   model="unavailable", reason="registration rejected or frame excluded")
        if accepted[i]:
            moving = translate(stack[i], shifts[i])[crop].astype(float)
            if i == anchor:
                row.update(available=True, model="reference", reason="reference identity, not a fit",
                           gain_to_reference=1.0, offset_to_reference_dn=0.0)
            elif registration_enabled:
                row.update(fit_brightness(reference, moving))
            else:
                row["reason"] = "registration disabled; correspondence is unverified"
            corrected = apply_brightness(moving, row["gain_to_reference"], row["offset_to_reference_dn"]) if row["available"] else moving.copy()
            row.update(before_mean_dn=float(moving.mean()),
                       after_mean_dn=float(corrected.mean()) if row["available"] else None,
                       before_reference_rmse_dn=float(np.sqrt(np.mean((moving - reference)**2))),
                       after_reference_rmse_dn=float(np.sqrt(np.mean((corrected - reference)**2))) if row["available"] else None)
            if i in examples:
                for name, array in dict(before=moving, after=corrected, correction=corrected - moving,
                                        residual_before=moving - reference, residual_after=corrected - reference).items():
                    maps[f"frame_{i}_{name}"] = array.astype(np.float32)
        rows.append(row)
    maps["reference"] = reference.astype(np.float32)
    valid = [r for r in rows if r["available"]]
    fitted = [r for r in valid if r["frame_position"] != anchor]
    summary = dict(enabled=True, reference_position=anchor, reference_frame_index=int(indices[anchor]),
                   convention="corrected = gain_to_reference * registered_frame + offset_to_reference_dn",
                   method="robust block-mean affine; held-out selection versus offset and identity",
                   fitted_frames=len(fitted), unavailable_frames=len(rows) - len(valid),
                   example_positions=sorted(int(i) for i in examples),
                   block_size_max=16, affine_gain_bounds=[0.25, 4.0], affine_min_relative_improvement=0.02,
                   limitation="Empirical global model, not a charging diagnosis. Reference is noisy. Block averaging reduces but does not eliminate errors-in-variables bias. Registration/interpolation and fitted coefficients affect noise statistics. Inspect spatial residuals; a flat mean alone does not validate the model.")
    if len(valid) > 1:
        order = np.array([r["frame_index"] for r in valid], dtype=float)
        order -= order[0]
        for label in ("before", "after"):
            summary[f"{label}_slope_dn_per_frame"] = float(np.polyfit(order, [r[f"{label}_mean_dn"] for r in valid], 1)[0])
    return summary, rows, maps
