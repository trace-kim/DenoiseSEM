"""Translation estimates and diagnostics; never deform the measured specimen."""

from __future__ import annotations

import numpy as np
from scipy import fft, ndimage, signal
from skimage.registration import phase_cross_correlation

from .config import AnalysisConfig


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


def _prepared(image: np.ndarray, config: AnalysisConfig) -> tuple[np.ndarray, int]:
    stride = max(1, int(np.ceil(max(image.shape) / config.registration_max_side)))
    centered = np.asarray(image - np.mean(image, dtype=np.float64), dtype=np.float32)
    smooth = ndimage.gaussian_filter(centered, max(config.registration_sigma, stride / 2))
    smooth = smooth[::stride, ::stride].copy()
    smooth -= smooth.mean(dtype=np.float64)
    return smooth, stride


def _refine_overlap(reference: np.ndarray, moving: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Refine on a fixed interior, jointly fitting translation, gain, and offset.

    Tapered Fourier peaks provide a robust starting point, but shifting a
    pattern relative to a fixed taper can bias that peak. Interior least squares
    avoids this effect. Only the registration copies enter this fit.
    """
    border = int(np.ceil(np.max(np.abs(initial)))) + 4
    if min(reference.shape) - 2 * border < 12:
        return initial
    stride = max(1, int(np.ceil(np.sqrt(reference.size / 16384))))
    yy, xx = np.mgrid[border:reference.shape[0] - border:stride,
                     border:reference.shape[1] - border:stride]
    target = reference[yy, xx].ravel().astype(float)
    gy, gx = np.gradient(moving)
    coefficients = ndimage.spline_filter(moving, order=3)
    shift, gain, offset = initial.copy(), 1.0, 0.0
    for _ in range(8):
        coordinates = np.array([yy.ravel() - shift[0], xx.ravel() - shift[1]])
        warped = ndimage.map_coordinates(coefficients, coordinates, order=3, prefilter=False)
        dy = ndimage.map_coordinates(gy, coordinates, order=1, prefilter=False)
        dx = ndimage.map_coordinates(gx, coordinates, order=1, prefilter=False)
        design = np.column_stack((-gain * dy, -gain * dx, warped, np.ones_like(warped)))
        update, _, rank, _ = np.linalg.lstsq(design, target - gain * warped - offset, rcond=None)
        if rank < 4 or not np.isfinite(update).all():
            return initial
        candidate = shift + update[:2]
        if np.max(np.abs(candidate - initial)) > 1.5:
            return initial
        shift, gain, offset = candidate, gain + update[2], offset + update[3]
        if np.linalg.norm(update[:2]) < 0.002:
            break
    return shift


def estimate_translation(reference: np.ndarray, moving: np.ndarray, config: AnalysisConfig) -> dict:
    """Return correction (dy, dx), overlap correlation and peak ambiguity proxy.

    The upsample grid is numerical precision, not a confidence interval. A
    separate peak within the allowed shift range can expose periodic patterns.
    """
    ref, stride = _prepared(reference, config)
    mov, _ = _prepared(moving, config)
    if float(np.std(ref)) < 1e-7 or float(np.std(mov)) < 1e-7:
        return {"shift": np.zeros(2), "correlation": 0.0, "peak_ratio": 0.0, "valid": False}
    # Suppress discontinuities at the rectangular boundary. Without tapering,
    # cropped, nonperiodic SEM patterns bias subpixel shifts toward integers.
    taper = np.outer(signal.windows.tukey(ref.shape[0], 0.5),
                     signal.windows.tukey(ref.shape[1], 0.5)).astype(np.float32)
    ref_fft, mov_fft = fft.fft2(ref * taper), fft.fft2(mov * taper)
    shift, _, _ = phase_cross_correlation(
        ref_fft, mov_fft, upsample_factor=config.upsample_factor,
        normalization=None, space="fourier",
    )
    shift = np.asarray(shift, dtype=float)
    shift = _refine_overlap(ref, mov, shift) * stride
    if not np.isfinite(shift).all() or np.max(np.abs(shift)) > config.max_shift_px:
        return {"shift": shift, "correlation": 0.0, "peak_ratio": 0.0, "valid": False}
    crop = common_crop(ref.shape, np.array([[0, 0], shift / stride]), margin=1)
    a = ref[crop].ravel().astype(float)
    b = translate(mov, shift / stride)[crop].ravel().astype(float)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    correlation = float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
    cross = fft.fftshift(fft.ifft2(ref_fft * mov_fft.conj()).real)
    yy, xx = np.ogrid[:cross.shape[0], :cross.shape[1]]
    yy, xx = yy - cross.shape[0] // 2, xx - cross.shape[1] // 2
    search = (np.abs(yy) <= config.max_shift_px / stride) & (np.abs(xx) <= config.max_shift_px / stride)
    peak_location = np.unravel_index(np.argmax(np.where(search, cross, -np.inf)), cross.shape)
    py, px = peak_location[0] - cross.shape[0] // 2, peak_location[1] - cross.shape[1] // 2
    other = search & ((yy - py)**2 + (xx - px)**2 > 9)
    peak = float(cross[peak_location])
    second = float(np.max(cross[other])) if other.any() else 0.0
    ratio = peak / second if second > 0 else None
    return {"shift": shift, "correlation": correlation, "peak_ratio": ratio,
            "valid": correlation >= config.min_correlation}


def register_stack(stack: np.ndarray, included: np.ndarray, config: AnalysisConfig) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Two passes: anchor registration, then refinement against leave-one-out means."""
    count = len(stack)
    shifts = np.zeros((count, 2), dtype=float)
    accepted = included.copy()
    anchor = int(np.flatnonzero(included)[0])
    rows = [{"frame_position": i, "accepted": bool(included[i]), "correlation": None,
             "peak_ratio": None, "reason": "" if included[i] else "excluded in manifest"}
            for i in range(count)]
    if config.registration == "none":
        return shifts, accepted, rows
    rows[anchor]["correlation"] = 1.0
    for i in np.flatnonzero(included):
        if i == anchor:
            continue
        result = estimate_translation(stack[anchor], stack[i], config)
        shifts[i] = result["shift"]
        accepted[i] = result["valid"]
        rows[i].update(correlation=result["correlation"], peak_ratio=result["peak_ratio"])
        if not accepted[i]:
            rows[i].update(accepted=False, reason="initial registration failed quality/shift limit")
    indices = np.flatnonzero(accepted)
    if len(indices) < config.min_frames:
        return shifts, accepted, rows
    crop = common_crop(stack.shape[1:], shifts[accepted])
    total = np.zeros(stack[anchor][crop].shape, dtype=np.float64)
    for i in indices:
        total += translate(stack[i], shifts[i])[crop]
    refined = shifts.copy()
    for i in indices:
        own = translate(stack[i], shifts[i])[crop]
        reference = (total - own) / (len(indices) - 1)
        result = estimate_translation(reference, stack[i][crop], config)
        refined[i] = result["shift"]
        accepted[i] = result["valid"]
        rows[i].update(correlation=result["correlation"], peak_ratio=result["peak_ratio"])
        if not accepted[i]:
            rows[i].update(accepted=False, reason="refined registration failed quality/shift limit")
    # Keep the first included frame as the coordinate origin, even if refinement
    # rejects it. A failed anchor is visible in the frame table.
    if accepted[anchor]:
        refined -= refined[anchor].copy()
    for i in indices:
        if accepted[i] and np.max(np.abs(refined[i])) > config.max_shift_px:
            accepted[i] = False
            rows[i].update(accepted=False, reason="anchor-relative shift exceeds max_shift_px")
    return refined, accepted, rows


def local_diagnostics(stack: np.ndarray, shifts: np.ndarray, accepted: np.ndarray,
                      mean: np.ndarray, crop: tuple[slice, slice], config: AnalysisConfig) -> list[dict]:
    """Measure remaining tile translations after global correction; do not warp."""
    from dataclasses import replace

    if config.registration == "none":
        return []
    indices = np.flatnonzero(accepted)
    selected = indices[np.unique(np.linspace(0, len(indices) - 1, min(config.local_frames, len(indices))).astype(int))]
    ys = np.linspace(0, mean.shape[0], config.local_grid + 1).astype(int)
    xs = np.linspace(0, mean.shape[1], config.local_grid + 1).astype(int)
    local_config = replace(config, max_shift_px=min(3.0, config.max_shift_px))
    rows = []
    for i in selected:
        moving = translate(stack[i], shifts[i])[crop]
        reference = (mean * len(indices) - moving) / (len(indices) - 1)
        for y0, y1 in zip(ys, ys[1:]):
            for x0, x1 in zip(xs, xs[1:]):
                if min(y1 - y0, x1 - x0) < 24:
                    continue
                result = estimate_translation(reference[y0:y1, x0:x1], moving[y0:y1, x0:x1], local_config)
                rows.append({"frame_position": int(i), "y_px": float((y0 + y1) / 2 + crop[0].start),
                             "x_px": float((x0 + x1) / 2 + crop[1].start),
                             "residual_dy_px": float(result["shift"][0]),
                             "residual_dx_px": float(result["shift"][1]),
                             "correlation": result["correlation"], "valid": result["valid"],
                             "peak_ratio": result["peak_ratio"]})
    return rows
