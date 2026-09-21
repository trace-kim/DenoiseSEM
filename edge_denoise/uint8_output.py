"""One range audit and quantization contract for offline and live SEM outputs."""

from __future__ import annotations

import numpy as np


RANGE_WARNING = (
    "Prediction outside the uint8 intensity range. Review the model training and output scaling. "
    "The measurements below use the clipped uint8 image and may be affected by clipping."
)


def prediction_uint8(normalized: np.ndarray, black: float, white: float) -> tuple[np.ndarray, dict]:
    """Audit restored DN before rounding/clipping, without an excursion tolerance."""
    if not np.isfinite([black, white]).all() or white <= black:
        raise ValueError("expected finite black < white")
    values = np.asarray(normalized, dtype=np.float64) * (white - black) + black
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Nonfinite prediction: NaN or infinity (or empty prediction); no image saved")
    below, above = int((values < 0).sum()), int((values > 255).sum())
    stats = {
        "minimum_dn": float(values.min()), "maximum_dn": float(values.max()),
        "pixels": int(values.size), "below_zero": below, "above_255": above,
        "below_fraction": below / values.size, "above_fraction": above / values.size,
        "below_percent": 100 * below / values.size, "above_percent": 100 * above / values.size,
        "out_of_range_fraction": (below + above) / values.size,
        "clipped": bool(below or above), "warning": RANGE_WARNING if below or above else "",
    }
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8), stats


def average_uint8(frames: np.ndarray) -> np.ndarray:
    """Raw pixelwise mean: float64 accumulation, exactly one rounding."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 3 or not len(frames):
        raise ValueError("averaging requires a nonempty uint8 [N, H, W] stack")
    return np.rint(frames.mean(axis=0, dtype=np.float64)).astype(np.uint8)
