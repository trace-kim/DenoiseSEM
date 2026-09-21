"""Small descriptive measurements of delivered uint8 images, without correction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def native_series_statistics(images: Iterable[np.ndarray]) -> tuple[list[dict], np.ndarray | None]:
    """Stream whole-image brightness and sample temporal SD in native coordinates.

    Float arithmetic is derived only from uint8 pixels, never network predictions.
    Temporal variation includes noise, motion, charging and specimen changes.
    A single reference image has no sample temporal standard deviation.
    """
    rows = []
    mean = m2 = None
    for image in images:
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 2 or not image.size:
            raise ValueError("comparison analysis requires nonempty 2-D uint8 images")
        if mean is None:
            mean = np.zeros(image.shape, dtype=np.float64)
            m2 = np.zeros_like(mean)
        if image.shape != mean.shape:
            raise ValueError("comparison images must have identical dimensions")
        values = image.astype(np.float64)
        rows.append({"mean_dn": float(values.mean()), "minimum_saved_dn": int(image.min()),
                     "maximum_saved_dn": int(image.max())})
        delta = values - mean
        mean += delta / len(rows)
        m2 += delta * (values - mean)
    if not rows:
        raise ValueError("comparison series is empty")
    return rows, np.sqrt(np.maximum(m2, 0) / (len(rows) - 1)) if len(rows) > 1 else None


def difference_rgb(output: np.ndarray, raw: np.ndarray, limit: float) -> np.ndarray:
    """Blue negative / white zero / red positive, one fixed symmetric DN scale.

    Saturation affects this visualization only. Neither source is modified.
    """
    if output.dtype != np.uint8 or raw.dtype != np.uint8 or output.shape != raw.shape:
        raise ValueError("difference requires matching uint8 images")
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("difference limit must be finite and positive")
    signed = np.clip((output.astype(float) - raw.astype(float)) / limit, -1, 1)
    rgb = np.ones((*output.shape, 3))
    rgb[..., 0] -= np.maximum(-signed, 0)
    rgb[..., 1] -= np.abs(signed)
    rgb[..., 2] -= np.maximum(signed, 0)
    return np.rint(rgb * 255).astype(np.uint8)
