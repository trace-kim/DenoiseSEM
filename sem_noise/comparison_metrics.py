"""Small descriptive measurements of delivered uint8 images, without correction."""

from __future__ import annotations

from collections.abc import Iterable
import re

import numpy as np


def native_series_statistics(images: Iterable[np.ndarray], *, device: str = "cpu") -> tuple[list[dict], np.ndarray | None]:
    """Stream whole-image brightness and sample temporal SD in native coordinates.

    Float arithmetic is derived only from uint8 pixels, never network predictions.
    Temporal variation includes noise, motion, charging and specimen changes.
    A single reference image has no sample temporal standard deviation.
    """
    if device != "cpu":
        return _cuda_series_statistics(images, device)
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


def _cuda_series_statistics(images: Iterable[np.ndarray], device: str) -> tuple[list[dict], np.ndarray | None]:
    """Same float64 Welford updates, retaining the full-field accumulators on GPU.

    Transfer uint8 inputs, then only small frame summaries and the final SD map
    back. No full floating-point working image makes a host round trip.
    """
    if not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("analysis device must be cpu, cuda, or cuda:N")
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError("CUDA native analysis requires CuPy") from error
    index = int(device.split(":")[1]) if ":" in device else 0
    summaries = []
    mean = m2 = None
    with cp.cuda.Device(index):
        for image in images:
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 2 or not image.size:
                raise ValueError("comparison analysis requires nonempty 2-D uint8 images")
            if mean is None:
                mean = cp.zeros(image.shape, dtype=cp.float64)
                m2 = cp.zeros_like(mean)
            if image.shape != mean.shape:
                raise ValueError("comparison images must have identical dimensions")
            values = cp.asarray(image).astype(cp.float64)
            summaries.append(cp.stack([values.mean(), values.min(), values.max()]))
            delta = values - mean
            mean += delta / len(summaries)
            m2 += delta * (values - mean)
        if not summaries:
            raise ValueError("comparison series is empty")
        stats = cp.asnumpy(cp.stack(summaries))
        std = cp.asnumpy(cp.sqrt(cp.maximum(m2, 0) / (len(summaries) - 1))) if len(summaries) > 1 else None
    rows = [{"mean_dn": float(row[0]), "minimum_saved_dn": int(row[1]), "maximum_saved_dn": int(row[2])}
            for row in stats]
    return rows, std


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
