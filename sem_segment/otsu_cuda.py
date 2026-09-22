"""Batched CUDA execution of the ordinary saved-uint8 Otsu detector.

Only compact integer labels and scalar diagnostics return to the CPU. Existing
scikit-image marching squares still defines the outlines, including open paths.
One producer overlaps decoding/GPU work with the caller's contour/metrology work.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator, Sequence
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from .config import InputConfig
from .image_io import read_native
from .otsu_baseline import OtsuResult, OtsuSettings


def histogram_thresholds(images: Any, xp: Any) -> tuple[Any, Any]:
    """Ordinary 256-bin float-image Otsu, independently along each batch plane.

    Match NumPy's image-range histogram (including bin-edge corrections) and
    scikit-image's float32 histogram weights, float64 centers and first maximum.
    ``xp`` is CuPy in production; NumPy exercises the same array expressions in
    tests against the independent scikit-image reference.
    """
    count = images.shape[0]
    values = images.reshape(count, -1)
    low, high = values.min(axis=1), values.max(axis=1)
    constant = low == high
    first = xp.where(constant, low - .5, low)
    last = xp.where(constant, high + .5, high)
    widths = last - first
    edges = first[:, None] + xp.arange(257, dtype=xp.float64)[None, :] * (widths[:, None] / 256)
    edges[:, -1] = last
    bins = ((values - first[:, None]) / widths[:, None] * 256).astype(xp.int64)
    bins = xp.minimum(bins, 255)
    row = xp.arange(count)[:, None]
    bins -= values < edges[row, bins]
    bins += (values >= edges[row, bins + 1]) & (bins != 255)
    hist = xp.bincount((bins + row * 256).ravel(), minlength=count * 256).reshape(count, 256)
    weights = hist.astype(xp.float32)
    centers = (edges[:, :-1] + edges[:, 1:]) / 2
    w1 = xp.cumsum(weights, axis=1)
    w2 = xp.cumsum(weights[:, ::-1], axis=1)[:, ::-1]
    products = weights * centers
    m1 = xp.cumsum(products, axis=1) / xp.maximum(w1, 1)
    m2 = (xp.cumsum(products[:, ::-1], axis=1) / xp.maximum(w2[:, ::-1], 1))[:, ::-1]
    variance = w1[:, :-1] * w2[:, 1:] * (m1[:, :-1] - m2[:, 1:]) ** 2
    thresholds = centers[xp.arange(count), xp.argmax(variance, axis=1)]
    return xp.where(constant, low, thresholds), constant


def cuda_masks(
    pixels: np.ndarray, settings: OtsuSettings, device: str,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, dict[str, float]]:
    """Run one bounded batch, with CUDA-event stage times and one label download."""
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or not pixels.size:
        raise ValueError("CUDA Otsu requires a nonempty batch of decoded uint8 images")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("CUDA Otsu device must be cuda or cuda:N")
    try:
        import cupy as cp
        from cupyx.scipy import ndimage
    except ImportError as error:
        raise RuntimeError("CUDA Otsu requires CuPy; see sem_segment/README.md, or select the CPU device") from error
    index = int(device.split(":")[1]) if ":" in device else 0
    started = time.perf_counter()
    with cp.cuda.Device(index):
        events = []

        def mark(name):
            event = cp.cuda.Event()
            event.record()
            events.append((name, event))

        mark("start")
        working = cp.asarray(pixels).astype(cp.float64)
        mark("upload")
        if settings.sigma_px:
            # Never smooth along the acquisition axis.
            working = ndimage.gaussian_filter(working, (0, settings.sigma_px, settings.sigma_px), mode="reflect")
        mark("gaussian")
        thresholds, constant = histogram_thresholds(working, cp)
        foreground = (working <= thresholds[:, None, None] if settings.polarity == "dark"
                      else working > thresholds[:, None, None])
        foreground &= ~constant[:, None, None]
        del working
        mark("otsu")
        labels_out, counts, retained = [], [], []
        structure = cp.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        for plane in foreground:
            # Independent 2-D components: acquisitions must never connect.
            labels, count = ndimage.label(plane, structure=structure)
            keep = cp.bincount(labels.ravel(), minlength=count + 1) >= settings.min_area_px
            keep[0] = False
            mapping = cp.cumsum(keep, dtype=cp.int32)
            retained.append(mapping[-1])
            mapping = cp.where(keep, mapping, 0)
            labels_out.append(mapping[labels])
            counts.append(int(count))
        mark("components")
        labels_cpu = cp.asnumpy(cp.stack(labels_out))
        thresholds_cpu = cp.asnumpy(thresholds)
        retained_cpu = cp.asnumpy(cp.stack(retained))
        mark("download")
        events[-1][1].synchronize()
        timings = {name: cp.cuda.get_elapsed_time(previous, event) / 1000
                   for (_, previous), (name, event) in zip(events, events[1:])}
    timings["batch_compute"] = time.perf_counter() - started
    return labels_cpu, thresholds_cpu, counts, retained_cpu, timings


def batch_capacity(shape: tuple[int, int], requested: int, memory_mb: int) -> int:
    """Conservative working-set estimate; bound both queued host and GPU batches.

    Includes histogram indices, smoothing, labels and temporary array operations.
    A single image exceeding the budget fails explicitly instead of ignoring it.
    """
    if requested < 1 or memory_mb < 1:
        raise ValueError("analysis batch size and memory budget must be positive")
    capacity = memory_mb * 1024**2 // (int(np.prod(shape)) * 96)
    if capacity < 1:
        raise ValueError("One image exceeds the analysis memory budget; increase --analysis-memory-mb")
    return min(requested, capacity)


def iter_saved_otsu(paths: Sequence[Path], settings: OtsuSettings, *,
                   crop: tuple[int, int, int, int] | None = None,
                   device: str = "cuda:0", batch_size: int = 16,
                   memory_mb: int = 8192, io_workers: int = 2) -> Iterator[OtsuResult]:
    """Decode saved pixels once; overlap the next GPU batch with CPU consumers.

    At most one batch is queued ahead of the consumer. GPU-stage times are amortized
    batch costs, not independent per-frame latency. Exceptions propagate and
    all worker pools are joined even when a consumer stops early.
    """
    from skimage.measure import find_contours

    if not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("CUDA Otsu device must be cuda or cuda:N")
    if io_workers < 1:
        raise ValueError("io_workers must be positive")
    crop = InputConfig(crop=crop).crop
    if not paths:
        return

    def decode(path):
        started = time.perf_counter()
        pixels, _ = read_native(path, crop=crop)
        if pixels.dtype != np.uint8:
            raise ValueError(f"Otsu baseline requires saved uint8 pixels: {path}")
        return pixels, time.perf_counter() - started

    first = decode(paths[0])
    capacity = batch_capacity(first[0].shape, batch_size, memory_mb)
    chunks = [paths[i:i + capacity] for i in range(0, len(paths), capacity)]
    with ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="sem-decode") as loader:
        def produce(chunk, initial=None):
            decoded = ([initial] if initial is not None else [])
            decoded.extend(loader.map(decode, chunk[1:] if initial is not None else chunk))
            if any(pixels.shape != first[0].shape for pixels, _ in decoded):
                raise ValueError("Saved images in one analysis series must have identical dimensions")
            try:
                result = cuda_masks(np.stack([p for p, _ in decoded]), settings, device)
            except MemoryError as error:
                raise RuntimeError("CUDA Otsu batch allocation failed; reduce --analysis-batch or --analysis-memory-mb") from error
            return result, [seconds for _, seconds in decoded]

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sem-otsu") as producer:
            future = producer.submit(produce, chunks[0], first)
            for batch_index, chunk in enumerate(chunks):
                (labels, thresholds, counts, retained, batch_times), decode_times = future.result()
                if batch_index + 1 < len(chunks):
                    future = producer.submit(produce, chunks[batch_index + 1])
                for i, label_map in enumerate(labels):
                    started = time.perf_counter()
                    mask = label_map > 0
                    outlines = find_contours(mask, level=.5, fully_connected="low")
                    if crop:
                        outlines = [p + [crop[0], crop[2]] for p in outlines]
                    timings = {key: seconds / len(chunk) for key, seconds in batch_times.items()}
                    timings.update(decode=decode_times[i], outlines=time.perf_counter() - started)
                    timings["total"] = timings["decode"] + timings["batch_compute"] + timings["outlines"]
                    yield OtsuResult(mask, outlines, float(thresholds[i]), counts[i], int(retained[i]),
                                     "cupy" if settings.sigma_px else "disabled", timings, labels=label_map,
                                     execution={"backend": "cupy", "batch_size": len(chunk),
                                                "batch_capacity": capacity, "memory_budget_mb": memory_mb,
                                                "io_workers": io_workers, "timing_basis": "amortized batch cost"})
                del labels, label_map, mask, outlines
