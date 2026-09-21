"""Optional single-device float64 contour refinement; no training dependencies.

One refiner owns a device, stream and allocation pool. Images remain independent:
only kernels and unused allocations are reused, never pixels or measurements.
Future image workers can each own a refiner without shared device/image state.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .config import RefineConfig
from .contours import Contour
from .refine import RefinedContour, _finish_refinement, _gradient_peak_arrays, refine_contour


def _load_cuda():
    try:
        import cupy as cp
        from cupyx.scipy import ndimage, signal
    except ImportError as error:
        raise RuntimeError(
            "CUDA refinement requires CuPy. Install the wheel matching the server's CUDA "
            "Toolkit (e.g. python -m pip install 'cupy-cuda12x>=13.4,<15'), "
            "or select refine.device: cpu. See sem_segment/README.md."
        ) from error
    return cp, ndimage, signal


class CudaRefiner:
    """Batched profiles and gradient diagnostics on exactly one visible GPU."""

    def __init__(self, config: RefineConfig) -> None:
        if config.device == "cpu" or not config.enabled or config.estimator != "gradient_peak":
            raise ValueError("CudaRefiner requires enabled gradient_peak refinement on cuda[:N]")
        self.config = config.model_copy(deep=True)
        self.cp, self.ndimage, self.signal = _load_cuda()
        self.device_index = int(config.device.split(":")[1]) if ":" in config.device else 0
        self.closed = False
        try:
            self.device = self.cp.cuda.Device(self.device_index)
            with self.device:
                self.stream = self.cp.cuda.Stream(non_blocking=True)
                self.pool = self.cp.cuda.MemoryPool()
                info = self.cp.cuda.runtime.getDeviceProperties(self.device_index)
                name = info["name"]
                self.device_name = name.decode() if isinstance(name, bytes) else str(name)
                # Exercise each required primitive before a long comparison,
                # including runtime compilation of the peak-finding kernels.
                with self.stream, self.cp.cuda.using_allocator(self.pool.malloc):
                    probe = self.cp.arange(9, dtype=self.cp.float64).reshape(3, 3)
                    filtered = self.ndimage.spline_filter(probe, order=3, mode="nearest")
                    self.cp.asnumpy(self.ndimage.map_coordinates(
                        filtered, self.cp.asarray([[1.], [1.]]), order=3, mode="nearest", prefilter=False))
                    self.cp.asnumpy(self.ndimage.gaussian_gradient_magnitude(probe, 1.))
                    profile = self.cp.asarray([[0., 0., 0., .1, .5, .9, 1., 1., 1.]])
                    positions, _, _ = _gradient_peak_arrays(
                        profile, self.cp.arange(9, dtype=self.cp.float64) - 4, 1., config,
                        xp=self.cp, gaussian_filter1d=self.ndimage.gaussian_filter1d,
                        find_peaks=self.signal.find_peaks)
                    self.cp.asnumpy(positions)
                self.stream.synchronize()
        except Exception as error:
            raise RuntimeError(
                f"Cannot initialize CUDA refinement on cuda:{self.device_index}: {error}. "
                "Check the visible GPU, CuPy wheel and CUDA Toolkit. No CPU fallback was used."
            ) from error

    def describe(self) -> dict:
        return {"backend": "cupy", "device": f"cuda:{self.device_index}",
                "device_name": self.device_name, "cupy_version": self.cp.__version__,
                "dtype": "float64", "batch_samples": self.config.cuda_batch_samples,
                "accelerated": ["spline_filter", "profile_sampling", "gradient_peak", "edge_strength"]}

    def __enter__(self) -> "CudaRefiner":
        if self.closed:
            raise RuntimeError("CUDA refiner is closed")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Release this refiner's cached allocations, leaving other libraries alone."""
        if not self.closed:
            with self.device:
                self.stream.synchronize()
                self.pool.free_all_blocks()
            self.closed = True

    def measure(self, contours: list[Contour], image01: np.ndarray, *,
                spacing_px: float, search_px: list[float]) -> tuple[list[RefinedContour], np.ndarray, dict]:
        """Return CPU contours, paired edge strengths and synchronized stage times.

        The image is uploaded once. Equal search radii share a profile batch;
        vertices are chunked by sample count to bound temporary device memory.
        Final per-contour coherence/gap handling uses the CPU reference rules.
        """
        if self.closed:
            raise RuntimeError("CUDA refiner is closed")
        if len(contours) != len(search_px):
            raise ValueError("one search radius is required per contour")
        with self.device, self.stream, self.cp.cuda.using_allocator(self.pool.malloc):
            result = self._measure(contours, image01, spacing_px, search_px)
            self.stream.synchronize()
        return result

    def _measure(self, contours, image01, spacing_px, radii):
        cp, ndimage, config = self.cp, self.ndimage, self.config
        started = time.perf_counter()
        if not contours:
            return [], np.empty((0, 2)), {"refine": 0.0, "edge_strength": 0.0}
        image = cp.asarray(image01, dtype=cp.float64)
        padding = 12 if config.interp_order == 3 else 0
        coefficients = (ndimage.spline_filter(cp.pad(image, padding, mode="edge"),
                                            order=3, mode="nearest") if padding else image)
        groups = defaultdict(list)
        refined = [None] * len(contours)
        for index, (contour, radius) in enumerate(zip(contours, radii)):
            if contour.normals is None:
                raise ValueError("contour has no normals; call contours.attach_normals first")
            if not len(contour.points):
                refined[index] = refine_contour(contour, image01, config)
            else:
                groups[float(radius)].append(index)
        for radius, indices in groups.items():
            counts = [len(contours[i].points) for i in indices]
            points = np.concatenate([contours[i].points for i in indices])
            normals = np.concatenate([contours[i].normals for i in indices])
            count = int(round(2.0 * radius / config.step_px)) + 1
            # Generate offsets on the host with the CPU's exact linspace rule.
            host_offsets = np.linspace(-radius, radius, count)
            offsets = cp.asarray(host_offsets)
            step = float(host_offsets[1] - host_offsets[0])
            batch_rows = max(1, config.cuda_batch_samples // count)
            fitted = np.empty((len(points), 6), dtype=np.float64)
            for start in range(0, len(points), batch_rows):
                stop = min(start + batch_rows, len(points))
                p = cp.asarray(points[start:stop], dtype=cp.float64)
                n = cp.asarray(normals[start:stop], dtype=cp.float64)
                rows = p[:, 0:1] + offsets[None, :] * n[:, 0:1]
                cols = p[:, 1:2] + offsets[None, :] * n[:, 1:2]
                h, w = image01.shape
                in_bounds = ((rows >= 0) & (rows <= h - 1) & (cols >= 0) & (cols <= w - 1)).all(axis=1)
                coordinates = cp.stack((rows.ravel() + padding, cols.ravel() + padding))
                profiles = ndimage.map_coordinates(coefficients, coordinates, order=config.interp_order,
                                                   mode="nearest", prefilter=False).reshape(rows.shape)
                positions, reasons, contrast = _gradient_peak_arrays(
                    profiles, offsets, step, config, xp=cp,
                    gaussian_filter1d=ndimage.gaussian_filter1d, find_peaks=self.signal.find_peaks)
                # Only six values per vertex return to the CPU, not all profile
                # samples. Polarity is metadata for this direction-free estimator.
                inner = profiles[:, offsets < 0].mean(axis=1)
                outer = profiles[:, offsets > 0].mean(axis=1)
                fitted[start:stop] = cp.asnumpy(cp.column_stack(
                    (positions, reasons, contrast, in_bounds, inner, outer)))
            start = 0
            for index, size in zip(indices, counts):
                values = fitted[start:start + size]
                start += size
                bounds = values[:, 3].astype(bool)
                selected = values[bounds] if bounds.any() else values
                polarity = 1.0 if selected[:, 5].mean() >= selected[:, 4].mean() else -1.0
                refined[index] = _finish_refinement(
                    contours[index], config, spacing_px=spacing_px, radius=radius,
                    positions=values[:, 0].copy(), reasons=values[:, 1].astype(np.int16),
                    contrast=values[:, 2].copy(), widths=np.full(size, np.nan),
                    in_bounds=bounds, polarity=polarity, profiles_shape=(size, count))
        self.stream.synchronize()
        times = {"refine": time.perf_counter() - started}
        started = time.perf_counter()
        strengths = self._edge_strengths(image, contours, refined)
        self.stream.synchronize()
        times["edge_strength"] = time.perf_counter() - started
        return refined, strengths, times

    def _edge_strengths(self, image, contours, refined):
        cp = self.cp
        magnitude = self.ndimage.gaussian_gradient_magnitude(image, 1.0)
        rings = [ring for base, ref in zip(contours, refined) for ring in (base.points, ref.polygon)]
        usable = [i for i, ring in enumerate(rings) if len(ring) >= 3 and np.isfinite(ring).all()]
        strengths = np.full(len(rings), np.nan)
        if usable:
            points = np.concatenate([rings[i] for i in usable])
            samples = np.empty(len(points))
            batch_rows = self.config.cuda_batch_samples
            for start in range(0, len(points), batch_rows):
                coords = cp.asarray(points[start:start + batch_rows].T)
                samples[start:start + batch_rows] = cp.asnumpy(self.ndimage.map_coordinates(
                    magnitude, coords, order=1, mode="nearest", prefilter=False))
            start = 0
            for index in usable:
                size = len(rings[index])
                strengths[index] = samples[start:start + size].mean()
                start += size
        return strengths.reshape(-1, 2)
