"""Pair-time corrections on immutable prepared raw frames and fixed splits."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from burst_diffusion.data import BurstCache, BurstSource
from burst_diffusion.real_data import normalize_native
from runctl.run_logging import atomic_write_json

from .config import Config, RealMatchingConfig
from .data import PairBatch, PairInfo, ValPairBatch
from .real_data import (LOSS_MARGIN, RealPairFactory, estimate_translations, fixed_windows,
                        registration_contrast, sample_region)

logger = logging.getLogger(__name__)
PERCENTILES = np.arange(10, 91, 5)
MEASUREMENT_VERSION = 2


def percentile_mapping(input_quantiles: np.ndarray, target_quantiles: np.ndarray) -> tuple[float, float]:
    """The report's full-image 10:5:90 percentile OLS, using cached points."""
    y, x = np.asarray(input_quantiles, dtype=np.float64), np.asarray(target_quantiles, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(centered @ centered)
    if denominator == 0:
        raise ValueError("target's 10th–90th percentiles are equal: percentile gain is not measurable")
    gain = float(centered @ (y - y.mean()) / denominator)
    offset = float(y.mean() - gain * x.mean())
    if not np.isfinite([gain, offset]).all():
        raise ValueError("percentile brightness mapping is nonfinite")
    return gain, offset


def compose_pair(matrices: np.ndarray, a: int, b: int) -> np.ndarray:
    """Map A's native coordinates directly to B's sampling coordinates."""
    return (matrices[b] @ np.linalg.inv(matrices[a]))[:2]


def crop_matrix(matrix: np.ndarray, origin_yx: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    result[:, 2] += matrix[:, :2] @ origin_yx[::-1]
    return result


def sampling_coordinates(matrix: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:size, :size]
    return (matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2],
            matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2])


def sample_target(frame: np.ndarray, matrix: np.ndarray, size: int,
                  black: float, white: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample B once into an A crop; mask pixels without cubic support.

    Pure shifts use the existing training sampler. Affine uses OpenCV cubic
    sampling on only the required source rectangle, never warping the input.
    """
    ys, xs = sampling_coordinates(matrix, size)
    h, w = frame.shape
    valid = (ys >= 2) & (ys <= h - 3) & (xs >= 2) & (xs <= w - 3)
    if not valid.any():
        raise ValueError("target has no valid overlap with this input crop")
    if np.array_equal(matrix[:, :2], np.eye(2)):
        return sample_region(frame, matrix[1, 2], matrix[0, 2], size,
                             normalization=(black, white)), valid
    import cv2

    y0, x0 = max(0, int(np.floor(ys.min())) - 2), max(0, int(np.floor(xs.min())) - 2)
    y1, x1 = min(h, int(np.ceil(ys.max())) + 3), min(w, int(np.ceil(xs.max())) + 3)
    patch = normalize_native(frame[y0:y1, x0:x1], black, white)
    local = matrix.copy()
    local[:, 2] -= [x0, y0]
    sampled = cv2.warpAffine(patch, local, (size, size), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sampled, valid


def match_prediction(prediction: torch.Tensor, matrices: torch.Tensor,
                     brightness: torch.Tensor) -> torch.Tensor:
    """Differentiably map a second native prediction to A's geometry/brightness."""
    height, width = prediction.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(height, device=prediction.device, dtype=prediction.dtype),
                            torch.arange(width, device=prediction.device, dtype=prediction.dtype), indexing="ij")
    coords = torch.stack((xs, ys, torch.ones_like(xs))).reshape(3, -1)
    sampled = matrices.to(prediction) @ coords
    gx = sampled[:, 0].reshape(-1, height, width) * (2 / (width - 1)) - 1
    gy = sampled[:, 1].reshape(-1, height, width) * (2 / (height - 1)) - 1
    moved = F.grid_sample(prediction, torch.stack((gx, gy), dim=-1), mode="bicubic", align_corners=True)
    identity = torch.eye(2, 3, device=prediction.device, dtype=prediction.dtype)
    unchanged = (matrices.to(prediction) == identity).all(dim=(1, 2))
    moved = torch.where(unchanged[:, None, None, None], prediction, moved)
    gain, offset = brightness.to(prediction).unbind(dim=1)
    return moved * gain[:, None, None, None] + offset[:, None, None, None]


class MatchedRealPairFactory(RealPairFactory):
    """Fit full-frame measurements once; correct each drawn target to its raw A.

    No prepared pixels or site assignments change. The old RealPairFactory
    remains the default when data.real_matching is absent. The leave-one-out
    path deliberately starts simply: sample/correct each other frame, then
    average on their common valid pixels; it never includes A in its target.
    """

    def __init__(self, cache: BurstCache, config: Config, *, seed: int,
                 measurements: dict | None = None) -> None:
        if cache.real_metadata is None or config.data.real_matching is None:
            raise ValueError("inline real matching requires a prepared real dataset and data.real_matching settings")
        if cache.real_metadata["registration"]["mode"] != "none":
            raise ValueError("inline real matching requires a prepare-real --align none dataset")
        super().__init__(cache, config, seed=seed)
        if self.image_size <= 2 * (LOSS_MARGIN + 2):
            raise ValueError("inline real matching needs crops larger than 10 pixels for valid consistency support")
        self.settings = config.data.real_matching
        self.sites = {key: dict(site) for key, site in self.sites.items()}
        self.measurements = {}
        self.matrices, self.quantiles, self.geometry_available, self.brightness_available = {}, {}, {}, {}
        measurement_path = config.data.real_matching_cache
        if measurements is not None:
            self.measurements = measurements
            self._set_measurements()
            logger.info("restored real matching measurements from checkpoint")
            return
        if measurement_path is not None and measurement_path.exists():
            record = json.loads(measurement_path.read_text(encoding="utf-8"))
            if (record.get("dataset_fingerprint") != cache.real_fingerprint
                    or record.get("settings") != self.settings.model_dump()
                    or record.get("percentiles") != PERCENTILES.tolist()):
                raise ValueError("real matching cache identity differs: use a new cache path for changed data/settings")
            if record.get("measurement_version") == MEASUREMENT_VERSION:
                self.measurements = record["sites"]
                self._set_measurements()
                logger.info("reused verified real matching measurements from %s", measurement_path)
                return
            if record.get("measurement_version") != 1:
                raise ValueError("unsupported real matching cache version; use a new cache path")
            logger.warning("refreshing version-1 matching cache with patternless-frame safeguards: %s", measurement_path)
        for source in cache.train_sources + cache.val_sources:
            site = self.sites[source.source_index]
            logger.info("measuring %s: geometry=%s, brightness=%s on %d full raw frames",
                        site["name"], self.settings.registration, self.settings.brightness, len(source.frames))
            try:
                record = self._measure(source.frames, cache.real_metadata["registration"])
            except ValueError as error:
                raise ValueError(f"site {site['name']}: {error}") from error
            self.measurements[str(source.source_index)] = record
            skipped = sum(row["status"].startswith("skipped") for row in record.get("diagnostics", []))
            if skipped:
                logger.warning("%s: %d/%d frames have unmeasured identity geometry; frames remain in training",
                               site["name"], skipped, len(source.frames))
            flat = sum(row["status"].startswith("skipped") for row in record.get("brightness_diagnostics", []))
            if flat:
                logger.warning("%s: %d/%d frames have flat percentiles; pairs involving them retain native brightness",
                               site["name"], flat, len(source.frames))
        self._set_measurements()
        if measurement_path is not None:
            atomic_write_json(measurement_path, self.measurement_record())

    def _measure(self, frames: np.ndarray, prepared: dict) -> dict:
        matrices = np.repeat(np.eye(3)[None], len(frames), axis=0)
        failure_policy = self.settings.registration_failure
        if failure_policy is None:
            failure_policy = prepared.get("failure_policy", "error") if self.settings.registration == "translation" else "skip"
        record = {"geometry": "none", "matrices": None, "percentiles_dn": None,
                  "failure_policy": failure_policy}
        if self.settings.registration == "translation":
            diagnostics = []
            shifts, _ = estimate_translations(
                frames, black=self.black, white=self.white, device="cpu",
                sigma=prepared["sigma"], radius=prepared["radius"], max_shift=prepared["max_shift"],
                min_contrast=prepared.get("min_contrast", 0.005), diagnostics=diagnostics,
                failure_policy=failure_policy, progress=logger.info)
            matrices[:, :2, 2] = shifts[:, ::-1]
            record.update(geometry="legacy translation", diagnostics=diagnostics, settings=prepared)
        elif self.settings.registration == "affine":
            from sem_noise.pair_matching import (GeometryEstimationError, check_geometry_reference,
                                                 estimate_geometry)
            from sem_noise.registration import clip_mask

            min_contrast = prepared.get("min_contrast", 0.005)
            if not np.isfinite(min_contrast) or min_contrast < 0:
                raise ValueError("min_contrast must be finite and nonnegative")
            reference = bad_reference = reference_index = None
            seed = np.eye(2, 3)
            diagnostics = []
            for i, frame in enumerate(frames):
                unit = torch.from_numpy(normalize_native(frame, self.black, self.white))[None, None]
                contrast = registration_contrast(unit)
                diagnostic = {"status": "registered", "contrast": contrast}
                diagnostics.append(diagnostic)
                if min_contrast > 0 and contrast < min_contrast:
                    diagnostic["status"] = "skipped_low_contrast"
                    continue
                bad = clip_mask(frame, (self.black, self.white))
                if reference is None:
                    try:
                        check_geometry_reference(frame, sigma=1, invalid=bad)
                    except GeometryEstimationError as error:
                        diagnostic.update(status="skipped_unmeasurable_reference", reason=str(error))
                        continue
                    reference, bad_reference, reference_index = frame, bad, i
                    diagnostic["status"] = "reference"
                    continue
                try:
                    candidate_seed, _ = estimate_geometry(reference, frame, motion="translation", initial=seed,
                                                          sigma=1, input_invalid=bad_reference, target_invalid=bad)
                    candidate, _ = estimate_geometry(reference, frame, motion="affine", initial=candidate_seed,
                                                     sigma=1, input_invalid=bad_reference, target_invalid=bad)
                except GeometryEstimationError as error:
                    if failure_policy == "error":
                        raise ValueError(f"frame {i}: {error}; use --registration-failure skip to retain failed frames "
                                         "with an unmeasured identity transform") from error
                    diagnostic.update(status="skipped_failed_registration", reason=str(error))
                    logger.warning("frame %d: %s; skipped registration (identity transform)", i, error)
                else:
                    matrices[i, :2] = candidate
                    seed = candidate_seed  # Only successful fits may seed the next acquisition.
                if i % 16 == 0:
                    logger.info("affine geometry: %d/%d frames", i, len(frames))
            record.update(geometry="translation-initialized affine ECC", reference_frame=reference_index, sigma_px=1,
                          min_contrast=min_contrast, diagnostics=diagnostics)
        record["matrices"] = matrices.tolist()
        if self.settings.brightness == "percentile":
            points, diagnostics = [], []
            for frame in frames:
                q = np.percentile(frame, PERCENTILES)
                flat = np.ptp(q) == 0
                diagnostics.append({"status": "skipped_flat_percentiles" if flat else "measured",
                                    "reason": "10th–90th percentiles are equal; gain is unmeasurable" if flat else ""})
                points.append(q.tolist())
            record["percentiles_dn"] = points
            record["brightness_diagnostics"] = diagnostics
        return record

    def _set_measurements(self) -> None:
        sources = {s.source_index: s for s in self.cache.train_sources + self.cache.val_sources}
        if set(self.measurements) != {str(index) for index in sources}:
            raise ValueError("matching measurements must cover exactly the train/val sites")
        for key, record in self.measurements.items():
            index = int(key)
            matrices = np.asarray(record["matrices"], dtype=np.float64)
            count = len(sources[index].frames)
            if (matrices.shape != (count, 3, 3) or not np.isfinite(matrices).all()
                    or not np.allclose(matrices[:, 2], [0, 0, 1])
                    or np.any(np.abs(np.linalg.det(matrices[:, :2, :2])) < 1e-12)):
                raise ValueError("invalid cached registration matrices")
            if self.settings.brightness == "percentile":
                points = np.asarray(record["percentiles_dn"], dtype=np.float64)
                if points.shape != (count, len(PERCENTILES)) or not np.isfinite(points).all():
                    raise ValueError("invalid cached percentile measurements")
                self.brightness_available[index] = np.ptp(points, axis=1) > 0
            self.matrices[index] = matrices
            self.quantiles[index] = np.asarray(record["percentiles_dn"], dtype=np.float64)
            diagnostics = record.get("diagnostics")
            # Older successful affine checkpoints did not store per-frame statuses.
            if diagnostics:
                if len(diagnostics) != len(matrices):
                    raise ValueError("registration diagnostics must identify every measured frame")
                self.geometry_available[index] = np.array([
                    row["status"] in {"registered", "reference"} for row in diagnostics], dtype=bool)
            else:
                self.geometry_available[index] = np.ones(len(matrices), dtype=bool)
            if self.settings.registration == "translation":
                source = next(s for s in self.cache.train_sources + self.cache.val_sources if s.source_index == index)
                shifts = matrices[:, :2, 2][:, ::-1]
                lower = np.ceil(4 - shifts.min(axis=0)).astype(int)
                upper = np.floor(np.array(source.frames.shape[1:]) - 4 - shifts.max(axis=0)).astype(int)
                self.sites[index]["bounds"] = [*lower.tolist(), *upper.tolist()]
            fixed_windows(self.sites[index]["bounds"], self.image_size)

    def _brightness(self, index: int, a: int, b: int) -> tuple[float, float]:
        if self.settings.brightness == "none":
            return 1.0, 0.0
        available = self.brightness_available[index]
        if not (available[a] and available[b]):
            return 1.0, 0.0
        gain, offset_dn = percentile_mapping(self.quantiles[index][a], self.quantiles[index][b])
        # Fixed dataset normalization: g*((B-black)/range) + offset_unit.
        return gain, (offset_dn + (gain - 1) * self.black) / (self.white - self.black)

    def _pair_matrix(self, index: int, a: int, b: int) -> np.ndarray:
        """An unavailable estimate disables geometry for the pair, never the frame."""
        available = self.geometry_available[index]
        if not (available[a] and available[b]):
            return np.eye(2, 3)
        return compose_pair(self.matrices[index], a, b)

    def _pair(self, source: BurstSource, window: tuple[int, int], a: int, b: int, second: int | None) -> tuple:
        size, index = self.image_size, source.source_index
        matrices = self.matrices[index]
        centre = np.asarray(window)[::-1] + (size - 1) / 2
        origin = np.rint((matrices[a] @ [*centre, 1])[:2][::-1] - (size - 1) / 2).astype(int)
        origin = np.clip(origin, 0, np.array(source.frames.shape[1:]) - size)
        inputs = normalize_native(source.frames[a, origin[0]:origin[0] + size, origin[1]:origin[1] + size],
                                  self.black, self.white)
        targets = np.zeros((size, size), dtype=np.float64)
        target_valid = np.ones((size, size), dtype=bool)
        selected = [b] if self.target == "noisy" else [j for j in range(len(source.frames)) if j != a]
        for j in selected:
            matrix = crop_matrix(self._pair_matrix(index, a, j), origin)
            sampled, valid = sample_target(source.frames[j], matrix, size, self.black, self.white)
            gain, offset = self._brightness(index, a, j)
            targets += gain * sampled + offset
            target_valid &= valid
        targets /= len(selected)
        second_input = second_matrix = second_brightness = second_valid = None
        if second is not None:
            matrix = crop_matrix(self._pair_matrix(index, a, second), origin)
            second_origin = np.rint((matrix @ [(size - 1) / 2, (size - 1) / 2, 1])[::-1]
                                    - (size - 1) / 2).astype(int)
            second_origin = np.clip(second_origin, 0, np.array(source.frames.shape[1:]) - size)
            second_input = normalize_native(
                source.frames[second, second_origin[0]:second_origin[0] + size,
                              second_origin[1]:second_origin[1] + size], self.black, self.white)
            second_matrix = matrix.copy()
            second_matrix[:, 2] -= second_origin[::-1]
            ys, xs = sampling_coordinates(second_matrix, size)
            border = LOSS_MARGIN + 2  # Exclude the second network's boundary and cubic footprint.
            second_valid = (ys >= border) & (ys <= size - border - 1) & (xs >= border) & (xs <= size - border - 1)
            gain, offset = self._brightness(index, a, second)
            second_brightness = np.array([gain, 2 * offset + gain - 1], dtype=np.float32)
        info = PairInfo(index, tuple(origin.tolist()), a, b if self.target == "noisy" else None, second)
        convert = lambda image: (np.asarray(image, dtype=np.float32) * 2 - 1)[None]
        return (convert(inputs), convert(targets), None if second_input is None else convert(second_input),
                target_valid, second_matrix, second_brightness, second_valid, info)

    def _batch(self, samples: list[tuple], *, validation: bool = False):
        inputs, targets, seconds, valid, matrices, brightness, second_valid, infos = zip(*samples)
        tensor = lambda arrays: torch.from_numpy(np.stack(arrays).astype(np.float32))
        values = dict(inputs=tensor(inputs), targets=tensor(targets),
                      second=None if seconds[0] is None else tensor(seconds), loss_margin=LOSS_MARGIN,
                      target_valid=torch.from_numpy(np.stack(valid))[:, None],
                      second_matrices=None if matrices[0] is None else tensor(matrices),
                      second_brightness=None if brightness[0] is None else tensor(brightness),
                      second_valid=None if second_valid[0] is None else torch.from_numpy(np.stack(second_valid))[:, None])
        return (ValPairBatch(**values, clean=None) if validation else PairBatch(**values)), list(infos)

    def state_dict(self) -> dict:
        return {**super().state_dict(), "matching_settings": self.settings.model_dump(),
                "matching_measurements": self.measurements}

    def load_state_dict(self, state: dict) -> None:
        if RealMatchingConfig.model_validate(state.get("matching_settings", self.settings.model_dump())) != self.settings:
            raise ValueError("cannot resume with different real pair matching settings")
        if "matching_measurements" in state:
            self.measurements = state["matching_measurements"]
            self._set_measurements()
        super().load_state_dict(state)

    def measurement_record(self) -> dict:
        return {"measurement_version": MEASUREMENT_VERSION,
                  "dataset_fingerprint": self.cache.real_fingerprint, "settings": self.settings.model_dump(),
                  "percentiles": PERCENTILES.tolist(), "sites": self.measurements,
                  "brightness": "full raw frame percentiles; B mapped directly to each sampled A; no output clipping",
                  "brightness_fallback": "if either frame has flat percentiles, use gain 1 and offset 0; retain frames",
                  "geometry": "matrices map the site reference to each raw frame; pair W = W_B @ inverse(W_A)",
                  "registration_fallback": "if either frame has unavailable geometry, the pair uses identity; "
                                           "all frames remain eligible, including for mean targets and consistency"}

    def write_measurements(self, path: Path) -> None:
        atomic_write_json(path, self.measurement_record())
        from .registration_report import write_registration_report

        write_registration_report(path.parent, self.cache, self.measurements, self.settings.registration)
