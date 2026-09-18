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

from .config import Config
from .data import PairBatch, PairInfo, ValPairBatch
from .real_data import LOSS_MARGIN, RealPairFactory, estimate_translations, fixed_windows, sample_region

logger = logging.getLogger(__name__)
PERCENTILES = np.arange(10, 91, 5)


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
    gain, offset = brightness.to(prediction).unbind(dim=1)
    return moved * gain[:, None, None, None] + offset[:, None, None, None]


class MatchedRealPairFactory(RealPairFactory):
    """Fit full-frame measurements once; correct each drawn target to its raw A.

    No prepared pixels or site assignments change. The old RealPairFactory
    remains the default when data.real_matching is absent. The leave-one-out
    path deliberately starts simply: sample/correct each other frame, then
    average on their common valid pixels; it never includes A in its target.
    """

    def __init__(self, cache: BurstCache, config: Config, *, seed: int) -> None:
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
        self.matrices, self.quantiles = {}, {}
        for source in cache.train_sources + cache.val_sources:
            site = self.sites[source.source_index]
            logger.info("measuring %s: geometry=%s, brightness=%s on %d full raw frames",
                        site["name"], self.settings.registration, self.settings.brightness, len(source.frames))
            try:
                record = self._measure(source.frames, cache.real_metadata["registration"])
            except ValueError as error:
                raise ValueError(f"site {site['name']}: {error}") from error
            self.measurements[str(source.source_index)] = record
        self._set_measurements()

    def _measure(self, frames: np.ndarray, prepared: dict) -> dict:
        matrices = np.repeat(np.eye(3)[None], len(frames), axis=0)
        record = {"geometry": "none", "matrices": None, "percentiles_dn": None}
        if self.settings.registration == "translation":
            diagnostics = []
            shifts, _ = estimate_translations(
                frames, black=self.black, white=self.white, device="cpu",
                sigma=prepared["sigma"], radius=prepared["radius"], max_shift=prepared["max_shift"],
                min_contrast=prepared.get("min_contrast", 0.005), diagnostics=diagnostics,
                failure_policy=prepared.get("failure_policy", "error"), progress=logger.info)
            matrices[:, :2, 2] = shifts[:, ::-1]
            record.update(geometry="legacy translation", diagnostics=diagnostics, settings=prepared)
        elif self.settings.registration == "affine":
            from sem_noise.pair_matching import estimate_geometry
            from sem_noise.registration import clip_mask

            reference = np.asarray(frames[0], dtype=np.float64)
            bad_reference = clip_mask(reference, (self.black, self.white))
            seed = np.eye(2, 3)
            for i in range(1, len(frames)):
                bad = clip_mask(frames[i], (self.black, self.white))
                try:
                    seed, _ = estimate_geometry(reference, frames[i], motion="translation", initial=seed,
                                                sigma=1, input_invalid=bad_reference, target_invalid=bad)
                    matrices[i, :2], _ = estimate_geometry(reference, frames[i], motion="affine", initial=seed,
                                                           sigma=1, input_invalid=bad_reference, target_invalid=bad)
                except ValueError as error:
                    raise ValueError(f"frame {i}: {error}") from error
                if i % 16 == 0:
                    logger.info("affine geometry: %d/%d frames", i, len(frames))
            record.update(geometry="translation-initialized affine ECC", reference_frame=0, sigma_px=1)
        record["matrices"] = matrices.tolist()
        if self.settings.brightness == "percentile":
            points = []
            for i, frame in enumerate(frames):
                q = np.percentile(frame, PERCENTILES)
                try:
                    percentile_mapping(q, q)  # Reject an unmeasurable target before training starts.
                except ValueError as error:
                    raise ValueError(f"frame {i}: {error}") from error
                points.append(q.tolist())
            record["percentiles_dn"] = points
        return record

    def _set_measurements(self) -> None:
        for key, record in self.measurements.items():
            index = int(key)
            matrices = np.asarray(record["matrices"], dtype=np.float64)
            self.matrices[index] = matrices
            self.quantiles[index] = np.asarray(record["percentiles_dn"], dtype=np.float64)
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
        gain, offset_dn = percentile_mapping(self.quantiles[index][a], self.quantiles[index][b])
        # Fixed dataset normalization: g*((B-black)/range) + offset_unit.
        return gain, (offset_dn + (gain - 1) * self.black) / (self.white - self.black)

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
            matrix = crop_matrix(compose_pair(matrices, a, j), origin)
            sampled, valid = sample_target(source.frames[j], matrix, size, self.black, self.white)
            gain, offset = self._brightness(index, a, j)
            targets += gain * sampled + offset
            target_valid &= valid
        targets /= len(selected)
        second_input = second_matrix = second_brightness = second_valid = None
        if second is not None:
            matrix = crop_matrix(compose_pair(matrices, a, second), origin)
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
        if state.get("matching_settings", self.settings.model_dump()) != self.settings.model_dump():
            raise ValueError("cannot resume with different real pair matching settings")
        if "matching_measurements" in state:
            self.measurements = state["matching_measurements"]
            self._set_measurements()
        super().load_state_dict(state)

    def write_measurements(self, path: Path) -> None:
        record = {"dataset_fingerprint": self.cache.real_fingerprint, "settings": self.settings.model_dump(),
                  "percentiles": PERCENTILES.tolist(), "sites": self.measurements,
                  "brightness": "full raw frame percentiles; B mapped directly to each sampled A; no output clipping",
                  "geometry": "matrices map the site reference to each raw frame; pair W = W_B @ inverse(W_A)"}
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
