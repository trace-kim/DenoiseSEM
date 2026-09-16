"""Native-frame SIFT correspondences and independently validated affine fits."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform

from .affine import _geometry, _parameters
from .config import AnalysisConfig


def extract_features(image: np.ndarray, config: AnalysisConfig) -> tuple[np.ndarray, np.ndarray]:
    """Normalize/smooth detector copies only; coordinates return in native pixels."""
    stride = max(1, int(np.ceil(max(image.shape) / config.registration_max_side)))
    prepared = ndimage.gaussian_filter(np.asarray(image, dtype=float), max(config.registration_sigma, stride / 2))[::stride, ::stride]
    low, high = np.percentile(prepared, [1, 99])
    if high - low <= max(1e-12, abs(high) * 1e-12) or min(prepared.shape) < 32:
        return np.empty((0, 2)), np.empty((0, 128), dtype=np.uint8)
    prepared = np.clip((prepared - low) / (high - low), 0, 1)
    detector = SIFT(upsampling=1)
    try:
        detector.detect_and_extract(prepared)
    except RuntimeError:  # SIFT reports no features this way.
        return np.empty((0, 2)), np.empty((0, 128), dtype=np.uint8)
    # Keep fractional keypoint positions; keypoints is the rounded alternative.
    points = detector.positions[:, ::-1] * stride
    # Spatially interleave features before capping, rather than keeping only
    # one octave or one concentrated region in the detector's output order.
    cells = np.minimum((points / np.array(image.shape[::-1]) * 8).astype(int), 7)
    groups = [np.flatnonzero((cells[:, 0] == x) & (cells[:, 1] == y)) for y in range(8) for x in range(8)]
    order = [group[j] for j in range(max(map(len, groups), default=0)) for group in groups if j < len(group)]
    order = np.asarray(order[:config.feature_max_keypoints], dtype=int)
    return points[order], detector.descriptors[order]


def _sample_valid(source: np.ndarray, target: np.ndarray) -> bool:
    # The three anchors must span an actual triangle in BOTH images.
    return all(abs(np.linalg.det((points[1:] - points[0]))) > 1 for points in (source, target))


def _model_valid(model: AffineTransform, *data) -> bool:
    a = model.params[:2, :2]
    return bool(np.isfinite(model.params).all() and np.linalg.det(a) > 0
                and np.linalg.cond(a) < 4)


def fit_correspondences(source: np.ndarray, target: np.ndarray, shape: tuple[int, int],
                        config: AnalysisConfig, *, seed: int) -> tuple[dict, list[dict]]:
    """Fit moving->reference on training matches; never refit on validation matches."""
    result = {"available": False, "reason": "insufficient unique matches", "matches": len(source)}
    matches = []
    if len(source) < config.feature_min_inliers + 4:
        return result, matches
    cells = np.minimum(np.maximum((target / np.array(shape[::-1]) * 4).astype(int), 0), 3)
    held = (cells[:, 0] + cells[:, 1]) % 3 == 0
    train = ~held
    result.update(training_matches=int(train.sum()), validation_matches=int(held.sum()))
    if train.sum() < config.feature_min_inliers or held.sum() < 4:
        result["reason"] = "insufficient matches in spatial training/validation regions"
        return result, matches
    if not all(_geometry(points[mask], shape) for points in (source, target) for mask in (train, held)):
        result["reason"] = "matches lack two-dimensional training/validation coverage"
        return result, matches
    model, inliers = ransac((source[train], target[train]), AffineTransform, min_samples=3,
                            residual_threshold=config.feature_residual_px,
                            is_data_valid=_sample_valid, is_model_valid=_model_valid,
                            max_trials=1000, stop_probability=0.999, rng=np.random.default_rng(seed))
    if model is None or inliers is None or not _model_valid(model):
        result["reason"] = "RANSAC found no valid affine model"
        return result, matches
    errors = model.residuals(source, target)
    # RANSAC refits on its consensus set. Recheck support after that refit.
    support = errors <= config.feature_residual_px
    train_support = train & support
    validation_fraction = float(np.mean(support[held]))
    for i, (p, q) in enumerate(zip(source, target)):
        matches.append({"moving_x_px": float(p[0]), "moving_y_px": float(p[1]),
                        "reference_x_px": float(q[0]), "reference_y_px": float(q[1]),
                        "partition": "validation" if held[i] else "training",
                        "residual_px": float(errors[i]), "within_threshold": bool(support[i])})
    result.update(training_inliers=int(train_support.sum()),
                  training_inlier_fraction=float(np.mean(support[train])),
                  validation_inliers=int(np.sum(support & held)), validation_inlier_fraction=validation_fraction,
                  validation_median_error_px=float(np.median(errors[held])),
                  validation_rms_error_px=float(np.sqrt(np.mean(errors[held]**2))))
    if (train_support.sum() < config.feature_min_inliers
            or np.mean(support[train]) < config.feature_min_inlier_fraction
            or (support & held).sum() < 4 or validation_fraction < config.feature_min_inlier_fraction
            or np.median(errors[held]) > config.feature_residual_px):
        result["reason"] = "insufficient RANSAC support or held-out match agreement"
        return result, matches
    if not all(_geometry(points[mask], shape) for points in (source, target)
               for mask in (train_support, held & support)):
        result["reason"] = "inliers lack spatial coverage"
        return result, matches
    parameters = _parameters(model.params[:2, :2], model.params[:2, 2], shape)
    if parameters["sampled_overlap_fraction"] < config.feature_min_overlap:
        result["reason"] = "insufficient estimated image overlap"
        return result, matches
    result.update(available=True, reason="validated feature correspondence fit", matrix=model.params.tolist(), **parameters)
    if config.pixel_size_nm is not None:
        result.update(correction_center_dx_nm=parameters["correction_center_dx_px"] * config.pixel_size_nm,
                      correction_center_dy_nm=parameters["correction_center_dy_px"] * config.pixel_size_nm)
    return result, matches


def match_features(reference: tuple[np.ndarray, np.ndarray], moving: tuple[np.ndarray, np.ndarray],
                   shape: tuple[int, int], config: AnalysisConfig, *, seed: int) -> tuple[dict, list[dict]]:
    if min(len(reference[0]), len(moving[0])) < 2:
        return {"available": False, "reason": "insufficient detected features", "matches": 0}, []
    pairs = match_descriptors(moving[1], reference[1], cross_check=True, max_ratio=config.feature_match_ratio)
    # SIFT may describe one keypoint under multiple orientations. Such repeats
    # must not inflate the number of independent correspondence locations.
    unique, seen_source, seen_target = [], set(), set()
    for i, j in pairs:
        p, q = tuple(np.round(moving[0][i], 1)), tuple(np.round(reference[0][j], 1))
        if p not in seen_source and q not in seen_target:
            unique.append((i, j))
            seen_source.add(p)
            seen_target.add(q)
    if not unique:
        return {"available": False, "reason": "no distinctive mutual feature matches", "matches": 0}, []
    indices = np.asarray(unique)
    return fit_correspondences(moving[0][indices[:, 0]], reference[0][indices[:, 1]], shape, config, seed=seed)


def warp_to_reference(image: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear pull sampling of native pixels with an explicit valid footprint."""
    inverse = np.linalg.inv(matrix)
    yy, xx = np.indices(image.shape, dtype=float)
    x = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
    y = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]
    valid = (x >= 0) & (y >= 0) & (x <= image.shape[1] - 1) & (y <= image.shape[0] - 1)
    warped = ndimage.map_coordinates(np.asarray(image, dtype=float), [y, x], order=1, mode="constant", cval=0, prefilter=False)
    return warped, valid


def analyze_features(stack: np.ndarray, included: np.ndarray, indices: np.ndarray, config: AnalysisConfig,
                     progress: Callable[[str], None]) -> tuple[dict, list[dict], list[dict], list[int]]:
    """Register to first included native frame, independently of translation QC."""
    positions = np.flatnonzero(included)
    anchor = int(positions[0])
    others = positions[1:]
    examples = others[np.unique(np.linspace(0, len(others) - 1, min(config.diff_examples, len(others))).astype(int))].tolist()
    sampled = positions if config.local_frames == 0 else positions[
        np.unique(np.linspace(0, len(positions) - 1, min(config.local_frames, len(positions))).astype(int))]
    selected = set(sampled.tolist()) | set(examples) | {anchor}
    reference = extract_features(stack[anchor], config)
    rows, matches = [], []
    for i in range(len(stack)):
        row = {"frame_position": i, "frame_index": int(indices[i]), "reference_position": anchor,
               "reference_frame_index": int(indices[anchor]), "available": False, "reason": "excluded from analysis"}
        if included[i] and i not in selected:
            row["reason"] = "not sampled"
        elif i == anchor:
            row.update(available=True, reason="reference identity (not an estimated fit)", matrix=np.eye(3).tolist(),
                       **_parameters(np.eye(2), np.zeros(2), stack.shape[1:]))
        elif included[i]:
            progress(f"Feature affine: frame {indices[i]} against reference {indices[anchor]}")
            moving = extract_features(stack[i], config)
            fit, pair_matches = match_features(reference, moving, stack.shape[1:], config, seed=config.seed + i)
            row.update(fit, reference_features=len(reference[0]), moving_features=len(moving[0]))
            matches.extend(dict(m, frame_position=i, frame_index=int(indices[i]), reference_frame_index=int(indices[anchor])) for m in pair_matches)
        rows.append(row)
    reliable = [r for r in rows if r["available"] and r["frame_position"] != anchor]
    summary = {"enabled": True, "method": "SIFT mutual ratio matches + affine RANSAC + spatially held-out matches",
               "reference_position": anchor, "reference_frame_index": int(indices[anchor]),
               "estimated_frames": len(reliable), "attempted_frames": len(selected) - 1,
               "coordinates": "native ROI (x right, y down) -> fixed native reference ROI; positive rotation clockwise",
               "validation": "4x4 reference cells with (column+row)%3 == 0 held out before RANSAC; no refit on held-out points",
               "decomposition": "A = R(theta) [[sx, shear*sy], [0, sy]]; scale changes are 100*(s-1) percent",
               "parameters": {key: {"min": float(np.min([r[key] for r in reliable])),
                                    "median": float(np.median([r[key] for r in reliable])),
                                    "max": float(np.max([r[key] for r in reliable]))} for key in
                              ("correction_rotation_deg", "correction_scale_x_percent", "correction_scale_y_percent",
                               "correction_shear", "correction_center_dx_px", "correction_center_dy_px")} if reliable else {}}
    return summary, rows, matches, examples


def difference_examples(stack: np.ndarray, rows: list[dict], examples: list[int], shifts: np.ndarray,
                        accepted: np.ndarray, translation_enabled: bool) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fixed frame pairs and common masks/scales; never substitute identity for failure."""
    anchor = rows[0]["reference_position"]
    reference = np.asarray(stack[anchor], dtype=float)
    summaries, arrays = [], {}
    for number, i in enumerate(examples):
        images = {"raw": np.asarray(stack[i], dtype=float)}
        mask = np.ones(reference.shape, dtype=bool)
        if translation_enabled and accepted[anchor] and accepted[i]:
            matrix = np.eye(3)
            matrix[:2, 2] = (shifts[i] - shifts[anchor])[::-1]
            images["translation"], valid = warp_to_reference(stack[i], matrix)
            mask &= valid
        if rows[i]["available"]:
            images["affine"], valid = warp_to_reference(stack[i], np.asarray(rows[i]["matrix"]))
            mask &= valid
        prefix = f"pair_{number:02d}"
        record = {"prefix": prefix, "reference_frame_index": rows[i]["reference_frame_index"],
                  "moving_frame_index": rows[i]["frame_index"], "available_modes": list(images),
                  "affine_reason": rows[i]["reason"], "valid_pixels": int(mask.sum()),
                  "translation_reason": "available" if "translation" in images else "translation disabled or frame rejected"}
        arrays[f"{prefix}_valid"] = mask
        arrays[f"{prefix}_reference"] = reference.astype(np.float32)
        arrays[f"{prefix}_moving"] = np.asarray(stack[i], dtype=np.float32)
        differences = []
        for mode, moved in images.items():
            delta = np.where(mask, moved - reference, np.nan).astype(np.float32)
            arrays[f"{prefix}_{mode}_diff"] = delta
            if mask.any():
                differences.append(np.abs(delta[mask]))
                record[f"{mode}_rms_dn"] = float(np.sqrt(np.mean(delta[mask].astype(float)**2)))
        record["color_limit_dn"] = max(float(np.percentile(np.concatenate(differences), 99)), 1e-9) if differences else 1.0
        summaries.append(record)
    return summaries, arrays
