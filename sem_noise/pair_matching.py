"""Small building blocks for matching a noisy target to an untouched input.

Matrices use OpenCV's sampling convention: W maps output/input coordinates
(x, y, 1) to coordinates to sample in the target. Neither raw image is mutated.
Geometry and two-region brightness are measured separately.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .registration import prepare_fit_images


def _image(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 8 or not np.isfinite(result).all():
        raise ValueError("expected a finite grayscale image of at least 8x8 pixels")
    return result


def _matrix(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("expected a finite 2x3 affine sampling matrix")
    if np.linalg.matrix_rank(result[:, :2]) < 2:
        raise ValueError("affine matrix is singular")
    return result


def identity_transform() -> np.ndarray:
    return np.eye(2, 3, dtype=np.float64)


def estimate_geometry(input_image: np.ndarray, target_image: np.ndarray, *,
                      motion: str = "affine", initial: np.ndarray | None = None,
                      sigma: float = 1.0, input_invalid: np.ndarray | None = None,
                      target_invalid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Fit geometry only with ECC; direct affine defaults to the identity seed.

    ``initial`` may instead be an existing translation estimate. Affine includes
    translation: a later translation warp is never necessary. The returned
    score is normalized correlation, not a parameter error or convergence proof.
    """
    fixed, moving = _image(input_image), _image(target_image)
    if fixed.shape != moving.shape:
        raise ValueError("input and target images must have the same shape")
    if motion not in ("translation", "affine"):
        raise ValueError("motion must be translation or affine")
    fixed, moving, bad_fixed, bad_moving = prepare_fit_images(fixed, moving, sigma, input_invalid, target_invalid)
    # Gradient support in ECC extends one pixel beyond the Gaussian support.
    kernel = np.ones((3, 3), dtype=bool)
    good_fixed = ~ndimage.binary_dilation(bad_fixed, structure=kernel)
    good_moving = ~ndimage.binary_dilation(bad_moving, structure=kernel)
    if min(good_fixed.sum(), good_moving.sum()) < 16:
        raise ValueError("too few unclipped pixels for ECC registration")
    if np.ptp(fixed[good_fixed]) == 0 or np.ptp(moving[good_moving]) == 0:
        raise ValueError("constant image: geometry is not measurable")
    matrix = identity_transform() if initial is None else _matrix(initial).copy()
    if motion == "translation" and not np.allclose(matrix[:, :2], np.eye(2)):
        raise ValueError("translation initial matrix must have an identity linear part")
    try:
        score, matrix = cv2.findTransformECCWithMask(
            fixed.astype(np.float32), moving.astype(np.float32),
            good_fixed.astype(np.uint8), good_moving.astype(np.uint8),
            matrix.astype(np.float32),
            cv2.MOTION_AFFINE if motion == "affine" else cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6), 1,
        )
    except cv2.error as error:
        raise ValueError(f"{motion} ECC failed: {error.err}") from error
    return _matrix(matrix).copy(), float(score)


def pair_transform(input_transform: np.ndarray, target_transform: np.ndarray) -> np.ndarray:
    """Compose reference->A and reference->B into A->B sampling coordinates."""
    a = np.vstack((_matrix(input_transform), [0.0, 0.0, 1.0]))
    b = np.vstack((_matrix(target_transform), [0.0, 0.0, 1.0]))
    return (b @ np.linalg.inv(a))[:2]


def warp_target(target_image: np.ndarray, matrix: np.ndarray, *,
                invalid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample the original target once, with a mask covering cubic support."""
    target, matrix = _image(target_image), _matrix(matrix)
    bad = np.zeros(target.shape, dtype=bool) if invalid is None else np.asarray(invalid, dtype=bool)
    if bad.shape != target.shape:
        raise ValueError("invalid mask must match the target shape")
    size = (target.shape[1], target.shape[0])
    corrected = cv2.warpAffine(target, matrix, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    good = cv2.erode((~bad).astype(np.float32), np.ones((5, 5), dtype=np.uint8),
                    borderType=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.warpAffine(good, matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return corrected, coverage >= 1.0 - 1e-6


def select_brightness_regions(reference: np.ndarray, valid: np.ndarray, *, sigma: float = 1.0) -> np.ndarray:
    """Label lower/upper quartiles of a blurred site mean: 1=low, 2=high.

    Labels are chosen ONCE per site and saved. No per-frame reclassification,
    tile fits, or intensity correction is performed here.
    """
    reference = _image(reference)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != reference.shape or not valid.any():
        raise ValueError("brightness regions require valid reference pixels")
    blurred, _, bad, _ = prepare_fit_images(reference, reference, sigma, ~valid, ~valid)
    usable = valid & ~bad
    if not usable.any():
        raise ValueError("no reference pixels with full blur support for brightness regions")
    low, high = np.quantile(blurred[usable], [0.25, 0.75])
    if low >= high:
        raise ValueError("reference has no distinct low/high brightness regions")
    labels = np.zeros(reference.shape, dtype=np.uint8)
    labels[usable & (blurred <= low)] = 1
    labels[usable & (blurred >= high)] = 2
    return labels


def regions_on_input(regions: np.ndarray, input_transform: np.ndarray) -> np.ndarray:
    """Move reference-grid region labels onto A; only labels are resampled."""
    inverse = pair_transform(input_transform, identity_transform())
    labels = np.asarray(regions, dtype=np.uint8)
    return cv2.warpAffine(labels, inverse, (labels.shape[1], labels.shape[0]),
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def measure_brightness(input_image: np.ndarray, aligned_target: np.ndarray,
                       regions: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    """Match two corresponding region means, using raw DN in both images."""
    fixed, target = _image(input_image), _image(aligned_target)
    if target.shape != fixed.shape or regions.shape != fixed.shape or valid.shape != fixed.shape:
        raise ValueError("images, regions and validity mask must have identical shapes")
    result = {}
    for label, name in ((1, "low"), (2, "high")):
        mask = (regions == label) & valid
        if not mask.any():
            raise ValueError(f"no shared pixels in the {name} brightness region")
        result[f"{name}_pixels"] = int(mask.sum())
        result[f"input_{name}_dn"] = float(fixed[mask].mean())
        result[f"target_{name}_dn"] = float(target[mask].mean())
    contrast = result["target_high_dn"] - result["target_low_dn"]
    if contrast == 0:
        raise ValueError("target region means are equal: gain is not measurable")
    result["gain"] = (result["input_high_dn"] - result["input_low_dn"]) / contrast
    result["offset_dn"] = result["input_low_dn"] - result["gain"] * result["target_low_dn"]
    if not np.isfinite([result["gain"], result["offset_dn"]]).all():
        raise ValueError("brightness mapping is nonfinite")
    return result


def match_target(input_image: np.ndarray, target_image: np.ndarray, matrix: np.ndarray,
                  regions: np.ndarray, *, input_invalid: np.ndarray | None = None,
                  target_invalid: np.ndarray | None = None) -> dict:
    """Build B's answer for A. ``regions`` must be on A's grid; A is untouched.

    No output clipping or quantization: target values remain in floating DN.
    Call for each selected input/target pair, even when frames share a reference.
    """
    fixed = _image(input_image)
    aligned, valid = warp_target(target_image, matrix, invalid=target_invalid)
    if fixed.shape != aligned.shape:
        raise ValueError("input and target images must have the same shape")
    if input_invalid is not None:
        if input_invalid.shape != fixed.shape:
            raise ValueError("invalid mask must match the input shape")
        valid &= ~input_invalid
    brightness = measure_brightness(fixed, aligned, regions, valid)
    corrected = brightness["gain"] * aligned + brightness["offset_dn"]
    return {"aligned_target": aligned, "corrected_target": corrected, "valid": valid,
            "brightness": brightness}
