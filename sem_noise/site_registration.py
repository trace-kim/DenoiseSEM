"""Two passes of the single fit over one site, and the outputs for every frame.

Pass 1 fits every frame against the first included frame. The registered mean
of all frames, built from those fits on the first frame's grid and brightness
scale, is the reference for pass 2, which is the reported result. Every frame
gets its numbers, its three native-resolution difference images and its region
table; nothing is selected, gated or dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .registration import (PARAMETERS, clip_mask, fit_frame, identity_result, parameter_vector,
                           shift_only, warp)

PANELS = ("before", "shift_only", "full_fit")
REGION_GRID = 4
PANEL_GAP_PX = 8
INVALID_INDEX = 255
CSV_EXCLUDED = {"covariance"}


def difference_palette() -> list[int]:
    """255 diverging entries (blue, white, red) and grey for invalid pixels."""
    t = np.linspace(-1, 1, 255)[:, None]
    blue, white, red = np.array([33, 102, 217.]), np.array([255, 255, 255.]), np.array([217, 51, 38.])
    colours = np.where(t < 0, white + (blue - white) * -t, white + (red - white) * t)
    colours = np.vstack((np.rint(colours), [[128, 128, 128]])).astype(np.uint8)
    return colours.ravel().tolist()


def difference_image(differences: list[np.ndarray], valid: np.ndarray, limit: float) -> np.ndarray:
    """Palette indices for the panels side by side, on one shared symmetric scale."""
    gap = np.full((valid.shape[0], PANEL_GAP_PX), INVALID_INDEX, dtype=np.uint8)
    panels = []
    for delta in differences:
        panel_valid = valid & np.isfinite(delta)
        index = np.clip(np.rint(127 + 127 * np.where(panel_valid, delta, 0) / limit), 0, 254).astype(np.uint8)
        index[~panel_valid] = INVALID_INDEX
        panels.extend((index, gap))
    return np.hstack(panels[:-1])


def difference_limit(differences: Sequence[np.ndarray], valid: np.ndarray, *, percentile: float = 95) -> float:
    """Shared symmetric display range: largest per-panel absolute percentile.

    Outliers saturate only in the rendered image. Use 100 for the full range;
    invalid pixels and failed (nonfinite) panels never set the scale.
    """
    if not 0 < percentile <= 100:
        raise ValueError("difference display percentile must be in (0, 100]")
    limit = 0.0
    for delta in differences:
        values = delta[valid & np.isfinite(delta)]
        if values.size:
            limit = max(limit, float(np.percentile(np.abs(values), percentile)))
    return max(limit, 1e-9)


def write_difference_png(path: Path, differences: list[np.ndarray], valid: np.ndarray, limit: float) -> None:
    image = Image.fromarray(difference_image(differences, valid, limit))
    image.putpalette(difference_palette())  # attaches the palette and makes the image mode "P"
    image.save(path, compress_level=6)


def region_edges(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    return (np.linspace(0, shape[0], REGION_GRID + 1).astype(int),
            np.linspace(0, shape[1], REGION_GRID + 1).astype(int))


def region_table(differences: list[np.ndarray], valid: np.ndarray) -> list[dict]:
    """RMS of each panel's difference in every cell of a 4x4 grid."""
    ys, xs = region_edges(valid.shape)
    rows = []
    for r, (y0, y1) in enumerate(zip(ys, ys[1:])):
        for c, (x0, x1) in enumerate(zip(xs, xs[1:])):
            cell = valid[y0:y1, x0:x1]
            for panel, delta in zip(PANELS, differences):
                values = delta[y0:y1, x0:x1][cell]
                rows.append({"panel": panel, "row": r, "col": c, "y0": int(y0), "y1": int(y1),
                             "x0": int(x0), "x1": int(x1), "pixels": int(cell.sum()),
                             "rms_dn": float(np.sqrt(np.mean(values ** 2))) if cell.any() else None})
    return rows


def frame_differences(frame: np.ndarray, bad: np.ndarray, mean: np.ndarray, mean_valid: np.ndarray,
                      parameters: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, float, dict]:
    """Frame minus reference before correction, after the shift alone, after the full fit.

    "Shift alone" applies only the centre translation, with unit gain and zero
    offset. All three panels use one pixel set and one scale.
    """
    frame, mean = np.asarray(frame, dtype=np.float64), np.asarray(mean, dtype=np.float64)
    shifted, valid_shift = warp(frame, shift_only(parameters), bad)
    full, valid_full = warp(frame, parameters, bad)
    valid = mean_valid & ~bad & valid_shift & valid_full
    if not valid.any():
        raise ValueError("no common valid pixels between the frame and the registered mean")
    differences = [np.where(valid, frame - mean, 0.0), np.where(valid, shifted - mean, 0.0),
                   np.where(valid, full - mean, 0.0)]
    limit = difference_limit(differences, valid)
    # "After" is the corrected copy itself (warped, gain, offset) on the same
    # pixels, so a good fit puts it on the reference mean; the raw frame's
    # mean also carries whatever its drift moved into or out of the region.
    brightness = {"mean_before_dn": float(frame[valid].mean()),
                  "mean_after_dn": float(full[valid].mean()),
                  "reference_mean_dn": float(mean[valid].mean())}
    return differences, valid, limit, brightness


def register_site(stack: np.ndarray, included: np.ndarray, levels: tuple[float | None, float | None], *,
                  sigma: float, progress: Callable[[str], None]) -> dict:
    """Run both passes; return per-frame fits, the registered mean and its mask."""
    positions = np.flatnonzero(included)
    if len(positions) == 0:
        raise ValueError("registration requires at least one included frame")
    anchor = int(positions[0])
    shape = stack.shape[1:]
    reference = np.asarray(stack[anchor], dtype=np.float64)
    reference_bad = clip_mask(reference, levels)
    pass1 = {}
    for number, i in enumerate(positions, 1):
        if i == anchor:
            pass1[i] = identity_result(shape)
            continue
        progress(f"pass 1: frame {number}/{len(positions)} against the first frame")
        moving = np.asarray(stack[i], dtype=np.float64)
        pass1[i] = fit_frame(reference, moving, sigma=sigma, reference_invalid=reference_bad,
                             moving_invalid=clip_mask(moving, levels))
    total, count = np.zeros(shape), np.zeros(shape, dtype=np.int32)
    for i in positions:
        moving = np.asarray(stack[i], dtype=np.float64)
        corrected, valid = warp(moving, parameter_vector(pass1[i]), clip_mask(moving, levels))
        total[valid] += corrected[valid]
        count[valid] += 1
    mean_valid = count == len(positions)
    if mean_valid.sum() < 64:
        raise ValueError("fewer than 64 pixels are covered by every registered frame; check drift and clipping")
    mean = np.where(mean_valid, total / np.maximum(count, 1), 0.0)
    pass2 = {}
    for number, i in enumerate(positions, 1):
        progress(f"pass 2: frame {number}/{len(positions)} against the registered mean")
        moving = np.asarray(stack[i], dtype=np.float64)
        pass2[i] = fit_frame(mean, moving, sigma=sigma, reference_invalid=~mean_valid,
                             moving_invalid=clip_mask(moving, levels))
    return {"anchor": anchor, "positions": positions, "pass1": pass1, "pass2": pass2,
            "mean": mean, "mean_valid": mean_valid}


def difference_outputs(stack: np.ndarray, fit: dict, levels: tuple[float | None, float | None],
                       directory: Path, frame_indices: np.ndarray,
                       progress: Callable[[str], None]) -> tuple[dict[int, dict], list[dict]]:
    """Write one native-resolution PNG per frame; return per-frame extras and region rows."""
    directory.mkdir(parents=True, exist_ok=True)
    mean, mean_valid = fit["mean"], fit["mean_valid"]
    extras, regions = {}, []
    for number, i in enumerate(fit["positions"], 1):
        progress(f"difference images: frame {number}/{len(fit['positions'])}")
        frame = np.asarray(stack[i], dtype=np.float64)
        bad = clip_mask(frame, levels)
        differences, valid, limit, brightness = frame_differences(frame, bad, mean, mean_valid,
                                                                  parameter_vector(fit["pass2"][i]))
        name = f"frame_{int(frame_indices[i]):04d}.png"
        write_difference_png(directory / name, differences, valid, limit)
        extra = dict(brightness, difference_image=f"{directory.name}/{name}", colour_limit_dn=limit,
                     difference_pixels=int(valid.sum()))
        for panel, delta in zip(PANELS, differences):
            extra[f"{panel}_rms_dn"] = float(np.sqrt(np.mean(delta[valid] ** 2)))
        extras[int(i)] = extra
        for row in region_table(differences, valid):
            regions.append({"frame_position": int(i), "frame_index": int(frame_indices[i]), **row})
    return extras, regions


def fit_rows(fits: dict[int, dict], anchor: int | None, frame_indices: np.ndarray,
             extras: dict[int, dict] | None = None) -> list[dict]:
    """Flatten per-frame fits into rows; the pass-1 anchor is labelled as such."""
    rows = []
    for i in sorted(fits):
        row = {"frame_position": int(i), "frame_index": int(frame_indices[i]),
               "role": "reference (identity, not a fit)" if i == anchor else "fitted"}
        row.update(fits[i])
        if extras and i in extras:
            row.update(extras[i])
        rows.append(row)
    return rows


def csv_rows(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in CSV_EXCLUDED} for row in rows]


def registration_summary(rows: list[dict], pass1_rows: list[dict], anchor_index: int, sigma: float,
                         registration_enabled: bool = True) -> tuple[dict, dict]:
    """Site-level description of the fit and the brightness track (pass 2)."""
    if not registration_enabled:
        return {"enabled": False}, {"enabled": False}
    drift = np.array([[r["dy_px"], r["dx_px"]] for r in rows])
    indices = np.array([r["frame_index"] for r in rows])
    steps = np.diff(drift, axis=0)[np.diff(indices) == 1]
    corner = [r["corner_max_px"] for r in rows]
    registration = {
        "enabled": True, "blur_sigma_px": sigma, "frames": len(rows),
        "reference_pass1": f"first included frame (acquisition {anchor_index})",
        "reference_pass2": f"registered mean of {len(rows)} frames on acquisition {anchor_index}'s grid and brightness scale",
        "parameters": list(PARAMETERS),
        "convention": "corrected(x, y) = gain * frame(x + dx + a11 u + a12 v, y + dy + a21 u + a22 v) + offset, "
                      "u and v measured from the ROI centre, x right, y down; (dy, dx) is where the reference "
                      "centre's content sits in the frame (its drift); positive rotation is clockwise",
        "standard_errors": "approximate weighted least-squares covariance of the fit, scaled by the measured residual "
                           "correlation area; conditional on the constructed reference, without accounting for "
                           "errors-in-variables bias, reference uncertainty, or local-minimum error",
        "max_drift_px": float(np.max(np.linalg.norm(drift, axis=1))),
        "drift_step_rms_px": float(np.sqrt(np.mean(np.sum(steps ** 2, axis=1)))) if len(steps) else None,
        "max_corner_effect_px": float(np.max(corner)),
        "median_residual_rms_dn": float(np.median([r["residual_rms_dn"] for r in rows])),
        "unconverged_frames": int(sum(not r["converged"] for r in rows)),
        "pass1_max_drift_px": float(np.max(np.linalg.norm([[r["dy_px"], r["dx_px"]] for r in pass1_rows], axis=1))),
    }
    gains = [r["gain"] for r in rows]
    offsets = [r["offset_dn"] for r in rows]
    brightness = {"enabled": True, "gain_min": float(min(gains)), "gain_max": float(max(gains)),
                  "offset_min_dn": float(min(offsets)), "offset_max_dn": float(max(offsets)),
                  "scale": f"gain and offset map each frame onto acquisition {anchor_index}'s brightness scale"}
    if len(rows) > 1 and all("mean_before_dn" in r for r in rows):
        order = indices - indices[0]
        for label in ("before", "after"):
            brightness[f"mean_{label}_slope_dn_per_frame"] = float(np.polyfit(order, [r[f"mean_{label}_dn"] for r in rows], 1)[0])
    return registration, brightness
