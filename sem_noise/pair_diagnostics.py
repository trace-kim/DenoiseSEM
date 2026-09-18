"""Raw-pair diagnostics using the same target-to-input matching primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .pair_matching import (estimate_geometry, identity_transform, match_target,
                            measure_quantile_brightness, pair_transform, regions_on_input,
                            select_brightness_regions, warp_target)
from .registration import clip_mask, prepare_fit_images
from .site_registration import difference_limit, region_edges, write_difference_png

PAIR_PANELS = ("before", "translation", "affine", "brightness")


def matrix_fields(matrix: np.ndarray) -> dict[str, float]:
    return {f"m{r}{c}": float(matrix[r, c]) for r in range(2) for c in range(3)}


def diagnose_pairs(stack: np.ndarray, included: np.ndarray, frame_indices: np.ndarray,
                   levels: tuple[float | None, float | None], directory: Path, *,
                   sigma: float, progress: Callable[[str], None]) -> dict:
    """Measure one transform per frame; diagnose one distinct target per input.

    Target indices are half a burst away, cyclically, to expose drift in both
    directions. Only example selection is reduced: every input gets a CSV row
    and, on success, a native difference PNG. Failed estimates remain explicit.
    """
    positions = np.flatnonzero(included).tolist()
    if len(positions) < 2:
        raise ValueError("pair diagnostics require two included frames")
    directory.mkdir(parents=True, exist_ok=True)
    anchor = positions[0]
    reference = np.asarray(stack[anchor], dtype=float)
    reference_bad = clip_mask(reference, levels)
    translations, affines, rows = {}, {}, []
    shifts = np.zeros((len(stack), 2))
    translation_seed = identity_transform()
    for number, i in enumerate(positions, 1):
        progress(f"geometry: frame {number}/{len(positions)}")
        frame = np.asarray(stack[i], dtype=float)
        bad = clip_mask(frame, levels)
        row = {"frame_position": i, "frame_index": int(frame_indices[i])}
        for motion, matrices in (("translation", translations), ("affine", affines)):
            try:
                if i == anchor:
                    matrix, score = identity_transform(), 1.0
                else:
                    matrix, score = estimate_geometry(reference, frame, motion=motion, sigma=sigma,
                                                       initial=translations.get(i) if motion == "affine" else translation_seed,
                                                       input_invalid=reference_bad, target_invalid=bad)
                matrices[i] = matrix
                row[f"{motion}_status"] = "identity reference" if i == anchor else "completed"
                row[f"{motion}_ecc"] = score
                if motion == "translation":
                    translation_seed = matrix.copy()
                    row.update(dy_px=float(matrix[1, 2]), dx_px=float(matrix[0, 2]))
                    shifts[i] = (-matrix[1, 2], -matrix[0, 2])
                else:
                    row.update(matrix_fields(matrix))
                    half_y, half_x = (np.asarray(reference.shape) - 1) / 2
                    corners = np.array([[-half_x, -half_y], [half_x, -half_y],
                                        [-half_x, half_y], [half_x, half_y]])
                    effects = corners @ (matrix[:, :2] - np.eye(2)).T
                    row["corner_max_px"] = float(np.linalg.norm(effects, axis=1).max())
            except ValueError as error:
                row[f"{motion}_status"] = "failed"
                row[f"{motion}_error"] = str(error)
        rows.append(row)

    # Mean used only for selecting shared physical regions, never for relabelling
    # the input or replacing a noisy target. No brightness correction enters it.
    total, counts = np.zeros(reference.shape), np.zeros(reference.shape, dtype=np.int32)
    for i, matrix in affines.items():
        aligned, valid = warp_target(stack[i], matrix, invalid=clip_mask(stack[i], levels))
        total[valid] += aligned[valid]
        counts[valid] += 1
    mean = total / np.maximum(counts, 1)
    mean_valid = counts == len(affines)
    labels = np.zeros(reference.shape, dtype=np.uint8)
    region_error = ""
    try:
        labels = select_brightness_regions(mean, mean_valid, sigma=sigma)
    except ValueError as error:
        region_error = str(error)
    np.savez_compressed(directory / "brightness_regions.npz", mean=mean, valid=mean_valid, labels=labels)

    examples = {positions[k] for k in (0, len(positions) // 2, len(positions) - 1)}
    pair_rows, region_rows, quantile_rows = [], [], []
    for number, a in enumerate(positions):
        b = positions[(number + len(positions) // 2) % len(positions)]
        progress(f"target-to-input diagnostics: pair {number + 1}/{len(positions)}")
        row = {"input_position": a, "target_position": b,
               "input_index": int(frame_indices[a]), "target_index": int(frame_indices[b])}
        pair_rows.append(row)
        # Keep the full-image distribution estimate independent of geometry and
        # of the original region-based estimate, including their failure cases.
        # Subtract signed floating-point DN, never uint8/uint16 storage values.
        input_image, target_image = np.asarray(stack[a], dtype=np.float64), np.asarray(stack[b], dtype=np.float64)
        stem = f"input_{frame_indices[a]:04d}_target_{frame_indices[b]:04d}"
        example = {"input": stack[a], "target": stack[b]} if a in examples else None
        try:
            quantile = measure_quantile_brightness(input_image, target_image)
            row["quantile_status"] = "complete"
            row.update({f"quantile_{key}": quantile[key] for key in
                        ("gain", "offset_dn", "fit_rms_dn", "input_pixels", "target_pixels")})
            quantile_rows.extend({"input_index": int(frame_indices[a]), "target_index": int(frame_indices[b]),
                                  "percentile": int(p), "input_dn": float(y), "target_dn": float(x),
                                  "corrected_target_dn": float(quantile["gain"] * x + quantile["offset_dn"])}
                                 for p, y, x in zip(quantile["percentiles"], quantile["input_quantiles_dn"],
                                                    quantile["target_quantiles_dn"]))
        except ValueError as error:
            row.update(quantile_status="failed", quantile_error=str(error))
        try:
            if a not in affines or b not in affines:
                raise ValueError("pair needs successful affine estimates for both frames; see geometry.csv")
            if region_error:
                raise ValueError(region_error)
            input_bad, target_bad = clip_mask(input_image, levels), clip_mask(target_image, levels)
            matrix = pair_transform(affines[a], affines[b])
            input_regions = regions_on_input(labels, affines[a])
            matched = match_target(input_image, target_image, matrix, input_regions,
                                   input_invalid=input_bad, target_invalid=target_bad)
            if a in translations and b in translations:
                translated, translation_valid = warp_target(target_image, pair_transform(translations[a], translations[b]), invalid=target_bad)
                row["translation_status"] = "complete"
            else:
                translated = np.full(input_image.shape, np.nan)
                translation_valid = np.ones(input_image.shape, dtype=bool)
                row["translation_status"] = "failed"
                row["translation_error"] = "Translation comparison failed; affine and brightness remain measured. See geometry.csv."
            valid = matched["valid"] & translation_valid & ~target_bad
            if not valid.any():
                raise ValueError("no shared pixels for the before/translation/affine/brightness comparison")
            differences = [image - input_image for image in (target_image, translated, matched["aligned_target"], matched["corrected_target"])]
            limit = difference_limit(differences, valid)
            write_difference_png(directory / f"{stem}_differences.png", differences, valid, limit)
            row.update(status="complete", **matrix_fields(matrix), **matched["brightness"],
                       difference_image=f"pairs/{stem}_differences.png", colour_limit_dn=limit,
                       difference_pixels=int(valid.sum()), input_mean_dn=float(input_image[valid].mean()),
                       target_mean_before_dn=float(target_image[valid].mean()),
                       target_mean_after_dn=float(matched["corrected_target"][valid].mean()))
            for name, delta in zip(PAIR_PANELS, differences):
                values = delta[valid & np.isfinite(delta)]
                row[f"{name}_rms_dn"] = float(np.sqrt(np.mean(values ** 2))) if len(values) else None
                row[f"{name}_min_dn"] = float(values.min()) if len(values) else None
                row[f"{name}_max_dn"] = float(values.max()) if len(values) else None
                row[f"{name}_abs_p99_dn"] = float(np.percentile(np.abs(values), 99)) if len(values) else None
            ys, xs = region_edges(reference.shape)
            for r, (y0, y1) in enumerate(zip(ys, ys[1:])):
                for c, (x0, x1) in enumerate(zip(xs, xs[1:])):
                    cell = valid[y0:y1, x0:x1]
                    for name, delta in zip(PAIR_PANELS, differences):
                        panel_valid = cell & np.isfinite(delta[y0:y1, x0:x1])
                        values = delta[y0:y1, x0:x1][panel_valid]
                        region_rows.append({"input_index": int(frame_indices[a]), "target_index": int(frame_indices[b]),
                                            "panel": name, "row": r, "col": c, "pixels": int(panel_valid.sum()),
                                            "rms_dn": float(np.sqrt(np.mean(values ** 2))) if len(values) else None})
            if example is not None:
                input_blur, target_blur, input_blur_bad, target_blur_bad = prepare_fit_images(
                    input_image, target_image, sigma, input_bad, target_bad)
                example.update(input_blurred=input_blur, target_blurred=target_blur,
                               input_blur_valid=~input_blur_bad, target_blur_valid=~target_blur_bad,
                               translated_target=translated, aligned_target=matched["aligned_target"],
                               corrected_target=matched["corrected_target"], matrix=matrix,
                               regions=input_regions, brightness_valid=matched["valid"], difference_valid=valid)
        except ValueError as error:
            row.update(status="failed", error=str(error))
        if example is not None:
            np.savez_compressed(directory / f"{stem}.npz", **example)
            row["example_arrays"] = f"pairs/{stem}.npz"

    completed = [row for row in pair_rows if row["status"] == "complete"]
    measured = [i for i in positions if i in translations]
    drift = -shifts[measured]
    steps = np.diff(drift, axis=0)[np.diff(frame_indices[measured]) == 1]
    registration = {"enabled": True, "method": "affine", "blur_sigma_px": sigma,
                    "anchor_index": int(frame_indices[anchor]), "frames": len(positions),
                    "max_drift_px": float(np.linalg.norm(drift, axis=1).max()),
                    "drift_step_rms_px": float(np.sqrt(np.mean(np.sum(steps ** 2, axis=1)))) if len(steps) else None,
                    "max_corner_effect_px": max((r.get("corner_max_px", 0) for r in rows), default=0),
                    "translation_failures": len(positions) - len(translations),
                    "affine_failures": len(positions) - len(affines),
                    "pair_failures": len(pair_rows) - len(completed), "region_error": region_error,
                    "quantile_failures": sum(r["quantile_status"] != "complete" for r in pair_rows),
                    "quantile_method": "OLS of input percentiles against target percentiles at 10,15,...,90%; all supplied raw pixels, including clipping bounds; no automatic spatial selection or registration",
                    "convention": "W maps input (x,y,1) to target sampling coordinates; pair W = W_target @ inverse(W_input)",
                    "region_selection": "lower and upper intensity quartiles of the geometry-only mean blurred by registration_sigma; labels selected once per site",
                    "pair_selection": "one target per input, half the included burst away cyclically; all frames remain included"}
    brightness = {"enabled": True, "scale": "each reported target is mapped to its paired raw input's brightness"}
    for key in ("gain", "offset_dn"):
        values = [r[key] for r in completed]
        prefix, suffix = ("offset", "_dn") if key == "offset_dn" else ("gain", "")
        brightness[f"{prefix}_min{suffix}"] = min(values) if values else None
        brightness[f"{prefix}_max{suffix}"] = max(values) if values else None
    return {"geometry_rows": rows, "pair_rows": pair_rows, "region_rows": region_rows, "quantile_rows": quantile_rows,
            "shifts": shifts, "registration": registration, "brightness": brightness,
            "maps": {"pair_reference_mean": mean, "pair_reference_valid": mean_valid, "brightness_regions": labels}}
