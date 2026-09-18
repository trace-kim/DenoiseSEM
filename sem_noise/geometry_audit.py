"""Independent direct fits versus composed reference fits, on identical support."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from .difference_viewer import write_difference_viewer
from .pair_matching import (estimate_geometry, measure_quantile_brightness,
                            pair_transform, warp_target)
from .registration import clip_mask
from .report import _figure, _number

DISPLAY_SIGMA = 2.0


def audit_pairs(positions: list[int], mode: str) -> list[tuple[int, int]]:
    """Sample both neighbouring directions and half-burst targets, or all pairs."""
    if mode not in {"sampled", "all"}:
        raise ValueError("registration audit mode must be sampled or all")
    if mode == "all":
        return [(a, b) for a in positions for b in positions if a != b]
    count = len(positions)
    return list(dict.fromkeys((a, positions[(i + gap) % count])
                             for i, a in enumerate(positions) for gap in (1, -1, count // 2)
                             if a != positions[(i + gap) % count]))


def comparison_metrics(input_image: np.ndarray, target_image: np.ndarray,
                       direct: np.ndarray, composed: np.ndarray,
                       input_bad: np.ndarray, target_bad: np.ndarray,
                       brightness: dict | None) -> tuple[dict, dict]:
    """Warp original B once per route; compare on one mask, also safe for blur."""
    direct_image, direct_valid = warp_target(target_image, direct, invalid=target_bad)
    composed_image, composed_valid = warp_target(target_image, composed, invalid=target_bad)
    common = ~input_bad & direct_valid & composed_valid
    radius = int(4 * DISPLAY_SIGMA + 0.5)
    support = ndimage.minimum_filter(common, size=2 * radius + 1, mode="constant", cval=0)
    y, x = np.nonzero(support)
    if not len(x):
        raise ValueError("no shared pixels with full interpolation and display-blur support")
    points = np.array([x, y, np.ones(len(x))])
    displacement = (direct - composed) @ points
    norms = np.linalg.norm(displacement, axis=0)
    h, w = input_image.shape
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]])
    result = {"comparison_pixels": len(x), "comparison_fraction": len(x) / input_image.size,
              "coordinate_rms_px": float(np.sqrt(np.mean(norms ** 2))),
              "coordinate_max_px": float(norms.max()),
              "corner_max_px": float(np.linalg.norm(corners @ (direct - composed).T, axis=1).max())}
    arrays = {"input": input_image, "target": target_image, "direct_target": direct_image,
              "composed_target": composed_image, "common_valid": common, "measurement_valid": support,
              "direct_matrix": direct, "composed_matrix": composed}
    for prefix, gain, offset in [("", 1.0, 0.0)] + (
            [("percentile_", brightness["gain"], brightness["offset_dn"])] if brightness else []):
        a, b = gain * direct_image + offset, gain * composed_image + offset
        for name, delta in (("direct", a - input_image), ("composed", b - input_image),
                            ("disagreement", a - b)):
            result[f"{prefix}{name}_rms_dn"] = float(np.sqrt(np.mean(delta[support] ** 2)))
            blurred = ndimage.gaussian_filter(np.where(common, delta, 0.0), DISPLAY_SIGMA)
            result[f"{prefix}{name}_blurred_rms_dn"] = float(np.sqrt(np.mean(blurred[support] ** 2)))
    return result, arrays


def compare_direct_registration(stack: np.ndarray, included: np.ndarray, frame_indices: np.ndarray,
                                levels: tuple[float | None, float | None],
                                translations: dict[int, np.ndarray], affines: dict[int, np.ndarray],
                                directory: Path, *, mode: str, sigma: float,
                                progress: Callable[[str], None]) -> list[dict]:
    """Direct translation starts at identity; direct affine starts at that fit.

    Never seed a direct fit with the composed answer. Failures of either route
    remain separate, so failure of the common anchor cannot hide a usable pair.
    The direct fit is a comparator, not ground truth.
    """
    positions = np.flatnonzero(included).tolist()
    pairs = audit_pairs(positions, mode)
    directory.mkdir(parents=True, exist_ok=True)
    examples = {(a, positions[(i + len(positions) // 2) % len(positions)])
                for i, a in enumerate(positions) if i in (0, len(positions) // 2, len(positions) - 1)}
    rows = []
    for number, (a, b) in enumerate(pairs, 1):
        progress(f"direct/composed audit: pair {number}/{len(pairs)}")
        fixed, moving = np.asarray(stack[a], dtype=float), np.asarray(stack[b], dtype=float)
        bad_a, bad_b = clip_mask(fixed, levels), clip_mask(moving, levels)
        brightness, brightness_error = None, ""
        try:
            brightness = measure_quantile_brightness(fixed, moving)
        except ValueError as error:
            brightness_error = str(error)
        direct_translation = None
        for motion, cached in (("translation", translations), ("affine", affines)):
            row = {"input_index": int(frame_indices[a]), "target_index": int(frame_indices[b]),
                   "anchor_index": int(frame_indices[positions[0]]), "motion": motion,
                   "frame_gap": int(frame_indices[b] - frame_indices[a]),
                   "direct_status": "failed", "composed_status": "failed", "comparison_status": "unavailable",
                   "brightness_status": "complete" if brightness else "failed", "brightness_error": brightness_error}
            if brightness:
                row.update(gain=brightness["gain"], offset_dn=brightness["offset_dn"])
            direct = composed = None
            try:
                if motion == "affine" and direct_translation is None:
                    raise ValueError("direct translation initialization failed")
                direct, score = estimate_geometry(fixed, moving, motion=motion, sigma=sigma,
                                                   initial=direct_translation if motion == "affine" else None,
                                                   input_invalid=bad_a, target_invalid=bad_b)
                row.update(direct_status="complete", direct_ecc=score)
                if motion == "translation":
                    direct_translation = direct
            except ValueError as error:
                row["direct_error"] = str(error)
            try:
                if a not in cached or b not in cached:
                    raise ValueError("input or target has no successful reference-relative fit; see geometry.csv")
                composed = pair_transform(cached[a], cached[b])
                row["composed_status"] = "complete"
            except ValueError as error:
                row["composed_error"] = str(error)
            for prefix, matrix in (("direct", direct), ("composed", composed)):
                if matrix is not None:
                    row.update({f"{prefix}_m{r}{c}": float(matrix[r, c]) for r in range(2) for c in range(3)})
            if direct is not None and composed is not None:
                try:
                    metrics, arrays = comparison_metrics(fixed, moving, direct, composed, bad_a, bad_b, brightness)
                    row.update(metrics, comparison_status="complete")
                    if (a, b) in examples:
                        stem = f"{motion}_input_{frame_indices[a]:04d}_target_{frame_indices[b]:04d}"
                        np.savez_compressed(directory / f"{stem}.npz", **arrays)
                        row["example_arrays"] = f"geometry_audit/{stem}.npz"
                        differences = [arrays["composed_target"] - fixed, arrays["direct_target"] - fixed,
                                       arrays["direct_target"] - arrays["composed_target"]]
                        write_difference_viewer(directory / f"{stem}.html", differences, arrays["common_valid"],
                                                ("Via reference - A", "Direct - A", "Direct - via reference"),
                                                f"{motion}: A {frame_indices[a]}, B {frame_indices[b]}",
                                                difference_label="Geometry-only difference (no brightness correction)")
                        row["difference_viewer"] = f"geometry_audit/{stem}.html"
                except ValueError as error:
                    row.update(comparison_status="failed", comparison_error=str(error))
            rows.append(row)
    return rows


def geometry_audit_report(out: Path, rows: list[dict], mode: str) -> str:
    """Show every selected pair, and preserve the three representative viewers."""
    body = '<section><h2>Direct versus composed registration</h2>'
    body += f'<p>Pair selection: {escape(mode)}; {len(rows) // 2} ordered pairs, each evaluated for translation and affine. <a href="geometry_comparison.csv">All matrices, errors, pixel counts and statuses (CSV)</a>.</p>'
    body += '<p>Direct translation starts from identity; direct affine starts from that independent direct translation. The composed route uses W_B @ inverse(W_A). Each route resamples original B once; A is untouched. Direct fits are comparators, not ground truth. The translation estimator here is ECC, not the training pipeline\'s legacy coarse/refine estimator.</p>'
    body += '<p>Both routes use exactly the same valid pixels, excluding clipping, cubic footprints and the full 2 px Gaussian blur support. Raw and blurred residual metrics use that same support. Coordinate disagreement is in native pixels. CSV percentile residuals apply the same full-image B-to-A brightness fit to both routes. A smaller noisy-image residual alone does not establish a more accurate transform.</p>'
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, motion in enumerate(("translation", "affine")):
        selected = [row for row in rows if row["motion"] == motion]
        x = np.arange(len(selected))
        axes[0, column].plot(x, [r.get("coordinate_rms_px", np.nan) for r in selected], ".-", label="RMS on shared support")
        axes[0, column].plot(x, [r.get("corner_max_px", np.nan) for r in selected], ".-", label="Maximum at image corners")
        axes[0, column].set(title=f"{motion.title()}: transform disagreement", ylabel="Pixels")
        for route in ("composed", "direct"):
            axes[1, column].plot(x, [r.get(f"{route}_blurred_rms_dn", np.nan) for r in selected], ".-", label=route)
        axes[1, column].set(title="Geometry-only blurred residual RMS", ylabel="DN")
        failures = {name: sum(r[f"{name}_status"] != "complete" for r in selected)
                    for name in ("direct", "composed", "comparison")}
        body += f'<p>{motion.title()}: {failures["direct"]} failed direct fits, {failures["composed"]} unavailable composed fits, {failures["comparison"]} unavailable comparisons. Gaps remain gaps, not zeros.</p>'
    for ax in axes.flat:
        ax.set_xlabel("Ordered-pair ordinal (indices in CSV)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    body += _figure(out, "geometry_audit/comparison.png", fig, "All selected pairs, on matched measurement support. Inspect both fit failures and the size of transform disagreement.")
    body += '<details><summary>Every comparison and failure</summary><div class="scroll"><table><tr><th>A / B</th><th>Motion</th><th>Coordinate RMS / max (px)</th><th>Pixels</th><th>Direct / composed blurred RMS (DN)</th><th>Status / reason</th></tr>'
    for row in rows:
        reasons = '; '.join(f'{name}: {row[f"{name}_status"]} {row.get(f"{name}_error", "")}' for name in ("direct", "composed", "comparison"))
        body += (f'<tr><td>{row["input_index"]} / {row["target_index"]}</td><td>{row["motion"]}</td>'
                 f'<td>{_number(row.get("coordinate_rms_px"))} / {_number(row.get("coordinate_max_px"))}</td>'
                 f'<td>{row.get("comparison_pixels", 0)}</td><td>{_number(row.get("direct_blurred_rms_dn"))} / {_number(row.get("composed_blurred_rms_dn"))}</td><td>{escape(reasons)}</td></tr>')
    body += '</table></div></details>'
    for row in rows:
        if "difference_viewer" in row:
            body += f'<p>{row["motion"].title()} A {row["input_index"]}, B {row["target_index"]}: <a href="{row["difference_viewer"]}">Interactive blurred differences</a> | <a href="{row["example_arrays"]}">Originals, single-warp results, matrices and masks</a></p>'
    return body + '</section>'
