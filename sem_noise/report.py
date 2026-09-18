"""Offline HTML reports and standalone diagnostic figures (no external assets)."""

from __future__ import annotations

import base64
import csv
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .difference_viewer import write_difference_viewer
from .registration import (CORNERS, PARAMETERS, clip_mask, fit_pixel_diagnostics,
                           parameter_vector, shift_only, warp)
from .site_registration import PANELS, REGION_GRID, difference_limit, difference_palette


STYLE = """
body {font: 16px/1.5 system-ui, sans-serif; color:#183047; background:#f3f6fa; margin:0}
main {max-width:1250px; margin:auto; padding:30px} h1,h2,h3 {line-height:1.2}
section {background:white; padding:22px; margin:20px 0; border:1px solid #d9e2eb; border-radius:8px}
img {width:100%; height:auto} table {width:100%; border-collapse:collapse; font-size:14px}
td,th {padding:10px; text-align:left; border-bottom:1px solid #d9e2eb} th {background:#eef3f8}
.scroll {overflow-x:auto} .muted {color:#51657a} a {color:#065c9f} code {background:#eef3f8;padding:2px 5px}
.frame {border-top:2px solid #d9e2eb; padding-top:14px; margin-top:18px}
.regions {display:flex; gap:24px; flex-wrap:wrap} .regions table {width:auto; font-size:13px}
.regions td, .regions th {padding:4px 10px; text-align:right}
"""

PARAMETER_LABELS = {"dy_px": "dy (px)", "dx_px": "dx (px)", "a11": "a11", "a12": "a12", "a21": "a21", "a22": "a22",
                    "gain": "gain", "offset_dn": "offset (DN)"}


def _number(value, digits: int = 4) -> str:
    return f"{value:.{digits}g}" if value is not None and isinstance(value, (int, float)) and np.isfinite(value) else "n/a"


def _error_key(key: str) -> str:
    return key[:-3] + "_se" if key.startswith("corner_") and key.endswith("_px") else f"{key}_se"


def _with_error(row: dict, key: str, digits: int = 4) -> str:
    error = row.get(_error_key(key))
    return f"{_number(row.get(key), digits)} ± {_number(error, 2)}" if error is not None else _number(row.get(key), digits)


def _document(title: str, body: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{STYLE}</style></head><body><main><h1>{escape(title)}</h1>{body}</main></body></html>'


def _figure(out: Path, name: str, figure, caption: str) -> str:
    figure.savefig(out / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    encoded = base64.b64encode((out / name).read_bytes()).decode("ascii")
    return f'<section><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}"><p>{escape(caption)}</p></section>'


def raw_histogram_examples(out: Path, stack: np.ndarray, included: np.ndarray,
                           indices: np.ndarray, source_dtype: str,
                           levels: tuple[float | None, float | None]) -> str:
    """Show all uncorrected ROI pixels, including clipping, for three raw frames."""
    positions = np.flatnonzero(included)
    selected = list(dict.fromkeys(positions[k] for k in (0, len(positions) // 2, len(positions) - 1)))
    images = [np.asarray(stack[i], dtype=np.float64) for i in selected]
    dtype = np.dtype(source_dtype)
    if dtype.kind in "ui" and dtype.itemsize == 1:
        limits = np.iinfo(dtype)
        edges = np.arange(limits.min - 0.5, limits.max + 1.5)
        bin_description = "One bin per integer DN across the full 8-bit storage range."
    else:
        lo, hi = min(float(a.min()) for a in images), max(float(a.max()) for a in images)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, 257)
        bin_description = "256 shared bins covering the full observed range of these examples."
    fig, axes = plt.subplots(1, len(selected), figsize=(15, 4.2), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    stats_rows, count_rows = [], []
    for ax, i, values in zip(axes.flat, selected, images):
        counts, _ = np.histogram(values, bins=edges)
        ax.stairs(counts, edges, fill=True, color="#315d83", alpha=0.8)
        ax.set(title=f"Raw acquisition {int(indices[i])}", xlabel="Raw intensity (DN)", xlim=(edges[0], edges[-1]))
        ax.grid(alpha=0.15)
        p01, median, p99 = np.percentile(values, [1, 50, 99])
        stats_rows.append([int(indices[i]), int(values.size), float(values.min()), p01, median, p99,
                           float(values.max()), float(values.mean()), float(values.std()),
                           int((values <= levels[0]).sum()) if levels[0] is not None else None,
                           int((values >= levels[1]).sum()) if levels[1] is not None else None])
        count_rows.extend({"frame_index": int(indices[i]), "bin_left_dn": float(left),
                           "bin_right_dn": float(right), "pixel_count": int(count)}
                          for left, right, count in zip(edges[:-1], edges[1:], counts))
    axes[0, 0].set_ylabel("Pixel count")
    with (out / "raw_histograms.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_index", "bin_left_dn", "bin_right_dn", "pixel_count"])
        writer.writeheader()
        writer.writerows(count_rows)
    body = '<section id="raw-image-histograms"><h2>Raw image histograms</h2><details open><summary>First, middle, and last included acquisitions</summary>'
    body += f'<p>Original dtype: <b>{escape(source_dtype)}</b>. These are all raw pixels within the configured ROI, or the whole image when no ROI is set. Clipped pixels are included. No registration, blur, brightness correction, normalization, or comparison mask is applied. Axes show intensity in DN and pixel counts on linear scales. {bin_description}</p>'
    body += _figure(out, "raw_histograms.png", fig,
                    "Histograms of the original pixel values, including clipping bounds. All pixels contribute; bins and axes are shared across examples.")
    body += '<div class="scroll"><table><tr>' + ''.join(f'<th>{label}</th>' for label in
             ("Frame", "Pixels", "Min", "P1", "Median", "P99", "Max", "Mean", "Std. dev.",
              f"Pixels ≤ {_number(levels[0])}", f"Pixels ≥ {_number(levels[1])}")) + '</tr>'
    for row in stats_rows:
        body += '<tr>' + ''.join(f'<td>{value if isinstance(value, int) else _number(value, 5)}</td>' for value in row) + '</tr>'
    body += '</table></div><p>Intensity statistics are in DN. Clipping counts use the configured or storage bounds. <a href="raw_histograms.csv">Download exact bin counts (CSV)</a> · <a href="raw_histograms.png">Open histogram figure</a>.</p></details></section>'
    return body


def _warnings(messages: list[str]) -> str:
    return "<section><h2>Interpretation and quality flags</h2><ul>" + "".join(f"<li>{escape(m)}</li>" for m in messages) + "</ul></section>"


def _errorbar(ax, x, rows: list[dict], key: str, label: str | None = None, color: str | None = None) -> None:
    y = np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows], dtype=float)
    e = np.array([r.get(_error_key(key)) if r.get(_error_key(key)) is not None else np.nan for r in rows], dtype=float)
    ax.errorbar(x, y, yerr=np.where(np.isfinite(e), e, 0), fmt=".", ms=5, lw=0.9, capsize=2, label=label, color=color)


def intermediate_examples(out: Path, stack: np.ndarray, fit: dict,
                          levels: tuple[float | None, float | None], indices: np.ndarray,
                          sigma: float) -> str:
    """Render diagnostic examples from actual fit pixels without re-fitting."""
    positions = list(fit["positions"])
    selected = list(dict.fromkeys((positions[0], min(positions, key=lambda i: fit["pass2"][i]["gain"]),
                                  max(positions, key=lambda i: fit["pass2"][i]["corner_max_px"]))))
    directory = out / "intermediates"
    directory.mkdir(exist_ok=True)
    body = '<section><h2>Intermediate examples: geometry and brightness</h2><p>Examples are the first included frame, the lowest fitted gain, and the largest affine corner effect (duplicates shown once). All frames remain in the tables and difference images. Image limits are shared within each example; every numerical array is exported at native resolution. Display rendering may reduce image size.</p></section>'
    reference = fit["mean"]
    for i in selected:
        row = fit["pass2"][i]
        p = parameter_vector(row)
        frame = np.asarray(stack[i], dtype=float)
        bad = clip_mask(frame, levels)
        geometry = p.copy()
        geometry[6:] = (1.0, 0.0)
        affine, affine_valid = warp(frame, geometry, bad)
        shifted, shift_valid = warp(frame, shift_only(p), bad)
        full = p[6] * affine + p[7]
        valid = fit["mean_valid"] & ~bad & affine_valid & shift_valid
        pixels = fit_pixel_diagnostics(reference, frame, p, sigma=sigma,
                                       reference_invalid=~fit["mean_valid"], moving_invalid=bad)
        fit_valid = pixels["fit_valid"]
        x = pixels["moving_blurred_warped"][fit_valid]
        y = pixels["reference_blurred"][fit_valid]
        weights = pixels["huber_weights"][fit_valid]
        stem = f"frame_{int(indices[i]):04d}"
        np.savez_compressed(directory / f"{stem}.npz", original=frame, reference=reference,
                            shift_corrected=shifted, affine_corrected=affine, full_corrected=full,
                            native_valid=valid, parameters=p, **pixels)
        body += f'<section><h3>Acquisition {indices[i]}</h3><p><a href="intermediates/{stem}.npz">Native arrays, final fit mask, Huber weights and parameters (NPZ)</a>. The affine-only image includes translation and all four affine terms, with gain 1 and offset 0. Brightness correction is then applied to that image.</p></section>'
        limits = np.percentile(np.concatenate((reference[valid], frame[valid])), [1, 99])
        if limits[1] <= limits[0]:
            limits[1] = limits[0] + 1
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        panels = ((frame, "Original moving frame"), (pixels["moving_blurred"], f"Moving frame blurred ({sigma:g} px)"),
                  (reference, "Fixed registered-mean reference"), (shifted, "Shift only: no brightness change"),
                  (affine, "Affine + shift: no brightness change"), (full, "Affine + shift + gain/offset"))
        for ax, (array, title) in zip(axes.flat, panels):
            ax.imshow(np.ma.array(array, mask=~valid), cmap="gray", vmin=limits[0], vmax=limits[1], interpolation="nearest")
            ax.set(title=title)
            ax.set_axis_off()
        body += _figure(out, f"intermediates/{stem}_images.png", fig,
                        f"Acquisition {indices[i]}: original, blurred and corrected images. Shared grayscale range {limits[0]:.4g} to {limits[1]:.4g} DN; masked pixels are blank. Filled clipped values are excluded from fit observations.")
        differences = (frame - reference, shifted - reference, affine - reference, full - reference)
        limit = difference_limit(differences, valid)
        fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
        for ax, delta, title in zip(axes, differences, ("Before", "Shift only", "Affine + shift only", "Full fit")):
            im = ax.imshow(np.ma.array(delta, mask=~valid), cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set(title=title)
            ax.set_axis_off()
        fig.colorbar(im, ax=axes, shrink=0.7, label="Corrected frame minus reference (DN)")
        body += _figure(out, f"intermediates/{stem}_differences.png", fig,
                        "Native-pixel differences on the same mask and full colour range. The affine-only and full-fit panels isolate what gain/offset changes, including any contrast suppression.")
        viewer = f"intermediates/{stem}_differences.html"
        write_difference_viewer(out / viewer, list(differences), valid, ("Raw", "Shift only", "Affine only", "Full fit"),
                                f"Acquisition {indices[i]}: legacy intermediate differences", difference_label="Frame − reference")
        body += f'<p><a href="{viewer}">Adjust the difference colour range interactively</a></p>'
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
        density = axes[0].hexbin(x, y, gridsize=65, bins="log", mincnt=1, cmap="viridis")
        xx = np.array([x.min(), x.max()])
        axes[0].plot(xx, p[6] * xx + p[7], color="#d62728", label=f"Fit: y = {p[6]:.5g} x + {p[7]:.5g}")
        axes[0].plot(xx, xx, "--", color="gray", label="Identity: y = x")
        axes[0].set(title="Actual brightness fit pixel pairs", xlabel="Blurred moving frame, geometrically warped (DN)", ylabel="Blurred fixed reference (DN)")
        axes[0].legend(fontsize=8)
        fig.colorbar(density, ax=axes[0], label="Pixel count (log scale)")
        weighted = axes[1].hexbin(x, y, C=weights, reduce_C_function=np.mean, gridsize=65,
                                  mincnt=1, vmin=0, vmax=1, cmap="viridis")
        axes[1].plot(xx, p[6] * xx + p[7], color="#d62728")
        axes[1].set(title="Final Huber weights on those pairs", xlabel="Warped blurred moving intensity (DN)", ylabel="Blurred reference intensity (DN)")
        fig.colorbar(weighted, ax=axes[1], label="Mean weight")
        xpad, ypad = max(float(np.ptp(x)) * 0.03, 1e-6), max(float(np.ptp(y)) * 0.03, 1e-6)
        for ax in axes[:2]:
            ax.set_xlim(x.min() - xpad, x.max() + xpad)
            ax.set_ylim(y.min() - ypad, y.max() + ypad)
        residual = pixels["fit_residual"][fit_valid]
        axes[2].hexbin(x, residual, gridsize=65, bins="log", mincnt=1, cmap="viridis")
        axes[2].axhline(0, color="#d62728")
        axes[2].set(title="Final blurred residual versus intensity", xlabel="Warped blurred moving intensity (DN)", ylabel="gain × moving + offset − reference (DN)")
        body += _figure(out, f"intermediates/{stem}_brightness_fit.png", fig,
                        f"All {len(x)} valid native-resolution fit pairs contribute to these binned plots; there is no additional regression or fitting subsample. The red line is the gain/offset from the joint eight-parameter fit. RMS of these exact blurred residuals: {np.sqrt(np.mean(residual ** 2)):.5g} DN. Density and weights are aggregated for display only.")
    return body


def _registration_report(out: Path, summary: dict, rows: list[dict], regions: list[dict]) -> str:
    diagnostic = summary["registration"]
    if diagnostic.get("method") == "affine":
        return ""  # The raw-pair report is rendered from its own pair records.
    if not diagnostic.get("enabled"):
        return '<section><h2>Registration fit</h2><p>Registration was disabled for this run; frames were taken as aligned. No registration or brightness correction was fitted. Raw acquisition brightness remains plotted above.</p></section>'
    body = '<section><h2>Registration fit: one least-squares fit per frame</h2>'
    body += '<p><b>Brightness bias:</b> this least-squares model treats the moving-image intensity as an error-free predictor, although both images contain noise. Gain can therefore shrink toward zero and offset compensate toward the reference mean even when true brightness is unchanged. Building the mean from pass-1 corrected images can compound this in pass 2. Huber loss, convergence and small standard errors do not remove this bias. The error bars are conditional approximations, not evidence that the fitted brightness change is physical. Inspect the pixel-pair plots and original/corrected contrast below. No gain is clamped or frame discarded.</p>'
    body += '<p><a href="registration.csv">Pass 2 per-frame numbers (CSV)</a> · <a href="registration_pass1.csv">Pass 1 (against the first frame)</a> · <a href="registration.json">Both passes with covariances</a> · <a href="regions.csv">4×4 region residuals</a> · <a href="differences/">Native-resolution difference images</a></p>'
    body += f'<p>Reference for the reported numbers: <b>{escape(diagnostic["reference_pass2"])}</b>. Pass 1 used {escape(diagnostic["reference_pass1"])} and is saved for comparison. Each frame is fitted once on native pixels, both copies blurred by {_number(diagnostic["blur_sigma_px"])} px, no subsampling, no search, starting from zero shift, with a Huber loss and clipped pixels masked. Eight parameters: dy, dx, the four affine terms a11 a12 a21 a22, gain, offset.</p>'
    body += f'<p>{escape(diagnostic["convention"])}. The affine terms are dimensionless; their effect is shown as the displacement they add at the ROI corners, in pixels. Every frame is reported. Compare numbers with their conditional error bars and residual images, while accounting for brightness bias and possible local minima; statistical precision alone does not establish a physical change.</p>'
    body += f'<p class="muted">Error bars: {escape(diagnostic["standard_errors"])}. The blur makes neighbouring residuals dependent, so each frame\'s error bars are scaled by its measured residual correlation area (column residual_correlation_area_px2; about 12.6 px² for white noise blurred by 1 px). Frames whose fit did not reach the step tolerance are marked in the converged column (termination_reason in CSV/JSON distinguishes iteration limit, stalled step and singular system); their numbers are shown as they stand.</p></section>'
    x = np.array([r["frame_index"] for r in rows])
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), constrained_layout=True)
    for ax, key in zip(axes[0, :2], ("dy_px", "dx_px")):
        _errorbar(ax, x, rows, key)
        ax.set(title=f"Drift {PARAMETER_LABELS[key]}", ylabel="Pixels")
    _errorbar(axes[0, 2], x, rows, "corner_max_px", color="#b0413e")
    axes[0, 2].set(title="Largest corner displacement from the affine terms", ylabel="Pixels")
    axes[0, 3].plot(x, [r["initial_rms_dn"] for r in rows], ".", color="#6b7280", label="before fit")
    axes[0, 3].plot(x, [r["residual_rms_dn"] for r in rows], ".", color="#087e8b", label="after fit")
    axes[0, 3].set(title="Residual RMS on the blurred copies", ylabel="DN")
    axes[0, 3].legend(fontsize=8)
    for ax, key in zip(axes[1], ("a11", "a12", "a21", "a22")):
        _errorbar(ax, x, rows, key)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(title=f"Affine term {key}", ylabel="Dimensionless")
    for ax in axes.ravel():
        ax.set_xlabel("Acquisition index")
        ax.grid(alpha=0.2)
    body += _figure(out, "registration.png", fig, "Pass 2 fit per frame with one-standard-error bars. dy/dx are the drift of the frame content relative to the reference centre. The affine panels show the raw terms; the corner panel converts them into pixels at the ROI corners.")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), constrained_layout=True)
    _errorbar(axes[0], x, rows, "gain")
    axes[0].axhline(1, color="gray", lw=0.7)
    axes[0].set(title="Gain to the reference scale", ylabel="Gain")
    _errorbar(axes[1], x, rows, "offset_dn")
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].set(title="Offset applied after gain", ylabel="DN")
    if all("mean_before_dn" in r for r in rows):
        axes[2].plot(x, [r["mean_before_dn"] for r in rows], ".-", color="#6b7280", label="frame mean, before")
        axes[2].plot(x, [r["mean_after_dn"] for r in rows], ".-", color="#087e8b", label="after gain and offset")
        axes[2].plot(x, [r["reference_mean_dn"] for r in rows], "--", color="#b0413e", lw=0.8, label="reference mean, same pixels")
        axes[2].legend(fontsize=8)
    axes[2].set(title="Brightness track", ylabel="Mean (DN)")
    for ax in axes:
        ax.set_xlabel("Acquisition index")
        ax.grid(alpha=0.2)
    body += _figure(out, "brightness.png", fig, "Per-frame gain and offset with one-standard-error bars, and the frame mean before and after applying them, on the pixels shared with the reference. A flat after-track is the fitted outcome, not independent proof; the residual images show what the two numbers cannot explain.")
    body += '<section><h3>All frames (pass 2)</h3><div class="scroll"><table><tr><th>Frame</th>' + ''.join(f'<th>{escape(PARAMETER_LABELS[k])}</th>' for k in PARAMETERS)
    body += '<th>Rotation (deg)</th><th>Shear</th><th>Corner max (px)</th><th>Residual RMS (DN)</th><th>Corr. area (px²)</th><th>Iter.</th><th>Converged</th></tr>'
    for row in rows:
        body += f'<tr><td>{row["frame_index"]}</td>' + ''.join(f'<td>{_with_error(row, k)}</td>' for k in PARAMETERS)
        body += f'<td>{_with_error(row, "rotation_deg", 3)}</td><td>{_with_error(row, "shear", 3)}</td><td>{_with_error(row, "corner_max_px", 3)} ({escape(str(row.get("corner_max_name")))})</td>'
        body += f'<td>{_number(row["initial_rms_dn"], 3)} → {_number(row["residual_rms_dn"], 3)}</td><td>{_number(row.get("residual_correlation_area_px2"), 3)}</td><td>{row["iterations"]}</td><td>{"yes" if row["converged"] else "no"}</td></tr>'
    body += '</table></div><p class="muted">Values are ± one standard error. Rotation is (a21 − a12)/2 and shear (a12 + a21)/2 in the small-motion limit, with propagated errors. The corner column names the corner with the largest displacement.</p></section>'
    body += '<section><h3>Affine effect at all four corners (pass 2)</h3><div class="scroll"><table><tr><th>Frame</th>' + ''.join(f'<th>{c.replace("_", " ")}: dy / dx / magnitude (px)</th>' for c in CORNERS) + '</tr>'
    for row in rows:
        body += f'<tr><td>{row["frame_index"]}</td>'
        for corner in CORNERS:
            body += '<td>' + ' / '.join(_with_error(row, f"corner_{corner}_{part}", 3) for part in ("dy_px", "dx_px", "px")) + '</td>'
        body += '</tr>'
    body += '</table></div></section>'
    body += _difference_report(out, rows, regions)
    return body


def _difference_report(out: Path, rows: list[dict], regions: list[dict]) -> str:
    palette = np.array(difference_palette(), dtype=np.uint8).reshape(-1, 3)[:255] / 255
    fig, ax = plt.subplots(figsize=(6, 0.6), constrained_layout=True)
    ax.imshow(palette[None, :, :], aspect="auto", extent=(-1, 1, 0, 1))
    ax.set_yticks([])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["−limit", "−½", "0", "+½", "+limit"])
    fig.savefig(out / "difference_scale.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    scale = base64.b64encode((out / "difference_scale.png").read_bytes()).decode("ascii")
    body = '<section><h3>Difference images for every frame</h3>'
    body += f'<p>Each row is one native-resolution PNG with three panels, left to right: <b>frame minus reference before correction</b>, <b>after the shift alone</b> (centre translation only; gain 1, offset 0), and <b>after the full fit</b>. The PNG shows the full symmetric range. Open its interactive viewer to enter any positive DN limit or adjust the continuous slider; all valid pixels remain in the measurements. Grey marks pixels outside the common valid area or touching clipped values. The full-fit panel adds both affine and brightness corrections; the intermediate examples separate their effects.</p>'
    body += f'<img src="data:image/png;base64,{scale}" alt="difference colour scale" style="max-width:420px">'
    body += f'<p>Below each image, the {REGION_GRID}×{REGION_GRID} tables give the RMS difference (DN) per region for the same three panels on the same pixels, so a corner that improves only under the full fit is visible as a number.</p>'
    by_frame: dict[int, list[dict]] = {}
    for region in regions:
        by_frame.setdefault(region["frame_position"], []).append(region)
    for row in rows:
        if "difference_image" not in row:
            continue
        body += f'<div class="frame"><h4>Acquisition {row["frame_index"]}</h4>'
        if row.get("difference_viewer"):
            body += f'<p><a href="{escape(row["difference_viewer"])}">Interactive difference colour range</a></p>'
        body += f'<a href="{escape(row["difference_image"])}"><img src="{escape(row["difference_image"])}" loading="lazy" alt="difference panels for acquisition {row["frame_index"]}"></a>'
        body += f'<p class="muted">Colour limit ± {_number(row["colour_limit_dn"], 3)} DN over {row["difference_pixels"]} shared pixels. RMS: before {_number(row["before_rms_dn"], 3)}, shift only {_number(row["shift_only_rms_dn"], 3)}, full fit {_number(row["full_fit_rms_dn"], 3)} DN. Drift ({_number(row["dy_px"], 3)}, {_number(row["dx_px"], 3)}) px, gain {_number(row["gain"], 4)}, offset {_number(row["offset_dn"], 3)} DN, corner max {_number(row["corner_max_px"], 3)} px.</p>'
        body += '<div class="regions">'
        cells = by_frame.get(row["frame_position"], [])
        for panel, title in zip(PANELS, ("Before correction", "Shift only", "Full fit")):
            body += f'<table><caption>{title}: RMS (DN)</caption>'
            for r in range(REGION_GRID):
                body += '<tr>' + ''.join(f'<td>{_number(next((c["rms_dn"] for c in cells if c["panel"] == panel and c["row"] == r and c["col"] == col), None), 3)}</td>' for col in range(REGION_GRID)) + '</tr>'
            body += '</table>'
        body += '</div></div>'
    return body + '</section>'


def _acquisition_figure(frames: list[dict], geometry: list[dict], registration: dict) -> plt.Figure:
    """Plot acquisition history, independently of which target each input uses."""
    affine = registration.get("method") == "affine"
    fig, axes = plt.subplots(2 if affine else 1, 2, figsize=(13, 8 if affine else 4), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    x = [r["frame_index"] for r in frames]
    by_index = {r["frame_index"]: r for r in geometry}
    def frame_values(key):
        return [r.get(key, np.nan) if r["included"] else np.nan for r in frames]
    def fitted_values(key):
        return [by_index.get(r["frame_index"], {}).get(key, np.nan) if r["included"] else np.nan for r in frames]
    for key, label in (("raw_mean_dn", "Raw full-image mean"), ("native_mean_dn", "Integer-aligned common-crop mean"),
                       ("aligned_mean_dn", "Subpixel-aligned common-crop mean")):
        axes[0].plot(x, frame_values(key), ".-", label=label)
    axes[0].set(title="Brightness evolution — no brightness correction", ylabel="Mean intensity (DN)")
    for key, label in (("dx_px", "x drift"), ("dy_px", "y drift")):
        axes[1].plot(x, fitted_values(key), ".-", label=label)
    axes[1].set(title="Translation drift from anchor" if affine else "Fitted centre displacement", ylabel="Pixels")
    if not registration.get("enabled"):
        axes[1].text(0.5, 0.5, "Registration disabled — no drift estimated", ha="center", transform=axes[1].transAxes)
    if affine:
        for key, label in (("affine_dx_px", "x drift"), ("affine_dy_px", "y drift")):
            axes[2].plot(x, fitted_values(key), ".-", label=label)
        axes[2].set(title="Affine centre drift from anchor", ylabel="Pixels")
        axes[3].plot(x, fitted_values("corner_max_px"), ".-", label="Maximum corner effect")
        axes[3].set(title="Affine deformation beyond translation", ylabel="Pixels at corners")
    for ax in axes:
        for row in frames:
            if not row["included"]:
                ax.axvline(row["frame_index"], color="gray", alpha=0.2)
        ax.set_xlabel("Acquisition index")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    return fig


def acquisition_brightness_report(out: Path, rows: list[dict]) -> str:
    """Show raw and actually corrected full-image means on one fixed reference."""
    anchor = rows[0]["reference_index"]
    x = [row["frame_index"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for key, label, style, color in (("raw_mean_dn", "Raw", ".-", "#777777"),
                                      ("two_region_mean_dn", "Two-region gain/offset", ".-", "#2878bd"),
                                      ("quantile_mean_dn", "Percentile gain/offset", "x--", "#d27a15")):
        axes[0].plot(x, [r.get(key, np.nan) for r in rows], style, label=label, color=color)
    axes[0].axhline(rows[0]["reference_mean_dn"], color="gray", ls=":",
                   label=f"Reference: acquisition {anchor}")
    axes[0].set(title=f"Acquisition brightness matched to acquisition {anchor}", ylabel="Mean intensity (DN)")
    for prefix, label, style, color in (("two_region_", "Two-region", ".-", "#2878bd"),
                                        ("quantile_", "Percentiles", "x--", "#d27a15")):
        for ax, key in zip(axes[1:], ("gain", "offset_dn")):
            ax.plot(x, [r.get(prefix + key, np.nan) for r in rows], style, label=label, color=color)
    axes[1].set(title="Gain to the same reference", ylabel="Gain")
    axes[2].set(title="Offset to the same reference", ylabel="Offset (DN)")
    for ax in axes:
        for row in rows:
            if not row["included"]:
                ax.axvline(row["frame_index"], color="gray", alpha=0.2)
        ax.set_xlabel("Acquisition index")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    body = f'<section id="acquisition-brightness"><h2>Before/after brightness: one fixed reference</h2><p>Every included acquisition is matched to <b>raw acquisition {anchor}</b>, the first included image. Training input A stays untouched; these are diagnostic copies. Excluded acquisitions retain their raw means and grey markers; missing corrections break the lines.</p>'
    body += '<p>All three curves average every supplied pixel, including clipping bounds, over the same full image (or explicitly configured ROI). Each corrected mean is measured from <code>gain × raw image + offset</code>, with no resampling, comparison crop, blur, output clipping, or additional mean normalization. Geometry is used only to estimate the two-region coefficients against the reference; its fixed low/high labels and corresponding valid pixels are the same as in the pair workflow. Percentiles use every supplied raw pixel. Neither estimator is changed.</p>'
    body += '<p>A flat corrected curve near the dotted reference level indicates consistent whole-image brightness. Remaining trends or excursions expose mismatch; motion bringing different content into the field, specimen changes, clipping, or noise-distribution changes can still affect these means. This is separate from the cyclic B→A pair tracks.</p>'
    body += '<p><a href="acquisition_brightness.csv">All plotted means, gains, offsets, pixel counts and status/reasons (CSV)</a> · <a href="pair_registration.json">Measurements JSON</a></p></section>'
    body += _figure(out, "acquisition_brightness.png", fig,
                    f"One brightness reference: acquisition {anchor}. Raw, two-region gain/offset, and percentile gain/offset means use identical full-image support. Corrected curves are measurements of corrected diagnostic images, not forced-flat means.")
    body += '<section><details><summary>Acquisition correction status and diagnostic arrays</summary><table><tr><th>Acquisition</th><th>Two-region status / reason</th><th>Percentile status / reason</th><th>Native diagnostic arrays</th></tr>'
    for row in rows:
        link = f'<a href="{escape(row["example_arrays"])}">Raw, reference and available corrected images (NPZ)</a>' if "example_arrays" in row else ""
        statuses = [escape(row.get(f"{prefix}_error", row[f"{prefix}_status"])) for prefix in ("two_region", "quantile")]
        body += f'<tr><td>{row["frame_index"]}</td><td>{statuses[0]}</td><td>{statuses[1]}</td><td>{link}</td></tr>'
    return body + '</table></details></section>'


def site_report(out: Path, summary: dict, maps: dict[str, np.ndarray], frames: list[dict],
                registration_rows: list[dict], regions: list[dict], intermediate_html: str = "",
                acquisition_html: str = "") -> None:
    """Render quantitative diagnostics without treating the repeat mean as truth."""
    native, aligned = summary["modes"]["native"], summary["modes"]["aligned"]
    body = '<p><a href="../index.html">All sites</a> · <a href="#acquisition-evolution">Acquisition evolution</a> · <a href="#raw-image-histograms">Raw image histograms</a> · <a href="summary.json">Metrics JSON</a> · <a href="frames.csv">Frame audit CSV</a> · <a href="maps.npz">Full-resolution maps (NumPy)</a></p>'
    body += f'<section><h2>{summary["accepted_frames"]} / {summary["input_frames"]} frames analysed</h2><p>Native flat-region temporal σ: <b>{_number(native["flat_temporal_sigma_dn"])} DN</b>. Aligned: <b>{_number(aligned["flat_temporal_sigma_dn"])} DN</b>. Largest drift: <b>{_number(summary["max_drift_px"])} px</b>. Largest corner displacement from the affine terms: <b>{_number(summary["max_corner_effect_px"])} px</b>.</p><p>DN means digital number: the original exported pixel units. Native statistics use integer translations; aligned statistics use bilinear translations of the fitted centre shift only (no affine warp, no brightness correction). The latter changes noise variance and spatial correlation. The predicted mean variance multiplier for independent white noise is {_number(summary["bilinear_white_noise_variance_factor_mean"])}; no universal correction is applied.</p></section>'
    body += _warnings(summary["warnings"])
    body += '<section id="acquisition-evolution"><h2>Acquisition brightness and motion</h2><p>All acquisitions in order. These tracks describe the measured sequence; the pair-correction tracks below describe different B→A mappings.</p></section>'
    body += acquisition_html
    body += _figure(out, "acquisition_evolution.png", _acquisition_figure(frames, registration_rows, summary["registration"]),
                    "Raw brightness uses the full supplied image. The two geometrically aligned means use the shared crop for noise statistics; none receives a gain/offset correction. Drift is content displacement (x right, y down), not the negative shift applied for alignment. Affine drift is measured at the image centre, separately from deformation at the corners. Excluded acquisitions and failed estimates break the lines; grey lines mark exclusions. Exact values are in frames.csv and geometry.csv (registration.csv in legacy fit mode).")
    body += _registration_report(out, summary, registration_rows, regions)
    body += intermediate_html
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [("unregistered_mean", "Unregistered mean", "gray"), ("aligned_mean", "Aligned repeat mean", "gray"),
              ("aligned_early_late_delta", "Last quarter − first quarter (DN)", "coolwarm"),
              ("native_std", "Native temporal standard deviation (DN)", "magma"),
              ("aligned_std", "Aligned temporal standard deviation (DN)", "magma"),
              ("native_flat_mask", "Native low-gradient, unclipped mask", "gray")]
    for ax, (key, title, cmap) in zip(axes.ravel(), panels):
        array = maps[key]
        stride = max(1, int(np.ceil(max(array.shape) / 600)))
        vmin, vmax = np.quantile(array.astype(float), [0.01, 0.99])
        if "delta" in key:
            vmax = max(float(np.max(np.abs(array))), 1e-9)
            vmin = -vmax
        im = ax.imshow(array[::stride, ::stride], cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-9))
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    body += _figure(out, "maps.png", fig, "Grayscale/variance display limits use the 1st–99th percentiles; maps.npz preserves numerical values. The early/late difference uses its full symmetric range and retains gain/offset changes.")
    delta = maps["aligned_early_late_delta"]
    write_difference_viewer(out / "early_late_difference.html", [delta], np.isfinite(delta), ("Last quarter − first quarter",),
                            "Early/late brightness and structure", difference_label="Last quarter − first quarter")
    body += '<p><a href="early_late_difference.html">Inspect early/late differences with an adjustable colour range</a></p>'
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    accepted = [r for r in frames if r["included"]]
    x = np.array([r["frame_index"] for r in accepted])
    for mode in ("native", "aligned"):
        ax.plot(x, [r[f"{mode}_flat_residual_rms_dn"] for r in accepted], ".-", label=mode)
    ax.set(title="Frame residual against repeat mean", xlabel="Acquisition index", ylabel="Flat-region RMS (DN)")
    ax.legend()
    ax.grid(alpha=0.2)
    body += _figure(out, "frame_residuals.png", fig, "Flat-region RMS of each frame against the repeat mean in both pixel domains; a frame that stands out here also stands out in the registration table.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for mode, color in (("native", "#136d90"), ("aligned", "#cc693a")):
        result = summary["modes"][mode]
        d = result["residual_distribution"]
        axes[0, 0].plot(d["histogram_centers_dn"], d["histogram_density"], label=mode, color=color)
        axes[0, 1].plot(d["normal_quantiles"], (np.array(d["observed_quantiles_dn"]) - d["mean_dn"]) / max(d["std_dn"], 1e-12), label=mode, color=color)
        bins = result["intensity_bins"]
        xx, yy = [b["signal_dn"] for b in bins], [b["variance_dn2"] for b in bins]
        axes[1, 0].plot(xx, yy, "o", label=mode, color=color)
        fit = result["mean_variance_fit"]
        if fit["available"]:
            axes[1, 0].plot(xx, np.array(xx) * fit["slope_dn"] + fit["intercept_dn2"], "--", color=color)
        pair = result["adjacent_difference_distribution"]
        if pair:
            axes[1, 1].plot(pair["histogram_centers_dn"], pair["histogram_density"], label=mode, color=color)
    d = native["residual_distribution"]
    xx = np.array(d["histogram_centers_dn"])
    if d["std_dn"] > 0:
        axes[0, 0].plot(xx, stats.norm.pdf(xx, d["mean_dn"], d["std_dn"]), "k--", label="Matched Gaussian")
    axes[0, 0].set(title="Corrected centered residual distribution", xlabel="Residual (DN)", ylabel="Density")
    axes[0, 1].plot([-3, 3], [-3, 3], "k--")
    axes[0, 1].set(title="Standardized Gaussian Q–Q", xlabel="Gaussian quantile", ylabel="Observed standardized quantile")
    axes[1, 0].set(title="Signal-dependent variance (flat regions)", xlabel="Signal from even positions (DN)", ylabel="Variance from odd positions (DN²)")
    axes[1, 1].set(title="Adjacent differences / √2", xlabel="Difference (DN)", ylabel="Density")
    for ax in axes.ravel():
        ax.legend(fontsize=9)
    body += _figure(out, "distribution.png", fig, "Residuals include specimen change and residual misalignment. The finite-repeat correction assumes independent stationary frames. Pair differences are symmetric and cannot identify the original distribution's skewness. A linear mean–variance fit is descriptive; it does not prove Poisson counting or determine electron gain.")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for mode in ("native", "aligned"):
        result = summary["modes"][mode]
        temporal = result["temporal"]
        acf, averaging = temporal["acf"], temporal["averaging"]
        if acf:
            axes[0, 0].plot([r["lag_frames"] for r in acf], [r["pixel_acf"] for r in acf], label=mode)
        if averaging:
            tau = [r["block_frames"] for r in averaging]
            axes[0, 1].loglog(tau, [r["pixel_allan_deviation_dn"] for r in averaging], "o-", label=mode)
            axes[0, 2].loglog(tau, [r["mean_allan_deviation_dn"] for r in averaging], "o-", label=mode)
            if mode == "native":
                axes[0, 1].loglog(tau, [r["white_noise_reference_dn"] for r in averaging], "k--", label="1/√N reference")
                axes[0, 1].loglog(tau, [r["shuffled_pixel_allan_deviation_dn"] for r in averaging], ":", label="Shuffled control")
        if temporal["periodogram_available"]:
            axes[1, 0].semilogy(temporal["mean_periodogram_frequency"], temporal["mean_periodogram_density"], label=mode)
    axes[0, 0].axhline(0, color="gray", lw=0.7)
    axes[0, 0].set(title="Pixel temporal autocorrelation", xlabel="Lag (frames)", ylabel="ACF")
    axes[0, 1].set(title="Averaging stability / pixel Allan deviation", xlabel="Frames per average", ylabel="DN")
    axes[0, 2].set(title="Mean-intensity Allan deviation", xlabel="Frames per average", ylabel="DN")
    axes[1, 0].set(title="Detrended mean-intensity periodogram", xlabel=native["temporal"]["frequency_unit"], ylabel="Power density")
    psd = maps["native_psd"]
    im = axes[1, 1].imshow(np.log10(np.maximum(psd, 1e-12)), extent=(-0.5, 0.5, 0.5, -0.5), cmap="magma")
    if native["spatial"]["pair_count"] == 0:
        axes[1, 1].text(0.5, 0.5, "No adjacent pairs; spectrum unavailable", ha="center", color="white", transform=axes[1, 1].transAxes)
    axes[1, 1].set(title="Native pair spatial spectrum (log₁₀)", xlabel="x cycles/pixel", ylabel="y cycles/pixel")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
    for mode in ("native", "aligned"):
        acf = summary["modes"][mode]["spatial"]["acf"]
        for direction, style in (("x", "-"), ("y", "--")):
            axes[1, 2].plot([r["lag_px"] for r in acf], [r[f"{direction}_acf"] for r in acf], style, label=f"{mode} {direction}")
    axes[1, 2].set(title="Directional spatial autocorrelation", xlabel="Lag (pixels)", ylabel="ACF")
    for ax in [*axes[0], axes[1, 0], axes[1, 2]]:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    body += _figure(out, "stability.png", fig, "Allan curves use differences of adjacent block averages, never an average compared with itself. Gaps are not joined. Overlapping pairs are dependent; the CSV also reports disjoint-pair counts. Spectra use an unmasked central crop, so residual structure can contribute. Mean removal biases finite-series ACF slightly negative. No line-frequency identification is possible without scan timing.")
    body += '<section><h2>Useful follow-up measurements</h2><p>Preserve acquisition order, original bit depth, dwell time, frame time, scan direction, detector, voltage/current, pixel size, working distance, and any automatic contrast, filtering, or averaging settings. Revisit sites after longer delays and acquire dark/blank and uniform-reference images to investigate fixed-pattern response and stability beyond this sequence.</p><p>See the package guide for estimator definitions, limitations, and references.</p></section>'
    (out / "report.html").write_text(_document(f'SEM noise analysis — {summary["site"]}', body), encoding="utf-8")


def index_report(out: Path, overview: dict) -> None:
    body = f'<p>{overview["successful_sites"]} of {overview["site_count"]} sites analyzed. <a href="summary.csv">Comparison CSV</a> · <a href="summary.json">All metrics</a> · <a href="input_manifest.json">Input hashes and ordering</a> · <a href="provenance.json">Configuration and versions</a></p>'
    warnings = list(overview["warnings"])
    if overview["cross_site_duplicate_groups"]:
        warnings.append(f'{len(overview["cross_site_duplicate_groups"])} decoded duplicate groups cross site boundaries; inspect summary.json before creating train/validation splits.')
    body += _warnings(warnings)
    body += '<section><div class="scroll"><table><thead><tr><th>Site</th><th>Frames</th><th>Native flat σ (DN)</th><th>Aligned flat σ (DN)</th><th>Max drift (px)</th><th>Max corner effect (px)</th><th>Two-region / legacy gain range</th><th>Percentile gain range</th><th>Flags / status</th></tr></thead><tbody>'
    for site in overview["sites"]:
        name = escape(site["site"])
        if site["status"] != "complete":
            body += f'<tr><td>{name}</td><td colspan="8">Failed: {escape(site["error"])}</td></tr>'
            continue
        brightness = site.get("brightness", {})
        gain = f'{_number(brightness.get("gain_min"), 3)} – {_number(brightness.get("gain_max"), 3)}' if brightness.get("enabled") else "n/a"
        quantile_gain = f'{_number(brightness.get("quantile_gain_min"), 3)} – {_number(brightness.get("quantile_gain_max"), 3)}'
        body += f'<tr><td><a href="{site["directory"]}/report.html">{name}</a></td><td>{site["accepted_frames"]}/{site["input_frames"]}</td>'
        for value in [site["modes"]["native"]["flat_temporal_sigma_dn"], site["modes"]["aligned"]["flat_temporal_sigma_dn"], site["max_drift_px"], site["max_corner_effect_px"]]:
            body += f'<td>{_number(value)}</td>'
        body += f'<td>{gain}</td><td>{quantile_gain}</td><td>{len(site["warnings"])} flags / complete</td></tr>'
    body += '</tbody></table></div><p>Compare sites with matched acquisition settings. These are descriptive observed-noise estimates; the site is the unit of comparison, not millions of independent pixels. In affine mode, gain ranges describe the reported target-to-input pairs; in legacy fit mode they describe frame-to-reference fits. Open a site for the geometry, brightness measurements and difference images.</p></section>'
    (out / "index.html").write_text(_document("Repeated SEM acquisition — noise and stability", body), encoding="utf-8")
