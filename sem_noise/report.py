"""Offline HTML reports and standalone diagnostic figures (no external assets)."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .registration import CORNERS, PARAMETERS
from .site_registration import PANELS, REGION_GRID, difference_palette


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


def _with_error(row: dict, key: str, digits: int = 4) -> str:
    error = row.get(f"{key}_se")
    return f"{_number(row.get(key), digits)} ± {_number(error, 2)}" if error is not None else _number(row.get(key), digits)


def _document(title: str, body: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{STYLE}</style></head><body><main><h1>{escape(title)}</h1>{body}</main></body></html>'


def _figure(out: Path, name: str, figure, caption: str) -> str:
    figure.savefig(out / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    encoded = base64.b64encode((out / name).read_bytes()).decode("ascii")
    return f'<section><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}"><p>{escape(caption)}</p></section>'


def _warnings(messages: list[str]) -> str:
    return "<section><h2>Interpretation and quality flags</h2><ul>" + "".join(f"<li>{escape(m)}</li>" for m in messages) + "</ul></section>"


def _errorbar(ax, x, rows: list[dict], key: str, label: str | None = None, color: str | None = None) -> None:
    y = np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows], dtype=float)
    e = np.array([r.get(f"{key}_se") if r.get(f"{key}_se") is not None else np.nan for r in rows], dtype=float)
    ax.errorbar(x, y, yerr=np.where(np.isfinite(e), e, 0), fmt=".", ms=5, lw=0.9, capsize=2, label=label, color=color)


def _registration_report(out: Path, summary: dict, rows: list[dict], regions: list[dict]) -> str:
    diagnostic = summary["registration"]
    if not diagnostic.get("enabled"):
        return '<section><h2>Registration fit</h2><p>Registration was disabled for this run; frames were taken as aligned and no fit, difference images or brightness track exist.</p></section>'
    body = '<section><h2>Registration fit: one least-squares fit per frame</h2>'
    body += '<p><a href="registration.csv">Pass 2 per-frame numbers (CSV)</a> · <a href="registration_pass1.csv">Pass 1 (against the first frame)</a> · <a href="registration.json">Both passes with covariances</a> · <a href="regions.csv">4×4 region residuals</a> · <a href="differences/">Native-resolution difference images</a></p>'
    body += f'<p>Reference for the reported numbers: <b>{escape(diagnostic["reference_pass2"])}</b>. Pass 1 used {escape(diagnostic["reference_pass1"])} and is saved for comparison. Each frame is fitted once on native pixels, both copies blurred by {_number(diagnostic["blur_sigma_px"])} px, no subsampling, no search, starting from zero shift, with a Huber loss and clipped pixels masked. Eight parameters: dy, dx, the four affine terms a11 a12 a21 a22, gain, offset.</p>'
    body += f'<p>{escape(diagnostic["convention"])}. The affine terms are dimensionless; their effect is shown as the displacement they add at the ROI corners, in pixels. Nothing is decided here: every frame is reported, and whether a rotation, shear, or gain is real is read by comparing the number with its error bar.</p>'
    body += f'<p class="muted">Error bars: {escape(diagnostic["standard_errors"])}. The blur makes neighbouring residuals dependent, so each frame\'s error bars are scaled by its measured residual correlation area (column residual_correlation_area_px2; about 12.6 px² for white noise blurred by 1 px). Frames whose fit did not reach the step tolerance are marked in the converged column; their numbers are shown as they stand.</p></section>'
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
    body += f'<p>Each row is one native-resolution PNG with three panels, left to right: <b>frame minus reference before correction</b>, <b>after the shift alone</b> (the fit\'s centre translation with its gain and offset, affine terms set to zero), and <b>after the full fit</b>. The three panels share one symmetric colour scale, set per frame to the 99th percentile of the uncorrected absolute difference and printed in the caption; grey marks pixels outside the common valid area or touching clipped values. The last two panels differ only by the four affine terms.</p>'
    body += f'<img src="data:image/png;base64,{scale}" alt="difference colour scale" style="max-width:420px">'
    body += f'<p>Below each image, the {REGION_GRID}×{REGION_GRID} tables give the RMS difference (DN) per region for the same three panels on the same pixels, so a corner that improves only under the full fit is visible as a number.</p>'
    by_frame: dict[int, list[dict]] = {}
    for region in regions:
        by_frame.setdefault(region["frame_position"], []).append(region)
    for row in rows:
        if "difference_image" not in row:
            continue
        body += f'<div class="frame"><h4>Acquisition {row["frame_index"]}</h4>'
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


def site_report(out: Path, summary: dict, maps: dict[str, np.ndarray], frames: list[dict],
                registration_rows: list[dict], regions: list[dict]) -> None:
    """Render quantitative diagnostics without treating the repeat mean as truth."""
    native, aligned = summary["modes"]["native"], summary["modes"]["aligned"]
    body = '<p><a href="../index.html">All sites</a> · <a href="summary.json">Metrics JSON</a> · <a href="frames.csv">Frame audit CSV</a> · <a href="maps.npz">Full-resolution maps (NumPy)</a></p>'
    body += f'<section><h2>{summary["accepted_frames"]} / {summary["input_frames"]} frames analysed</h2><p>Native flat-region temporal σ: <b>{_number(native["flat_temporal_sigma_dn"])} DN</b>. Aligned: <b>{_number(aligned["flat_temporal_sigma_dn"])} DN</b>. Largest drift: <b>{_number(summary["max_drift_px"])} px</b>. Largest corner displacement from the affine terms: <b>{_number(summary["max_corner_effect_px"])} px</b>.</p><p>DN means digital number: the original exported pixel units. Native statistics use integer translations; aligned statistics use bilinear translations of the fitted centre shift only (no affine warp, no brightness correction). The latter changes noise variance and spatial correlation. The predicted mean variance multiplier for independent white noise is {_number(summary["bilinear_white_noise_variance_factor_mean"])}; no universal correction is applied.</p></section>'
    body += _warnings(summary["warnings"])
    body += _registration_report(out, summary, registration_rows, regions)
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
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        im = ax.imshow(array[::stride, ::stride], cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-9))
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    body += _figure(out, "maps.png", fig, "Display limits use the 1st–99th percentiles; maps.npz preserves numerical values. The early/late difference is descriptive and retains gain/offset changes.")
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
    body += '<section><div class="scroll"><table><thead><tr><th>Site</th><th>Frames</th><th>Native flat σ (DN)</th><th>Aligned flat σ (DN)</th><th>Max drift (px)</th><th>Max corner effect (px)</th><th>Gain range</th><th>Flags / status</th></tr></thead><tbody>'
    for site in overview["sites"]:
        name = escape(site["site"])
        if site["status"] != "complete":
            body += f'<tr><td>{name}</td><td colspan="7">Failed: {escape(site["error"])}</td></tr>'
            continue
        brightness = site.get("brightness", {})
        gain = f'{_number(brightness.get("gain_min"), 3)} – {_number(brightness.get("gain_max"), 3)}' if brightness.get("enabled") else "n/a"
        body += f'<tr><td><a href="{site["directory"]}/report.html">{name}</a></td><td>{site["accepted_frames"]}/{site["input_frames"]}</td>'
        for value in [site["modes"]["native"]["flat_temporal_sigma_dn"], site["modes"]["aligned"]["flat_temporal_sigma_dn"], site["max_drift_px"], site["max_corner_effect_px"]]:
            body += f'<td>{_number(value)}</td>'
        body += f'<td>{gain}</td><td>{len(site["warnings"])} flags / complete</td></tr>'
    body += '</tbody></table></div><p>Compare sites with matched acquisition settings. These are descriptive observed-noise estimates; the site is the unit of comparison, not millions of independent pixels. Drift, corner effect and gain come from the per-frame registration fit; open a site for the numbers with their error bars and the difference images.</p></section>'
    (out / "index.html").write_text(_document("Repeated SEM acquisition — noise and stability", body), encoding="utf-8")
