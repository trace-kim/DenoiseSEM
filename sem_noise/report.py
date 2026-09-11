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


STYLE = """
body {font: 16px/1.5 system-ui, sans-serif; color:#183047; background:#f3f6fa; margin:0}
main {max-width:1250px; margin:auto; padding:30px} h1,h2,h3 {line-height:1.2}
section {background:white; padding:22px; margin:20px 0; border:1px solid #d9e2eb; border-radius:8px}
img {width:100%; height:auto} table {width:100%; border-collapse:collapse; font-size:14px}
td,th {padding:10px; text-align:left; border-bottom:1px solid #d9e2eb} th {background:#eef3f8}
.scroll {overflow-x:auto} .muted {color:#51657a} a {color:#065c9f} code {background:#eef3f8;padding:2px 5px}
"""


def _number(value) -> str:
    return f"{value:.4g}" if value is not None and isinstance(value, (int, float)) else "unavailable"


def _document(title: str, body: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{STYLE}</style></head><body><main><h1>{escape(title)}</h1>{body}</main></body></html>'


def _figure(out: Path, name: str, figure, caption: str) -> str:
    figure.savefig(out / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    encoded = base64.b64encode((out / name).read_bytes()).decode("ascii")
    return f'<section><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}"><p>{escape(caption)}</p></section>'


def _warnings(messages: list[str]) -> str:
    return "<section><h2>Interpretation and quality flags</h2><ul>" + "".join(f"<li>{escape(m)}</li>" for m in messages) + "</ul></section>"


def site_report(out: Path, summary: dict, maps: dict[str, np.ndarray], frames: list[dict], local: list[dict]) -> None:
    """Render quantitative diagnostics without treating the repeat mean as truth."""
    native, aligned = summary["modes"]["native"], summary["modes"]["aligned"]
    body = '<p><a href="../index.html">All sites</a> · <a href="summary.json">Metrics JSON</a> · <a href="frames.csv">Frame audit CSV</a> · <a href="maps.npz">Full-resolution maps (NumPy)</a></p>'
    body += f'<section><h2>{summary["accepted_frames"]} / {summary["input_frames"]} frames accepted</h2><p>Native flat-region temporal σ: <b>{_number(native["flat_temporal_sigma_dn"])} DN</b>. Aligned: <b>{_number(aligned["flat_temporal_sigma_dn"])} DN</b>. Maximum estimated drift: <b>{_number(summary["max_drift_px"])} px</b>. Local residual: <b>{_number(summary["local_residual_rms_px"])} px RMS</b>.</p><p>DN means the original exported pixel units. Native statistics use integer translations; aligned statistics use bilinear interpolation. The latter changes noise variance and spatial correlation. The predicted mean variance multiplier for independent white noise is {_number(summary["bilinear_white_noise_variance_factor_mean"])}; no universal correction is applied.</p></section>'
    body += _warnings(summary["warnings"])
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    accepted = [r for r in frames if r["accepted"]]
    x = np.array([r["frame_index"] for r in accepted])
    for name, label in (("drift_dy_px", "y drift"), ("drift_dx_px", "x drift")):
        axes[0, 0].plot(x, [r[name] for r in accepted], ".-", label=label)
    rejected = [r for r in frames if not r["accepted"]]
    for r in rejected:
        axes[0, 0].axvline(r["frame_index"], color="red", alpha=0.18)
    axes[0, 0].set(title="Estimated displacement from anchor", xlabel="Acquisition index", ylabel="Pixels")
    axes[0, 0].legend()
    axes[0, 1].plot(x, [r["aligned_mean_dn"] for r in accepted], ".-", label="Mean intensity")
    axes[0, 1].set(title="Brightness evolution (no normalization)", xlabel="Acquisition index", ylabel="DN")
    for mode in ("native", "aligned"):
        axes[1, 0].plot(x, [r[f"{mode}_flat_residual_rms_dn"] for r in accepted], ".-", label=mode)
    axes[1, 0].set(title="Frame residual against repeat mean", xlabel="Acquisition index", ylabel="Flat-region RMS (DN)")
    axes[1, 0].legend()
    valid_local = [r for r in local if r["valid"]]
    if valid_local:
        axes[1, 1].scatter([r["frame_index"] for r in valid_local], [np.hypot(r["residual_dy_px"], r["residual_dx_px"]) for r in valid_local], s=12, alpha=0.5)
    else:
        axes[1, 1].text(0.5, 0.5, "No valid local tiles", ha="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set(title="Residual local translations after alignment", xlabel="Acquisition index", ylabel="Tile displacement (px)")
    body += _figure(out, "registration.png", fig, "Red lines mark excluded frames. Drift is the negative of the applied (dy, dx) correction. Tile shifts diagnose departures from translation; they are not applied as warps or interpreted as calibrated uncertainty.")
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
    body += '<section><div class="scroll"><table><thead><tr><th>Site</th><th>Accepted</th><th>Native flat σ (DN)</th><th>Aligned flat σ (DN)</th><th>Max drift (px)</th><th>Local residual (px)</th><th>Flags / status</th></tr></thead><tbody>'
    for site in overview["sites"]:
        name = escape(site["site"])
        if site["status"] != "complete":
            body += f'<tr><td>{name}</td><td colspan="6">Failed: {escape(site["error"])}</td></tr>'
            continue
        body += f'<tr><td><a href="{site["directory"]}/report.html">{name}</a></td><td>{site["accepted_frames"]}/{site["input_frames"]}</td>'
        for value in [site["modes"]["native"]["flat_temporal_sigma_dn"], site["modes"]["aligned"]["flat_temporal_sigma_dn"], site["max_drift_px"], site["local_residual_rms_px"]]:
            body += f'<td>{_number(value)}</td>'
        body += f'<td>{len(site["warnings"])} flags / complete</td></tr>'
    body += '</tbody></table></div><p>Compare sites with matched acquisition settings. These are descriptive observed-noise estimates; the site is the unit of comparison, not millions of independent pixels.</p></section>'
    (out / "index.html").write_text(_document("Repeated SEM acquisition — noise and stability", body), encoding="utf-8")
