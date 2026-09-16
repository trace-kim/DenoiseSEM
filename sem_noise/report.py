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


def _affine_report(out: Path, summary: dict, rows: list[dict], local: list[dict]) -> str:
    diagnostic = summary.get("affine_diagnostics", {})
    if not diagnostic.get("enabled"):
        return ""
    counts = diagnostic["selected_model_counts"]
    body = '<section><h2>Approximate tile-based affine diagnostics</h2><p>These earlier tile-displacement estimates are approximate. Use the separately validated registration estimate above for affine parameters and difference images.</p><p><a href="affine.json">Matrices and metadata (JSON)</a> | <a href="affine_models.csv">Model comparison CSV</a> | <a href="affine_frames.csv">Frame decisions CSV</a></p>'
    body += '<p>Supported frame counts: ' + ', '.join(f'{escape(k)}: {v}' for k, v in counts.items()) + '.</p>'
    body += f'<p>{diagnostic["unavailable_frames"]} measured frames lack a supported model; {diagnostic.get("not_assessed_frames", 0)} frames were not assessed. Missing estimates are unavailable, not zero motion.</p>'
    body += '<p>Use the simplest supported model. A richer model must improve held-out tile error by both the configured absolute and relative thresholds, and pass error and tile-deletion stability limits. Translation support means no sufficiently large improvement was detected; it does not prove zero rotation.</p>'
    body += '<p>Parameters below are from reliable full-affine candidates, including those where a simpler model was selected. They describe corrections into each frame’s leave-one-out repeat mean, not absolute stage motion or a shared anchor shape. Coordinates are within the analysis ROI; x points right, y down, and positive rotation is clockwise. Center offsets include the original translation. Scale/shear can also absorb specimen changes.</p>'
    body += '<p>These tile estimates do not change noise measurements or training data and are not used for the affine difference panels. Tile shifts approximate small motion and are limited to 3 px per axis. Tile-deletion spread measures sensitivity, not a confidence interval.</p></section>'
    reliable = [r for r in rows if r["model"] == "affine" and r["reliable"]]
    if not reliable:
        return body + '<section><p>No reliable affine parameter estimates. Inspect frame decisions, texture, tile size, and the selected ROI.</p></section>'
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    x = [r["frame_index"] for r in reliable]
    panels = [(("correction_rotation_deg",), "Correction rotation", "Degrees"),
              (("correction_scale_x_percent", "correction_scale_y_percent"), "Correction scale change", "Percent"),
              (("correction_center_dx_px", "correction_center_dy_px"), "Correction at ROI center", "Pixels"),
              (("correction_shear",), "Correction shear", "Dimensionless")]
    for ax, (keys, title, unit) in zip(axes.ravel(), panels):
        for key in keys:
            ax.scatter(x, [r[key] for r in reliable], s=15, label=key.replace("correction_", ""))
        ax.set(title=title, xlabel="Acquisition index", ylabel=unit)
        ax.legend(fontsize=8)
    for model in counts:
        model_rows = [r for r in rows if r["model"] == model and "cv_median_error_px" in r]
        axes[2, 0].scatter([r["frame_index"] for r in model_rows],
                           [r["cv_median_error_px"] for r in model_rows], s=12, label=model)
    axes[2, 0].set(title="Held-out tile prediction error", xlabel="Acquisition index", ylabel="Median error (px)")
    axes[2, 0].legend(fontsize=8)
    chosen = max(reliable, key=lambda r: abs(r["correction_rotation_deg"]) +
                 abs(r["correction_scale_x_percent"]) + abs(r["correction_scale_y_percent"]))
    tiles = [r for r in local if r["frame_position"] == chosen["frame_position"] and r["valid"]]
    axes[2, 1].quiver([r["x_px"] for r in tiles], [r["y_px"] for r in tiles],
                      [r["residual_dx_px"] for r in tiles], [r["residual_dy_px"] for r in tiles],
                      angles="xy")
    axes[2, 1].invert_yaxis()
    axes[2, 1].margins(0.2)
    axes[2, 1].set_aspect("equal")
    axes[2, 1].set(title=f'Local residual field, frame {chosen["frame_index"]}', xlabel="ROI x (px)", ylabel="ROI y (px)")
    body += _figure(out, "affine.png", fig, "Points omit unavailable estimates and do not bridge gaps. Residual arrows are visually autoscaled; numerical vectors are in local_registration.csv. Held-out errors include unreliable candidates so failures remain visible.")
    body += '<section><h3>Within-site affine parameter distributions</h3><table><tr><th>Correction parameter</th><th>Min</th><th>Median</th><th>Max</th><th>Std</th></tr>'
    for key, values in diagnostic["affine_parameter_distributions"].items():
        if values:
            body += '<tr><td>' + escape(key) + '</td>' + ''.join(f'<td>{_number(values[k])}</td>' for k in ("min", "median", "max", "std")) + '</tr>'
    return body + '</table><p>Descriptive distribution over reliable sampled frames, not independent-site uncertainty. Do not pool unrelated sites into one geometric reference.</p></section>'


def _feature_report(out: Path, summary: dict, rows: list[dict], differences: dict[str, np.ndarray]) -> str:
    diagnostic = summary.get("feature_affine", {})
    if not diagnostic.get("enabled"):
        return ""
    intensity = diagnostic.get("estimator") == "intensity"
    title = "Translation-initialized affine registration" if intensity else "Feature-based affine registration"
    body = f'<section><h2>{title}</h2><p><a href="feature_affine.json">Transforms and convention (JSON)</a> | <a href="feature_affine.csv">Per-frame parameters and quality (CSV)</a></p>'
    if not intensity:
        body += '<p><a href="feature_matches.csv">Matched point coordinates (CSV)</a></p>'
    body += f'<p>{diagnostic["estimated_frames"]} / {diagnostic["attempted_frames"]} attempted moving frames passed validation against fixed native reference frame {diagnostic["reference_frame_index"]}.</p>'
    if intensity:
        body += '<p>Accepted translations initialize small affine corrections on smoothed image copies, from coarse to fine, with brightness gain and offset fitted as nuisance parameters. Spatial validation regions are excluded from optimization. Affine must improve their RMS over a separately refined translation and remain stable across disjoint training regions. Unsupported corrections retain translation; no affine matrix is reported. Validation is conditional on the whole-image translation initializer, not an independent uncertainty estimate.</p>'
    else:
        body += '<p>SIFT features are matched directly between native-frame detector copies, without translation pre-alignment. Mutual descriptor matches pass a ratio test; affine RANSAC rejects outliers. Spatially separate matches are held out before fitting and are never used to refit the matrix. Failed or unsampled frames have no reported transform.</p>'
    body += '<p>Parameters are corrections from native ROI coordinates into the fixed reference: x right, y down, positive rotation clockwise. Scale changes are percentages; shear is dimensionless using A = R(theta) [[sx, shear*sy], [0, sy]]. Offsets describe displacement at the ROI center. The reference identity is not an estimated fit. Repeated patterns can still produce ambiguous registration; inspect errors and difference panels. These are acquisition-relative measurements, not calibrated stage motion.</p>'
    error_label = "Held-out RMS improvement (%)" if intensity else "Held-out median (px)"
    body += f'<div class="scroll"><table><tr><th>Frame</th><th>Status</th><th>Rotation (deg)</th><th>Scale x / y (%)</th><th>Shear</th><th>Center dx / dy (px)</th><th>{error_label}</th></tr>'
    for row in rows:
        body += f'<tr><td>{row["frame_index"]}</td><td>{escape(row["reason"])}</td>'
        body += f'<td>{_number(row.get("correction_rotation_deg"))}</td><td>{_number(row.get("correction_scale_x_percent"))} / {_number(row.get("correction_scale_y_percent"))}</td>'
        error = row.get("validation_relative_improvement") if intensity else row.get("validation_median_error_px")
        if intensity and error is not None:
            error *= 100
        body += f'<td>{_number(row.get("correction_shear"))}</td><td>{_number(row.get("correction_center_dx_px"))} / {_number(row.get("correction_center_dy_px"))}</td><td>{_number(error)}</td></tr>'
    body += '</table></div></section>'
    valid = [r for r in rows if r["available"] and r["frame_position"] != diagnostic["reference_position"]]
    if valid:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        panels = [(("correction_rotation_deg",), "Affine rotation", "Degrees"),
                  (("correction_scale_x_percent", "correction_scale_y_percent"), "Affine scale change", "Percent"),
                  (("correction_center_dx_px", "correction_center_dy_px"), "Affine center displacement", "Pixels"),
                  (("correction_shear",), "Affine shear", "Dimensionless")]
        for ax, (keys, title, unit) in zip(axes.ravel(), panels):
            for key in keys:
                ax.scatter([r["frame_index"] for r in valid], [r[key] for r in valid], label=key.replace("correction_", ""), s=18)
            ax.set(title=title, xlabel="Acquisition index", ylabel=unit)
            ax.legend(fontsize=8)
        body += _figure(out, "feature_affine.png", fig, "Validated affine estimates relative to one fixed native reference frame. Unavailable frames and reference identity are omitted.")
    body += '<section><h2>Frame-pair difference comparison</h2><p><a href="difference_examples.json">Pair identities, shared limits, and RMS (JSON)</a> | <a href="difference_examples.npz">Numerical differences and masks (NumPy)</a></p><p>Pairs are selected uniformly in acquisition order before inspecting fit quality. Each row shows moving minus reference in original exported units (DN), on the intersection of valid pixels for all available modes. All three differences in a row share one symmetric color scale. No brightness matching or smoothing is applied to these displayed measurements. Both correction modes use one bilinear resampling of the original moving image.</p><p>RMS is computed over the same pixels in each row. Interpolation changes noise; lower difference RMS alone does not prove a more accurate transform. Existing native and translation-aligned noise statistics are unchanged.</p></section>'
    for pair in summary.get("difference_examples", []):
        prefix = pair["prefix"]
        fig, axes = plt.subplots(1, 5, figsize=(19, 4.5), constrained_layout=True)
        ref, moving = differences[f"{prefix}_reference"], differences[f"{prefix}_moving"]
        stride = max(1, int(np.ceil(max(ref.shape) / 600)))
        low, high = np.percentile(np.concatenate((ref.ravel(), moving.ravel())), [1, 99])
        for ax, image, title in zip(axes[:2], (ref, moving), (f'Reference {pair["reference_frame_index"]}', f'Moving {pair["moving_frame_index"]}')):
            ax.imshow(image[::stride, ::stride], cmap="gray", vmin=low, vmax=max(high, low + 1e-9), interpolation="nearest")
            ax.set_title(title)
        im = None
        for ax, mode, label in zip(axes[2:], ("raw", "translation", "affine"), ("No registration fix", "Translation only", "Validated affine fix")):
            if mode in pair["available_modes"] and pair["valid_pixels"]:
                delta = differences[f"{prefix}_{mode}_diff"]
                im = ax.imshow(np.ma.masked_invalid(delta[::stride, ::stride]), cmap="coolwarm",
                               vmin=-pair["color_limit_dn"], vmax=pair["color_limit_dn"], interpolation="nearest")
                ax.set_title(f'{label}\nRMS {pair[f"{mode}_rms_dn"]:.3g} DN')
            else:
                ax.set_title(label)
                reason = pair["affine_reason"] if mode == "affine" else pair["translation_reason"]
                ax.text(0.5, 0.5, "Unavailable\n" + (reason if pair["valid_pixels"] else "No common valid pixels"),
                        ha="center", va="center", wrap=True, fontsize=8, transform=ax.transAxes)
            ax.set_facecolor("#dddddd")
        for ax in axes:
            ax.set_axis_off()
        if im is not None:
            fig.colorbar(im, ax=list(axes[2:]), shrink=0.65, label="Moving - reference (DN)")
        body += _figure(out, f"difference_{prefix}.png", fig,
                        f'Frame {pair["moving_frame_index"]} minus frame {pair["reference_frame_index"]}; {pair["valid_pixels"]} shared valid pixels. Limits are +/- {pair["color_limit_dn"]:.4g} DN (pooled 99th percentile absolute difference). Masked borders are blank; numerical arrays retain unclipped differences.')
    return body


def site_report(out: Path, summary: dict, maps: dict[str, np.ndarray], frames: list[dict], local: list[dict],
                affine: list[dict] | None = None, feature_rows: list[dict] | None = None,
                differences: dict[str, np.ndarray] | None = None) -> None:
    """Render quantitative diagnostics without treating the repeat mean as truth."""
    native, aligned = summary["modes"]["native"], summary["modes"]["aligned"]
    body = '<p><a href="../index.html">All sites</a> · <a href="summary.json">Metrics JSON</a> · <a href="frames.csv">Frame audit CSV</a> · <a href="maps.npz">Full-resolution maps (NumPy)</a></p>'
    body += f'<section><h2>{summary["accepted_frames"]} / {summary["input_frames"]} frames accepted</h2><p>Native flat-region temporal σ: <b>{_number(native["flat_temporal_sigma_dn"])} DN</b>. Aligned: <b>{_number(aligned["flat_temporal_sigma_dn"])} DN</b>. Maximum estimated drift: <b>{_number(summary["max_drift_px"])} px</b>. Local residual: <b>{_number(summary["local_residual_rms_px"])} px RMS</b>.</p><p>DN means the original exported pixel units. Native statistics use integer translations; aligned statistics use bilinear interpolation. The latter changes noise variance and spatial correlation. The predicted mean variance multiplier for independent white noise is {_number(summary["bilinear_white_noise_variance_factor_mean"])}; no universal correction is applied.</p></section>'
    body += _warnings(summary["warnings"])
    body += _feature_report(out, summary, feature_rows or [], differences or {})
    if summary.get("affine_diagnostics", {}).get("enabled"):
        body += '<details><summary>Earlier approximate tile diagnostics (supplementary)</summary>'
        body += _affine_report(out, summary, affine or [], local) + '</details>'
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
    diagnostics = [s for s in overview["sites"] if s.get("affine_diagnostics", {}).get("enabled")]
    if diagnostics:
        body += '<section><h2>Validated affine estimates</h2><table><tr><th>Site</th><th>Reference frame</th><th>Passed / attempted</th></tr>'
        for site in diagnostics:
            result = site.get("feature_affine", {})
            if result.get("enabled"):
                body += f'<tr><td><a href="{site["directory"]}/report.html">{escape(site["site"])}</a></td><td>{result["reference_frame_index"]}</td><td>{result["estimated_frames"]} / {result["attempted_frames"]}</td></tr>'
        body += '</table><p>Open each site for affine parameters, validation quality, and raw/translation/affine difference panels. Noise statistics remain translation-based.</p></section>'
        body += '<section><h2>Motion model support by site (approximate tiles)</h2><table><tr><th>Site</th><th>Translation</th><th>Rigid</th><th>Similarity</th><th>Affine</th><th>Unavailable / not assessed</th></tr>'
        for site in diagnostics:
            result = site["affine_diagnostics"]
            body += '<tr><td>' + escape(site["site"]) + '</td>'
            body += ''.join(f'<td>{result["selected_model_counts"][model]}</td>' for model in ("translation", "rigid", "similarity", "affine"))
            body += f'<td>{result["unavailable_frames"]} / {result.get("not_assessed_frames", 0)}</td></tr>'
        body += '</table><p>These supplementary counts use approximate tile correspondences. Affine registration fits are validated separately and applied only to the example difference images.</p></section>'
    (out / "index.html").write_text(_document("Repeated SEM acquisition — noise and stability", body), encoding="utf-8")
