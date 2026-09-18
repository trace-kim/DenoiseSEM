"""Report actual noisy input/target pairs, with A fixed throughout."""

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from .pair_diagnostics import PAIR_PANELS
from .report import _document, _figure, _number
from .site_registration import difference_palette

REGION_COLOURS = ListedColormap(["#eeeeee", "#2878bd", "#e89a27"])


def pair_report(out: Path, result: dict) -> str:
    """Return representative examples and write the full comparison report."""
    registration = result["registration"]
    rows = result["pair_rows"]
    body = '<section><h2>Raw target-to-input matching</h2>'
    body += '<p><b>A is the untouched noisy input. B is a different noisy target.</b> Only B is resampled and brightness-corrected. A registered mean selects shared brightness regions; it is never the answer image in these diagnostics.</p>'
    body += '<p><a href="geometry.csv">Per-frame geometry CSV</a> · <a href="target_pairs.csv">Target-to-input measurements CSV</a> · <a href="pair_regions.csv">4×4 residual tables CSV</a> · <a href="pair_registration.json">All measurements JSON</a> · <a href="pairs/brightness_regions.npz">Native mean and region labels</a></p>'
    body += f'<p>Geometry: translation ECC, then full affine ECC initialized by that translation, using {_number(registration["blur_sigma_px"])} px blurred copies against acquisition {registration["anchor_index"]}. The full affine contains translation. Pair transforms compose the saved matrices and resample the original target once. {escape(registration["pair_selection"])}.</p>'
    body += '<p>Brightness: blue and orange regions are the lower and upper intensity quartiles of the blurred, geometrically aligned site mean. Their labels are fixed once per site and moved into A’s coordinates. Means are measured on the same valid pixels of raw A and affine-aligned B: <code>gain = (high_A − low_A) / (high_B − low_B)</code>, <code>offset = low_A − gain × low_B</code>. No noisy-pixel regression is performed.</p>'
    body += '<p>These are diagnostic estimates, not proof of unchanged specimen structure. Low/high regions must represent stable content. ECC reports correlation, not parameter standard errors; this method does not manufacture error bars. Failed geometry or brightness measurements retain their rows and reasons. Native/aligned noise statistics in the site report still use translations only and receive no brightness correction.</p>'
    body += f'<p>All {len(rows)} pairs were measured: {registration["pair_failures"]} failed pair measurements, {registration["translation_failures"]} failed translations, and {registration["affine_failures"]} failed affine estimates. See the full tables for every failure reason.</p></section>'
    mean = result["maps"]["pair_reference_mean"]
    valid = result["maps"]["pair_reference_valid"]
    labels = result["maps"]["brightness_regions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].imshow(np.ma.array(mean, mask=~valid), cmap="gray")
    axes[0].set_title("Geometry-only mean: region selection")
    axes[1].imshow(labels, cmap=REGION_COLOURS, vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title(f"Fixed regions: low {(labels == 1).sum()} / high {(labels == 2).sum()} pixels")
    for ax in axes:
        ax.set_axis_off()
    body += _figure(out, "pairs/brightness_regions.png", fig,
                    "Blue: lower-intensity region. Orange: higher-intensity region. Grey: unused or invalid. The mean contains no fitted gain/offset correction. Pair measurements further intersect these regions with valid input and target pixels.")
    good = [row for row in rows if row["status"] == "complete"]
    if good:
        x = [row["input_index"] for row in good]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        axes[0].plot(x, [row["gain"] for row in good], ".-")
        axes[0].set(title="Gain applied to each paired target", ylabel="Gain")
        axes[1].plot(x, [row["offset_dn"] for row in good], ".-")
        axes[1].set(title="Offset applied to each paired target", ylabel="DN")
        for key, label in (("input_mean_dn", "Fixed input A"), ("target_mean_before_dn", "Raw target B"),
                           ("target_mean_after_dn", "Corrected target B")):
            axes[2].plot(x, [row[key] for row in good], ".-", label=label)
        axes[2].set(title="Pair brightness on common pixels", ylabel="Mean DN")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("Input acquisition index")
            ax.grid(alpha=0.2)
        body += _figure(out, "pairs/brightness_track.png", fig,
                        "Each point describes B→A for the target listed in the pair table. It does not normalize A to a common reference. Missing points correspond to explicit failed pairs.")
    selected = {rows[k]["input_index"] for k in (0, len(rows) // 2, len(rows) - 1)} if rows else set()
    examples = [row for row in rows if row["input_index"] in selected]
    overview = '<section><h3>Representative comparisons</h3><p>Only the first, middle, and last included inputs are shown here. The tracks use all measured pairs. <a href="pair_report_full.html">Open the full report: every pair, difference image, and measurement table</a>.</p></section>' + body
    overview += _geometry_table([row for row in result["geometry_rows"] if row["frame_index"] in selected])
    overview += _pair_table(examples)
    full = '<p><a href="report.html">Back to the site summary and representative examples</a> · <a href="report.html#raw-image-histograms">Raw image histograms</a></p>' + body
    full += _geometry_table(result["geometry_rows"]) + _pair_table(rows)
    for row in rows:
        if row["status"] != "complete":
            continue
        section = _pair_section(row, result["region_rows"])
        if "example_arrays" in row:
            section += _example(out, row)
        full += section
        if row["input_index"] in selected:
            overview += section
    (out / "pair_report_full.html").write_text(_document("Full raw target-to-input comparisons", full), encoding="utf-8")
    return overview


def _geometry_table(rows: list[dict]) -> str:
    body = '<section><h3>Geometry measurements</h3><div class="scroll"><table><tr><th>Frame</th><th>Translation dy / dx (px)</th><th>Translation ECC</th><th>Affine matrix (sampling x,y)</th><th>Affine ECC</th><th>Max corner effect (px)</th><th>Status / reason</th></tr>'
    for row in rows:
        matrix = '; '.join(', '.join(_number(row.get(f"m{r}{c}"), 6) for c in range(3)) for r in range(2))
        status = '; '.join(f'{name}: {row[f"{name}_status"]} {row.get(f"{name}_error", "")}' for name in ("translation", "affine"))
        body += f'<tr><td>{row["frame_index"]}</td><td>{_number(row.get("dy_px"))} / {_number(row.get("dx_px"))}</td><td>{_number(row.get("translation_ecc"))}</td><td>{matrix}</td><td>{_number(row.get("affine_ecc"))}</td><td>{_number(row.get("corner_max_px"))}</td><td>{escape(status)}</td></tr>'
    return body + '</table></div></section>'


def _pair_table(rows: list[dict]) -> str:
    body = '<section><h3>Pair measurements</h3><div class="scroll"><table><tr><th>Input A</th><th>Target B</th><th>Gain</th><th>Offset DN</th><th>A low / high DN</th><th>B low / high DN</th><th>Low / high pixels</th><th>Status / reason</th></tr>'
    for row in rows:
        body += f'<tr><td>{row["input_index"]}</td><td>{row["target_index"]}</td><td>{_number(row.get("gain"))}</td><td>{_number(row.get("offset_dn"))}</td>'
        for prefix in ("input", "target"):
            body += f'<td>{_number(row.get(prefix + "_low_dn"))} / {_number(row.get(prefix + "_high_dn"))}</td>'
        body += f'<td>{row.get("low_pixels", "—")} / {row.get("high_pixels", "—")}</td><td>{escape(row.get("error", row["status"]))}</td></tr>'
    return body + '</table></div></section>'


def _pair_section(row: dict, regions: list[dict]) -> str:
    body = f'<section><h3>Input A = {row["input_index"]}, target B = {row["target_index"]}</h3>'
    if row.get("translation_error"):
        body += f'<p>{escape(row["translation_error"])} Its panel is grey.</p>'
    body += f'<a href="{row["difference_image"]}"><img loading="lazy" src="{row["difference_image"]}" alt="Target minus fixed input: before, translation, affine, brightness"></a>'
    palette = np.array(difference_palette()).reshape(-1, 3)
    colours = [f'rgb({r},{g},{b})' for r, g, b in palette[[0, 127, 254]]]
    limit = row["colour_limit_dn"]
    body += '<div style="max-width:480px;margin:12px auto" aria-label="Difference colour scale in DN">'
    body += f'<div style="height:16px;background:linear-gradient(to right,{",".join(colours)})"></div>'
    body += '<div style="display:flex;justify-content:space-between">' + ''.join(f'<span>{_number(v, 3)}</span>' for v in (-limit, 0, limit)) + '</div><div style="text-align:center">Target − input (DN)</div></div>'
    body += f'<p>Left to right: target minus input <b>before</b>, after <b>translation</b>, after <b>affine</b>, after <b>affine + brightness</b>. Automatic shared scale ±{_number(limit)} DN covers the largest absolute valid difference across all four maps; grey is invalid. Subtraction uses signed floating-point DN. One extreme pixel can set the range; the absolute-difference P99 below shows how much smaller most differences are. Corrected target values are not clipped to the original storage range.</p>'
    body += '<table><tr><th>Stage</th><th>Signed minimum (DN)</th><th>Signed maximum (DN)</th><th>P99 |difference| (DN)</th><th>RMS (DN)</th></tr>'
    for panel in PAIR_PANELS:
        body += f'<tr><td>{panel}</td>' + ''.join(f'<td>{_number(row.get(panel + suffix))}</td>' for suffix in
                                               ("_min_dn", "_max_dn", "_abs_p99_dn", "_rms_dn")) + '</tr>'
    body += '</table><div class="regions">'
    cells = [cell for cell in regions if cell["input_index"] == row["input_index"]]
    for panel in PAIR_PANELS[1:]:
        body += f'<table><caption>{panel}: RMS (DN)</caption>'
        for r in range(4):
            body += '<tr>' + ''.join('<td>' + _number(next((c["rms_dn"] for c in cells if c["panel"] == panel and c["row"] == r and c["col"] == col), None), 3) + '</td>' for col in range(4)) + '</tr>'
        body += '</table>'
    body += '</div></section>'
    return body


def _example(out: Path, row: dict) -> str:
    with np.load(out / row["example_arrays"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    fixed, target = arrays["input"], arrays["target"]
    valid, labels = arrays["brightness_valid"], arrays["regions"]
    lo, hi = np.percentile(np.r_[fixed[valid], target[valid]], [1, 99])
    panels = ((fixed, "Raw input A — untouched"), (target, "Raw target B"),
              (np.ma.array(arrays["input_blurred"], mask=~arrays["input_blur_valid"]), "Blurred input copy"),
              (np.ma.array(arrays["target_blurred"], mask=~arrays["target_blur_valid"]), "Blurred target copy"),
              (arrays["translated_target"], "Target after translation"), (arrays["aligned_target"], "Target after affine"),
              (arrays["corrected_target"], "Target after affine + brightness"), (labels, "Region labels on input A"))
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, (array, title) in zip(axes.flat, panels):
        if title.startswith("Region"):
            ax.imshow(np.where(valid, array, 0), cmap=REGION_COLOURS, vmin=0, vmax=2, interpolation="nearest")
        else:
            ax.imshow(np.ma.array(array, mask=~valid), cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-9), interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    stem = Path(row["example_arrays"]).with_suffix("").as_posix()
    body = _figure(out, f"{stem}_images.png", fig,
                   f"Raw-pair intermediate images: input {row['input_index']}, target {row['target_index']}. Shared grayscale limits; these are the blurred copies used to estimate each frame's geometry against the anchor. Brightness means use raw A and the unblurred, geometrically aligned B.")
    fig = _pixel_scatter(arrays, row)
    body += _figure(out, f"{stem}_brightness.png", fig,
                    f"All {int(arrays['difference_valid'].sum())} corresponding valid pixels are plotted for each available stage, using the same mask and shared linear axes. Each dot is one pixel pair; no binning or subsampling. The affine panel's line passes through the two measured region means from the unchanged brightness calculation. No line is fitted to the scatter. Neither image is assumed clean.")
    return body + f'<section><a href="{row["example_arrays"]}">Native original images, corrected targets, masks and pair matrix (NPZ)</a></section>'


def _pixel_scatter(arrays: dict[str, np.ndarray], row: dict) -> plt.Figure:
    """Show individual corresponding pixels at each geometric stage, without fitting."""
    valid = arrays["difference_valid"]
    y = arrays["input"][valid]
    stages = (("target", "Raw target"), ("translated_target", "Translation corrected"),
              ("aligned_target", "Affine corrected"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True, sharey=True, constrained_layout=True)
    xmin, xmax = np.inf, -np.inf
    for ax, (key, title) in zip(axes, stages):
        x = arrays[key][valid]
        finite = np.isfinite(x)
        if finite.any():
            ax.scatter(x[finite], y[finite], s=1, alpha=0.2, color="#315d83", edgecolors="none", rasterized=True)
            xmin, xmax = min(xmin, float(x[finite].min())), max(xmax, float(x[finite].max()))
        else:
            ax.text(0.5, 0.5, "Translation estimate failed", ha="center", transform=ax.transAxes)
        ax.set(title=title, xlabel="Target B (DN)")
        ax.grid(alpha=0.15)
    xx = np.array([xmin, xmax])
    for ax in axes:
        ax.plot(xx, xx, "--", color="gray", lw=0.9, label="Identity: y = x")
    axes[2].plot(xx, row["gain"] * xx + row["offset_dn"], color="#c23928",
                 label=f"Region fit: y = {row['gain']:.5g} x {row['offset_dn']:+.5g}")
    axes[2].scatter([row["target_low_dn"], row["target_high_dn"]],
                    [row["input_low_dn"], row["input_high_dn"]],
                    c=["#2878bd", "#e89a27"], s=60, edgecolors="black", zorder=3, label="Measured low/high means")
    xpad = max((xmax - xmin) * 0.03, 1e-6)
    ymin, ymax = float(y.min()), float(y.max())
    ypad = max((ymax - ymin) * 0.03, 1e-6)
    axes[0].set(xlim=(xmin - xpad, xmax + xpad), ylim=(ymin - ypad, ymax + ypad), ylabel="Untouched input A (DN)")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Pixel-to-pixel comparison: target B {row['target_index']} → input A {row['input_index']}")
    return fig
