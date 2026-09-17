"""Report actual noisy input/target pairs, with A fixed throughout."""

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from .pair_diagnostics import PAIR_PANELS
from .report import _figure, _number

REGION_COLOURS = ListedColormap(["#eeeeee", "#2878bd", "#e89a27"])


def pair_report(out: Path, result: dict) -> str:
    registration = result["registration"]
    rows = result["pair_rows"]
    body = '<section><h2>Raw target-to-input matching</h2>'
    body += '<p><b>A is the untouched noisy input. B is a different noisy target.</b> Only B is resampled and brightness-corrected. A registered mean selects shared brightness regions; it is never the answer image in these diagnostics.</p>'
    body += '<p><a href="geometry.csv">Per-frame geometry CSV</a> · <a href="target_pairs.csv">Target-to-input measurements CSV</a> · <a href="pair_regions.csv">4×4 residual tables CSV</a> · <a href="pair_registration.json">All measurements JSON</a> · <a href="pairs/brightness_regions.npz">Native mean and region labels</a></p>'
    body += f'<p>Geometry: translation ECC, then full affine ECC initialized by that translation, using {_number(registration["blur_sigma_px"])} px blurred copies against acquisition {registration["anchor_index"]}. The full affine contains translation. Pair transforms compose the saved matrices and resample the original target once. {escape(registration["pair_selection"])}.</p>'
    body += '<p>Brightness: blue and orange regions are the lower and upper intensity quartiles of the blurred, geometrically aligned site mean. Their labels are fixed once per site and moved into A’s coordinates. Means are measured on the same valid pixels of raw A and affine-aligned B: <code>gain = (high_A − low_A) / (high_B − low_B)</code>, <code>offset = low_A − gain × low_B</code>. No noisy-pixel regression is performed.</p>'
    body += '<p>These are diagnostic estimates, not proof of unchanged specimen structure. Low/high regions must represent stable content. ECC reports correlation, not parameter standard errors; this method does not manufacture error bars. Failed geometry or brightness measurements retain their rows and reasons. Native/aligned noise statistics below still use translations only and receive no brightness correction.</p></section>'
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
    body += '<section><h3>Geometry measurements</h3><div class="scroll"><table><tr><th>Frame</th><th>Translation dy / dx (px)</th><th>Translation ECC</th><th>Affine matrix (sampling x,y)</th><th>Affine ECC</th><th>Max corner effect (px)</th><th>Status / reason</th></tr>'
    for row in result["geometry_rows"]:
        matrix = '; '.join(', '.join(_number(row.get(f"m{r}{c}"), 6) for c in range(3)) for r in range(2))
        status = '; '.join(f'{name}: {row[f"{name}_status"]} {row.get(f"{name}_error", "")}' for name in ("translation", "affine"))
        body += f'<tr><td>{row["frame_index"]}</td><td>{_number(row.get("dy_px"))} / {_number(row.get("dx_px"))}</td><td>{_number(row.get("translation_ecc"))}</td><td>{matrix}</td><td>{_number(row.get("affine_ecc"))}</td><td>{_number(row.get("corner_max_px"))}</td><td>{escape(status)}</td></tr>'
    body += '</table></div></section>'
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
    body += '<section><h3>Pair measurements</h3><div class="scroll"><table><tr><th>Input A</th><th>Target B</th><th>Gain</th><th>Offset DN</th><th>A low / high DN</th><th>B low / high DN</th><th>Low / high pixels</th><th>Status / reason</th></tr>'
    for row in rows:
        body += f'<tr><td>{row["input_index"]}</td><td>{row["target_index"]}</td><td>{_number(row.get("gain"))}</td><td>{_number(row.get("offset_dn"))}</td>'
        for prefix in ("input", "target"):
            body += f'<td>{_number(row.get(prefix + "_low_dn"))} / {_number(row.get(prefix + "_high_dn"))}</td>'
        body += f'<td>{row.get("low_pixels", "—")} / {row.get("high_pixels", "—")}</td><td>{escape(row.get("error", row["status"]))}</td></tr>'
    body += '</table></div></section>'
    for row in rows:
        if row["status"] != "complete":
            continue
        body += f'<section><h3>Input A = {row["input_index"]}, target B = {row["target_index"]}</h3>'
        if row.get("translation_error"):
            body += f'<p>{escape(row["translation_error"])} Its panel is grey.</p>'
        body += f'<a href="{row["difference_image"]}"><img loading="lazy" src="{row["difference_image"]}" alt="Target minus fixed input: before, translation, affine, brightness"></a>'
        body += f'<p>Left to right: target minus input <b>before</b>, after <b>translation</b>, after <b>affine</b>, after <b>affine + brightness</b>. Shared scale ±{_number(row["colour_limit_dn"])} DN; grey is invalid. RMS: ' + ', '.join(f'{name} {_number(row[name + "_rms_dn"])}' for name in PAIR_PANELS) + ' DN.</p><div class="regions">'
        cells = [cell for cell in result["region_rows"] if cell["input_index"] == row["input_index"]]
        for panel in PAIR_PANELS[1:]:
            body += f'<table><caption>{panel}: RMS (DN)</caption>'
            for r in range(4):
                body += '<tr>' + ''.join('<td>' + _number(next((c["rms_dn"] for c in cells if c["panel"] == panel and c["row"] == r and c["col"] == col), None), 3) + '</td>' for col in range(4)) + '</tr>'
            body += '</table>'
        body += '</div></section>'
        if "example_arrays" in row:
            body += _example(out, row)
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
    x, y = arrays["aligned_target"][valid], fixed[valid]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    density = ax.hexbin(x, y, gridsize=70, bins="log", mincnt=1, cmap="Greys")
    xx = [row["target_low_dn"], row["target_high_dn"]]
    yy = [row["input_low_dn"], row["input_high_dn"]]
    ax.scatter(xx, yy, c=["#2878bd", "#e89a27"], s=90, edgecolors="black", zorder=3, label="Measured low/high means")
    limits = np.array([x.min(), x.max()])
    ax.plot(limits, row["gain"] * limits + row["offset_dn"], color="#c23928", label=f"y = {row['gain']:.5g} x {row['offset_dn']:+.5g}")
    ax.set(title=f"Brightness from two region means: B {row['target_index']} → A {row['input_index']}",
           xlabel="Affine-aligned noisy target B (DN)", ylabel="Untouched noisy input A (DN)")
    ax.legend(fontsize=9)
    fig.colorbar(density, ax=ax, label="Pixel count (log scale)")
    body += _figure(out, f"{stem}_brightness.png", fig,
                    "The line passes through the two measured region means. The pixel cloud is shown for context and is not fitted. Neither image is assumed clean.")
    return body + f'<section><a href="{row["example_arrays"]}">Native original images, corrected targets, masks and pair matrix (NPZ)</a></section>'
