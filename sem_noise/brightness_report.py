"""Report panels for the separately audited brightness calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def brightness_report(out: Path, summary: dict) -> str:
    """Render saved brightness artifacts into the site's offline report."""
    import json
    from html import escape

    from .report import _figure, plt

    if "brightness_correction" not in summary:
        return ""
    diagnostic = summary["brightness_correction"]
    rows = json.loads((out / "brightness.json").read_text(encoding="utf-8"))["frames"]
    body = '<section><h2>Brightness correction comparison</h2><p><a href="brightness.csv">Per-frame fits and means</a> | <a href="brightness.json">Method and validation</a> | <a href="brightness_examples.npz">Full-resolution difference arrays</a></p>'
    body += f'<p>Reference acquisition: {diagnostic["reference_frame_index"]}. Validated fits: {diagnostic["fitted_frames"]}. Unavailable frames: {diagnostic["unavailable_frames"]}.</p>'
    body += '<p>Analysis copies are mapped to one fixed reference level. Training instead maps each target into its sampled input level. Original images and the native/aligned noise statistics are unchanged. Missing corrections remain gaps in the corrected track.</p>'
    body += '<p>' + escape(diagnostic["limitation"]) + '</p></section>'
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    x = [r["frame_index"] for r in rows]
    values = lambda key: [r.get(key) if r.get(key) is not None else np.nan for r in rows]
    for label, color in (("before", "#6b7280"), ("after", "#087e8b")):
        axes[0, 0].plot(x, values(f"{label}_mean_dn"), ".-", color=color, label=label)
        axes[0, 1].plot(x, values(f"{label}_reference_rmse_dn"), ".-", color=color, label=label)
    axes[0, 0].set(title="Mean on the same registered support", ylabel="Mean (DN)")
    axes[0, 1].set(title="Residual against noisy reference", ylabel="Pixel RMSE (DN)")
    axes[1, 0].plot(x, values("gain_to_reference"), ".-")
    axes[1, 0].set(title="Gain applied to frame", ylabel="Gain")
    axes[1, 1].plot(x, values("offset_to_reference_dn"), ".-")
    axes[1, 1].set(title="Offset applied after gain", ylabel="Offset (DN)")
    for ax in axes.ravel():
        ax.set_xlabel("Acquisition index (gaps retained)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[0, 1].legend()
    body += _figure(out, "brightness_evolution.png", fig,
                    "Before/after brightness evolution. Flattening is partly imposed by fitting and is not independent proof of correctness. RMSE includes noise in both images.")
    with np.load(out / "brightness_examples.npz", allow_pickle=False) as maps:
        for i in diagnostic["example_positions"]:
            row = rows[i]
            prefix = f"frame_{i}_"
            names = ("before", "after", "correction", "residual_before", "residual_after")
            arrays = [maps[prefix + name] for name in names]
            low, high = np.percentile(np.stack(arrays[:2]), [1, 99])
            correction_limit = max(float(np.max(np.abs(arrays[2]))), 1e-6)
            residual_limit = max(float(np.percentile(np.abs(np.stack(arrays[3:])), 99)), 1e-6)
            titles = ("Before (registered)", "After brightness correction", "After minus before", "Before minus reference", "After minus reference")
            fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
            for j, (ax, array, title) in enumerate(zip(axes, arrays, titles)):
                limit = correction_limit if j == 2 else residual_limit
                im = ax.imshow(array, cmap="gray" if j < 2 else "RdBu_r",
                               vmin=low if j < 2 else -limit, vmax=high if j < 2 else limit)
                ax.set_title(title, fontsize=10)
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, shrink=0.65, label="DN")
            status = row["model"] if row["available"] else "UNAVAILABLE: unchanged copy; " + row["reason"]
            body += _figure(out, f"brightness_difference_{i}.png", fig,
                            f'Acquisition {row["frame_index"]}: {status}. Image panels share one display range; reference residuals share a symmetric range. Correction has its own labeled scale. All differences are signed floating-point DN; no clipping or per-image contrast normalization is applied to the data.')
    return body
