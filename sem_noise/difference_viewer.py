"""Offline, native-pixel difference viewer with a user-controlled DN range."""

from __future__ import annotations

import base64
import gzip
from html import escape
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from .site_registration import difference_palette


def write_difference_viewer(path: Path, differences: list[np.ndarray], valid: np.ndarray,
                            labels: tuple[str, ...], title: str, *, difference_label: str = "Target − input",
                            blur_sigma_px: float = 2.0) -> None:
    """Save signed native-pixel display data without clipping or subsampling.

    Gaussian filtering is linear: filtering a signed difference equals
    subtracting images blurred after their respective correction stages.
    Only pixels with full finite, valid Gaussian support are displayed, so
    masked values and warp borders cannot leak into the visible differences.
    Numerical analysis continues to use the original arrays. Each viewer is a
    standalone HTML file; opening it locally requires neither a server nor CDN.
    Display values use float32, or float64 when float32 would overflow or erase
    a nonzero difference.
    """
    if not np.isfinite(blur_sigma_px) or blur_sigma_px < 0:
        raise ValueError("difference blur sigma must be finite and nonnegative")
    panels = []
    radius = int(4 * blur_sigma_px + 0.5)
    for delta in differences:
        delta = np.asarray(delta, dtype=np.float64)
        usable = valid & np.isfinite(delta)
        if blur_sigma_px > 0:
            delta = ndimage.gaussian_filter(np.where(usable, delta, 0.0), blur_sigma_px)
            usable = ndimage.minimum_filter(usable, size=2 * radius + 1, mode="constant", cval=0)
        panels.append(np.where(usable, delta, np.nan))
    original = np.stack(panels)
    del panels
    with np.errstate(over="ignore", under="ignore"):
        values = original.astype("<f4")
    # Do not turn valid extreme float64 inputs into invalid pixels or zeros.
    if np.any(np.isfinite(original) & (~np.isfinite(values) | ((original != 0) & (values == 0)))):
        values = original.astype("<f8")
    finite = np.isfinite(values)
    maximum = float(np.max(np.abs(values[finite]))) if finite.any() else 0.0
    payload = {"shape": list(values.shape), "bytes_per_value": values.dtype.itemsize,
               "labels": list(labels), "maximum": maximum, "blur_sigma_px": blur_sigma_px,
               "palette": difference_palette(),
               "data": base64.b64encode(gzip.compress(values.tobytes(), compresslevel=3, mtime=0)).decode("ascii")}
    script = (Path(__file__).parent / "assets" / "difference_viewer.js").read_text(encoding="utf-8")
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    description = (f"Differences of Gaussian-blurred diagnostic images (σ = {blur_sigma_px:g} px), "
                   "blurred after each correction stage. This suppresses noise to expose misaligned edges. "
                   f"Only pixels with complete ±{radius} px blur support are displayed. "
                   "The statistics and pixel readout here describe blurred differences; "
                   "the report's static PNGs and residual tables remain unblurred."
                   if blur_sigma_px > 0 else "Unblurred differences on the supplied valid pixels.")
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>
body {{font:14px/1.45 system-ui,sans-serif;color:#183047;margin:16px;background:white}}
h1 {{font-size:18px}} .controls {{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
input[type=range] {{width:260px;max-width:70vw}} input[type=number] {{width:160px}}
button,input {{font:inherit}} #panels {{display:grid;gap:8px;margin-top:12px}}
figure {{margin:0;min-width:0}} figcaption {{text-align:center;font-weight:600}}
canvas {{width:100%;height:auto;display:block;image-rendering:pixelated}}
#viewport {{overflow:auto}} #bar {{max-width:480px;margin:12px auto}}
#gradient {{height:16px;background:linear-gradient(to right,rgb(33,102,217),white,rgb(217,51,38))}}
#ticks {{display:flex;justify-content:space-between}} #readout {{min-height:1.5em}}
table {{border-collapse:collapse;width:100%;font-size:13px}} th,td {{text-align:left;padding:4px;border-bottom:1px solid #ddd}}
[role=alert] {{color:#b02020}} .note {{color:#51657a}} [hidden] {{display:none!important}}
</style></head><body><h1>{escape(title)}</h1>
<p class="note">{escape(description)}</p>
<div class="controls">
<label>Colour limit ± <input id="limit" type="number" step="any" aria-label="Colour limit in DN" disabled> DN</label>
<input id="range" type="range" min="0" max="1" step="any" aria-label="Adjust colour limit" disabled>
<button id="full" disabled>Full range</button>
<label><input id="native" type="checkbox"> Native pixels</label>
<button id="download" disabled>Save current PNG</button></div>
<p id="error" role="alert" hidden></p><p id="loading">Loading native differences…</p>
<div id="viewport"><div id="panels"></div></div>
<div id="bar"><div id="gradient"></div><div id="ticks"><span id="negative"></span><span>0</span><span id="positive"></span></div><div style="text-align:center">{escape(difference_label)} (DN)</div></div>
<p id="readout">Point at a pixel to read its signed difference.</p>
<table><thead><tr><th>Stage</th><th>Minimum DN</th><th>Maximum DN</th><th>RMS DN</th><th>Beyond current range (%)</th></tr></thead><tbody id="statistics"></tbody></table>
<p class="note">Enter any positive DN limit, or drag the slider. All panels use the same zero-centred linear scale. Blue/red saturation changes only the display; grey means invalid or a failed stage. Full range includes every finite difference. Native pixels enables scrolling at original resolution. Colour controls do not refit, reblur or change brightness estimates.</p>
<script id="difference-data" type="application/json">{encoded}</script><script>{script}</script></body></html>'''
    path.write_text(html, encoding="utf-8")
