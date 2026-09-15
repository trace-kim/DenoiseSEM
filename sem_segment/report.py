"""A self-contained HTML report: figures embedded, no external assets.

Everything is base64-embedded into a single ``index.html`` with an inline
stylesheet.  That is not decoration - this report is generated on a remote GPU
machine and read after copying one file, so a page that referenced a ``figures/``
directory would arrive broken.  It is also how this repository already ships
figure-heavy documents through git, which otherwise excludes ``*.png`` globally.

The panels are ordered by what a reader needs to decide: what was found, where
the two boundaries disagree, whether the edge fits can be trusted, and only then
the distributions.  The quality-flag list comes first, because a result whose
caveats are buried is a result that gets quoted without them.
"""

from __future__ import annotations

import base64
import io
from html import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .pipeline import SegmentationResult

STYLE = """
body {font: 16px/1.5 system-ui, sans-serif; color:#183047; background:#f3f6fa; margin:0}
main {max-width:1250px; margin:auto; padding:30px} h1,h2,h3 {line-height:1.2}
section {background:white; padding:22px; margin:20px 0; border:1px solid #d9e2eb; border-radius:8px}
img {width:100%; height:auto} table {width:100%; border-collapse:collapse; font-size:13px}
td,th {padding:7px 9px; text-align:left; border-bottom:1px solid #d9e2eb} th {background:#eef3f8}
.scroll {overflow-x:auto; max-height:520px; overflow-y:auto} .muted {color:#51657a}
a {color:#065c9f} code {background:#eef3f8; padding:2px 5px; border-radius:3px}
.kpis {display:flex; flex-wrap:wrap; gap:18px; margin:0}
.kpi {flex:1 1 150px; background:#f7fafd; border:1px solid #e1e9f1; border-radius:6px; padding:12px}
.kpi b {display:block; font-size:22px; line-height:1.3}
ul.flags {margin:0; padding-left:20px} ul.flags li {margin:6px 0}
"""


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "unavailable" if not np.isfinite(value) else f"{float(value):.{digits}g}"
    return escape(str(value))


def _document(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{STYLE}</style></head>"
        f"<body><main><h1>{escape(title)}</h1>{body}</main></body></html>"
    )


def _figure(figure, caption: str, *, dpi: int = 130) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<section><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}">'
        f'<p class="muted">{escape(caption)}</p></section>'
    )


def _colour_labels(labels: np.ndarray, count: int) -> np.ndarray:
    """A stable random colour per region id, black background."""
    rng = np.random.default_rng(12345)
    palette = rng.uniform(0.35, 1.0, size=(max(count, 1) + 1, 3))
    palette[0] = 0.0
    return palette[np.clip(labels, 0, count)]


def _draw_contours(axis, result: SegmentationResult, *, linewidth: float = 0.9) -> None:
    for index, contour in enumerate(result.coarse):
        if len(contour) < 3:
            continue
        ring = contour.closed
        axis.plot(ring[:, 1], ring[:, 0], "-", color="#ff9d3c", linewidth=linewidth, alpha=0.9)
        for hole in result.holes[index]:
            hring = hole.closed
            axis.plot(hring[:, 1], hring[:, 0], "-", color="#ffd27f", linewidth=linewidth, alpha=0.8)
    for refined in result.refined:
        polygon = refined.polygon
        if polygon.shape[0] < 3:
            continue
        ring = np.vstack([polygon, polygon[:1]])
        axis.plot(ring[:, 1], ring[:, 0], "-", color="#2ad4a0", linewidth=linewidth)


def _overview_figure(image01: np.ndarray, result: SegmentationResult, dpi: int) -> str:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    vmin, vmax = np.quantile(image01, [0.01, 0.99])

    axes[0].imshow(image01, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Input (measured pixels)")

    labels = result.label_map()
    axes[1].imshow(_colour_labels(labels, result.region_count))
    axes[1].set_title(f"{result.region_count} instance masks")

    axes[2].imshow(image01, cmap="gray", vmin=vmin, vmax=vmax)
    _draw_contours(axes[2], result)
    axes[2].set_title("Contours: mask boundary vs refined")
    for region in result.regions:
        measures = region.refined or region.coarse
        if measures is None:
            continue
        axes[2].annotate(
            str(region.region_id),
            (measures.centroid_x, measures.centroid_y),
            color="#ffffff", fontsize=7, ha="center", va="center",
        )
    for axis in axes:
        axis.set_axis_off()
    return _figure(
        figure,
        "Left: the array every measurement is taken from. Middle: one colour per instance. "
        "Right: method 1 (orange, mask boundary; pale orange for holes) and method 2 (green, "
        "gradient-refined) with region ids.",
        dpi=dpi,
    )


def _zoom_figure(image01: np.ndarray, result: SegmentationResult, config: Config, dpi: int) -> str:
    """Close-ups where the two boundaries can actually be told apart."""
    ranked = sorted(
        (r for r in result.regions if (r.refined or r.coarse)),
        key=lambda r: -(r.shift_abs_mean_px if np.isfinite(r.shift_abs_mean_px) else 0.0),
    )[: config.report.zoom_insets]
    if not ranked:
        return ""

    half = config.report.zoom_half_width_px
    figure, axes = plt.subplots(
        1, len(ranked), figsize=(5.0 * len(ranked), 5.2), constrained_layout=True, squeeze=False
    )
    vmin, vmax = np.quantile(image01, [0.01, 0.99])
    for axis, region in zip(axes[0], ranked):
        measures = region.refined or region.coarse
        cy, cx = measures.centroid_y, measures.centroid_x
        # Slide the window back inside the frame rather than letting it hang off
        # the edge, which would pad the panel with blank space exactly for the
        # border regions whose close-up is most worth seeing.
        height, width = image01.shape
        span = min(half, height / 2.0, width / 2.0)
        cy = float(np.clip(cy, span, max(span, height - span)))
        cx = float(np.clip(cx, span, max(span, width - span)))
        axis.imshow(image01, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        _draw_contours(axis, result, linewidth=1.6)
        axis.set_xlim(cx - span, cx + span)
        axis.set_ylim(cy + span, cy - span)
        axis.set_title(
            f"region {region.region_id}: mean shift {region.shift_abs_mean_px:.3f} px", fontsize=10
        )
        axis.set_xticks([])
        axis.set_yticks([])
    return _figure(
        figure,
        "Close-ups of the regions the refinement moved most, at pixel resolution. Orange is the "
        "mask boundary, green the refined edge. The gap between them is the measurement error the "
        "segmentation alone would have carried.",
        dpi=dpi,
    )


def _refinement_figure(
    image01: np.ndarray, result: SegmentationResult, config: Config, dpi: int
) -> str:
    """Whether the edge fits can be trusted, not just what they produced."""
    from .refine import sample_profiles

    shifts = np.concatenate(
        [r.displacement[r.valid] for r in result.refined if r.valid.any()]
    ) if any(r.valid.any() for r in result.refined) else np.zeros(0)
    if shifts.size == 0:
        return ""

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    axes[0].hist(shifts, bins=40, color="#2ad4a0", edgecolor="#12705a")
    axes[0].axvline(0.0, color="#ff9d3c", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("refinement displacement (px)")
    axes[0].set_ylabel("vertices")
    axes[0].set_title(
        f"shift: mean {shifts.mean():+.3f}, 3σ {3 * shifts.std():.3f} px", fontsize=10
    )

    vmin, vmax = np.quantile(image01, [0.01, 0.99])
    axes[1].imshow(image01, cmap="gray", vmin=vmin, vmax=vmax)
    points = np.vstack([r.base_points[r.valid] for r in result.refined if r.valid.any()])
    limit = float(np.percentile(np.abs(shifts), 98)) or 1.0
    scatter = axes[1].scatter(
        points[:, 1], points[:, 0], c=shifts, cmap="coolwarm", s=3, vmin=-limit, vmax=limit
    )
    figure.colorbar(scatter, ax=axes[1], shrink=0.8, label="displacement (px)")
    axes[1].set_axis_off()
    axes[1].set_title("where the boundary moved", fontsize=10)

    drawn = 0
    wanted = config.report.profile_samples
    for refined in result.refined:
        if drawn >= wanted or not refined.valid.any():
            continue
        indices = np.flatnonzero(refined.valid)
        for index in indices[:: max(1, len(indices) // 2)]:
            if drawn >= wanted:
                break
            profiles, offsets, _ = sample_profiles(
                image01,
                refined.base_points[index : index + 1],
                refined.normals[index : index + 1],
                search_px=config.refine.search_px,
                step_px=config.refine.step_px,
                interp_order=config.refine.interp_order,
            )
            axes[2].plot(offsets, profiles[0], linewidth=1.0, alpha=0.85)
            axes[2].plot([refined.displacement[index]], [np.interp(
                refined.displacement[index], offsets, profiles[0])], "o", markersize=5)
            drawn += 1
    axes[2].axvline(0.0, color="#ff9d3c", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("distance along outward normal (px)")
    axes[2].set_ylabel("intensity")
    axes[2].set_title(f"sampled edge profiles ({config.refine.estimator})", fontsize=10)

    return _figure(
        figure,
        "Left: how far refinement moved each vertex - a distribution centred well away from zero "
        "means the mask boundary was systematically biased. Middle: the same, in place. Right: raw "
        "intensity profiles along the normal with the fitted edge marked; the dashed line is the "
        "mask boundary. If these profiles do not look like edges, no number in this report is "
        "trustworthy.",
        dpi=dpi,
    )


def _distribution_figure(result: SegmentationResult, config: Config, dpi: int) -> str:
    usable = [r for r in result.regions if r.refined is not None]
    if len(usable) < 2:
        return ""
    figure, axes = plt.subplots(1, 4, figsize=(17, 3.9), constrained_layout=True)
    panels = [
        ("cd_px", f"CD ({config.metrology.cd_definition}) (px)"),
        ("area_px2", "area (px²)"),
        ("circularity", "circularity"),
        ("equivalent_diameter_px", "equivalent diameter (px)"),
    ]
    for axis, (attribute, label) in zip(axes, panels):
        coarse = [getattr(r.coarse, attribute) for r in usable if r.coarse]
        refined = [getattr(r.refined, attribute) for r in usable]
        bins = min(20, max(5, len(usable) // 2))
        axis.hist(coarse, bins=bins, alpha=0.55, label="mask boundary", color="#ff9d3c")
        axis.hist(refined, bins=bins, alpha=0.75, label="refined", color="#2ad4a0")
        axis.set_xlabel(label)
        axis.set_ylabel("regions")
    axes[0].legend(fontsize=8)
    return _figure(
        figure,
        "Distributions by both methods. A visible offset between the two histograms is the bias "
        "the mask boundary would have introduced into every number derived from it.",
        dpi=dpi,
    )


def _kpis(result: SegmentationResult, config: Config) -> str:
    stats = result.image_stats
    unit = "nm" if config.input.pixel_size_nm else "px"
    cd_key = f"cd_{unit}_refined_mean" if f"cd_{unit}_refined_mean" in stats else "cd_px_refined_mean"
    tiles = [
        ("Regions", result.region_count),
        ("Aggregated", stats.get("region_count_aggregated")),
        (f"Mean CD ({unit})", stats.get(cd_key)),
        ("Mean |shift| (px)", stats.get("shift_abs_mean_px_mean")),
        ("Valid vertices", result.diagnostics.valid_fraction),
        (f"Mean LER 3σ ({unit})", stats.get(f"ler_3sigma_{unit}_mean", stats.get("ler_3sigma_px_mean"))),
    ]
    cells = "".join(
        f'<div class="kpi"><span class="muted">{escape(name)}</span><b>{_number(value)}</b></div>'
        for name, value in tiles
    )
    return f'<section><div class="kpis">{cells}</div></section>'


def _flags(result: SegmentationResult) -> str:
    diagnostics = result.diagnostics
    items = list(diagnostics.warnings)
    if diagnostics.rejections.get("instances_rejected"):
        reasons = ", ".join(
            f"{k.replace('_', ' ')}: {v}"
            for k, v in diagnostics.rejections.get("rejected_by_reason", {}).items()
        )
        items.append(
            f"{diagnostics.rejections['instances_rejected']} of "
            f"{diagnostics.rejections.get('instances_in')} raw masks were filtered out ({reasons})."
        )
    if diagnostics.refine_rejections:
        reasons = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in diagnostics.refine_rejections.items())
        items.append(f"Contour vertices rejected during edge fitting - {reasons}.")
    items.append(
        "Method 1 (mask boundary) interpolates the 0.5 level of a binary mask, so its precision is "
        "bounded by the resolution the mask was produced at. Method 2 measures the original pixels. "
        "Where the two disagree, method 2 is the measurement."
    )
    if not result.provenance.get("config", {}).get("input", {}).get("pixel_size_nm"):
        items.append(
            "No pixel size was supplied, so every value is in pixels. Nothing in this package reads "
            "a physical scale from an image file; pass --pixel-size-nm to add nanometre columns."
        )
    body = "".join(f"<li>{escape(text)}</li>" for text in items)
    return f'<section><h2>Interpretation and quality flags</h2><ul class="flags">{body}</ul></section>'


def _table(result: SegmentationResult, config: Config) -> str:
    rows = result.rows(pixel_size_nm=config.input.pixel_size_nm)
    if not rows:
        return "<section><h2>Regions</h2><p>No regions were measured.</p></section>"
    unit = "nm" if config.input.pixel_size_nm else "px"
    columns = [
        ("region_id", "id"),
        (f"cd_{unit}_refined", f"CD ({unit})"),
        (f"cd_{unit}_coarse", f"CD coarse ({unit})"),
        (f"equivalent_diameter_{unit}_refined", f"eq. diam ({unit})"),
        (f"area_{unit}2_refined", f"area ({unit}²)"),
        ("circularity_refined", "circularity"),
        ("solidity_refined", "solidity"),
        ("aspect_ratio_refined", "aspect"),
        ("shift_abs_mean_px", "|shift| px"),
        (f"ler_3sigma_{unit}", f"LER 3σ ({unit})"),
        ("valid_fraction", "valid"),
        ("touches_border", "border"),
    ]
    present = [(key, label) for key, label in columns if key in rows[0]]
    header = "".join(f"<th>{escape(label)}</th>" for _, label in present)
    body = ""
    for row in rows[: config.report.max_table_rows]:
        cells = "".join(f"<td>{_number(row.get(key))}</td>" for key, _ in present)
        body += f"<tr>{cells}</tr>"
    more = ""
    if len(rows) > config.report.max_table_rows:
        more = f'<p class="muted">Showing {config.report.max_table_rows} of {len(rows)} regions; the full table is in <code>metrology.csv</code>.</p>'
    return (
        f"<section><h2>Regions</h2>{more}"
        f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'
        "</section>"
    )


def render_html(
    image01: np.ndarray,
    result: SegmentationResult,
    config: Config,
    *,
    title: str = "SEM segmentation and metrology",
) -> str:
    """Build the complete report as a single self-contained HTML string."""
    dpi = config.report.dpi
    backend = result.diagnostics.backend
    body = (
        f'<p class="muted">Backend <code>{escape(str(backend.get("backend", "?")))}</code>'
        f'{" · model <code>" + escape(str(backend.get("model_id"))) + "</code>" if backend.get("model_id") else ""}'
        f' · {result.shape[0]}x{result.shape[1]} px'
        f' · estimator <code>{escape(config.refine.estimator)}</code>'
        " · <a href=\"metrology.csv\">metrology.csv</a>"
        " · <a href=\"contours.json\">contours.json</a>"
        " · <a href=\"summary.json\">summary.json</a></p>"
    )
    body += _kpis(result, config)
    body += _flags(result)
    body += _overview_figure(image01, result, dpi)
    body += _zoom_figure(image01, result, config, dpi)
    body += _refinement_figure(image01, result, config, dpi)
    body += _distribution_figure(result, config, dpi)
    body += _table(result, config)
    return _document(title, body)


def write_report(path, image01: np.ndarray, result: SegmentationResult, config: Config, **kwargs):
    """Render and write ``index.html``."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(image01, result, config, **kwargs), encoding="utf-8")
    return path
