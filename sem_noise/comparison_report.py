"""Visual audit of delivered uint8 images; no training imports or corrections."""

from __future__ import annotations

from collections import defaultdict
from html import escape
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw


def comparison_metrics(record: dict) -> list[dict]:
    rows = []
    for site in record["sites"]:
        for name, series in site["series"].items():
            frames = series["frames"]
            deltas = [f["brightness_delta_dn"] for f in frames if "brightness_delta_dn" in f]
            drift = [f["output_minus_raw_dy_px"] ** 2 + f["output_minus_raw_dx_px"] ** 2
                     for f in frames if f.get("output_minus_raw_dy_px") is not None
                     and f.get("output_minus_raw_dx_px") is not None]
            values = {"frames": len(frames), "clipped_images": sum(f.get("clipped", False) for f in frames),
                      "frames_with_complete_contours": sum(f.get("contour_status") == "available" for f in frames),
                      "brightness_bias_dn": float(np.mean(deltas)) if deltas else None,
                      "temporal_variation_dn": series.get("native", {}).get("temporal_rms_dn"),
                      "output_vs_raw_translation_rms_px": float(np.sqrt(np.mean(drift))) if drift else None,
                      "output_vs_raw_registration_count": len(drift)}
            for summary in site.get("repeatability", []):
                if summary["series"] == name:
                    values[f"{summary['method']}/ecd_sd"] = summary["median_cd_std"]
                    values[f"{summary['method']}/holes"] = summary["common_hole_count"]
            arm = record["models"].get(name, {}).get("arm", {})
            rows.append({"site": site["name"], "series": name, "step": series["step"],
                         "refinement_device": series.get("refinement_backend", {}).get("device", "cpu"),
                         "training_registration": arm.get("registration"),
                         "training_brightness": arm.get("brightness"), "values": values})
    return rows


def comparison_arms(record: dict) -> list[dict]:
    return [{"arm": name, "registration": model["arm"]["registration"],
             "brightness": model["arm"]["brightness"], "step": model["step"],
             "ema": model["ema"], "settings_source": model["arm"]["settings_source"],
             "checkpoint": model["checkpoint"], "checkpoint_sha256": model["sha256"]}
            for name, model in record["models"].items()]


def _table(rows: list[dict], columns: list[str]) -> str:
    def cell(value):
        return "—" if value is None else repr(value) if isinstance(value, float) else str(value)
    return ("<table><thead><tr>" + "".join(f"<th>{escape(k)}</th>" for k in columns) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{escape(cell(r.get(k)))}</td>" for k in columns) + "</tr>" for r in rows)
            + "</tbody></table>")


def _warning(record: dict, rows: list[dict]) -> str:
    if not rows:
        return ""
    return ('<details class="notice"><summary>' + str(len(rows)) + ' exported images have clipping flags</summary><p>'
            + escape(record["range_warning"]) + '</p>'
            + _table(rows, ["site", "model", "filename", "minimum_dn", "maximum_dn", "below_zero", "above_255"]) + '</details>')


def _script(path: Path, assignment: str, value: object) -> None:
    from .pipeline import _json_value
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_value(value), ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    path.write_text(f"{assignment} = {payload};\n", encoding="utf-8")


def _overlay(image: Image.Image, contours: list[dict]) -> Image.Image:
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for contour in contours:
        partial = contour["status"]["coarse"] != "valid"
        for key, color in (("coarse", "#d7ad56" if partial else "#43dbe2"), ("refined", "#ff9d50")):
            points = [(p[1], p[0]) for p in contour[key]]
            if len(points) > 2:
                if key == "coarse":
                    draw.line([*points, points[0]], fill=color, width=1)
                else:
                    valid = contour["refined_valid"]
                    for i in range(len(points)):
                        j = (i + 1) % len(points)
                        if valid[i] and valid[j]:
                            draw.line([points[i], points[j]], fill=color, width=1)
        for ring in contour["holes"]:
            points = [(p[1], p[0]) for p in ring]
            if points:
                draw.line([*points, points[0]], fill="#43dbe2", width=1)
        for path in contour.get("open_paths", []):
            points = [(p[1], p[0]) for p in path]
            if len(points) >= 2:
                draw.line(points, fill="#d7ad56", width=1)
    return image


def _contour_band(path: Path, shape: list[int], name: str, frames: list[dict],
                  grouped: dict, method: str) -> None:
    """Export every saved outline in native coordinates, without remeasurement.

    SVG keeps subpixel vertices visible when enlarged. Write one acquisition at
    a time rather than assembling another copy of the complete contour set.
    """
    height, width = shape
    scale = max(width, height) / 800
    margin, footer = 18 * scale, 90 * scale
    orders = [f["order"] for f in frames]
    first, last = min(orders), max(orders)
    stops = [(0, (35, 86, 180)), (.5, (20, 145, 150)), (1, (230, 75, 25))]

    def outline(points: list, close: bool = True) -> str:
        return ("M" + "L".join(f"{x},{y}" for y, x in points) + ("Z" if close else "")) if points else ""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + 2 * margin}" '
                     f'height="{height + footer}" viewBox="{-margin} {-margin} {width + 2 * margin} {height + footer}">'
                     f'<title>{escape(name)}: all {len(frames)} images, {method} contours</title>'
                     '<desc>Saved outlines in native pixel coordinates. No registration or alignment. '
                     'Interior rings and unmatched regions included; partial paths stay open. '
                     'Dashed lines mark partial boundaries or mask fallback.</desc>'
                     '<rect x="-100%" y="-100%" width="300%" height="300%" fill="white"/>'
                     '<g fill="none" stroke-width="0.8" stroke-opacity="0.45">')
        for frame in frames:
            t = (frame["order"] - first) / (last - first) if last != first else .5
            (a, start), (b, end) = stops[:2] if t <= .5 else stops[1:]
            color = "#" + "".join(f"{round(x + (y - x) * (t - a) / (b - a)):02x}" for x, y in zip(start, end))
            solid, partial = [], []
            for contour in grouped[name, frame["index"]]:
                partial.extend(outline(p, False) for p in contour.get("open_paths", []))
                rings = [outline(p) for p in contour["holes"]]
                if method == "coarse":
                    target = solid if contour["status"]["coarse"] == "valid" else partial
                    target.extend([outline(contour["coarse"]), *rings])
                elif contour["refined"]:
                    partial.append(outline(contour["refined"]))
                    partial.extend(rings)
                    points, valid = contour["refined"], contour["refined_valid"]
                    for i, point in enumerate(points):
                        j = (i + 1) % len(points)
                        if valid[i] and valid[j]:
                            solid.append(outline([point, points[j]], False))
                else:
                    partial.extend([outline(contour["coarse"]), *rings])
            stream.write(f'<g data-frame="{frame["index"]}" stroke="{color}">'
                         f'<title>Image {frame["index"]}; acquisition / block center {frame["order"]}</title>')
            for paths, dash in ((solid, ""), (partial, ' stroke-dasharray="3 3"')):
                data = " ".join(p for p in paths if p)
                if data:
                    stream.write(f'<path d="{data}"{dash}/>')
            stream.write('</g>')
        bar_y, bar_width = height + 12 * scale, width * .6
        stream.write('</g><defs><linearGradient id="acquisitions">')
        for offset, rgb in stops:
            color = "#" + "".join(f"{v:02x}" for v in rgb)
            stream.write(f'<stop offset="{offset}" stop-color="{color}"/>')
        stream.write('</linearGradient></defs>'
                     f'<rect x="0" y="{bar_y}" width="{bar_width}" height="{10 * scale}" '
                     'fill="url(#acquisitions)" fill-opacity="0.45"/>'
                     f'<g font-family="sans-serif" font-size="{12 * scale}" fill="#213044">'
                     f'<text x="0" y="{bar_y + 25 * scale}">{first:g}</text>'
                     f'<text x="{bar_width}" y="{bar_y + 25 * scale}" text-anchor="end">{last:g}</text>'
                     f'<text x="0" y="{bar_y + 42 * scale}">Acquisition / block center · {len(frames)} images · native pixels</text>'
                     f'<text x="0" y="{bar_y + 58 * scale}">Dashed: partial boundary / mask fallback. Overlapping lines accumulate.</text></g></svg>')


def render_comparison(root: Path, record: dict) -> Path:
    """Write a static, offline-capable viewer with lazy per-frame contour assets."""
    from .pipeline import write_csv, write_json

    root = Path(root)
    record["metrics"], record["arms"], record["artifacts"] = comparison_metrics(record), comparison_arms(record), []
    write_json(root / "metrics.json", record["metrics"])
    write_csv(root / "metrics.csv", [{k: v for k, v in r.items() if k != "values"} | r["values"] for r in record["metrics"]])
    write_csv(root / "arms.csv", record["arms"])
    write_json(root / "arms.json", record["arms"])
    (root / "viewer").mkdir(exist_ok=True)
    for filename in ("comparison.js", "comparison.css"):
        shutil.copyfile(Path(__file__).parent / "assets" / filename, root / "viewer" / filename)
    view = {"unit": record["unit"], "arms": record["arms"], "sites": [],
            "contour_method": record.get("contour_method", "current"),
            "otsu_settings": record.get("otsu_settings")}
    frame_keys = ("index", "order", "timestamp_s", "first_acquisition", "last_acquisition", "path", "mean_dn",
                  "brightness_delta_dn", "difference_path", "output_minus_raw_dy_px", "output_minus_raw_dx_px",
                  "dy_px", "dx_px", "output_dy_px", "output_dx_px", "contour_counts", "contour_status",
                  "correspondence_status", "clipped", "otsu_threshold_dn", "gaussian_backend")
    for site in record["sites"]:
        contours = site.get("contours")
        if contours is None:
            contours = json.loads((root / site["contours_path"]).read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for contour in contours:
            grouped[contour["series"], contour["frame"]].append(contour)
        traces = {}
        for row in site["observations"]:
            series = traces.setdefault(row["series"], {})
            hole = series.setdefault(str(row["hole"]), {"coarse": [], "refined": []})
            hole[row["method"]].append([row["frame"], row["cd"] if row["status"] == "valid" else None])
        output = {"name": site["name"], "shape": site["image_shape"], "series": {}, "traces": traces,
                  "contour_status": site["contour_status"], "warnings": site["warnings"],
                  "repeatability": site["repeatability"], "hole_count": len(site["template_centroids"])}
        maps = {name: np.load(root / s["temporal_std"], allow_pickle=False)
                for name, s in site["series"].items() if s.get("temporal_std")}
        limit = max((float(m.max()) for m in maps.values()), default=1) or 1
        for name, series in site["series"].items():
            frames = []
            for frame in series["frames"]:
                key = f"{site['name']}/{name}/{frame['index']}"
                relative = f"{site['name']}/viewer/{name}/{frame['index']:03d}.js"
                _script(root / relative, f"window.SEM_CONTOURS[{json.dumps(key)}]", grouped[name, frame["index"]])
                frames.append({**{k: frame.get(k) for k in frame_keys}, "overlay": relative, "overlay_key": key})
            item = {"frames": frames, "step": series["step"],
                    "difference_limit_dn": series.get("difference_limit_dn"),
                    "temporal_rms_dn": series.get("native", {}).get("temporal_rms_dn")}
            item["contour_bands"] = {}
            for method in (["coarse"] if view["contour_method"] == "otsu" else ["coarse", "refined"]):
                relative = f"{site['name']}/viewer/{name}/{method}_band.svg"
                _contour_band(root / relative, site["image_shape"], name, series["frames"], grouped, method)
                item["contour_bands"][method] = relative
            if name in maps:
                relative = f"{site['name']}/viewer/{name}/temporal_std.png"
                values = np.rint(np.clip(maps[name] / limit, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(values).save(root / relative)
                item.update(temporal_image=relative, temporal_limit_dn=limit)
            output["series"][name] = item
            first = series["frames"][0]
            relative = f"{site['name']}/viewer/{name}/overview.png"
            with Image.open(root / first["path"]) as original:
                _overlay(original, grouped[name, first["index"]]).save(root / relative)
            record["artifacts"].append({"path": relative, "site": site["name"], "series": name, "label": "Full-image contours"})
        view["sites"].append(output)
    _script(root / "viewer/data.js", "window.SEM_REPORT", view)
    html = (Path(__file__).parent / "assets/comparison.html").read_text(encoding="utf-8")
    details = _warning(record, [r for r in record["prediction_ranges"] if r.get("clipped")])
    details += _table(record["arms"], ["arm", "registration", "brightness", "step", "ema", "settings_source"])
    details += "".join(f"<p>{escape(w)}</p>" for w in record.get("warnings", []))
    destination = root / "index.html"
    destination.write_text(html.replace("<!-- AUDIT_DETAILS -->", details), encoding="utf-8")
    return destination


def write_tensorboard(root: Path, record: dict, *, writer_factory=None) -> None:
    """Separate acquisition steps from optimizer steps; reuse saved pixels/contours."""
    if writer_factory is None:
        from torch.utils.tensorboard import SummaryWriter
        writer_factory = SummaryWriter
    writer = writer_factory(log_dir=str(root / "tensorboard_comparison"))
    try:
        writer.add_text("comparison/protocol", "All analysis measures saved uint8 images. Acquisition image steps are acquisition numbers; "
                        "average8 uses each block's first acquisition. Average128 is a static reference, not ground truth. "
                        "Start TensorBoard with --samples_per_plugin images=128 to retain every acquisition per tag. "
                        "Open index.html for linked sliders, wipe, measurement areas and complete contour status.", 0)
        for arm in record["arms"]:
            writer.add_text(f"comparison/{arm['arm']}/training_treatment", _table([arm], list(arm)), arm["step"])
        for row in record["metrics"]:
            for key, value in row["values"].items():
                if value is not None and np.isfinite(value):
                    writer.add_scalar(f"summary/{row['site']}/{row['series']}/{key}", value, row["step"])
        for site in record["sites"]:
            grouped = defaultdict(list)
            for contour in site["contours"]:
                grouped[contour["series"], contour["frame"]].append(contour)
            for name, series in site["series"].items():
                for frame in series["frames"]:
                    prefix = f"acquisitions/{site['name']}/{name}"
                    step = frame.get("first_acquisition", frame["index"]) if name != "average128" else 0
                    with Image.open(root / frame["path"]) as original:
                        writer.add_image(f"{prefix}/pixels", np.asarray(original.convert("RGB")), step, dataformats="HWC")
                        annotated = _overlay(original, grouped[name, frame["index"]])
                        writer.add_image(f"{prefix}/contours", np.asarray(annotated), step, dataformats="HWC")
                    for key in ("mean_dn", "brightness_delta_dn", "output_minus_raw_dy_px", "output_minus_raw_dx_px"):
                        if frame.get(key) is not None:
                            writer.add_scalar(f"{prefix}/{key}", frame[key], step)
                if any(f.get("clipped") for f in series["frames"]):
                    writer.add_text(f"{site['name']}/{name}/range_warning", record["range_warning"], series["step"])
    finally:
        writer.close()
