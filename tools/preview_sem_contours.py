"""Preview saved comparison images with stored outlines and an Otsu baseline.

This command reads only the report and its saved image/contour assets. It does
not import training packages, load a checkpoint, or run the current detector.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time
from xml.sax.saxutils import escape, quoteattr

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sem_segment.config import InputConfig
from sem_segment.otsu_baseline import OtsuSettings, otsu_baseline

ASSETS = ROOT / "sem_segment" / "assets"
COLORS = {"coarse": "#43dbe2", "refined": "#ffa25d", "baseline": "#92f069"}


def _source_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Expected a report-relative asset path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset path escapes the source report: {relative}")
    return path


def _polyline(points: list | np.ndarray, color: str, *, closed: bool = False,
              dashed: bool = False) -> str:
    values = np.asarray(points, dtype=float)
    if not values.size:
        return ""
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Stored contours must contain finite (y, x) coordinates")
    if closed:
        values = np.vstack([values, values[:1]])
    # Pixel centres are integer (y, x) in both estimators, and half-integer in
    # an SVG whose raster image starts at (0, 0). Apply only that display shift.
    coords = " ".join(f"{x + .5:.6f},{y + .5:.6f}" for y, x in values)
    dash = ' stroke-dasharray="3 3"' if dashed else ""
    return f'<polyline points="{coords}" stroke="{color}"{dash}/>'


def _current_lines(contours: list[dict]) -> str:
    lines = []
    for contour in contours:
        # Existing stored rings have implicit closure and full-image offsets.
        for ring in [contour.get("coarse", []), *contour.get("holes", [])]:
            lines.append(_polyline(ring, COLORS["coarse"], closed=True))
        refined = contour.get("refined", [])
        if not refined:
            continue
        valid = contour.get("refined_valid", [])
        if not valid:
            lines.append(_polyline(refined, COLORS["refined"], closed=True))
            continue
        if len(valid) != len(refined):
            raise ValueError("Stored refined_valid length differs from refined contour")
        lines.append(_polyline(refined, COLORS["refined"], closed=True, dashed=True))
        for i, point in enumerate(refined):
            j = (i + 1) % len(refined)
            if valid[i] and valid[j]:
                lines.append(_polyline([point, refined[j]], COLORS["refined"]))
    return "\n".join(lines)


def _write_overlay(path: Path, image_url: str, size: tuple[int, int], lines: str,
                   title: str) -> None:
    width, height = size
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><title>{escape(title)}</title>'
        f'<image width="{width}" height="{height}" href={quoteattr(image_url)} '
        'style="image-rendering:pixelated"/>'
        f'<g fill="none" stroke-width="1">{lines}</g></svg>', encoding="utf-8")


def _stored_contours(site: dict, root: Path) -> tuple[dict, bool]:
    contours = site.get("contours")
    if contours is None and site.get("contours_path"):
        path = _source_path(root, site["contours_path"])
        if path.is_file():
            contours = json.loads(path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for contour in contours or []:
        groups[(contour["series"], contour["frame"])].append(contour)
    return groups, contours is not None


def _series(site: dict) -> dict:
    series = dict(site["series"])
    # Older comparisons saved the full average outside the series mapping.
    if "average128" not in series and site.get("full_average"):
        raw = series.get("raw", {}).get("frames", [])
        series["average128"] = {"frames": [{
            "path": site["full_average"], "index": 1,
            "first_acquisition": raw[0].get("order", raw[0]["index"]) if raw else None,
            "last_acquisition": raw[-1].get("order", raw[-1]["index"]) if raw else None,
        }]}
    names = [name for name in ("raw", "average8", "average128") if name in series]
    names.extend(name for name in series if name not in names)
    return {name: series[name] for name in names}


def build_preview(record_path: Path, output_dir: Path,
                  settings: OtsuSettings | None = None, *, device: str = "cpu") -> dict:
    """Write a portable, separate preview; source files are never written."""
    started = time.perf_counter()
    settings = settings or OtsuSettings()
    record_path = Path(record_path).expanduser().resolve()
    source_root = record_path.parent
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError("Output must be separate from and not overlap the source report")
    if output_dir.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output_dir}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    crop = InputConfig(crop=record.get("segmentation_settings", {}).get("input", {}).get("crop")).crop
    sites = record.get("sites", [])
    total = 0
    for site in sites:
        for series in _series(site).values():
            indices = [frame["index"] for frame in series["frames"]]
            if len(indices) != len(set(indices)):
                raise ValueError("Duplicate acquisition index within a saved source")
            for frame in series["frames"]:
                if not _source_path(source_root, frame["path"]).is_file():
                    raise ValueError(f"Saved image unavailable: {frame['path']}")
            total += len(indices)
    if not total:
        raise ValueError("The comparison contains no saved images")

    metadata = {
        "kind": "segmentation_preview", "source_comparison": str(record_path),
        "synthetic": bool(record.get("synthetic", False)),
        "study": record.get("study", "Saved SEM comparison"),
        "settings": settings.model_dump(), "device_requested": device,
        "measurement_crop_y0_y1_x0_x1": crop, "sites": [],
    }
    output_dir.mkdir(parents=True)
    image_dir = output_dir / "images"
    image_dir.mkdir()
    number, last_progress = 0, started
    for site in sites:
        stored, have_storage = _stored_contours(site, source_root)
        target_site = {"name": site["name"], "series": {}}
        metadata["sites"].append(target_site)
        for name, series in _series(site).items():
            frames = []
            target_site["series"][name] = {"frames": frames}
            for frame in series["frames"]:
                number += 1
                frame_started = time.perf_counter()
                source = _source_path(source_root, frame["path"])
                original = image_dir / f"{number:06d}{source.suffix.lower()}"
                # Decode the copy that the preview delivers, with exactly the
                # same file bytes as the report. No image adjustments are made.
                shutil.copyfile(source, original)
                result = otsu_baseline(original, settings, crop=crop, device=device)
                render_started = time.perf_counter()
                with Image.open(original) as image:
                    size = image.size
                    display = original
                    # Browsers do not display TIFF. A PNG copy preserves every
                    # decoded uint8 pixel; the native file remains available.
                    if source.suffix.lower() in {".tif", ".tiff"}:
                        display = image_dir / f"{number:06d}_display.png"
                        image.save(display)
                    mime = "image/png" if display != original else Image.MIME[image.format]
                image_url = f"data:{mime};base64," + base64.b64encode(display.read_bytes()).decode("ascii")
                current = stored.get((name, frame["index"]), [])
                has_points = any(c.get("coarse") or c.get("refined") or c.get("holes") for c in current)
                if has_points:
                    status = "Stored outlines reused."
                elif have_storage and frame.get("contour_counts", {}).get("detected") == 0:
                    status = "Stored result: no contours detected."
                else:
                    status = "Current outlines unavailable in the saved report."
                current_path = image_dir / f"{number:06d}_current.svg"
                baseline_path = image_dir / f"{number:06d}_otsu.svg"
                _write_overlay(current_path, image_url, size, _current_lines(current), "Current outlines: " + status)
                _write_overlay(baseline_path, image_url, size,
                               "\n".join(_polyline(p, COLORS["baseline"]) for p in result.outlines),
                               "Otsu baseline: binary-mask outlines, no edge refinement")
                timing = {**result.timings_s, "render": time.perf_counter() - render_started,
                          "preview_frame_total": time.perf_counter() - frame_started}
                item = {key: frame[key] for key in (
                    "index", "order", "first_acquisition", "last_acquisition", "timestamp_s", "note") if key in frame}
                item.update(source_image=frame["path"], shape=[size[1], size[0]],
                            original=original.relative_to(output_dir).as_posix(),
                            display=display.relative_to(output_dir).as_posix(),
                            current=current_path.relative_to(output_dir).as_posix(),
                            baseline=baseline_path.relative_to(output_dir).as_posix(),
                            current_status=status, threshold_dn=result.threshold_dn,
                            component_count=result.component_count, retained_count=result.retained_count,
                            outline_count=len(result.outlines), gaussian_backend=result.gaussian_backend,
                            timings_s=timing)
                frames.append(item)
                now = time.perf_counter()
                if number == 1 or number % 16 == 0 or number == total or now - last_progress >= 30:
                    print(f"Contour preview {number}/{total}: {site['name']}/{name} "
                          f"index {frame['index']}; {now - started:.1f}s elapsed", flush=True)
                    last_progress = now
    metadata["timings_s"] = {"build_total": time.perf_counter() - started}
    payload = json.dumps(metadata, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    (output_dir / "metadata.json").write_text(payload + "\n", encoding="utf-8")
    (output_dir / "data.js").write_text("window.CONTOUR_PREVIEW = " + payload + ";\n", encoding="utf-8")
    for asset, target in (("contour_preview.html", "index.html"),
                          ("contour_preview.css", "preview.css"), ("contour_preview.js", "preview.js")):
        shutil.copyfile(ASSETS / asset, output_dir / target)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "sem_segment/configs/contour_preview.yml")
    parser.add_argument("--from-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Gaussian smoothing device: cpu, cuda, or cuda:N")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = OtsuSettings.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
        build_preview(args.from_comparison, args.output_dir, settings, device=args.device)
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(2, f"Contour preview failed: {error}\n")
    print(f"Open {args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
