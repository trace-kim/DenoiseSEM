"""Connect the model-free template precision diagnostic to saved comparisons."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from sem_noise.template_precision import fit_template_batch

SERIES_NAME = "single_frame_template_limit"


def measure_template_limit(root: Path, site: dict, *, device: str) -> None:
    """Read each saved raw frame once; batch annulus fits across holes and frames."""
    from scipy import ndimage
    from skimage.draw import polygon
    from sem_noise.io import file_hash
    from tools.real_sem_compare import read_uint8

    started = time.perf_counter()
    template = read_uint8(root / site["full_average"])
    reference_rows = {r["hole"]: r for r in site["observations"]
                      if r["series"] == "average128" and r["method"] == "refined"}
    outlines = {r["hole"]: r for r in site["contours"] if r["series"] == "average128" and r.get("hole") is not None}
    raw = site["series"]["raw"]["frames"]
    holes = []
    for hole in range(1, len(site["template_centroids"]) + 1):
        row, outline = reference_rows.get(hole), outlines.get(hole)
        item = {"hole": hole, "reference": row, "reason": "template_refinement_failed"}
        if row and row["status"] == "valid" and outline and outline["refined"]:
            points = np.asarray(outline["refined"])
            # The annulus is fixed in native observation coordinates; the window
            # includes space for the documented small drift and cubic support.
            y0, x0 = np.maximum(np.floor(points.min(0)).astype(int) - 14, 0)
            y1, x1 = np.minimum(np.ceil(points.max(0)).astype(int) + 15, template.shape)
            shape = (y1 - y0, x1 - x0)
            mask = np.zeros(shape, dtype=bool)
            rr, cc = polygon(points[:, 0] - y0, points[:, 1] - x0, shape=shape)
            mask[rr, cc] = True
            annulus = (ndimage.distance_transform_edt(mask) <= 5) & (ndimage.distance_transform_edt(~mask) <= 5)
            item.update(reason=None, crop=(y0, y1, x0, x1), template=template[y0:y1, x0:x1],
                        annulus=annulus, center=points.mean(0) - [y0, x0])
        holes.append(item)
    fits, observations, pending = [], [], []

    def append(frame, item, fit):
        reference = item["reference"] or {}
        valid = fit["status"] == "valid"
        scale = fit.get("scale")
        cd = reference.get("cd")
        fits.append({"hole": item["hole"], "frame": frame["index"], "filename": frame["path"],
                     "template_ecd": cd, "ecd_se": fit.get("scale_se") * cd if valid else None, **fit})
        observations.append({"series": SERIES_NAME, "frame": frame["index"], "order": frame["order"],
            "timestamp_s": frame["timestamp_s"], "filename": frame["path"], "hole": item["hole"],
            "method": "refined", "status": fit["status"], "unit": reference.get("unit", "px"),
            "cd": scale * cd if valid else None,
            "major_axis": scale * reference["major_axis"] if valid and reference.get("major_axis") is not None else None,
            "minor_axis": scale * reference["minor_axis"] if valid and reference.get("minor_axis") is not None else None,
            "mean_dn": frame.get("mean_dn"), "otsu_threshold_dn": frame.get("otsu_threshold_dn"),
            "clipped": frame.get("clipped", False), "diagnostic": True})

    def flush():
        groups = defaultdict(list)
        for job in pending:
            groups[job[1]["template"].shape].append(job)
        for jobs in groups.values():
            results = fit_template_batch(np.stack([item["template"] for _, item, _, _ in jobs]),
                np.stack([pixels for _, _, pixels, _ in jobs]), np.stack([item["annulus"] for _, item, _, _ in jobs]),
                centers_yx=np.array([item["center"] for _, item, _, _ in jobs]),
                initial_shifts_yx=np.array([shift for _, _, _, shift in jobs]), device=device)
            for (frame, item, _, _), result in zip(jobs, results):
                append(frame, item, result)
        pending.clear()

    print(f"{site['name']}: single-frame template limit ({len(holes)} holes x {len(raw)} frames) on {device}; diagnostic only", flush=True)
    for number, frame in enumerate(raw, 1):
        pixels = read_uint8(root / frame["path"])
        shift = np.array([frame.get("dy_px"), frame.get("dx_px")], dtype=float)
        if not np.isfinite(shift).all():
            shift = np.zeros(2)  # The independent fit can still measure geometry.
        for item in holes:
            if item["reason"]:
                append(frame, item, {"status": item["reason"], "scale": None, "scale_se": None})
                continue
            y0, y1, x0, x1 = item["crop"]
            pending.append((frame, item, pixels[y0:y1, x0:x1].copy(), shift))
            if len(pending) >= 64:
                flush()
        if number % 16 == 0 or number == len(raw):
            print(f"{site['name']}: template diagnostic {number}/{len(raw)}; {time.perf_counter()-started:.1f}s", flush=True)
    flush()
    site["observations"] = [r for r in site["observations"] if r["series"] != SERIES_NAME] + observations
    site["template_precision"] = {"series": SERIES_NAME, "label": "single-frame template limit",
        "diagnostic": "not deployable; uses this site's own saved full average",
        "template": site["full_average"], "template_sha256": file_hash(root / site["full_average"]),
        "source": "decoded saved uint8 raw PNGs and full average", "device": str(device),
        "annulus_half_width_px": 5, "scale_convention": "I(x)=g*T(c+(x-c-d)/s)+o; ECD=s*ECD_template",
        "standard_errors": "Gauss-Newton residual sandwich HC1; conditional on template; independent pixel noise",
        "limitations": "not a universal lower bound; template blur, self-inclusion, correlated noise and shape changes can bias the diagnostic",
        "fits": fits, "seconds": time.perf_counter() - started}
    site.setdefault("timings_s", {})["template_precision"] = time.perf_counter() - started
