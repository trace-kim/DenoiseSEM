"""Persisting a :class:`~sem_segment.pipeline.SegmentationResult`.

Kept separate from the measurement so the pipeline stays a pure function, and so
a caller that only wants the numbers - a notebook, a wrapper that aggregates
across a wafer - never has to write a file to get them.

Each image gets one self-contained directory.  There is deliberately no
cross-image rollup: two images in a folder may be entirely unrelated, and a
summary averaging their feature counts would mean nothing.  A wrapper that knows
the images *are* related is the right place for that.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .pipeline import SegmentationResult

CONTOURS_NAME = "contours.json"
METROLOGY_NAME = "metrology.csv"
MASKS_NAME = "masks.npz"
SUMMARY_NAME = "summary.json"
PROVENANCE_NAME = "provenance.json"
REPORT_NAME = "index.html"


def _jsonable(value):
    """Convert numpy scalars and arrays into plain JSON types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_metrology_csv(
    path: Path, result: SegmentationResult, *, pixel_size_nm: float | None = None
) -> Path:
    """One row per region, with both contour methods side by side."""
    rows = result.rows(pixel_size_nm=pixel_size_nm)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("region_id\n", encoding="utf-8")
        return path
    # Union of keys: reject-reason columns appear only on regions that had them.
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})
    return path


def _csv_value(value):
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def write_contours_json(path: Path, result: SegmentationResult) -> Path:
    """Both contours per region, on a shared parametrisation.

    ``displacement`` is indexed against ``coarse``, so a consumer can line the
    two boundaries up vertex by vertex rather than having to re-match them.
    """
    payload = {
        "schema_version": 1,
        "coordinate_order": "yx",
        "image_shape": list(result.shape),
        "regions": [],
    }
    for index, contour in enumerate(result.coarse):
        refined = result.refined[index] if index < len(result.refined) else None
        entry = {
            "region_id": index + 1,
            "coarse": contour.points.tolist(),
            "holes": [h.points.tolist() for h in result.holes[index]],
            "normals": contour.normals.tolist() if contour.normals is not None else None,
        }
        if refined is not None:
            entry["refined"] = _jsonable(refined.polygon)
            entry["displacement"] = _jsonable(refined.displacement)
            entry["valid"] = refined.valid.astype(int).tolist()
            entry["valid_fraction"] = refined.valid_fraction
            entry["reject_counts"] = refined.reason_counts()
        payload["regions"].append(entry)
    return write_json(path, payload)


def write_masks_npz(path: Path, result: SegmentationResult) -> Path:
    """A uint16 label map plus per-region scores, compressed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=result.label_map(),
        scores=np.array([m.score for m in result.instances], dtype=np.float32),
        region_ids=np.arange(1, len(result.instances) + 1, dtype=np.uint16),
    )
    return path


def write_result(
    out_dir: Path,
    result: SegmentationResult,
    *,
    pixel_size_nm: float | None = None,
    source: Path | None = None,
    extra_provenance: dict | None = None,
    exist_ok: bool = False,
) -> Path:
    """Write every artifact for one image into its own directory.

    Refuses to write into an existing directory by default: a rerun that
    silently overwrote half of a previous result and left the other half in
    place would be indistinguishable from a complete one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=exist_ok)

    write_metrology_csv(out_dir / METROLOGY_NAME, result, pixel_size_nm=pixel_size_nm)
    write_contours_json(out_dir / CONTOURS_NAME, result)
    write_masks_npz(out_dir / MASKS_NAME, result)
    write_json(
        out_dir / SUMMARY_NAME,
        {
            "schema_version": 1,
            "source": str(source) if source else None,
            "image_shape": list(result.shape),
            "region_count": result.region_count,
            "image_stats": result.image_stats,
            "warnings": result.diagnostics.warnings,
            "rejections": result.diagnostics.rejections,
            "refine_rejections": result.diagnostics.refine_rejections,
            "valid_fraction": result.diagnostics.valid_fraction,
            "timings_s": result.diagnostics.timings_s,
            "scaling": result.diagnostics.scaling,
        },
    )
    provenance = dict(result.provenance)
    provenance.update(extra_provenance or {})
    write_json(out_dir / PROVENANCE_NAME, provenance)
    return out_dir
