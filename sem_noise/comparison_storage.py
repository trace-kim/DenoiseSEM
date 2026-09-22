"""Incremental contour checkpoints without re-encoding completed measurements.

New running records reference immutable JSON-lines parts. Final reports retain
the existing site-wide JSON array, assembled by copying those encoded rows.
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import islice
import json
from pathlib import Path
import tempfile
from typing import BinaryIO, Iterator

from .progress import Progress


def asset_path(root: Path, relative: str) -> Path:
    """Resolve a report asset without permitting paths outside the report."""
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Report asset must be inside its directory: {relative}")
    return path


@contextmanager
def atomic_binary(path: Path) -> Iterator[BinaryIO]:
    """Publish a complete file; a failed write leaves the previous file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_record(path: Path, value: object) -> None:
    """Atomically save comparison metadata, excluding external contour data."""
    from .pipeline import _json_value

    with atomic_binary(path) as stream:
        stream.write(json.dumps(_json_value(value), allow_nan=False, separators=(",", ":")).encode("utf-8"))
        stream.write(b"\n")


def checkpoint_contours(root: Path, site: dict) -> None:
    """Serialize only the appended suffix of an in-memory contour list.

    The coordinator appends completed measurements and never edits them. A
    remeasurement must clear both storage references before building a new list.
    """
    from .pipeline import _json_value

    if "contours_path" in site:
        return
    contours = site.get("contours", [])
    parts = site.setdefault("contours_parts", [])
    saved = sum(part["count"] for part in parts)
    if saved > len(contours):
        raise ValueError("Contour checkpoint exceeds current measurements; reset storage for remeasurement")
    if saved == len(contours):
        return
    relative = f"{site['name']}/contour_parts/{len(parts):06d}.jsonl"
    count = len(contours) - saved
    with Progress(f"{relative}: saving {count:,} new contours", timings=site.setdefault("timings_s", {}),
                  key="checkpoint_contours"):
        with atomic_binary(asset_path(root, relative)) as stream:
            for contour in islice(contours, saved, None):
                payload = json.dumps(_json_value(contour), allow_nan=False, separators=(",", ":"))
                stream.write(payload.encode("utf-8") + b"\n")
    # Publish the reference only after the complete part exists.
    parts.append({"path": relative, "count": count})


def finish_contours(root: Path, site: dict) -> None:
    """Build the compatible final array without converting any coordinates again."""
    checkpoint_contours(root, site)
    relative = f"{site['name']}/contours.json"
    with Progress(f"{relative}: assembling encoded contours", timings=site.setdefault("timings_s", {}),
                  key="export_contours_json"):
        with atomic_binary(asset_path(root, relative)) as output:
            output.write(b"[")
            first = True
            for part in site["contours_parts"]:
                count = 0
                with asset_path(root, part["path"]).open("rb") as source:
                    for line in source:
                        if not first:
                            output.write(b",")
                        output.write(line.rstrip(b"\r\n"))
                        first = False
                        count += 1
                if count != part["count"]:
                    raise ValueError(f"Incomplete contour checkpoint: {part['path']}")
            output.write(b"]\n")
    site["contours_path"] = relative


def load_contours(root: Path, site: dict) -> list[dict]:
    """Read old inline/final contours or new incremental checkpoint parts."""
    if "contours" in site:
        return site["contours"]
    if "contours_path" in site:
        return json.loads(asset_path(root, site["contours_path"]).read_text(encoding="utf-8"))
    if "contours_parts" not in site:
        raise ValueError("Missing saved contours: expected inline contours, contours_path or contours_parts")
    contours = []
    for part in site["contours_parts"]:
        count = 0
        with asset_path(root, part["path"]).open("r", encoding="utf-8") as stream:
            for line in stream:
                contours.append(json.loads(line))
                count += 1
        if count != part["count"]:
            raise ValueError(f"Incomplete contour checkpoint: {part['path']}")
    return contours
