"""Saved-image multi-acquisition baselines for the real SEM comparison."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from PIL import Image

from edge_denoise.device_sampling import sample_native_crops
from edge_denoise.real_data import registration_contrast
from edge_denoise.uint8_output import average_uint8
from sem_noise.pair_matching import GeometryEstimationError, check_geometry_reference, estimate_geometry
from sem_noise.registration import clip_mask


BLOCK_MANIFEST = "block_average.json"


def load_block_manifest(folder: Path, files: list[Path]) -> dict | None:
    """Verify optional block provenance before using a derived image folder."""
    from burst_diffusion.real_data import file_digest

    path = folder / BLOCK_MANIFEST
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    frames = record.get("frames", [])
    count = record.get("frames_per_output")
    if (record.get("format") != 1 or count not in (2, 4)
            or not frames or record.get("source_frame_count") != len(frames) * count
            or type(record.get("registered")) is not bool
            or record.get("registration") != ("affine" if record["registered"] else "none")
            or [f["name"] for f in frames] != [p.name for p in files]):
        raise ValueError(f"invalid block manifest or changed image list: {path}")
    for index, (frame, file) in enumerate(zip(frames, files)):
        if file_digest(file) != frame["file_sha256"]:
            raise ValueError(f"block image changed: {file}")
        hashes = frame.get("source_sha256", [])
        if (len(hashes) != count or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
                or frame.get("first_acquisition") != index * count + 1
                or frame.get("last_acquisition") != (index + 1) * count
                or bool(frame.get("support")) != record["registered"]):
            raise ValueError(f"invalid block source hashes: {file}")
        if frame.get("support"):
            support = (folder / frame["support"]).resolve()
            if support.parent != folder.resolve() or file_digest(support) != frame["support_sha256"]:
                raise ValueError(f"block support changed or escapes site: {support}")
    return record


def block_geometry(frames: np.ndarray) -> tuple[np.ndarray, list[dict], int | None]:
    """Training's translation-seeded affine estimator, with first usable anchor."""
    matrices = np.repeat(np.eye(2, 3)[None], len(frames), axis=0)
    diagnostics, reference = [], None
    seed = np.eye(2, 3)
    for index, frame in enumerate(frames):
        row = {"frame": index + 1, "status": "registered", "reason": ""}
        diagnostics.append(row)
        contrast = registration_contrast(torch.from_numpy(frame.astype(np.float32) / 255)[None, None])
        row["contrast"] = contrast
        if contrast < .005:
            row.update(status="skipped_low_contrast", reason="insufficient structured contrast; retained in native coordinates")
            continue
        invalid = clip_mask(frame, (0, 255))
        try:
            check_geometry_reference(frame, sigma=1, invalid=invalid)
            if reference is None:
                reference = index
                row["status"] = "reference"
                continue
            translation, _ = estimate_geometry(frames[reference], frame, motion="translation", initial=seed,
                sigma=1, input_invalid=clip_mask(frames[reference], (0, 255)), target_invalid=invalid)
            matrix, score = estimate_geometry(frames[reference], frame, motion="affine", initial=translation,
                sigma=1, input_invalid=clip_mask(frames[reference], (0, 255)), target_invalid=invalid)
        except GeometryEstimationError as error:
            row.update(status="skipped_failed_registration", reason=str(error))
            continue
        matrices[index], seed = matrix, translation
        row["score"] = score
    for row, matrix in zip(diagnostics, matrices):
        row["matrix"] = matrix.tolist()
    return matrices, diagnostics, reference


@torch.no_grad()
def registered_average(frames: np.ndarray, *, device: str = "cpu",
                       input_supports: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Affine-align saved uint8 frames and round their float64 mean once.

    Unmeasurable acquisitions contribute natively. Outside the common cubic
    support, retain the ordinary native mean and exclude that support from ECD.
    Tiles bound memory for large or rectangular SEM images.
    """
    if frames.dtype != np.uint8 or frames.ndim != 3 or len(frames) < 2:
        raise ValueError("registered averaging requires at least two saved uint8 images")
    supports = None
    if input_supports is not None:
        import cv2

        if input_supports.dtype != bool or input_supports.shape != frames.shape:
            raise ValueError("input supports must be boolean masks matching the frames")
        # A nearest sample in this eroded mask requires the whole cubic
        # footprint to lie inside the previous production stage's support.
        supports = np.array([cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8),
                                      borderType=cv2.BORDER_CONSTANT, borderValue=0) for m in input_supports])
    started = time.perf_counter()
    matrices, diagnostics, reference = block_geometry(frames)
    geometry_s = time.perf_counter() - started
    started = time.perf_counter()
    count, height, width = frames.shape
    output = average_uint8(frames)
    common = np.zeros((height, width), dtype=bool)
    size = min(512, height, width)
    for y in range(0, height, size):
        for x in range(0, width, size):
            local = matrices.copy()
            local[:, :, 2] += local[:, :, :2] @ np.array([x, y])
            total = torch.zeros((size, size), device=device, dtype=torch.float64)
            valid = torch.ones((size, size), device=device, dtype=torch.bool)
            for _, crops, masks in sample_native_crops(frames, local, size, 0, 255, device,
                                                      normalize=False, require_overlap=False):
                total += crops[:, 0].sum(0, dtype=torch.float64)
                valid &= masks[:, 0].all(0)
            if supports is not None:
                for _, crops, masks in sample_native_crops(supports, local, size, 0, 1, device,
                        normalize=False, require_overlap=False, interpolation="nearest"):
                    valid &= (crops[:, 0] == 1).all(0) & masks[:, 0].all(0)
            h, w = min(size, height - y), min(size, width - x)
            mask = valid[:h, :w].cpu().numpy()
            values = (total[:h, :w] / count).cpu().numpy()
            rounded = np.rint(values).clip(0, 255).astype(np.uint8)
            output[y:y + h, x:x + w][mask] = rounded[mask]
            common[y:y + h, x:x + w] = mask
    return output, common, {"reference_frame": None if reference is None else reference + 1,
        "frames": diagnostics, "device": str(device), "geometry_seconds": geometry_s,
        "warp_average_seconds": time.perf_counter() - started, "common_valid_fraction": float(common.mean()),
        "outside_support": "native unregistered mean; excluded from contour measurements",
        "brightness": "native; no gain or offset", "source": "decoded saved uint8 images"}


def add_average_series(root: Path, site: dict, counts: list[int], *, device: str,
                       model: str | None = None) -> list[str]:
    """Add missing saved baselines; retain old average8 bytes and all model frames."""
    from tools.real_sem_compare import read_uint8, save_rgb

    created = []
    for source_name in ("raw", model) if model is not None else ("raw",):
        source = site["series"][source_name]
        for count in counts:
            # Old small fixture/report series can contain fewer than one block.
            # Documented remainder policy: ignore the trailing incomplete group.
            blocks, remainder = divmod(len(source["frames"]), count)
            if not blocks:
                print(f"{site['name']}/{source_name}: no complete average{count} block; {remainder} frames retained only in single-frame series", flush=True)
                continue
            if remainder:
                print(f"{site['name']}/{source_name}: average{count} omits {remainder} trailing acquisitions", flush=True)
            for registered in ((False, True) if source_name == "raw" else (True,)):
                name = (f"average{count}" + ("_registered" if registered else "") if source_name == "raw"
                        else f"{source_name}_average{count}")
                if name in site["series"]:
                    continue
                print(f"{site['name']}: creating {name} from saved {source_name} PNGs", flush=True)
                series = {"step": source["step"], "frames": [], "frames_per_output": count,
                          "source_series": source_name, "registered": registered, "remainder_frames": remainder,
                          "family": ("raw_registered" if registered else "raw") if source_name == "raw" else source_name}
                site["series"][name] = series
                for block in range(blocks):
                    inputs = source["frames"][block * count:(block + 1) * count]
                    pixels = np.stack([read_uint8(root / f["path"]) for f in inputs])
                    source_supports = None
                    if any(f.get("common_support_path") for f in inputs):
                        masks = []
                        for frame, pixel in zip(inputs, pixels):
                            if frame.get("common_support_path"):
                                with Image.open(root / frame["common_support_path"]) as saved:
                                    mask = np.asarray(saved) > 0
                                if mask.shape != pixel.shape:
                                    raise ValueError("source support dimensions differ from the saved image")
                            else:
                                mask = np.ones(pixel.shape, dtype=bool)
                            masks.append(mask)
                        source_supports = np.stack(masks)
                    first, last = inputs[0]["index"], inputs[-1]["index"]
                    relative = Path(site["name"]) / name / f"block_{block + 1:03d}_{first:03d}-{last:03d}.png"
                    row = {"index": block + 1, "order": float(np.mean([f["order"] for f in inputs])),
                           "first_acquisition": first, "last_acquisition": last, "path": relative.as_posix(),
                           "timestamp_s": (float(np.mean([f["timestamp_s"] for f in inputs]))
                                           if all(f["timestamp_s"] is not None for f in inputs) else None), "clipped": False}
                    if registered:
                        values, valid, record = registered_average(pixels, device=device, input_supports=source_supports)
                        row["averaging_registration"] = record
                    else:
                        values = average_uint8(pixels)
                        valid = None if source_supports is None else source_supports.all(0)
                    if valid is not None:
                        support_path = relative.with_name(relative.stem + "_support.png")
                        (root / support_path).parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(valid.astype(np.uint8) * 255).save(root / support_path)
                        row["common_support_path"] = support_path.as_posix()
                    save_rgb(root / relative, values)
                    series["frames"].append(row)
                created.append(name)
    return created


def frames_vs_precision(site: dict, models: dict) -> list[dict]:
    """Use identical contributing holes across K within each family/measurement."""
    from sem_segment.repeatability import bootstrap_median

    families: dict[str, list[tuple[str, int]]] = {"raw": [("raw", 1)], "raw_registered": [("raw", 1)]}
    families.update({name: [(name, 1)] for name in models if name in site["series"]})
    for name, series in site["series"].items():
        if series.get("frames_per_output"):
            families.setdefault(series["family"], []).append((name, series["frames_per_output"]))
        elif name == "average8":  # Older reports have no family metadata.
            families["raw"].append((name, 8))
    rows = []
    for method in ("coarse", "refined"):
        for family, members in families.items():
            selected = {name: {r["hole"]: r for r in site["per_hole"] if r["series"] == name
                               and r["method"] == method and r["cd_std"] is not None} for name, _ in members}
            measurable = [set(holes) for holes in selected.values() if holes]
            common = set.intersection(*measurable) if measurable else set()
            for name, count in sorted(members, key=lambda pair: pair[1]):
                holes = [selected[name][h] for h in sorted(common) if h in selected[name]]
                values = [r["cd_std"] for r in holes]
                median = float(np.median(values)) if values else None
                interval = bootstrap_median(values)
                rows.append({"family": family, "series": name, "frames_per_output": count, "method": method,
                    "unit": holes[0]["unit"] if holes else None, "common_holes": [h["hole"] for h in holes],
                    "common_hole_count": len(holes), "comparison_series": [n for n in selected if selected[n]],
                    "group_count": len(site["series"][name]["frames"]),
                    "observations_per_hole": {str(r["hole"]): r["valid_count"] for r in holes},
                    "median_cd_3sigma": None if median is None else median * 3,
                    "ci95_low": None if interval is None else interval[0] * 3,
                    "ci95_high": None if interval is None else interval[1] * 3})
    return rows
