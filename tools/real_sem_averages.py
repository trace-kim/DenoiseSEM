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
# Shared with fused-model inference so both use one block estimator.
from edge_denoise.real_fusion import block_geometry
from edge_denoise.uint8_output import average_uint8
from sem_noise.pair_matching import GeometryEstimationError  # noqa: F401 -- re-exported for callers


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


def add_fused_series(root: Path, site: dict, model: str, denoiser, counts: list[int], *, stride: int,
                     tile_batch: int, prediction_ranges: list[dict]) -> list[str]:
    """One burst-fusion output per consecutive raw block, beside ``average{m}``.

    Blocks are exactly those of :func:`add_average_series` (trailing
    incomplete groups omitted). Each output is the model's denoised m-frame
    mean in the block anchor's coordinates, exported once to uint8; pixels
    some member does not support are saved as a support mask and excluded from
    contours. The series joins the model's single-frame series in one family.
    """
    from tools.real_sem_compare import read_uint8, save_rgb
    from edge_denoise.uint8_output import prediction_uint8

    raw = site["series"]["raw"]
    if any(frame.get("common_support_path") for frame in raw["frames"]):
        raise ValueError("fused series need raw acquisitions, not derived block images with support masks")
    black, white = denoiser.config.data.black_level, denoiser.config.data.white_level
    created = []
    for count in counts:
        name = f"{model}_fuse{count}"
        if name in site["series"]:
            continue
        blocks, remainder = divmod(len(raw["frames"]), count)
        if not blocks:
            print(f"{site['name']}/{name}: no complete {count}-frame block", flush=True)
            continue
        if remainder:
            print(f"{site['name']}/{name}: omits {remainder} trailing acquisitions", flush=True)
        series = {"step": denoiser.checkpoint_step, "frames": [], "frames_per_output": count, "source_series": "raw",
                  "registered": True, "remainder_frames": remainder, "family": model, "fusion_model": model}
        site["series"][name] = series
        started = time.perf_counter()
        for block in range(blocks):
            inputs = raw["frames"][block * count:(block + 1) * count]
            pixels = np.stack([read_uint8(root / f["path"]) for f in inputs])
            first, last = inputs[0]["index"], inputs[-1]["index"]
            relative = Path(site["name"]) / name / f"block_{block + 1:03d}_{first:03d}-{last:03d}.png"
            audit = {"site": site["name"], "model": name, "filename": relative.as_posix(), "status": "pending"}
            prediction_ranges.append(audit)
            try:
                prediction, valid, fusion = denoiser.denoise_frames(pixels, stride=stride, tile_batch=tile_batch,
                                                                    clip_output=False)
                quantized, stats = prediction_uint8(prediction, black, white)
            except Exception as error:
                audit.update(status="failed", error=str(error))
                raise
            audit.update(stats, status="complete")
            save_rgb(root / relative, quantized)
            row = {"index": block + 1, "order": float(np.mean([f["order"] for f in inputs])),
                   "first_acquisition": first, "last_acquisition": last, "path": relative.as_posix(),
                   "timestamp_s": (float(np.mean([f["timestamp_s"] for f in inputs]))
                                   if all(f["timestamp_s"] is not None for f in inputs) else None),
                   **stats, "fusion_input": fusion}
            if not valid.all():
                support_path = relative.with_name(relative.stem + "_support.png")
                Image.fromarray(valid.astype(np.uint8) * 255).save(root / support_path)
                row["common_support_path"] = support_path.as_posix()
            series["frames"].append(row)
        series.setdefault("timings_s", {})["fusion_inference_and_pngs"] = time.perf_counter() - started
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
