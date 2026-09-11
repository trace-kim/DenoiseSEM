"""Reproducible, site-at-a-time analysis of repeated SEM acquisitions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Callable

import numpy as np

from .config import AnalysisConfig
from .io import Frame, discover_sites, file_hash, pixel_hash, read_frame
from .metrics import analyze_mode
from .registration import common_crop, local_diagnostics, register_stack


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(_json_value(rows))


def _provenance(config: AnalysisConfig, order_source: str) -> dict:
    versions = {}
    for package in ("numpy", "scipy", "scikit-image", "matplotlib", "Pillow", "tifffile"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unknown"
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "python": platform.python_version(), "platform": platform.system(),
            "dependencies": versions, "order_source": order_source,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob("*.py"))}}


def _timing(frames: list[Frame], config: AnalysisConfig) -> tuple[float | None, bool, list[str]]:
    warnings = []
    if frames[0].timestamp_s is None:
        if config.frame_interval_s is None:
            warnings.append("No timestamps or frame interval: temporal axes are acquisition indices, not elapsed time.")
        return config.frame_interval_s, False, warnings
    indices = np.array([f.index for f in frames])
    stamps = np.array([f.timestamp_s for f in frames])
    steps = np.diff(stamps) / np.diff(indices)
    interval = float(np.median(steps))
    regular = bool(np.allclose(steps, interval, rtol=0.01, atol=1e-9))
    if not regular:
        warnings.append("Irregular timestamps: seconds-based Allan curves and temporal FFT are disabled; frame-lag curves remain descriptive.")
    if config.frame_interval_s is not None and not np.allclose(steps, config.frame_interval_s, rtol=0.01):
        warnings.append("Manifest timestamps override the conflicting configured frame interval.")
    return interval if regular else None, not regular, warnings


def _load_stack(frames: list[Frame], config: AnalysisConfig, scratch: Path) -> tuple[np.memmap, list[dict], tuple]:
    first = read_frame(frames[0])
    shape, dtype = first.shape, first.dtype
    y0, y1, x0, x1 = config.roi or (0, shape[0], 0, shape[1])
    if y1 > shape[0] or x1 > shape[1] or min(y1 - y0, x1 - x0) < 16:
        raise ValueError("ROI must fit inside the image and be at least 16x16")
    cache_dtype = np.float64 if dtype.kind == "f" and dtype.itemsize > 4 else np.float32
    cache_shape = (len(frames), y1 - y0, x1 - x0)
    required_bytes = int(np.prod(cache_shape)) * np.dtype(cache_dtype).itemsize
    if required_bytes > shutil.disk_usage(scratch).free * 0.9:
        raise ValueError(f"insufficient scratch space for {required_bytes / 2**30:.2f} GiB site cache")
    stack = np.memmap(scratch / "raw.dat", dtype=cache_dtype, mode="w+", shape=cache_shape)
    levels = (config.black_level, config.white_level)
    if dtype.kind in "ui":
        limits = np.iinfo(dtype)
        levels = (limits.min if levels[0] is None else levels[0], limits.max if levels[1] is None else levels[1])
    if all(level is not None for level in levels) and levels[0] >= levels[1]:
        stack._mmap.close()
        raise ValueError("effective black level must be below the white level")
    rows, hashes, seen_pixels = [], {}, {}
    try:
        for i, frame in enumerate(frames):
            before = frame.path.stat()
            signature = (before.st_size, before.st_mtime_ns)
            if frame.path not in hashes:
                hashes[frame.path] = (signature, file_hash(frame.path))
            elif hashes[frame.path][0] != signature:
                raise ValueError(f"input changed during analysis: {frame.relative_path}")
            array = read_frame(frame)
            after = frame.path.stat()
            if signature != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"input changed during analysis: {frame.relative_path}")
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"{frame.relative_path}: all frames in a site must have identical shape and dtype")
            digest = pixel_hash(array)
            duplicate = seen_pixels.get(digest)
            if frame.include and duplicate is None:
                seen_pixels[digest] = frame.index
            region = array[y0:y1, x0:x1]
            stack[i] = region
            rows.append({"site": frame.site, "path": frame.relative_path, "page": frame.page,
                         "frame_index": frame.index, "frame_position": i, "timestamp_s": frame.timestamp_s,
                         "included": frame.include and duplicate is None,
                         "exclusion_reason": "excluded in manifest" if not frame.include else
                                             (f"decoded duplicate of frame {duplicate}" if duplicate is not None else ""),
                         "file_sha256": hashes[frame.path][1], "pixel_sha256": digest,
                         "duplicate_of_frame_index": duplicate, "dtype": str(dtype), "height": shape[0], "width": shape[1],
                         "minimum_dn": float(region.min()), "maximum_dn": float(region.max()),
                         "raw_mean_dn": float(region.mean(dtype=float)),
                         "low_clip_fraction": float(np.mean(region <= levels[0])) if levels[0] is not None else None,
                         "high_clip_fraction": float(np.mean(region >= levels[1])) if levels[1] is not None else None,
                         "metadata": frame.metadata})
        stack.flush()
        return stack, rows, levels
    except BaseException:
        stack._mmap.close()
        raise


def _analyze_site(frames: list[Frame], out: Path, config: AnalysisConfig,
                  progress: Callable[[str], None]) -> dict:
    from .report import site_report

    if len(frames) < config.min_frames:
        raise ValueError(f"requires at least {config.min_frames} frames, found {len(frames)}")
    warnings = ["The repeat mean is a noisy specimen estimate, not ground truth. Acquisition order must be verified."]
    interval, irregular, timing_warnings = _timing(frames, config)
    warnings.extend(timing_warnings)
    if len(frames) != config.expected_frames:
        warnings.append(f"Expected {config.expected_frames} frames; found {len(frames)}.")
    if any(f.path.suffix.lower() in {".jpg", ".jpeg"} for f in frames):
        warnings.append("Lossy JPEG input: compression contributes to the observed noise statistics.")
    if any(len({f.metadata.get(key, "") for f in frames}) > 1 for key in {k for f in frames for k in f.metadata}):
        warnings.append("Metadata varies within this site. Verify acquisition settings and split different settings into separate site groups.")
    with tempfile.TemporaryDirectory(prefix=".cache-", dir=out) as temporary:
        stack, audit, levels = _load_stack(frames, config, Path(temporary))
        try:
            write_json(out / "inputs.json", audit)
            included = np.array([r["included"] for r in audit])
            if included.sum() < config.min_frames:
                raise ValueError(f"only {included.sum()} distinct included frames remain after duplicate detection")
            if (~included).any():
                warnings.append(f"{int((~included).sum())} frames excluded by manifest or exact decoded duplication; see inputs.json.")
            progress(f"{frames[0].site}: estimating registration for {int(included.sum())} frames")
            shifts, accepted, registration = register_stack(stack, included, config)
            for row in registration:
                i = row["frame_position"]
                row.update(frame_index=frames[i].index, path=frames[i].relative_path, page=frames[i].page,
                           correction_dy_px=float(shifts[i, 0]), correction_dx_px=float(shifts[i, 1]),
                           drift_dy_px=float(-shifts[i, 0]), drift_dx_px=float(-shifts[i, 1]))
                if not included[i]:
                    row["reason"] = audit[i]["exclusion_reason"]
                if config.pixel_size_nm:
                    row.update(drift_dy_nm=float(-shifts[i, 0] * config.pixel_size_nm),
                               drift_dx_nm=float(-shifts[i, 1] * config.pixel_size_nm))
            write_csv(out / "registration.csv", registration)
            if accepted.sum() < config.min_frames:
                raise ValueError(f"only {accepted.sum()} frames pass registration; see registration.csv")
            if (included & ~accepted).any():
                warnings.append(f"{int((included & ~accepted).sum())} additional frames failed registration; temporal gaps are retained.")
            if any(r["peak_ratio"] is not None and r["accepted"] and r["peak_ratio"] < 1.05 for r in registration):
                warnings.append("Multiple similar correlation peaks: periodic-pattern registration may be ambiguous. Inspect drift and local residuals.")
            if config.registration == "none":
                warnings.append("Registration disabled: shift values are placeholders; specimen motion can inflate noise statistics.")
            positions = np.flatnonzero(accepted)
            indices = np.array([frames[i].index for i in positions])
            # The same crop is valid for both nearest-integer and linear shifts.
            combined_shifts = np.concatenate((shifts[accepted], np.rint(shifts[accepted]), np.zeros((1, 2))))
            crop = common_crop(stack.shape[1:], combined_shifts, margin=2 if config.registration != "none" else 0)
            summaries, maps, frame_metrics = {}, {}, {}
            for mode, integer in (("native", True), ("aligned", False)):
                progress(f"{frames[0].site}: measuring {mode} noise, spectra, and temporal stability")
                result, mode_maps, rows = analyze_mode(stack, shifts, accepted, indices, crop, integer, levels, config, interval, irregular)
                summaries[mode] = result
                maps.update({f"{mode}_{key}": value for key, value in mode_maps.items()})
                frame_metrics[mode] = rows
                write_csv(out / f"{mode}_intensity_bins.csv", result["intensity_bins"])
                write_csv(out / f"{mode}_averaging.csv", result["temporal"]["averaging"])
                write_csv(out / f"{mode}_temporal_acf.csv", result["temporal"]["acf"])
                write_csv(out / f"{mode}_spatial_acf.csv", result["spatial"]["acf"])
            progress(f"{frames[0].site}: checking local distortion and writing report")
            local = local_diagnostics(stack, shifts, accepted, maps["aligned_mean"], crop, config)
            for row in local:
                row["frame_index"] = frames[row["frame_position"]].index
            write_csv(out / "local_registration.csv", local)
            valid_local = [r for r in local if r["valid"]]
            local_rms = float(np.sqrt(np.mean([r["residual_dy_px"]**2 + r["residual_dx_px"]**2 for r in valid_local]))) if valid_local else None
            if local_rms is not None and local_rms > 0.5:
                warnings.append("Local residual shifts exceed 0.5 px RMS; translation may not explain charging distortion, rotation, or pattern ambiguity.")
            if not valid_local and config.registration != "none":
                warnings.append("No valid local registration tiles; local distortion was not assessed.")
            if any((r["low_clip_fraction"] or 0) + (r["high_clip_fraction"] or 0) > 0.001 for r in audit):
                warnings.append("Some frames have >0.1% pixels at configured/storage clipping bounds. Clipped pixels are excluded from noise-distribution masks.")
            if config.white_level is None or config.black_level is None:
                warnings.append("Clipping bounds default to integer storage limits (unknown for float inputs). Set actual ADC/export bounds for packed or rescaled data.")
            for mode in ("native", "aligned"):
                acf = summaries[mode]["temporal"]["acf"]
                if acf and acf[0]["lag_frames"] == 1 and acf[0]["pixel_acf"] is not None and acf[0]["pixel_acf"] > 0.1:
                    warnings.append(f"{mode}: positive lag-1 correlation exceeds 0.1; adjacent pairs may underestimate independent-frame variance.")
            unregistered_mean, unregistered_m2 = np.zeros(stack[0][crop].shape), np.zeros(stack[0][crop].shape)
            for j, i in enumerate(positions):
                frame = stack[i][crop]
                delta = frame - unregistered_mean
                unregistered_mean += delta / (j + 1)
                unregistered_m2 += delta * (frame - unregistered_mean)
            maps["unregistered_mean"] = unregistered_mean.astype(np.float32)
            maps["unregistered_std"] = np.sqrt(unregistered_m2 / (len(positions) - 1)).astype(np.float32)
            metrics_by_position = {int(i): dict(audit[i], **registration[i]) for i in range(len(frames))}
            for mode, rows in frame_metrics.items():
                for row in rows:
                    metrics_by_position[row["frame_position"]].update({f"{mode}_{key}": value for key, value in row.items() if key != "frame_position"})
            for row in metrics_by_position.values():
                row.pop("metadata", None)
            write_csv(out / "frames.csv", list(metrics_by_position.values()))
            stamps = np.array([frames[i].timestamp_s if frames[i].timestamp_s is not None else frames[i].index * (interval or 1) for i in positions])
            stamps -= stamps[0]
            means = np.array([r["mean_dn"] for r in frame_metrics["aligned"]])
            brightness_slope = float(np.polyfit(stamps, means, 1)[0])
            fractions = shifts[accepted] - np.floor(shifts[accepted])
            variance_factors = np.prod((1 - fractions)**2 + fractions**2, axis=1)
            summary = {"site": frames[0].site, "status": "complete", "input_frames": len(frames),
                       "accepted_frames": int(accepted.sum()), "dtype": audit[0]["dtype"],
                       "original_shape": [audit[0]["height"], audit[0]["width"]],
                       "roi_y0_y1_x0_x1": list(config.roi) if config.roi else [0, stack.shape[1], 0, stack.shape[2]],
                       "common_crop_within_roi": [crop[0].start, crop[0].stop, crop[1].start, crop[1].stop],
                       "black_level_dn": levels[0], "white_level_dn": levels[1],
                       "interval_s": interval, "duration_s": float(stamps[-1]) if frames[0].timestamp_s is not None or interval else None,
                       "pixel_size_nm": config.pixel_size_nm,
                       "max_drift_px": float(np.max(np.linalg.norm(shifts[accepted], axis=1))) if config.registration != "none" else None,
                       "drift_step_rms_px": float(np.sqrt(np.mean(np.sum(np.diff(shifts[accepted], axis=0)[np.diff(indices) == 1]**2, axis=1)))) if np.any(np.diff(indices) == 1) and config.registration != "none" else None,
                       "local_residual_rms_px": local_rms, "valid_local_tiles": len(valid_local),
                       "brightness_slope_dn_per_unit": brightness_slope,
                       "brightness_slope_time_unit": "second" if frames[0].timestamp_s is not None or interval else "frame",
                       "bilinear_white_noise_variance_factor_mean": float(variance_factors.mean()),
                       "unregistered_temporal_sigma_dn": float(np.sqrt(np.mean(unregistered_m2 / (len(positions) - 1)))),
                       "modes": summaries, "warnings": warnings}
            np.savez_compressed(out / "maps.npz", **maps)
            write_json(out / "summary.json", summary)
            site_report(out, summary, maps, list(metrics_by_position.values()), local)
            return summary
        finally:
            stack._mmap.close()


def analyze_dataset(input_path: str | Path, output_path: str | Path, *,
                    config: AnalysisConfig | None = None, manifest: str | Path | None = None,
                    progress: Callable[[str], None] | None = None) -> dict:
    """Write a fresh report bundle, continuing other sites if a site's data fails.

    Input data are never changed. Existing output directories are rejected.
    Each site uses one temporary disk-backed stack, closed before cleanup.
    """
    from .report import index_report

    config = config or AnalysisConfig()
    progress = progress or (lambda message: None)
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if output == source or (source.is_dir() and output.is_relative_to(source)):
        raise ValueError("output must be outside the input directory")
    sites = discover_sites(source, manifest)
    output.mkdir(parents=True, exist_ok=False)
    provenance = _provenance(config, "manifest frame_index" if manifest else "natural filename order, then stack page")
    if manifest:
        provenance["manifest_sha256"] = file_hash(Path(manifest))
    write_json(output / "provenance.json", provenance)
    results, identities, input_manifest = [], {}, []
    for number, (site, frames) in enumerate(sites.items(), 1):
        directory = f"site_{number:03d}"
        site_out = output / directory
        site_out.mkdir()
        progress(f"[{number}/{len(sites)}] {site}")
        try:
            result = _analyze_site(frames, site_out, config, progress)
        except (ValueError, OSError, IndexError) as error:
            result = {"site": site, "status": "failed", "error": str(error)}
            write_json(site_out / "summary.json", result)
            progress(f"{site}: failed: {error}")
        result["directory"] = directory
        results.append(result)
        if (site_out / "inputs.json").exists():
            rows = json.loads((site_out / "inputs.json").read_text(encoding="utf-8"))
            input_manifest.extend(rows)
            for row in rows:
                identities.setdefault(row["pixel_sha256"], []).append({"site": site, "frame_index": row["frame_index"], "path": row["path"], "page": row["page"]})
    duplicates = [rows for rows in identities.values() if len({r["site"] for r in rows}) > 1]
    overview = {"schema_version": 1, "status": "complete" if all(r["status"] == "complete" for r in results) else "partial_failure",
                "site_count": len(results), "successful_sites": sum(r["status"] == "complete" for r in results),
                "cross_site_duplicate_groups": duplicates,
                "warnings": ["Noise parameters are reported per site. Different patterns/settings are not pooled.",
                             "128 repeats characterize only their acquisition duration; longer-term stability needs repeated sessions.",
                             "Stable detector fixed-pattern noise cannot be separated from specimen structure without dark/flat references."],
                "sites": results}
    write_json(output / "input_manifest.json", input_manifest)
    write_json(output / "summary.json", overview)
    write_csv(output / "summary.csv", [{"site": r["site"], "status": r["status"], "directory": r["directory"],
                                       "accepted_frames": r.get("accepted_frames"), "max_drift_px": r.get("max_drift_px"),
                                       "native_flat_sigma_dn": r.get("modes", {}).get("native", {}).get("flat_temporal_sigma_dn"),
                                       "aligned_flat_sigma_dn": r.get("modes", {}).get("aligned", {}).get("flat_temporal_sigma_dn"),
                                       "local_residual_rms_px": r.get("local_residual_rms_px"), "error": r.get("error", "")}
                                      for r in results])
    index_report(output, overview)
    return overview
