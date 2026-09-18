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
from .registration import common_crop
from .site_registration import (csv_rows, difference_outputs, fit_rows, register_site,
                                registration_summary)

FRAME_TABLE_KEYS = ("dy_px", "dy_px_se", "dx_px", "dx_px_se", "gain", "gain_se", "offset_dn", "offset_dn_se",
                    "residual_rms_dn", "corner_max_px", "corner_max_se", "converged",
                    "affine_dx_px", "affine_dy_px", "translation_status", "affine_status")


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
    package_root = Path(__file__).parent
    sources = [*package_root.glob("*.py"), *package_root.glob("assets/*.js")]
    versions = {}
    for package in ("numpy", "scipy", "scikit-image", "opencv-python-headless", "matplotlib", "Pillow", "tifffile"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unknown"
    return {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "python": platform.python_version(), "platform": platform.system(),
            "dependencies": versions, "order_source": order_source,
            "source_sha256": {p.relative_to(package_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(sources)}}


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
    from .report import intermediate_examples, raw_histogram_examples, site_report

    site = frames[0].site
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
            frame_indices = np.array([f.index for f in frames])
            positions = np.flatnonzero(included)
            shifts = np.zeros((len(frames), 2), dtype=float)
            registration_rows, pass1_rows, regions, maps = [], [], [], {}
            progress(f"{site}: rendering raw image histograms")
            intermediate_html = raw_histogram_examples(out, stack, included, frame_indices, audit[0]["dtype"], levels)
            if config.registration == "affine":
                from .pair_diagnostics import diagnose_pairs
                from .pair_report import pair_report

                pairs = diagnose_pairs(stack, included, frame_indices, levels, out / "pairs",
                                       sigma=config.registration_sigma,
                                       progress=lambda message: progress(f"{site}: {message}"))
                registration_rows = pairs["geometry_rows"]
                shifts = pairs["shifts"]
                registration, brightness = pairs["registration"], pairs["brightness"]
                maps.update(pairs["maps"])
                write_csv(out / "geometry.csv", registration_rows)
                write_csv(out / "target_pairs.csv", pairs["pair_rows"])
                write_csv(out / "pair_regions.csv", pairs["region_rows"])
                write_csv(out / "pair_quantiles.csv", pairs["quantile_rows"])
                write_json(out / "pair_registration.json", {key: pairs[key] for key in
                           ("registration", "brightness", "geometry_rows", "pair_rows", "region_rows", "quantile_rows")})
                progress(f"{site}: rendering raw target-to-input examples")
                intermediate_html += pair_report(out, pairs)
                if registration["translation_failures"]:
                    warnings.append(f"{registration['translation_failures']} translation estimates failed. Those frames "
                                    "remain unshifted in the noise statistics; inspect geometry.csv before interpreting aligned statistics.")
                if registration["pair_failures"]:
                    warnings.append(f"{registration['pair_failures']} two-region pair corrections failed; every pair and "
                                    "its reason remain in target_pairs.csv. Successful geometric and percentile stages are retained; failed stages are grey.")
                if registration["quantile_failures"]:
                    warnings.append(f"{registration['quantile_failures']} full-image percentile brightness fits failed; "
                                    "see quantile_status and quantile_error in target_pairs.csv.")
            elif config.registration == "fit":
                progress(f"{site}: fitting {len(positions)} frames in two passes")
                fit = register_site(stack, included, levels, sigma=config.registration_sigma,
                                    progress=lambda message: progress(f"{site}: {message}"))
                extras, regions = difference_outputs(stack, fit, levels, out / "differences", frame_indices,
                                                     lambda message: progress(f"{site}: {message}"))
                registration_rows = fit_rows(fit["pass2"], None, frame_indices, extras)
                pass1_rows = fit_rows(fit["pass1"], fit["anchor"], frame_indices)
                write_csv(out / "registration.csv", csv_rows(registration_rows))
                write_csv(out / "registration_pass1.csv", csv_rows(pass1_rows))
                write_csv(out / "regions.csv", regions)
                registration, brightness = registration_summary(registration_rows, pass1_rows,
                                                                int(frame_indices[fit["anchor"]]), config.registration_sigma)
                write_json(out / "registration.json", {"registration": registration, "brightness": brightness,
                                                       "pass1": pass1_rows, "pass2": registration_rows})
                for row in registration_rows:
                    # The noise statistics apply the centre shift alone. The fit's (dy, dx) is
                    # the drift; translating a frame by its negative aligns it to the reference.
                    shifts[row["frame_position"]] = (-row["dy_px"], -row["dx_px"])
                maps["reference_mean"] = fit["mean"].astype(np.float32)
                maps["reference_valid"] = fit["mean_valid"]
                progress(f"{site}: rendering intermediate image and brightness-fit examples")
                intermediate_html += intermediate_examples(out, stack, fit, levels, frame_indices, config.registration_sigma)
                warnings.append("Least-squares gain is biased by noise in the moving image; pass-1 contrast suppression "
                                "can propagate into the pass-2 reference. Convergence and small standard errors do not "
                                "establish physical brightness changes. See the intermediate pixel-pair plots.")
                if registration["unconverged_frames"]:
                    warnings.append(f"{registration['unconverged_frames']} frames did not reach the step tolerance "
                                    "(iteration limit, stalled step, or singular system); their numbers are reported as they stand "
                                    "(see converged and termination_reason).")
            else:
                registration, brightness = registration_summary([], [], 0, config.registration_sigma, registration_enabled=False)
                warnings.append("Registration disabled: frames are taken as aligned; specimen motion can inflate noise statistics.")
            indices = frame_indices[positions]
            # The same crop is valid for both nearest-integer and linear shifts.
            combined_shifts = np.concatenate((shifts[included], np.rint(shifts[included]), np.zeros((1, 2))))
            crop = common_crop(stack.shape[1:], combined_shifts, margin=2 if config.registration != "none" else 0)
            summaries, frame_metrics = {}, {}
            for mode, integer in (("native", True), ("aligned", False)):
                progress(f"{site}: measuring {mode} noise, spectra, and temporal stability")
                result, mode_maps, rows = analyze_mode(stack, shifts, included, indices, crop, integer, levels, config, interval, irregular)
                summaries[mode] = result
                maps.update({f"{mode}_{key}": value for key, value in mode_maps.items()})
                frame_metrics[mode] = rows
                write_csv(out / f"{mode}_intensity_bins.csv", result["intensity_bins"])
                write_csv(out / f"{mode}_averaging.csv", result["temporal"]["averaging"])
                write_csv(out / f"{mode}_temporal_acf.csv", result["temporal"]["acf"])
                write_csv(out / f"{mode}_spatial_acf.csv", result["spatial"]["acf"])
            if any((r["low_clip_fraction"] or 0) + (r["high_clip_fraction"] or 0) > 0.001 for r in audit):
                warnings.append("Some frames have >0.1% pixels at configured/storage clipping bounds. Clipped pixels are excluded from the registration fit and from noise-distribution masks.")
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
            by_position = {int(i): dict(audit[i]) for i in range(len(frames))}
            for row in registration_rows:
                by_position[row["frame_position"]].update({key: row[key] for key in FRAME_TABLE_KEYS if key in row})
            for mode, rows in frame_metrics.items():
                for row in rows:
                    by_position[row["frame_position"]].update({f"{mode}_{key}": value for key, value in row.items() if key != "frame_position"})
            for row in by_position.values():
                row.pop("metadata", None)
            write_csv(out / "frames.csv", list(by_position.values()))
            stamps = np.array([frames[i].timestamp_s if frames[i].timestamp_s is not None else frames[i].index * (interval or 1) for i in positions])
            stamps -= stamps[0]
            means = np.array([r["mean_dn"] for r in frame_metrics["aligned"]])
            fractions = shifts[included] - np.floor(shifts[included])
            variance_factors = np.prod((1 - fractions)**2 + fractions**2, axis=1)
            summary = {"site": site, "status": "complete", "input_frames": len(frames),
                       "accepted_frames": int(included.sum()), "dtype": audit[0]["dtype"],
                       "original_shape": [audit[0]["height"], audit[0]["width"]],
                       "roi_y0_y1_x0_x1": list(config.roi) if config.roi else [0, stack.shape[1], 0, stack.shape[2]],
                       "common_crop_within_roi": [crop[0].start, crop[0].stop, crop[1].start, crop[1].stop],
                       "black_level_dn": levels[0], "white_level_dn": levels[1],
                       "interval_s": interval, "duration_s": float(stamps[-1]) if frames[0].timestamp_s is not None or interval else None,
                       "pixel_size_nm": config.pixel_size_nm,
                       "max_drift_px": registration.get("max_drift_px"),
                       "drift_step_rms_px": registration.get("drift_step_rms_px"),
                       "max_corner_effect_px": registration.get("max_corner_effect_px"),
                       "brightness_slope_dn_per_unit": float(np.polyfit(stamps, means, 1)[0]),
                       "brightness_slope_time_unit": "second" if frames[0].timestamp_s is not None or interval else "frame",
                       "bilinear_white_noise_variance_factor_mean": float(variance_factors.mean()),
                       "unregistered_temporal_sigma_dn": float(np.sqrt(np.mean(unregistered_m2 / (len(positions) - 1)))),
                       "modes": summaries, "registration": registration, "brightness": brightness, "warnings": warnings}
            np.savez_compressed(out / "maps.npz", **maps)
            write_json(out / "summary.json", summary)
            progress(f"{site}: writing report")
            site_report(out, summary, maps, list(by_position.values()), registration_rows, regions, intermediate_html)
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
    overview = {"schema_version": 2, "status": "complete" if all(r["status"] == "complete" for r in results) else "partial_failure",
                "site_count": len(results), "successful_sites": sum(r["status"] == "complete" for r in results),
                "cross_site_duplicate_groups": duplicates,
                "warnings": ["Noise parameters are reported per site. Different patterns/settings are not pooled.",
                             "128 repeats characterize only their acquisition duration; longer-term stability needs repeated sessions.",
                             "Stable detector fixed-pattern noise cannot be separated from specimen structure without dark/flat references."],
                "sites": results}
    write_json(output / "input_manifest.json", input_manifest)
    write_json(output / "summary.json", overview)
    write_csv(output / "summary.csv", [{"site": r["site"], "status": r["status"], "directory": r["directory"],
                                       "frames": r.get("accepted_frames"), "max_drift_px": r.get("max_drift_px"),
                                       "max_corner_effect_px": r.get("max_corner_effect_px"),
                                       "gain_min": r.get("brightness", {}).get("gain_min"),
                                       "gain_max": r.get("brightness", {}).get("gain_max"),
                                       "quantile_gain_min": r.get("brightness", {}).get("quantile_gain_min"),
                                       "quantile_gain_max": r.get("brightness", {}).get("quantile_gain_max"),
                                       "native_flat_sigma_dn": r.get("modes", {}).get("native", {}).get("flat_temporal_sigma_dn"),
                                       "aligned_flat_sigma_dn": r.get("modes", {}).get("aligned", {}).get("flat_temporal_sigma_dn"),
                                       "error": r.get("error", "")}
                                      for r in results])
    index_report(output, overview)
    return overview
